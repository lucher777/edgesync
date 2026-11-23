"""Independent double-precision Python reference implementation for EdgeSync.

Implements the exact origin-centered Huber-weighted Recursive Least Squares
(H-RLS) clock-drift estimator, power-of-two static ring buffer semantics, and
4-tap Lagrange cubic Farrow fractional-delay resampler for bit-level (<1e-12)
verification against the compiled freestanding ANSI C11 microkernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ReferenceConfig:
    """Configuration parameters for the Python reference EdgeSync solver.

    Attributes:
        fs_in: Nominal input sampling frequency in Hz.
        lambda_: RLS forgetting factor in ``(0.5, 1.0]``.
        huber_delta: Huber outlier residual threshold in seconds.
        max_drift_ppm: Physical crystal frequency drift clamp in ppm.
        init_p_offset: Initial inverse-information offset variance.
        init_p_drift: Initial inverse-information drift variance.
        gap_factor: Packet-drop gap threshold as a multiple of ``1 / fs_in``.
        ring_cap: Power-of-two ring buffer capacity (default 512).
    """

    fs_in: float = 64.0
    lambda_: float = 0.995
    huber_delta: float = 0.015
    max_drift_ppm: float = 500.0
    init_p_offset: float = 100.0
    init_p_drift: float = 0.05
    gap_factor: float = 1.5
    ring_cap: int = 512


def farrow_cubic_4tap_py(
    xm1: float, x0: float, x1: float, x2: float, mu: float
) -> float:
    """Evaluate the 4-tap Lagrange cubic Farrow polynomial in Horner form.

    Computes the unique degree-3 polynomial interpolating ``(-1, xm1)``,
    ``(0, x0)``, ``(1, x1)``, and ``(2, x2)`` at fractional delay
    ``mu`` in ``[0, 1]``.

    Args:
        xm1: Sample amplitude at index ``k - 1``.
        x0: Sample amplitude at index ``k``.
        x1: Sample amplitude at index ``k + 1``.
        x2: Sample amplitude at index ``k + 2``.
        mu: Fractional delay in ``[0, 1]``.

    Returns:
        Interpolated double-precision amplitude at ``k + mu``.
    """
    d01 = x1 - x0
    c2 = 0.5 * (xm1 + x1) - x0
    c3 = (1.0 / 6.0) * (x2 - xm1) - 0.5 * d01
    c1 = d01 - c2 - c3
    return ((c3 * mu + c2) * mu + c1) * mu + x0


class ReferenceEdgeSync:
    """Pure-Python exact mathematical reference for the EdgeSync microkernel.

    Mirrors the exact fixed-capacity ring semantics, Huber-weighted RLS
    recursion with effective forgetting, and 4-tap cubic Farrow resampling of
    ``csrc/edgesync.c`` for independent differential verification.
    """

    def __init__(self, cfg: ReferenceConfig | None = None) -> None:
        """Initialize the reference microkernel state.

        Args:
            cfg: Optional :class:`ReferenceConfig`; defaults to 64 Hz BLE config.
        """
        self.cfg = cfg or ReferenceConfig()
        if self.cfg.ring_cap < 8 or (self.cfg.ring_cap & (self.cfg.ring_cap - 1)) != 0:
            raise ValueError("ring_cap must be a power of two >= 8")
        if self.cfg.fs_in <= 0.0 or not math.isfinite(self.cfg.fs_in):
            raise ValueError("fs_in must be finite and positive")
        self._mask = self.cfg.ring_cap - 1
        self._h_nom = 1.0 / self.cfg.fs_in
        self.reset()

    def reset(self) -> None:
        """Clear ring storage, clock parameters, and counters."""
        self.ring_t = np.zeros(self.cfg.ring_cap, dtype=np.float64)
        self.ring_x = np.zeros(self.cfg.ring_cap, dtype=np.float64)
        self.write_seq = 0
        self.read_seq = 0
        self.t_loc_origin = 0.0
        self.t_ref_origin = 0.0
        self.delta0 = 0.0
        self.alpha = 0.0
        self.p00 = float(self.cfg.init_p_offset)
        self.p01 = 0.0
        self.p11 = float(self.cfg.init_p_drift)
        self.last_residual = 0.0
        self.last_weight = 1.0
        self.n_sync = 0
        self.n_outliers = 0
        self.initialized = False
        self.n_farrow_cubic = 0
        self.n_gap_linear = 0

    def update_sync(self, t_loc: float, t_ref: float) -> int:
        """Process one asynchronous synchronization beacon pair in O(1) time.

        Args:
            t_loc: Local wearable node timestamp (seconds).
            t_ref: Host/reference clock timestamp (seconds).

        Returns:
            Status code ``0`` on success or ``-2`` on invalid input.
        """
        if not (math.isfinite(t_loc) and math.isfinite(t_ref)):
            return -2
        if not self.initialized:
            self.t_loc_origin = float(t_loc)
            self.t_ref_origin = float(t_ref)
            self.delta0 = 0.0
            self.alpha = 0.0
            self.p00 = float(self.cfg.init_p_offset)
            self.p01 = 0.0
            self.p11 = float(self.cfg.init_p_drift)
            self.last_residual = 0.0
            self.last_weight = 1.0
            self.n_sync = 1
            self.n_outliers = 0
            self.initialized = True
            return 0

        u = float(t_loc) - self.t_loc_origin
        y = (float(t_ref) - self.t_ref_origin) - u
        y_hat = self.delta0 + self.alpha * u
        err = y - y_hat
        abs_err = abs(err)

        w = 1.0
        if self.n_sync >= 3 and abs_err > self.cfg.huber_delta:
            self.n_outliers += 1
            if abs_err > 3.0 * self.cfg.huber_delta:
                self.last_residual = err
                self.last_weight = 0.0
                self.n_sync += 1
                return 0
            w = self.cfg.huber_delta / abs_err

        lam_eff = 1.0 - w * (1.0 - self.cfg.lambda_)
        inv_lam = 1.0 / lam_eff

        v0 = self.p00 + self.p01 * u
        v1 = self.p01 + self.p11 * u
        denom = lam_eff + w * (v0 + v1 * u)
        if denom <= 1e-18 or not math.isfinite(denom):
            return -2

        gain_scale = w / denom
        k0 = gain_scale * v0
        k1 = gain_scale * v1

        max_alpha = self.cfg.max_drift_ppm * 1e-6
        alpha_raw = self.alpha + k1 * err
        alpha_clamped = min(max(alpha_raw, -max_alpha), max_alpha)
        if alpha_clamped == alpha_raw:
            self.delta0 += k0 * err
            self.alpha = alpha_raw
        else:
            d_alpha = alpha_clamped - self.alpha
            step_y = (k0 + k1 * u) * err
            self.delta0 += step_y - d_alpha * u
            self.alpha = alpha_clamped

        np00 = (self.p00 - k0 * v0) * inv_lam
        np01 = (self.p01 - k0 * v1) * inv_lam
        np11 = (self.p11 - k1 * v1) * inv_lam

        np00 = min(max(np00, 1e-15), self.cfg.init_p_offset * 10.0)
        np11 = min(max(np11, 1e-18), self.cfg.init_p_drift * 10.0)
        if (np01 * np01) > 0.999999 * np00 * np11:
            np01 *= 0.95

        self.p00 = np00
        self.p01 = np01
        self.p11 = np11
        self.last_residual = err
        self.last_weight = w
        self.n_sync += 1
        return 0

    def loc_to_ref(self, t_loc: float) -> float:
        """Map local timestamp to reference timeline.

        Args:
            t_loc: Local clock timestamp (seconds).

        Returns:
            Estimated reference timestamp (seconds).
        """
        if not self.initialized:
            return float(t_loc)
        return self.t_ref_origin + self.delta0 + (1.0 + self.alpha) * (
            float(t_loc) - self.t_loc_origin
        )

    def ref_to_loc(self, t_ref: float) -> float:
        """Map reference timestamp to local node timeline.

        Args:
            t_ref: Target reference timestamp (seconds).

        Returns:
            Estimated local node timestamp (seconds).
        """
        if not self.initialized:
            return float(t_ref)
        return self.t_loc_origin + (
            float(t_ref) - self.t_ref_origin - self.delta0
        ) / (1.0 + self.alpha)

    @property
    def drift_ppm(self) -> float:
        """Return current estimated relative crystal drift in ppm."""
        return self.alpha * 1e6 if self.initialized else 0.0

    @property
    def offset_s(self) -> float:
        """Return current estimated clock offset at ``t_loc = 0`` in seconds."""
        if not self.initialized:
            return 0.0
        return (
            self.t_ref_origin
            + self.delta0
            - (1.0 + self.alpha) * self.t_loc_origin
        )

    def push_sample(self, t_loc: float, x: float) -> int:
        """Append one sample to the fixed-capacity ring buffer.

        Args:
            t_loc: Strictly increasing local timestamp (seconds).
            x: Sensor amplitude.

        Returns:
            Status code ``0`` on success, ``-2`` on non-finite input, or ``-5``
            on non-monotonic timestamp.
        """
        if not (math.isfinite(t_loc) and math.isfinite(x)):
            return -2
        if self.write_seq > 0:
            prev_slot = (self.write_seq - 1) & self._mask
            if t_loc <= self.ring_t[prev_slot]:
                return -5
        slot = self.write_seq & self._mask
        self.ring_t[slot] = float(t_loc)
        self.ring_x[slot] = float(x)
        self.write_seq += 1
        return 0

    def _find_bracket(self, oldest: int, newest: int, t_loc: float) -> int:
        """Locate the bracketing ring sequence index using bounded search."""
        cur = min(max(self.read_seq, oldest), newest - 1)
        if self.ring_t[cur & self._mask] <= t_loc <= self.ring_t[(cur + 1) & self._mask]:
            return cur
        if cur + 2 <= newest:
            nxt = cur + 1
            if (
                self.ring_t[nxt & self._mask]
                <= t_loc
                <= self.ring_t[(nxt + 1) & self._mask]
            ):
                return nxt
        lo, hi = oldest, newest - 1
        for _ in range(16):
            if lo >= hi:
                break
            mid = lo + ((hi - lo) >> 1)
            if self.ring_t[(mid + 1) & self._mask] < t_loc:
                lo = mid + 1
            elif self.ring_t[mid & self._mask] > t_loc:
                hi = mid
            else:
                return mid
        return lo

    def resample_at(self, t_ref: float) -> Tuple[int, float]:
        """Resample buffered stream at target reference timestamp ``t_ref``.

        Args:
            t_ref: Target reference timestamp (seconds).

        Returns:
            Tuple ``(status, value)`` matching the C11 microkernel status codes.
        """
        if not math.isfinite(t_ref):
            return -2, 0.0
        if self.write_seq < 2:
            return -3, 0.0

        t_loc = self.ref_to_loc(t_ref)
        newest = self.write_seq - 1
        oldest = max(0, self.write_seq - self.cfg.ring_cap)
        if t_loc < self.ring_t[oldest & self._mask]:
            return -3, 0.0
        if t_loc > self.ring_t[newest & self._mask]:
            return -4, 0.0

        idx = self._find_bracket(oldest, newest, t_loc)
        self.read_seq = idx

        s0 = idx & self._mask
        s1 = (idx + 1) & self._mask
        t0 = float(self.ring_t[s0])
        t1 = float(self.ring_t[s1])
        dt = t1 - t0
        mu = min(max((t_loc - t0) / dt, 0.0), 1.0) if dt > 0.0 else 0.0
        gap_lim = self.cfg.gap_factor * self._h_nom

        if idx > oldest and (idx + 2) <= newest:
            sm1 = (idx - 1) & self._mask
            s2 = (idx + 2) & self._mask
            dt_left = t0 - float(self.ring_t[sm1])
            dt_right = float(self.ring_t[s2]) - t1
            if dt <= gap_lim and dt_left <= gap_lim and dt_right <= gap_lim:
                val = farrow_cubic_4tap_py(
                    float(self.ring_x[sm1]),
                    float(self.ring_x[s0]),
                    float(self.ring_x[s1]),
                    float(self.ring_x[s2]),
                    mu,
                )
                self.n_farrow_cubic += 1
                return 0, val
            self.n_gap_linear += 1
            return 1, float(self.ring_x[s0]) + mu * float(
                self.ring_x[s1] - self.ring_x[s0]
            )

        self.n_gap_linear += 1
        return 2, float(self.ring_x[s0]) + mu * float(
            self.ring_x[s1] - self.ring_x[s0]
        )
