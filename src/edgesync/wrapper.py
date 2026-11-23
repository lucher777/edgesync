"""Thin ``ctypes`` wrapper for the compiled freestanding ANSI C11 EdgeSync library.

Loads ``libedgesync`` (auto-building from ``csrc/edgesync.c`` with ``gcc`` when
absent), binds exact C11 struct layouts without heap allocation in the kernel,
and exposes high-level Pythonic classes :class:`EdgeSyncKernel` (FP64) and
:class:`EdgeSyncKernelF32` (FP32).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Optional, Tuple

import numpy as np

ES_RING_CAP = 512

ES_OK = 0
ES_OK_GAP_LINEAR = 1
ES_OK_EDGE_LINEAR = 2
ES_ERR_NULL = -1
ES_ERR_PARAM = -2
ES_ERR_UNDERRUN = -3
ES_ERR_OVERRUN = -4
ES_ERR_NONMONOTONIC = -5


class _EsConfigC(ctypes.Structure):
    """Exact ctypes mirror of C11 ``es_config_t``."""

    _fields_ = [
        ("fs_in", ctypes.c_double),
        ("lambda_", ctypes.c_double),
        ("huber_delta", ctypes.c_double),
        ("max_drift_ppm", ctypes.c_double),
        ("init_p_offset", ctypes.c_double),
        ("init_p_drift", ctypes.c_double),
        ("gap_factor", ctypes.c_double),
    ]


class _EsClockC(ctypes.Structure):
    """Exact ctypes mirror of C11 ``es_clock_t``."""

    _fields_ = [
        ("t_loc_origin", ctypes.c_double),
        ("t_ref_origin", ctypes.c_double),
        ("delta0", ctypes.c_double),
        ("alpha", ctypes.c_double),
        ("p00", ctypes.c_double),
        ("p01", ctypes.c_double),
        ("p11", ctypes.c_double),
        ("last_residual", ctypes.c_double),
        ("last_weight", ctypes.c_double),
        ("n_sync", ctypes.c_uint64),
        ("n_outliers", ctypes.c_uint64),
        ("initialized", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8 * 7),
    ]


class _EsStatsC(ctypes.Structure):
    """Exact ctypes mirror of C11 ``es_stats_t``."""

    _fields_ = [
        ("n_pushed", ctypes.c_uint64),
        ("n_resampled", ctypes.c_uint64),
        ("n_farrow_cubic", ctypes.c_uint64),
        ("n_gap_linear", ctypes.c_uint64),
        ("n_underrun", ctypes.c_uint64),
        ("n_overrun", ctypes.c_uint64),
        ("ring_capacity", ctypes.c_uint32),
        ("kernel_bytes", ctypes.c_uint32),
    ]


class _EsKernelC(ctypes.Structure):
    """Exact ctypes mirror of C11 ``es_kernel_t``."""

    _fields_ = [
        ("cfg", _EsConfigC),
        ("clk", _EsClockC),
        ("stats", _EsStatsC),
        ("ring_t", ctypes.c_double * ES_RING_CAP),
        ("ring_x", ctypes.c_double * ES_RING_CAP),
        ("write_seq", ctypes.c_uint64),
        ("read_seq", ctypes.c_uint64),
        ("h_nom", ctypes.c_double),
    ]


class _EsKernelF32C(ctypes.Structure):
    """Exact ctypes mirror of C11 ``es_kernel_f32_t``."""

    _fields_ = [
        ("cfg", _EsConfigC),
        ("clk", _EsClockC),
        ("stats", _EsStatsC),
        ("t_base", ctypes.c_double * 2),
        ("ring_dt", ctypes.c_float * ES_RING_CAP),
        ("ring_x", ctypes.c_float * ES_RING_CAP),
        ("write_seq", ctypes.c_uint64),
        ("read_seq", ctypes.c_uint64),
        ("h_nom", ctypes.c_double),
    ]


@dataclass(frozen=True)
class EdgeSyncConfig:
    """User-facing configuration for the C11 EdgeSync microkernel.

    Attributes:
        fs_in: Nominal input sampling frequency in Hz.
        lambda_: RLS forgetting factor in ``(0.5, 1.0]``.
        huber_delta: Huber outlier residual threshold in seconds.
        max_drift_ppm: Physical crystal frequency drift clamp in ppm.
        init_p_offset: Initial inverse-information offset variance.
        init_p_drift: Initial inverse-information drift variance.
        gap_factor: Packet-drop gap threshold multiple of nominal period.
    """

    fs_in: float = 64.0
    lambda_: float = 0.995
    huber_delta: float = 0.015
    max_drift_ppm: float = 500.0
    init_p_offset: float = 100.0
    init_p_drift: float = 0.05
    gap_factor: float = 1.5

    def to_c(self) -> _EsConfigC:
        """Convert this configuration into its C11 ``es_config_t`` structure.

        Returns:
            Populated :class:`_EsConfigC` instance.
        """
        return _EsConfigC(
            fs_in=self.fs_in,
            lambda_=self.lambda_,
            huber_delta=self.huber_delta,
            max_drift_ppm=self.max_drift_ppm,
            init_p_offset=self.init_p_offset,
            init_p_drift=self.init_p_drift,
            gap_factor=self.gap_factor,
        )


def _locate_csrc_dir() -> Path:
    """Return the package ``csrc`` directory containing ``edgesync.c``."""
    pkg_root = Path(__file__).resolve().parents[2]
    csrc = pkg_root / "csrc"
    if (csrc / "edgesync.c").is_file():
        return csrc
    alt = Path(__file__).resolve().parent / "csrc"
    if (alt / "edgesync.c").is_file():
        return alt
    raise FileNotFoundError("Could not locate csrc/edgesync.c in package tree.")


def _shared_lib_filename() -> str:
    """Return the platform-specific compiled shared library filename."""
    sys_name = platform.system()
    if sys_name == "Windows":
        return "libedgesync.dll"
    if sys_name == "Darwin":
        return "libedgesync.dylib"
    return "libedgesync.so"


def _find_gcc_executable() -> Optional[str]:
    """Locate a working ``gcc`` compiler binary on PATH or standard locations."""
    env_cc = os.environ.get("EDGESYNC_CC")
    if env_cc and Path(env_cc).is_file():
        return env_cc
    found = shutil.which("gcc")
    if found:
        return found
    if platform.system() == "Windows":
        for candidate in (
            Path("C:/msys64/ucrt64/bin/gcc.exe"),
            Path("C:/msys64/mingw64/bin/gcc.exe"),
            Path("C:/msys64/usr/bin/gcc.exe"),
        ):
            if candidate.is_file():
                return str(candidate)
    return None


def compile_native_library(force: bool = False) -> Path:
    """Compile the freestanding C11 microkernel into a shared library.

    Runs ``gcc -std=c11 -O3 -Wall -Wextra -Werror -ffreestanding -fno-builtin``
    from inside ``csrc/`` so zero libc/heap symbols are linked.

    Args:
        force: If True, rebuild even when the shared library already exists.

    Returns:
        Path to the compiled shared library.
    """
    csrc = _locate_csrc_dir()
    lib_path = csrc / _shared_lib_filename()
    if lib_path.is_file() and not force:
        return lib_path

    gcc = _find_gcc_executable()
    if gcc is None:
        raise RuntimeError(
            "No C11 compiler (gcc) found. Set EDGESYNC_CC or install gcc."
        )

    cmd = [
        gcc,
        "-std=c11",
        "-O3",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-ffreestanding",
        "-fno-builtin",
    ]
    sys_name = platform.system()
    if sys_name == "Windows":
        cmd.extend(
            ["-shared", "-nostdlib", "-Wl,-e,0", "-Wl,--export-all-symbols"]
        )
    elif sys_name == "Darwin":
        cmd.extend(["-dynamiclib", "-fPIC"])
    else:
        cmd.extend(["-shared", "-fPIC"])
    cmd.extend(["edgesync.c", "-o", lib_path.name])

    env = os.environ.copy()
    gcc_dir = str(Path(gcc).parent)
    env["PATH"] = gcc_dir + os.pathsep + env.get("PATH", "")

    proc = subprocess.run(
        cmd,
        cwd=str(csrc),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"C11 compilation failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return lib_path


_LIB_CACHE: Optional[ctypes.CDLL] = None


def _configure_signatures(lib: ctypes.CDLL) -> None:
    """Bind exact C11 argument and return types on the loaded CDLL handle."""
    lib.es_sizeof_kernel.argtypes = []
    lib.es_sizeof_kernel.restype = ctypes.c_size_t
    lib.es_sizeof_kernel_f32.argtypes = []
    lib.es_sizeof_kernel_f32.restype = ctypes.c_size_t
    lib.es_ring_capacity.argtypes = []
    lib.es_ring_capacity.restype = ctypes.c_uint32
    lib.es_default_config.argtypes = [ctypes.c_double]
    lib.es_default_config.restype = _EsConfigC

    lib.es_init.argtypes = [ctypes.POINTER(_EsKernelC), ctypes.POINTER(_EsConfigC)]
    lib.es_init.restype = ctypes.c_int32
    lib.es_reset.argtypes = [ctypes.POINTER(_EsKernelC)]
    lib.es_reset.restype = None
    lib.es_update_sync.argtypes = [
        ctypes.POINTER(_EsKernelC),
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.es_update_sync.restype = ctypes.c_int32
    lib.es_loc_to_ref.argtypes = [ctypes.POINTER(_EsKernelC), ctypes.c_double]
    lib.es_loc_to_ref.restype = ctypes.c_double
    lib.es_ref_to_loc.argtypes = [ctypes.POINTER(_EsKernelC), ctypes.c_double]
    lib.es_ref_to_loc.restype = ctypes.c_double
    lib.es_get_drift_ppm.argtypes = [ctypes.POINTER(_EsKernelC)]
    lib.es_get_drift_ppm.restype = ctypes.c_double
    lib.es_get_offset_s.argtypes = [ctypes.POINTER(_EsKernelC)]
    lib.es_get_offset_s.restype = ctypes.c_double
    lib.es_push_sample.argtypes = [
        ctypes.POINTER(_EsKernelC),
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.es_push_sample.restype = ctypes.c_int32
    lib.es_farrow_cubic_4tap.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.es_farrow_cubic_4tap.restype = ctypes.c_double
    lib.es_resample_at.argtypes = [
        ctypes.POINTER(_EsKernelC),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.es_resample_at.restype = ctypes.c_int32
    lib.es_resample_batch.argtypes = [
        ctypes.POINTER(_EsKernelC),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.es_resample_batch.restype = ctypes.c_int32

    lib.es_f32_init.argtypes = [
        ctypes.POINTER(_EsKernelF32C),
        ctypes.POINTER(_EsConfigC),
    ]
    lib.es_f32_init.restype = ctypes.c_int32
    lib.es_f32_update_sync.argtypes = [
        ctypes.POINTER(_EsKernelF32C),
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.es_f32_update_sync.restype = ctypes.c_int32
    lib.es_f32_push_sample.argtypes = [
        ctypes.POINTER(_EsKernelF32C),
        ctypes.c_double,
        ctypes.c_float,
    ]
    lib.es_f32_push_sample.restype = ctypes.c_int32
    lib.es_farrow_cubic_4tap_f32.argtypes = [
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.es_farrow_cubic_4tap_f32.restype = ctypes.c_float
    lib.es_f32_resample_at.argtypes = [
        ctypes.POINTER(_EsKernelF32C),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.es_f32_resample_at.restype = ctypes.c_int32
    lib.es_f32_resample_batch.argtypes = [
        ctypes.POINTER(_EsKernelF32C),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.es_f32_resample_batch.restype = ctypes.c_int32


def load_native_library(auto_build: bool = True) -> ctypes.CDLL:
    """Load and cache the compiled C11 EdgeSync shared library.

    Args:
        auto_build: Compile from ``csrc/edgesync.c`` automatically if missing.

    Returns:
        Configured :class:`ctypes.CDLL` handle with validated ABI sizes.
    """
    global _LIB_CACHE
    if _LIB_CACHE is not None:
        return _LIB_CACHE

    csrc = _locate_csrc_dir()
    lib_path = csrc / _shared_lib_filename()
    if not lib_path.is_file():
        if not auto_build:
            raise FileNotFoundError(f"Compiled library not found: {lib_path.name}")
        lib_path = compile_native_library(force=False)

    lib = ctypes.CDLL(str(lib_path))
    _configure_signatures(lib)

    if lib.es_sizeof_kernel() != ctypes.sizeof(_EsKernelC):
        raise RuntimeError(
            "ABI mismatch between C11 es_kernel_t and Python ctypes mirror."
        )
    if lib.es_sizeof_kernel_f32() != ctypes.sizeof(_EsKernelF32C):
        raise RuntimeError(
            "ABI mismatch between C11 es_kernel_f32_t and Python ctypes mirror."
        )

    _LIB_CACHE = lib
    return lib


def farrow_cubic_4tap(
    xm1: float, x0: float, x1: float, x2: float, mu: float
) -> float:
    """Call the compiled C11 branch-free 4-tap Lagrange cubic Farrow kernel.

    Args:
        xm1: Sample amplitude at index ``k - 1``.
        x0: Sample amplitude at index ``k``.
        x1: Sample amplitude at index ``k + 1``.
        x2: Sample amplitude at index ``k + 2``.
        mu: Fractional delay in ``[0, 1]``.

    Returns:
        Interpolated double-precision amplitude computed by C11.
    """
    lib = load_native_library()
    return float(lib.es_farrow_cubic_4tap(xm1, x0, x1, x2, mu))


class EdgeSyncKernel:
    """Double-precision C11 static-memory streaming synchronization microkernel.

    Holds one fixed-size ``es_kernel_t`` buffer (~8.4 KB) and delegates every
    streaming operation directly to the zero-heap freestanding C11 library.
    """

    def __init__(self, cfg: EdgeSyncConfig | None = None) -> None:
        """Allocate and initialize a static ``es_kernel_t`` instance.

        Args:
            cfg: Optional :class:`EdgeSyncConfig`; defaults to 64 Hz BLE config.
        """
        self._lib = load_native_library()
        self.cfg = cfg or EdgeSyncConfig()
        self._state = _EsKernelC()
        c_cfg = self.cfg.to_c()
        rc = int(self._lib.es_init(ctypes.byref(self._state), ctypes.byref(c_cfg)))
        if rc != ES_OK:
            raise ValueError(f"es_init rejected configuration with status {rc}")

    @property
    def kernel_bytes(self) -> int:
        """Return exact C11 ``sizeof(es_kernel_t)`` in bytes."""
        return int(self._lib.es_sizeof_kernel())

    @property
    def ring_capacity(self) -> int:
        """Return compile-time ring capacity ``ES_RING_CAP``."""
        return int(self._lib.es_ring_capacity())

    @property
    def drift_ppm(self) -> float:
        """Return current estimated relative crystal drift in ppm."""
        return float(self._lib.es_get_drift_ppm(ctypes.byref(self._state)))

    @property
    def offset_s(self) -> float:
        """Return current estimated clock offset at ``t_loc = 0`` in seconds."""
        return float(self._lib.es_get_offset_s(ctypes.byref(self._state)))

    @property
    def n_sync(self) -> int:
        """Return number of processed synchronization beacons."""
        return int(self._state.clk.n_sync)

    @property
    def n_outliers(self) -> int:
        """Return number of Huber-downweighted synchronization outliers."""
        return int(self._state.clk.n_outliers)

    @property
    def stats(self) -> dict[str, int]:
        """Return a dictionary snapshot of deterministic C11 telemetry counters."""
        st = self._state.stats
        return {
            "n_pushed": int(st.n_pushed),
            "n_resampled": int(st.n_resampled),
            "n_farrow_cubic": int(st.n_farrow_cubic),
            "n_gap_linear": int(st.n_gap_linear),
            "n_underrun": int(st.n_underrun),
            "n_overrun": int(st.n_overrun),
            "ring_capacity": int(st.ring_capacity),
            "kernel_bytes": int(st.kernel_bytes),
        }

    def reset(self) -> None:
        """Reset ring buffer, clock state, and telemetry while keeping config."""
        self._lib.es_reset(ctypes.byref(self._state))

    def update_sync(self, t_loc: float, t_ref: float) -> int:
        """Incorporate one synchronization beacon pair in O(1) time.

        Args:
            t_loc: Local wearable node timestamp (seconds).
            t_ref: Reference timeline timestamp (seconds).

        Returns:
            C11 status code (``ES_OK`` on success).
        """
        return int(self._lib.es_update_sync(ctypes.byref(self._state), t_loc, t_ref))

    def loc_to_ref(self, t_loc: float) -> float:
        """Map local timestamp to reference timeline.

        Args:
            t_loc: Local node timestamp (seconds).

        Returns:
            Estimated reference timestamp (seconds).
        """
        return float(self._lib.es_loc_to_ref(ctypes.byref(self._state), t_loc))

    def ref_to_loc(self, t_ref: float) -> float:
        """Map target reference timestamp to local clock.

        Args:
            t_ref: Target reference timestamp (seconds).

        Returns:
            Estimated local node timestamp (seconds).
        """
        return float(self._lib.es_ref_to_loc(ctypes.byref(self._state), t_ref))

    def push_sample(self, t_loc: float, x: float) -> int:
        """Push one incoming sensor sample into the static ring buffer.

        Args:
            t_loc: Monotonically increasing local sample timestamp (seconds).
            x: Double-precision sensor amplitude.

        Returns:
            C11 status code (``ES_OK`` on success).
        """
        return int(self._lib.es_push_sample(ctypes.byref(self._state), t_loc, x))

    def resample_at(self, t_ref: float) -> Tuple[int, float]:
        """Resample the buffered stream at target reference timestamp ``t_ref``.

        Args:
            t_ref: Target reference timestamp (seconds).

        Returns:
            Tuple ``(status, value)`` with C11 status code and resampled value.
        """
        out = ctypes.c_double(0.0)
        rc = int(
            self._lib.es_resample_at(
                ctypes.byref(self._state), t_ref, ctypes.byref(out)
            )
        )
        return rc, float(out.value)

    def resample_batch(
        self, t_ref_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample a contiguous 1-D NumPy array of reference timestamps in C11.

        Args:
            t_ref_arr: 1-D float64 array of target reference timestamps.

        Returns:
            Tuple ``(values, statuses)`` as float64 and int32 NumPy arrays.
        """
        t_arr = np.ascontiguousarray(t_ref_arr, dtype=np.float64)
        n = int(t_arr.size)
        out_x = np.empty(n, dtype=np.float64)
        out_st = np.empty(n, dtype=np.int32)
        if n > 0:
            self._lib.es_resample_batch(
                ctypes.byref(self._state),
                t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_size_t(n),
                out_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                out_st.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
        return out_x, out_st


class EdgeSyncKernelF32:
    """Single-precision FPU C11 static-memory EdgeSync microkernel (~4.3 KB).

    Uses IEEE-754 float32 sample amplitudes and float32 timestamp offsets from two
    double-precision epoch bases; clock arithmetic remains double precision.
    """

    def __init__(self, cfg: EdgeSyncConfig | None = None) -> None:
        """Allocate and initialize a static ``es_kernel_f32_t`` instance.

        Args:
            cfg: Optional :class:`EdgeSyncConfig`; defaults to 64 Hz BLE config.
        """
        self._lib = load_native_library()
        self.cfg = cfg or EdgeSyncConfig()
        self._state = _EsKernelF32C()
        c_cfg = self.cfg.to_c()
        rc = int(self._lib.es_f32_init(ctypes.byref(self._state), ctypes.byref(c_cfg)))
        if rc != ES_OK:
            raise ValueError(f"es_f32_init rejected configuration with status {rc}")

    @property
    def kernel_bytes(self) -> int:
        """Return exact C11 ``sizeof(es_kernel_f32_t)`` in bytes."""
        return int(self._lib.es_sizeof_kernel_f32())

    def update_sync(self, t_loc: float, t_ref: float) -> int:
        """Incorporate one synchronization beacon pair in O(1) time.

        Args:
            t_loc: Local wearable node timestamp (seconds).
            t_ref: Reference timeline timestamp (seconds).

        Returns:
            C11 status code (``ES_OK`` on success).
        """
        return int(
            self._lib.es_f32_update_sync(ctypes.byref(self._state), t_loc, t_ref)
        )

    def push_sample(self, t_loc: float, x: float) -> int:
        """Push one float32 sensor sample into the static ring buffer.

        Args:
            t_loc: Monotonically increasing local sample timestamp (seconds).
            x: Sensor amplitude converted to float32.

        Returns:
            C11 status code (``ES_OK`` on success).
        """
        return int(
            self._lib.es_f32_push_sample(
                ctypes.byref(self._state), t_loc, ctypes.c_float(float(x))
            )
        )

    def resample_at(self, t_ref: float) -> Tuple[int, float]:
        """Resample the float32 ring buffer at target reference timestamp.

        Args:
            t_ref: Target reference timestamp (seconds).

        Returns:
            Tuple ``(status, value)`` with C11 status code and float32 value.
        """
        out = ctypes.c_float(0.0)
        rc = int(
            self._lib.es_f32_resample_at(
                ctypes.byref(self._state), t_ref, ctypes.byref(out)
            )
        )
        return rc, float(out.value)

    def resample_batch(
        self, t_ref_arr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch-resample a 1-D array of reference timestamps in float32 C11.

        Args:
            t_ref_arr: 1-D float64 array of target reference timestamps.

        Returns:
            Tuple ``(values, statuses)`` as float32 and int32 NumPy arrays.
        """
        t_arr = np.ascontiguousarray(t_ref_arr, dtype=np.float64)
        n = int(t_arr.size)
        out_x = np.empty(n, dtype=np.float32)
        out_st = np.empty(n, dtype=np.int32)
        if n > 0:
            self._lib.es_f32_resample_batch(
                ctypes.byref(self._state),
                t_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_size_t(n),
                out_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out_st.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
        return out_x, out_st
