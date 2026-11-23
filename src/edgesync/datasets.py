"""Signal sources and BLE stream simulator for EdgeSync benchmarks.

Two kinds of reference signals are supported:

* Synthetic PPG and ECG waveforms (:func:`synthetic_ppg`, :func:`synthetic_ecg`),
  generated deterministically in code. They ship with the package and drive the
  demo, quickstart and test suite.
* Excerpts of the WESAD wearable dataset (Schmidt et al., ICMI 2018,
  DOI: 10.1145/3242969.3242985), used for the paper's benchmarks. WESAD is
  licensed for scientific, non-commercial use only, so it is **not**
  redistributed with this MIT-licensed package. Point the loaders at a local
  directory with ``data_dir=...`` or the ``EDGESYNC_WESAD_DIR`` environment
  variable.

:func:`simulate_asynchronous_ble_stream` injects deterministic crystal drift,
beacon jitter, congestion outliers and burst packet loss onto any reference
signal for reproducible streaming benchmarks.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

WESAD_ENV_VAR = "EDGESYNC_WESAD_DIR"
WESAD_URL = "https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/"
_S10_FILE = "wesad_s10_slice.npz"
_EVENTS_FILE = "wesad_5subj_events.json"

PathLike = Union[str, "os.PathLike[str]"]


# --------------------------------------------------------------------------
# Synthetic physiological signals (bundled with the package)
# --------------------------------------------------------------------------


def _beat_times(duration_s: float, hr_bpm: float, hrv_s: float, rng: Any) -> np.ndarray:
    """Return beat onset times with respiratory sinus arrhythmia and random HRV."""
    rr_mean = 60.0 / float(hr_bpm)
    beats = []
    t = -rr_mean
    while t < duration_s + rr_mean:
        rsa = 0.04 * rr_mean * np.sin(2.0 * np.pi * 0.25 * t)
        t += rr_mean + rsa + float(rng.normal(0.0, hrv_s))
        beats.append(t)
    return np.asarray(beats, dtype=np.float64)


def synthetic_ppg(
    duration_s: float = 120.0,
    fs_hz: float = 64.0,
    hr_bpm: float = 72.0,
    noise_std: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Generate a deterministic synthetic photoplethysmogram (PPG/BVP) waveform.

    Each beat is a systolic Gaussian pulse followed by a smaller dicrotic wave,
    with heart-rate variability, respiratory baseline wander and white noise.

    Args:
        duration_s: Signal length in seconds.
        fs_hz: Sampling frequency in Hz.
        hr_bpm: Mean heart rate in beats per minute.
        noise_std: Standard deviation of additive white noise.
        seed: Deterministic RNG seed.

    Returns:
        Float64 array of ``round(duration_s * fs_hz)`` samples.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs_hz))
    t = np.arange(n, dtype=np.float64) / float(fs_hz)
    x = 0.15 * np.sin(2.0 * np.pi * 0.25 * t + 0.3)
    for tb in _beat_times(duration_s, hr_bpm, 0.02, rng):
        amp = 1.0 + float(rng.normal(0.0, 0.05))
        x += amp * np.exp(-0.5 * ((t - tb - 0.18) / 0.07) ** 2)
        x += 0.35 * amp * np.exp(-0.5 * ((t - tb - 0.42) / 0.09) ** 2)
    x += rng.normal(0.0, noise_std, size=n)
    return x


def synthetic_ecg(
    duration_s: float = 60.0,
    fs_hz: float = 250.0,
    hr_bpm: float = 72.0,
    noise_std: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Generate a deterministic synthetic electrocardiogram (ECG) waveform.

    Each beat is a sum of Gaussian P, Q, R, S and T waves, with heart-rate
    variability, baseline wander and white noise.

    Args:
        duration_s: Signal length in seconds.
        fs_hz: Sampling frequency in Hz.
        hr_bpm: Mean heart rate in beats per minute.
        noise_std: Standard deviation of additive white noise.
        seed: Deterministic RNG seed.

    Returns:
        Float64 array of ``round(duration_s * fs_hz)`` samples.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs_hz))
    t = np.arange(n, dtype=np.float64) / float(fs_hz)
    x = 0.05 * np.sin(2.0 * np.pi * 0.3 * t)
    # (relative time s, amplitude, width s) for P, Q, R, S, T
    waves = ((-0.20, 0.12, 0.025), (-0.03, -0.15, 0.010), (0.0, 1.0, 0.012),
             (0.03, -0.25, 0.010), (0.25, 0.30, 0.040))
    for tb in _beat_times(duration_s, hr_bpm, 0.02, rng):
        for dt, a, w in waves:
            x += a * np.exp(-0.5 * ((t - tb - dt) / w) ** 2)
    x += rng.normal(0.0, noise_std, size=n)
    return x


# --------------------------------------------------------------------------
# WESAD excerpts (not redistributed; loaded from a user directory)
# --------------------------------------------------------------------------


def wesad_data_dir(data_dir: Optional[PathLike] = None) -> Path:
    """Resolve the directory that holds the WESAD excerpt files.

    Resolution order: the ``data_dir`` argument, then the ``EDGESYNC_WESAD_DIR``
    environment variable.

    Raises:
        FileNotFoundError: If no directory is configured or it does not exist.
    """
    raw = data_dir if data_dir is not None else os.environ.get(WESAD_ENV_VAR)
    if not raw:
        raise FileNotFoundError(
            "WESAD data are not bundled with edgesync (the dataset is licensed for "
            "scientific, non-commercial use only). Obtain WESAD from "
            f"{WESAD_URL} and pass data_dir=... or set {WESAD_ENV_VAR} to the "
            f"directory containing {_S10_FILE} and {_EVENTS_FILE}."
        )
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"WESAD data directory does not exist: {path}")
    return path


def wesad_available(data_dir: Optional[PathLike] = None) -> bool:
    """Return ``True`` if both WESAD excerpt files can be found."""
    try:
        d = wesad_data_dir(data_dir)
    except FileNotFoundError:
        return False
    return (d / _S10_FILE).is_file() and (d / _EVENTS_FILE).is_file()


def load_wesad_s10_slice(data_dir: Optional[PathLike] = None) -> Dict[str, Any]:
    """Load the 360-second WESAD Subject S10 multimodal excerpt.

    Args:
        data_dir: Directory containing ``wesad_s10_slice.npz``. Defaults to the
            ``EDGESYNC_WESAD_DIR`` environment variable.

    Returns:
        Dictionary containing float64 arrays ``ecg_250hz``, ``bvp_64hz``,
        ``chest_acc_64hz``, ``wrist_acc_64hz``, sampling rates ``fs_ecg_hz``,
        ``fs_bvp_hz``, ``fs_acc_hz``, ``duration_s``, and ``subject_id``.
    """
    path = wesad_data_dir(data_dir) / _S10_FILE
    if not path.is_file():
        raise FileNotFoundError(f"WESAD excerpt not found: {path}")
    with np.load(path, allow_pickle=False) as d:
        return {
            "ecg_250hz": np.asarray(d["ecg_250hz"], dtype=np.float64),
            "bvp_64hz": np.asarray(d["bvp_64hz"], dtype=np.float64),
            "chest_acc_64hz": np.asarray(d["chest_acc_64hz"], dtype=np.float64),
            "wrist_acc_64hz": np.asarray(d["wrist_acc_64hz"], dtype=np.float64),
            "fs_ecg_hz": float(d["fs_ecg_hz"]),
            "fs_bvp_hz": float(d["fs_bvp_hz"]),
            "fs_acc_hz": float(d["fs_acc_hz"]),
            "duration_s": float(d["duration_s"]),
            "subject_id": str(d["subject_id"]),
        }


def load_wesad_5subj_events(data_dir: Optional[PathLike] = None) -> List[Dict[str, Any]]:
    """Load WESAD 5-subject ECG R-peak and PPG peak timelines.

    Args:
        data_dir: Directory containing ``wesad_5subj_events.json``. Defaults to
            the ``EDGESYNC_WESAD_DIR`` environment variable.

    Returns:
        List of 5 subject dictionaries (``S2``, ``S10``, ``S11``, ``S13``,
        ``S16``) containing ``ecg_rpeaks_s``, ``ppg_peaks_s``, and metadata.
    """
    path = wesad_data_dir(data_dir) / _EVENTS_FILE
    if not path.is_file():
        raise FileNotFoundError(f"WESAD events file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return list(payload["records"])


def simulate_asynchronous_ble_stream(
    signal_ref: np.ndarray,
    fs_hz: float,
    drift_ppm: float = 120.0,
    offset_s: float = 1.75,
    beacon_interval_s: float = 1.0,
    beacon_jitter_s: float = 0.002,
    outlier_rate: float = 0.0,
    outlier_delay_s: float = 0.18,
    packet_drop_rate: float = 0.0,
    burst_len: int = 4,
    seed: int = 0,
) -> Dict[str, Any]:
    """Generate a drifted asynchronous BLE wearable stream from a reference signal.

    Resamples ``signal_ref`` onto the wearable node's drifted local crystal
    timeline ``t_ref = (1 + alpha) * t_loc + offset_s`` using exact cubic
    spline interpolation, injects optional BLE packet burst dropouts on the
    sample stream, and synthesizes noisy/outlier-corrupted sync beacons.

    Args:
        signal_ref: Ground-truth reference-clock signal sampled at ``fs_hz``.
        fs_hz: Nominal sampling frequency in Hz.
        drift_ppm: True relative crystal frequency drift in ppm.
        offset_s: True startup clock offset in seconds.
        beacon_interval_s: Nominal synchronization beacon period in seconds.
        beacon_jitter_s: Zero-mean Gaussian BLE beacon arrival jitter std (s).
        outlier_rate: Fraction of beacons delayed by burst congestion spikes.
        outlier_delay_s: Mean BLE congestion delay magnitude for outlier beacons.
        packet_drop_rate: Fraction of sensor samples lost in BLE packet bursts.
        burst_len: Number of consecutive samples dropped per lost BLE packet.
        seed: Deterministic RNG seed.

    Returns:
        Dictionary with local sample arrays (``t_loc``, ``x_loc``),
        synchronization beacons (``beacon_t_loc``, ``beacon_t_ref``),
        ground-truth reference grid (``t_ref_grid``, ``x_ref_true``),
        and exact injected parameters.
    """
    from scipy.interpolate import CubicSpline

    rng = np.random.default_rng(seed)
    y_ref = np.asarray(signal_ref, dtype=np.float64)
    n_ref = y_ref.size
    h = 1.0 / float(fs_hz)
    t_ref_grid = np.arange(n_ref, dtype=np.float64) * h
    spline = CubicSpline(t_ref_grid, y_ref)

    alpha = float(drift_ppm) * 1e-6
    t_start_loc = -float(offset_s) / (1.0 + alpha)
    t_end_loc = (t_ref_grid[-1] - float(offset_s)) / (1.0 + alpha)
    n_loc = int(np.floor((t_end_loc - t_start_loc) * fs_hz))
    t_loc_full = t_start_loc + np.arange(n_loc, dtype=np.float64) * h
    t_ref_at_loc = (1.0 + alpha) * t_loc_full + float(offset_s)
    valid = (t_ref_at_loc >= t_ref_grid[0]) & (t_ref_at_loc <= t_ref_grid[-1])
    t_loc_full = t_loc_full[valid]
    t_ref_at_loc = t_ref_at_loc[valid]
    x_loc_full = spline(t_ref_at_loc)

    # Inject optional burst packet dropouts (keeping first 8 and last 8 samples)
    keep = np.ones(t_loc_full.size, dtype=bool)
    if packet_drop_rate > 0.0 and t_loc_full.size > 32:
        p_start = min(max(packet_drop_rate / max(burst_len, 1), 0.0), 0.5)
        i = 8
        limit = t_loc_full.size - 8
        while i < limit:
            if rng.random() < p_start:
                end = min(i + burst_len, limit)
                keep[i:end] = False
                i = end + 1
            else:
                i += 1

    t_loc = t_loc_full[keep]
    x_loc = x_loc_full[keep]

    # Generate periodic BLE synchronization beacons with Gaussian jitter & bursts
    step_b = max(1, int(round(beacon_interval_s * fs_hz)))
    b_indices = np.arange(0, t_loc_full.size, step_b)
    beacon_t_loc = t_loc_full[b_indices]
    beacon_t_ref = (1.0 + alpha) * beacon_t_loc + float(offset_s)
    if beacon_jitter_s > 0.0:
        beacon_t_ref = beacon_t_ref + rng.normal(
            0.0, beacon_jitter_s, size=beacon_t_loc.size
        )

    is_outlier = np.zeros(beacon_t_loc.size, dtype=bool)
    if outlier_rate > 0.0 and beacon_t_loc.size > 6:
        # Inject bursty asymmetric BLE congestion delays after warm-up (k >= 5)
        b_idx = 5
        while b_idx < beacon_t_loc.size:
            if rng.random() < (outlier_rate / 2.0):
                b_len = int(rng.integers(1, 4))
                for j in range(b_idx, min(b_idx + b_len, beacon_t_loc.size)):
                    delay = float(
                        rng.uniform(0.5 * outlier_delay_s, 1.5 * outlier_delay_s)
                    )
                    beacon_t_ref[j] += delay
                    is_outlier[j] = True
                b_idx += b_len + 1
            else:
                b_idx += 1

    return {
        "t_loc": t_loc,
        "x_loc": x_loc,
        "beacon_t_loc": beacon_t_loc,
        "beacon_t_ref": beacon_t_ref,
        "is_outlier": is_outlier,
        "t_ref_grid": t_ref_grid,
        "x_ref_true": y_ref,
        "true_drift_ppm": float(drift_ppm),
        "true_offset_s": float(offset_s),
        "dropped_fraction": float(1.0 - np.mean(keep)),
    }
