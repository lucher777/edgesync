"""Comprehensive pytest suite for the EdgeSync C11 microkernel and Python API."""

from __future__ import annotations

import math

import numpy as np
import pytest

from edgesync import (
    ES_ERR_NONMONOTONIC,
    ES_ERR_OVERRUN,
    ES_ERR_PARAM,
    ES_ERR_UNDERRUN,
    ES_OK,
    ES_OK_EDGE_LINEAR,
    ES_OK_GAP_LINEAR,
    EdgeSyncConfig,
    EdgeSyncKernel,
    EdgeSyncKernelF32,
    ReferenceConfig,
    ReferenceEdgeSync,
    audit_zero_heap_symbols,
    farrow_cubic_4tap,
    farrow_cubic_4tap_py,
    load_wesad_5subj_events,
    load_wesad_s10_slice,
    measure_object_sections,
    simulate_asynchronous_ble_stream,
    synthetic_ecg,
    synthetic_ppg,
    verify_c11_python_parity,
    verify_f32_quantization,
    verify_scipy_cubic_parity,
    wesad_available,
)
from edgesync.cli import main as cli_main
from edgesync.wrapper import _find_gcc_executable

requires_wesad = pytest.mark.skipif(
    not wesad_available(), reason="WESAD excerpts not found (set EDGESYNC_WESAD_DIR)"
)

requires_cc = pytest.mark.skipif(
    _find_gcc_executable() is None, reason="needs a C compiler (gcc) and binutils"
)


# ---- 1. Static Footprint & Zero-Heap Binary Audit (Property 1) ----


@requires_cc
def test_zero_heap_symbol_audit() -> None:
    """Compiled edgesync.o must import zero heap allocators and zero externals."""
    res = audit_zero_heap_symbols()
    assert res["zero_heap_verified"] is True
    assert res["freestanding_zero_undef"] is True
    assert res["heap_symbols"] == []
    assert res["undefined_symbols"] == []


@requires_cc
def test_static_footprint_under_16kb_o3() -> None:
    """Static kernel sizes and .data/.bss must obey strict < 16 KB SRAM budget."""
    sec = measure_object_sections("-O3")
    assert sec["data_bytes"] == 0
    assert sec["bss_bytes"] == 0
    assert sec["sizeof_kernel_fp64_bytes"] < 16384
    assert sec["sizeof_kernel_fp64_bytes"] <= 12800
    assert sec["sizeof_kernel_fp32_bytes"] < 8192


@requires_cc
def test_static_footprint_flash_os() -> None:
    """Code size under -Os must remain compact (< 8 KB Flash)."""
    sec = measure_object_sections("-Os")
    assert sec["text_bytes"] < 8192
    assert sec["data_bytes"] == 0
    assert sec["bss_bytes"] == 0


def test_ring_capacity_is_power_of_two() -> None:
    """Compile-time ring capacity must be a power of two >= 8."""
    k = EdgeSyncKernel()
    cap = k.ring_capacity
    assert cap >= 8
    assert (cap & (cap - 1)) == 0
    assert k.kernel_bytes < 16384


# ---- 2. Configuration Validation & Error Handling ----


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fs_in": 0.0},
        {"fs_in": -64.0},
        {"fs_in": float("nan")},
        {"lambda_": 0.5},
        {"lambda_": 1.01},
        {"huber_delta": 0.0},
        {"max_drift_ppm": -10.0},
        {"gap_factor": 1.0},
    ],
)
def test_invalid_configs_rejected(kwargs: dict[str, float]) -> None:
    """Invalid configuration parameters must be rejected deterministically."""
    with pytest.raises(ValueError):
        EdgeSyncKernel(EdgeSyncConfig(**kwargs))


def test_nonmonotonic_and_nonfinite_push_rejected() -> None:
    """Backward, duplicate, and non-finite sample timestamps must fail cleanly."""
    k = EdgeSyncKernel(EdgeSyncConfig(fs_in=64.0))
    assert k.push_sample(1.0, 0.5) == ES_OK
    assert k.push_sample(1.0, 0.6) == ES_ERR_NONMONOTONIC
    assert k.push_sample(0.5, 0.6) == ES_ERR_NONMONOTONIC
    assert k.push_sample(float("nan"), 0.1) == ES_ERR_PARAM
    assert k.push_sample(2.0, float("inf")) == ES_ERR_PARAM


def test_underrun_and_overrun_status_codes() -> None:
    """Queries outside buffered time span must return exact error codes."""
    k = EdgeSyncKernel(EdgeSyncConfig(fs_in=64.0))
    st, _ = k.resample_at(0.5)
    assert st == ES_ERR_UNDERRUN
    for i in range(10):
        k.push_sample(i / 64.0, float(i))
    st_lo, _ = k.resample_at(-0.01)
    st_hi, _ = k.resample_at(1.0)
    assert st_lo == ES_ERR_UNDERRUN
    assert st_hi == ES_ERR_OVERRUN
    assert k.stats["n_underrun"] == 2
    assert k.stats["n_overrun"] == 1


def test_reset_restores_clean_state() -> None:
    """Reset must zero telemetry and clock state while retaining config."""
    k = EdgeSyncKernel(EdgeSyncConfig(fs_in=100.0))
    k.update_sync(0.0, 1.0)
    k.update_sync(1.0, 2.0001)
    k.push_sample(0.0, 1.0)
    k.reset()
    assert k.n_sync == 0
    assert k.drift_ppm == 0.0
    assert k.stats["n_pushed"] == 0


# ---- 3. 4-Tap Lagrange Cubic Farrow Exactness & Error Bound (Property 3) ----


def test_farrow_exact_on_cubic_polynomial() -> None:
    """4-tap Lagrange cubic Farrow must match exact cubics to < 1e-14."""
    res = verify_scipy_cubic_parity(n_points=250)
    assert res["max_cubic_poly_error"] < 1e-14
    assert res["max_scipy_lagrange_diff"] < 1e-14


def test_farrow_c11_vs_python_bit_exact() -> None:
    """C11 and Python farrow_cubic_4tap must agree to < 1e-15."""
    rng = np.random.default_rng(123)
    for _ in range(100):
        vals = rng.normal(0.0, 10.0, size=4)
        mu = float(rng.uniform(0.0, 1.0))
        yc = farrow_cubic_4tap(vals[0], vals[1], vals[2], vals[3], mu)
        yp = farrow_cubic_4tap_py(vals[0], vals[1], vals[2], vals[3], mu)
        assert abs(yc - yp) < 1e-14


def test_farrow_theoretical_oh4_bound_holds() -> None:
    """Measured cubic Farrow error on sinusoid must obey (3/128) h^4 ||x^(4)||."""
    fs = 64.0
    h = 1.0 / fs
    f_sig = 3.0
    omega = 2.0 * math.pi * f_sig
    bound = (3.0 / 128.0) * (h**4) * (omega**4)

    max_err = 0.0
    for k_idx in range(10, 40):
        ts = (np.arange(k_idx - 1, k_idx + 3, dtype=np.float64)) * h
        xs = np.sin(omega * ts)
        for mu in np.linspace(0.0, 1.0, 21):
            y_hat = farrow_cubic_4tap(xs[0], xs[1], xs[2], xs[3], float(mu))
            y_true = math.sin(omega * (k_idx + mu) * h)
            max_err = max(max_err, abs(y_hat - y_true))
    assert max_err <= bound * 1.0001
    assert max_err > bound * 0.25


def test_edge_and_gap_linear_fallbacks() -> None:
    """Boundary taps and BLE burst gaps must trigger exact fallback status codes."""
    k = EdgeSyncKernel(EdgeSyncConfig(fs_in=64.0))
    h = 1.0 / 64.0
    # Push 0..4, drop 5..8 (gap), push 9..14
    for i in list(range(5)) + list(range(9, 15)):
        assert k.push_sample(i * h, float(i)) == ES_OK
    st_edge, y_edge = k.resample_at(0.5 * h)
    assert st_edge == ES_OK_EDGE_LINEAR
    assert abs(y_edge - 0.5) < 1e-12
    st_cubic, y_cubic = k.resample_at(2.5 * h)
    assert st_cubic == ES_OK
    assert abs(y_cubic - 2.5) < 1e-12
    st_gap, y_gap = k.resample_at(6.5 * h)
    assert st_gap == ES_OK_GAP_LINEAR
    assert abs(y_gap - 6.5) < 1e-12


# ---- 4. Numerical Parity (C11 vs Python < 1e-12 & FP32 Quantization) ----


@pytest.mark.parametrize("drift_ppm", [-180.0, 0.0, 135.0, 200.0])
def test_c11_vs_python_strict_parity(drift_ppm: float) -> None:
    """C11 microkernel and Python reference must match strictly < 1e-12."""
    res = verify_c11_python_parity(
        n_samples=1500, drift_ppm=drift_ppm, outlier_fraction=0.15, seed=9
    )
    assert res["status_mismatches"] == 0.0
    assert res["max_abs_sample_diff"] < 1e-12
    assert res["drift_ppm_diff"] < 1e-12
    assert res["offset_s_diff"] < 1e-12


def test_fp32_quantization_accuracy() -> None:
    """FP32 microkernel must match FP64 within < 2e-5 max / < 5e-6 RMSE."""
    res = verify_f32_quantization(n_samples=2000)
    assert res["n_compared"] > 1900
    assert res["rmse_diff"] < 5e-6
    assert res["max_abs_diff"] < 2e-5


def test_batch_resample_matches_scalar_api() -> None:
    """es_resample_batch must produce identical values and status codes."""
    cfg = EdgeSyncConfig(fs_in=64.0)
    k1 = EdgeSyncKernel(cfg)
    k2 = EdgeSyncKernel(cfg)
    h = 1.0 / 64.0
    for i in range(64):
        t = i * h
        x = math.sin(2.0 * math.pi * 1.5 * t)
        k1.push_sample(t, x)
        k2.push_sample(t, x)
    qs = np.linspace(0.05, 0.90, 50)
    vals_b, sts_b = k1.resample_batch(qs)
    for idx, q in enumerate(qs):
        st, val = k2.resample_at(float(q))
        assert st == int(sts_b[idx])
        assert abs(val - float(vals_b[idx])) < 1e-15


# ---- 5. Affine Crystal Drift Recovery & Huber Burst Robustness (Property 4) ----


@pytest.mark.parametrize("true_ppm", [-200.0, -75.0, 0.0, 50.0, 150.0, 200.0])
def test_exact_affine_drift_recovery_noiseless(true_ppm: float) -> None:
    """Noiseless affine beacons must recover crystal drift within 0.2 ppm."""
    k = EdgeSyncKernel(EdgeSyncConfig(fs_in=64.0, lambda_=0.995))
    alpha = true_ppm * 1e-6
    offset = -12.345
    for sec in range(120):
        t_loc = float(sec)
        t_ref = (1.0 + alpha) * t_loc + offset
        assert k.update_sync(t_loc, t_ref) == ES_OK
    assert abs(k.drift_ppm - true_ppm) < 0.2
    assert abs(k.offset_s - offset) < 1e-4


@pytest.mark.parametrize("outlier_rate", [0.05, 0.10, 0.20])
def test_huber_robustness_under_burst_outliers(outlier_rate: float) -> None:
    """Redescending Huber-RLS must reject 5%-20% BLE burst spikes cleanly."""
    rng = np.random.default_rng(77)
    k = EdgeSyncKernel(EdgeSyncConfig(fs_in=64.0, lambda_=0.995, huber_delta=0.015))
    true_ppm = 140.0
    alpha = true_ppm * 1e-6
    offset = 3.2
    for sec in range(240):
        t_loc = float(sec)
        jitter = float(rng.normal(0.0, 0.002))
        if sec >= 5 and rng.random() < outlier_rate:
            jitter += float(rng.uniform(0.10, 0.35))
        k.update_sync(t_loc, (1.0 + alpha) * t_loc + offset + jitter)
    assert k.n_outliers >= 1
    assert abs(k.drift_ppm - true_ppm) < 8.0
    t_test = 239.0
    err_ms = abs(k.loc_to_ref(t_test) - ((1.0 + alpha) * t_test + offset)) * 1000.0
    assert err_ms < 2.5


def test_physical_drift_clamp_prevents_runaway() -> None:
    """Drift estimate must never exceed max_drift_ppm even under adversarial input."""
    k = EdgeSyncKernel(EdgeSyncConfig(fs_in=64.0, max_drift_ppm=300.0))
    k.update_sync(0.0, 0.0)
    k.update_sync(1.0, 1.5)
    k.update_sync(2.0, 3.0)
    assert abs(k.drift_ppm) <= 300.0 + 1e-9


def test_long_horizon_o1_memory_stability_500k_samples() -> None:
    """Streaming 500,000 samples must keep exact constant footprint and drift lock."""
    k = EdgeSyncKernel(EdgeSyncConfig(fs_in=250.0))
    h = 1.0 / 250.0
    alpha = 115e-6
    offset = 0.85
    bytes_before = k.kernel_bytes
    for i in range(500_000):
        tl = i * h
        tr = (1.0 + alpha) * tl + offset
        k.push_sample(tl, math.sin(0.02 * i))
        if (i & 255) == 0:
            k.update_sync(tl, tr)
        if i >= 16 and (i & 15) == 0:
            st, _ = k.resample_at(tr - 4.0 * h)
            assert st == ES_OK
    assert k.kernel_bytes == bytes_before
    assert k.stats["n_pushed"] == 500_000
    assert abs(k.drift_ppm - 115.0) < 0.05


# ---- 6. Physiological Signal Validation (synthetic; real WESAD if available) ----

SIGNAL_SOURCES = [
    pytest.param("synthetic", id="synthetic"),
    pytest.param("wesad", id="wesad", marks=requires_wesad),
]


def _ppg(source: str, seconds: int, seed: int) -> tuple[np.ndarray, float]:
    """Return a 64 Hz PPG segment from the synthetic generator or WESAD S10."""
    if source == "wesad":
        wesad = load_wesad_s10_slice()
        return wesad["bvp_64hz"][: 64 * seconds], wesad["fs_bvp_hz"]
    return synthetic_ppg(duration_s=float(seconds), fs_hz=64.0, seed=seed), 64.0


def _ecg(source: str, seconds: int, seed: int) -> tuple[np.ndarray, float]:
    """Return a 250 Hz ECG segment from the synthetic generator or WESAD S10."""
    if source == "wesad":
        wesad = load_wesad_s10_slice()
        return wesad["ecg_250hz"][: 250 * seconds], wesad["fs_ecg_hz"]
    return synthetic_ecg(duration_s=float(seconds), fs_hz=250.0, seed=seed), 250.0


def test_synthetic_signals_are_deterministic() -> None:
    """Synthetic generators must be reproducible and physiologically plausible."""
    a = synthetic_ppg(duration_s=30.0, fs_hz=64.0, seed=3)
    b = synthetic_ppg(duration_s=30.0, fs_hz=64.0, seed=3)
    assert a.shape == (1920,)
    assert np.array_equal(a, b)
    ecg = synthetic_ecg(duration_s=30.0, fs_hz=250.0, seed=3)
    assert ecg.shape == (7500,)
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(ecg, height=0.6, distance=int(0.4 * 250))
    assert 30 <= peaks.size <= 42  # ~72 bpm over 30 s


@pytest.mark.parametrize("source", SIGNAL_SOURCES)
def test_ppg_streaming_accuracy(source: str) -> None:
    """64 Hz PPG under 150 ppm drift + 10% burst outliers."""
    bvp, fs = _ppg(source, 120, seed=11)
    sim = simulate_asynchronous_ble_stream(
        bvp,
        fs_hz=fs,
        drift_ppm=150.0,
        offset_s=2.1,
        outlier_rate=0.10,
        seed=11,
    )
    k = EdgeSyncKernel(EdgeSyncConfig(fs_in=fs))
    b_ptr, q_ptr = 0, 0
    preds, trues = [], []
    for tl, xl in zip(sim["t_loc"], sim["x_loc"]):
        k.push_sample(float(tl), float(xl))
        while b_ptr < sim["beacon_t_loc"].size and sim["beacon_t_loc"][b_ptr] <= tl:
            k.update_sync(
                float(sim["beacon_t_loc"][b_ptr]), float(sim["beacon_t_ref"][b_ptr])
            )
            b_ptr += 1
        cur_ref = k.loc_to_ref(float(tl)) - 0.25
        while q_ptr < sim["t_ref_grid"].size and sim["t_ref_grid"][q_ptr] <= cur_ref:
            st, y = k.resample_at(float(sim["t_ref_grid"][q_ptr]))
            if st >= 0 and q_ptr >= int(15 * fs):
                preds.append(y)
                trues.append(float(sim["x_ref_true"][q_ptr]))
            q_ptr += 1

    p, t = np.asarray(preds), np.asarray(trues)
    snr_db = 10.0 * np.log10(np.mean(t**2) / np.mean((p - t) ** 2))
    corr = float(np.corrcoef(p, t)[0, 1])
    assert snr_db > 40.0
    assert corr > 0.9999


@pytest.mark.parametrize("source", SIGNAL_SOURCES)
def test_ecg_250hz_streaming_accuracy(source: str) -> None:
    """250 Hz ECG under -180 ppm drift + 5% packet drop."""
    ecg, fs = _ecg(source, 60, seed=19)
    sim = simulate_asynchronous_ble_stream(
        ecg,
        fs_hz=fs,
        drift_ppm=-180.0,
        offset_s=-0.75,
        packet_drop_rate=0.05,
        burst_len=3,
        seed=19,
    )
    k = EdgeSyncKernel(EdgeSyncConfig(fs_in=fs))
    b_ptr, q_ptr = 0, 0
    preds, trues = [], []
    for tl, xl in zip(sim["t_loc"], sim["x_loc"]):
        k.push_sample(float(tl), float(xl))
        while b_ptr < sim["beacon_t_loc"].size and sim["beacon_t_loc"][b_ptr] <= tl:
            k.update_sync(
                float(sim["beacon_t_loc"][b_ptr]), float(sim["beacon_t_ref"][b_ptr])
            )
            b_ptr += 1
        cur_ref = k.loc_to_ref(float(tl)) - 0.15
        while q_ptr < sim["t_ref_grid"].size and sim["t_ref_grid"][q_ptr] <= cur_ref:
            st, y = k.resample_at(float(sim["t_ref_grid"][q_ptr]))
            if st >= 0 and q_ptr >= int(10 * fs):
                preds.append(y)
                trues.append(float(sim["x_ref_true"][q_ptr]))
            q_ptr += 1

    p, t = np.asarray(preds), np.asarray(trues)
    corr = float(np.corrcoef(p, t)[0, 1])
    assert corr > 0.99
    assert k.stats["n_gap_linear"] > 0


@requires_wesad
def test_wesad_5subj_cardiac_events_alignment() -> None:
    """All 5 real WESAD subjects must align sub-2-ms under 120 ppm + 10% bursts."""
    records = load_wesad_5subj_events()
    assert len(records) == 5
    rng = np.random.default_rng(2026)
    for rec in records:
        rpeaks = np.asarray(rec["ecg_rpeaks_s"], dtype=np.float64)
        alpha = 120e-6
        offset = 1.5
        t_loc = (rpeaks - offset) / (1.0 + alpha)
        k = EdgeSyncKernel(EdgeSyncConfig(fs_in=64.0))
        errs_ms = []
        for idx, (tl, tr) in enumerate(zip(t_loc, rpeaks)):
            jitter = float(rng.normal(0.0, 0.002))
            if idx >= 5 and rng.random() < 0.10:
                jitter += float(rng.uniform(0.10, 0.25))
            k.update_sync(float(tl), float(tr + jitter))
            if idx >= 20:
                errs_ms.append(abs(k.loc_to_ref(float(tl)) - float(tr)) * 1000.0)
        assert float(np.median(errs_ms)) < 2.0


@pytest.mark.parametrize("source", SIGNAL_SOURCES)
def test_f32_kernel_ppg_streaming(source: str) -> None:
    """Single-precision FP32 microkernel (~4.3 KB) must achieve > 40 dB SNR."""
    bvp, fs = _ppg(source, 60, seed=31)
    sim = simulate_asynchronous_ble_stream(
        bvp, fs_hz=fs, drift_ppm=95.0, offset_s=0.6, seed=31
    )
    k32 = EdgeSyncKernelF32(EdgeSyncConfig(fs_in=fs))
    assert k32.kernel_bytes < 5000
    b_ptr, q_ptr = 0, 0
    preds, trues = [], []
    for tl, xl in zip(sim["t_loc"], sim["x_loc"]):
        k32.push_sample(float(tl), float(xl))
        while b_ptr < sim["beacon_t_loc"].size and sim["beacon_t_loc"][b_ptr] <= tl:
            k32.update_sync(
                float(sim["beacon_t_loc"][b_ptr]), float(sim["beacon_t_ref"][b_ptr])
            )
            b_ptr += 1
        cur_ref = (1.0 + 95e-6) * float(tl) + 0.6 - 0.25
        while q_ptr < sim["t_ref_grid"].size and sim["t_ref_grid"][q_ptr] <= cur_ref:
            st, y = k32.resample_at(float(sim["t_ref_grid"][q_ptr]))
            if st >= 0 and q_ptr >= int(12 * fs):
                preds.append(y)
                trues.append(float(sim["x_ref_true"][q_ptr]))
            q_ptr += 1
    p, t = np.asarray(preds), np.asarray(trues)
    snr_db = 10.0 * np.log10(np.mean(t**2) / np.mean((p - t) ** 2))
    assert snr_db > 40.0


def test_reference_kernel_standalone_consistency() -> None:
    """Pure-Python ReferenceEdgeSync must handle inverse mapping and edge guards."""
    ref = ReferenceEdgeSync(ReferenceConfig(fs_in=64.0))
    assert ref.loc_to_ref(5.0) == 5.0
    assert ref.ref_to_loc(5.0) == 5.0
    ref.update_sync(0.0, 2.0)
    ref.update_sync(10.0, 12.001)
    assert abs(ref.ref_to_loc(ref.loc_to_ref(7.5)) - 7.5) < 1e-12


# ---- 7. CLI Smoke Tests ----


@pytest.mark.parametrize(
    "args",
    [pytest.param(["audit"], marks=requires_cc), ["parity"], ["demo", "--drift-ppm", "80"]],
)
def test_cli_subcommands_exit_zero(args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    """All CLI diagnostic subcommands must execute and emit valid JSON."""
    rc = cli_main(args)
    assert rc == 0
    captured = capsys.readouterr()
    assert "{" in captured.out


@requires_cc
def test_fp32_timestamps_stay_precise_on_long_streams() -> None:
    """FP32 timestamps must not lose resolution hours after the first sample."""
    fs = 250.0
    h = 1.0 / fs
    t_start = 20000.0  # about 5.6 h after the first sample
    k32 = EdgeSyncKernelF32(EdgeSyncConfig(fs_in=fs))
    k64 = EdgeSyncKernel(EdgeSyncConfig(fs_in=fs))
    assert k32.push_sample(0.0, 0.0) == 0
    assert k64.push_sample(0.0, 0.0) == 0
    f0 = 2.5
    for i in range(1200):
        t = t_start + i * h
        x = math.sin(2.0 * math.pi * f0 * t)
        assert k32.push_sample(t, x) == 0
        assert k64.push_sample(t, x) == 0
    max_err32 = 0.0
    for j in range(200):
        tq = t_start + (900 + j + 0.37) * h
        truth = math.sin(2.0 * math.pi * f0 * tq)
        s32, v32 = k32.resample_at(tq)
        s64, v64 = k64.resample_at(tq)
        assert s32 >= 0 and s64 >= 0
        assert abs(v64 - truth) < 1e-6
        max_err32 = max(max_err32, abs(v32 - truth))
    assert max_err32 < 1e-4
