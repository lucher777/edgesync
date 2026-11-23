"""Reproduce every number, table and figure input of the edgesync paper.

Run from any directory with the package importable (``pip install -e .``) and a
C toolchain on PATH (gcc, objdump, nm, size; override the compiler with
EDGESYNC_CC). Results go to ``paper/results/results_summary.json``; then run
``make_figures.py``. Runtime is a few minutes on one CPU core.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import tracemalloc

import numpy as np
from scipy.interpolate import interp1d

from edgesync import (
    EdgeSyncConfig,
    EdgeSyncKernel,
    EdgeSyncKernelF32,
    audit_zero_heap_symbols,
    farrow_cubic_4tap,
    load_wesad_5subj_events,
    load_wesad_s10_slice,
    measure_object_sections,
    simulate_asynchronous_ble_stream,
    verify_c11_python_parity,
    verify_f32_quantization,
    verify_scipy_cubic_parity,
)


HERE = Path(__file__).resolve().parent
PKG_DIR = HERE.parent.parent
CSRC_DIR = PKG_DIR / "csrc"
BENCH_DIR = PKG_DIR / "benchmarks"
OUT_DIR = HERE.parent / "results"


def _tool(name: str) -> str:
    """Return a toolchain executable; honours EDGESYNC_CC for the compiler."""
    if name == "gcc":
        return os.environ.get("EDGESYNC_CC", shutil.which("gcc") or "gcc")
    return shutil.which(name) or name


def run_native_bench() -> dict:
    """Build and run benchmarks/bench_compare.c (edgesync vs. two plain-C baselines)."""
    exe = BENCH_DIR / "bench_compare"
    subprocess.run(
        [_tool("gcc"), "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror", f"-I{CSRC_DIR}",
         str(BENCH_DIR / "bench_compare.c"), str(CSRC_DIR / "edgesync.c"), "-lm", "-o", str(exe)],
        check=True,
    )
    cmd = [str(exe)]
    if shutil.which("taskset"):
        cmd = ["taskset", "-c", "2"] + cmd
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = json.loads(proc.stdout)
    cc = subprocess.run([_tool("gcc"), "--version"], capture_output=True, text=True, check=True)
    out["compiler"] = cc.stdout.splitlines()[0]
    out["host"] = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    out["host"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return out


def count_objdump_instructions() -> dict:
    """Count compiled x86-64 instructions per API function via objdump."""
    obj = CSRC_DIR / "edgesync.o"
    subprocess.run(
        [_tool("gcc"), "-std=c11", "-O3", "-Wall", "-Wextra", "-Werror", "-ffreestanding",
         "-fno-builtin", "-c", str(CSRC_DIR / "edgesync.c"), "-o", str(obj)],
        check=True,
    )
    proc = subprocess.run(
        [_tool("objdump"), "-d", str(obj)], capture_output=True, text=True, check=True
    )
    targets = {
        "es_push_sample",
        "es_update_sync",
        "es_farrow_cubic_4tap",
        "es_resample_at",
        "es_farrow_cubic_4tap_f32",
        "es_f32_resample_at",
    }
    counts = {k: {"insns": 0, "branches": 0} for k in targets}
    cur = None
    for line in proc.stdout.splitlines():
        if line.endswith(">:"):
            fn = line.split("<", 1)[1].split(">", 1)[0]
            cur = fn if fn in targets else None
            continue
        if cur is not None and "\t" in line:
            parts = line.split("\t")
            if len(parts) >= 3:
                mnem = parts[2].strip().split()[0]
                if mnem not in ("nop", "nopw", "nopl"):
                    counts[cur]["insns"] += 1
                    if mnem.startswith("j") or mnem == "call":
                        counts[cur]["branches"] += 1
    return counts


def benchmark_python_baselines() -> dict:
    """Measure peak heap memory of offline SciPy cubic resampling versus recording length."""
    fs = 250.0
    h = 1.0 / fs
    horizons_s = [10, 60, 300, 1200]
    scipy_cubic_rows = []

    for dur_s in horizons_s:
        n = int(dur_s * fs)
        t_loc = np.arange(n, dtype=np.float64) * h
        x_loc = np.sin(2.0 * np.pi * 1.2 * t_loc)
        t_ref = (1.0 + 120e-6) * t_loc + 0.5
        t_q = np.linspace(t_ref[4], t_ref[-5], n)

        # Offline SciPy cubic interp1d
        tracemalloc.start()
        t0 = time.perf_counter_ns()
        f_sp = interp1d(t_ref, x_loc, kind="cubic", copy=True, assume_sorted=True)
        y_sp = f_sp(t_q)
        t1 = time.perf_counter_ns()
        _, peak_sp = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        _ = float(y_sp[0])
        scipy_cubic_rows.append(
            {
                "duration_s": dur_s,
                "n_samples": n,
                "total_ms": (t1 - t0) * 1e-6,
                "ns_per_sample": (t1 - t0) / n,
                "peak_heap_kb": peak_sp / 1024.0,
            }
        )

    return {"scipy_cubic_offline": scipy_cubic_rows}


def eval_stream_methods(
    signal_ref: np.ndarray,
    fs: float,
    drift_ppm: float,
    offset_s: float,
    outlier_rate: float = 0.0,
    packet_drop_rate: float = 0.0,
    seed: int = 0,
    warmup_s: float = 20.0,
) -> dict:
    """Compare EdgeSync FP64, FP32, Linear-only, Standard-RLS, and Unsynchronized."""
    sim = simulate_asynchronous_ble_stream(
        signal_ref,
        fs_hz=fs,
        drift_ppm=drift_ppm,
        offset_s=offset_s,
        beacon_interval_s=1.0,
        beacon_jitter_s=0.002,
        outlier_rate=outlier_rate,
        outlier_delay_s=0.18,
        packet_drop_rate=packet_drop_rate,
        burst_len=4,
        seed=seed,
    )
    t_loc = sim["t_loc"]
    x_loc = sim["x_loc"]
    b_loc = sim["beacon_t_loc"]
    b_ref = sim["beacon_t_ref"]
    t_grid = sim["t_ref_grid"]
    x_true = sim["x_ref_true"]

    # 1. EdgeSync C11 FP64 (Redescending Huber-RLS + 4-tap Cubic Farrow)
    k_fp64 = EdgeSyncKernel(EdgeSyncConfig(fs_in=fs, lambda_=0.995, huber_delta=0.015))
    # 2. EdgeSync C11 FP32 (4,344 B state)
    k_fp32 = EdgeSyncKernelF32(EdgeSyncConfig(fs_in=fs, lambda_=0.995, huber_delta=0.015))
    # 3. Standard Unweighted RLS (huber_delta = 1e6 -> disables Huber rejection)
    k_rls = EdgeSyncKernel(EdgeSyncConfig(fs_in=fs, lambda_=0.995, huber_delta=1e6))
    # 4. H-RLS + Linear interpolation only (gap_factor = 1e-6? no, must be >1; compute directly)
    k_lin = EdgeSyncKernel(EdgeSyncConfig(fs_in=fs, lambda_=0.995, huber_delta=0.015))

    b_ptr = 0
    q_ptr = 0
    delay_s = max(0.20, 6.0 / fs)
    warmup_idx = int(warmup_s * fs)

    y_fp64, y_fp32, y_rls, y_lin, y_unsync, y_gt = [], [], [], [], [], []
    err_t_fp64_ms, err_t_rls_ms, err_t_unsync_ms = [], [], []
    alpha_true = drift_ppm * 1e-6

    # Keep rolling arrays for linear-only ablation and unsync baseline
    loc_idx_map = 0
    for i in range(t_loc.size):
        tl = float(t_loc[i])
        xl = float(x_loc[i])
        k_fp64.push_sample(tl, xl)
        k_fp32.push_sample(tl, xl)
        k_rls.push_sample(tl, xl)
        k_lin.push_sample(tl, xl)

        while b_ptr < b_loc.size and b_loc[b_ptr] <= tl:
            bl = float(b_loc[b_ptr])
            br = float(b_ref[b_ptr])
            k_fp64.update_sync(bl, br)
            k_fp32.update_sync(bl, br)
            k_rls.update_sync(bl, br)
            k_lin.update_sync(bl, br)
            b_ptr += 1

        cur_ref_true = (1.0 + alpha_true) * tl + offset_s - delay_s
        while q_ptr < t_grid.size and t_grid[q_ptr] <= cur_ref_true:
            tq = float(t_grid[q_ptr])
            s64, v64 = k_fp64.resample_at(tq)
            s32, v32 = k_fp32.resample_at(tq)
            sr, vr = k_rls.resample_at(tq)

            if q_ptr >= warmup_idx and s64 >= 0 and s32 >= 0:
                y_gt.append(float(x_true[q_ptr]))
                y_fp64.append(v64)
                y_fp32.append(v32)
                y_rls.append(vr if sr >= 0 else 0.0)

                # Timing errors at target tq
                tloc_true = (tq - offset_s) / (1.0 + alpha_true)
                err_t_fp64_ms.append(abs(k_fp64.ref_to_loc(tq) - tloc_true) * 1000.0)
                err_t_rls_ms.append(abs(k_rls.ref_to_loc(tq) - tloc_true) * 1000.0)
                # Unsynchronized node assumes alpha=0 with initial handshake offset only
                tloc_unsync = tq - offset_s
                err_t_unsync_ms.append(abs(tloc_unsync - tloc_true) * 1000.0)

                # Advance linear-only and unsync pointer
                while loc_idx_map + 1 < i and t_loc[loc_idx_map + 1] < k_lin.ref_to_loc(tq):
                    loc_idx_map += 1
                # Linear interpolation with H-RLS clock
                tloc_est = k_lin.ref_to_loc(tq)
                idx_l = int(np.searchsorted(t_loc[: i + 1], tloc_est)) - 1
                idx_l = min(max(idx_l, 0), i - 1)
                dt_l = t_loc[idx_l + 1] - t_loc[idx_l]
                mu_l = min(max((tloc_est - t_loc[idx_l]) / dt_l, 0.0), 1.0) if dt_l > 0 else 0.0
                y_lin.append(float(x_loc[idx_l] + mu_l * (x_loc[idx_l + 1] - x_loc[idx_l])))

                # Unsynchronized local sample at tloc_unsync (initial offset only, no drift)
                idx_u = int(np.searchsorted(t_loc[: i + 1], tloc_unsync)) - 1
                idx_u = min(max(idx_u, 0), i - 1)
                dt_u = t_loc[idx_u + 1] - t_loc[idx_u]
                mu_u = min(max((tloc_unsync - t_loc[idx_u]) / dt_u, 0.0), 1.0) if dt_u > 0 else 0.0
                y_unsync.append(float(x_loc[idx_u] + mu_u * (x_loc[idx_u + 1] - x_loc[idx_u])))

            q_ptr += 1

    gt = np.asarray(y_gt)
    sig_pow = float(np.mean(gt**2))

    def metrics(pred_list: list[float]) -> dict:
        pr = np.asarray(pred_list)
        err = pr - gt
        mse = float(np.mean(err**2))
        snr = 10.0 * math.log10(sig_pow / max(mse, 1e-18))
        r = float(np.corrcoef(pr, gt)[0, 1]) if np.std(pr) > 1e-12 else 0.0
        return {"snr_db": snr, "rmse": math.sqrt(mse), "pearson_r": r}

    return {
        "n_eval": int(gt.size),
        "dropped_fraction": sim["dropped_fraction"],
        "fp64": {
            **metrics(y_fp64),
            "drift_err_ppm": abs(k_fp64.drift_ppm - drift_ppm),
            "timing_med_ms": float(np.median(err_t_fp64_ms)),
            "timing_p95_ms": float(np.percentile(err_t_fp64_ms, 95)),
            "n_outliers_rejected": k_fp64.n_outliers,
            "n_farrow_cubic": k_fp64.stats["n_farrow_cubic"],
            "n_gap_linear": k_fp64.stats["n_gap_linear"],
        },
        "fp32": {
            **metrics(y_fp32),
        },
        "hrls_linear": {
            **metrics(y_lin),
        },
        "std_rls_cubic": {
            **metrics(y_rls),
            "drift_err_ppm": abs(k_rls.drift_ppm - drift_ppm),
            "timing_med_ms": float(np.median(err_t_rls_ms)),
            "timing_p95_ms": float(np.percentile(err_t_rls_ms, 95)),
        },
        "unsync": {
            **metrics(y_unsync),
            "timing_med_ms": float(np.median(err_t_unsync_ms)),
            "timing_p95_ms": float(np.percentile(err_t_unsync_ms, 95)),
        },
    }


def eval_wesad_5subj_cohort() -> list[dict]:
    """Beacon-timing test: beacons at the R-peak times of 5 WESAD subjects, +/-150 ppm, 10% bursts."""
    records = load_wesad_5subj_events()
    rows = []
    for idx, rec in enumerate(records):
        rpeaks = np.asarray(rec["ecg_rpeaks_s"], dtype=np.float64)
        drift_ppm = 150.0 if idx % 2 == 0 else -150.0
        alpha = drift_ppm * 1e-6
        offset = 2.25
        t_loc = (rpeaks - offset) / (1.0 + alpha)
        rng = np.random.default_rng(100 + idx)

        k_es = EdgeSyncKernel(EdgeSyncConfig(fs_in=64.0, lambda_=0.995, huber_delta=0.015))
        k_rls = EdgeSyncKernel(EdgeSyncConfig(fs_in=64.0, lambda_=0.995, huber_delta=1e6))

        err_es, err_rls, err_un = [], [], []
        for j, (tl, tr) in enumerate(zip(t_loc, rpeaks)):
            jitter = float(rng.normal(0.0, 0.002))
            if j >= 5 and rng.random() < 0.10:
                jitter += float(rng.uniform(0.10, 0.25))
            k_es.update_sync(float(tl), float(tr + jitter))
            k_rls.update_sync(float(tl), float(tr + jitter))
            if j >= 20:
                err_es.append(abs(k_es.loc_to_ref(float(tl)) - float(tr)) * 1000.0)
                err_rls.append(abs(k_rls.loc_to_ref(float(tl)) - float(tr)) * 1000.0)
                err_un.append(abs((float(tl) + offset) - float(tr)) * 1000.0)

        rows.append(
            {
                "subject_id": rec["subject_id"],
                "n_beats": int(rpeaks.size),
                "duration_s": float(rec["duration_s"]),
                "true_drift_ppm": drift_ppm,
                "edgesync_med_ms": float(np.median(err_es)),
                "edgesync_p95_ms": float(np.percentile(err_es, 95)),
                "edgesync_drift_err_ppm": abs(k_es.drift_ppm - drift_ppm),
                "std_rls_med_ms": float(np.median(err_rls)),
                "std_rls_p95_ms": float(np.percentile(err_rls, 95)),
                "unsync_med_ms": float(np.median(err_un)),
            }
        )
    return rows


def eval_farrow_oh4_scaling() -> list[dict]:
    """Verify Property 2 O(h^4) Farrow truncation bound vs linear O(h^2)."""
    f_sig = 2.5
    omega = 2.0 * math.pi * f_sig
    rows = []
    for fs in [32.0, 64.0, 128.0, 250.0, 500.0]:
        h = 1.0 / fs
        bound_cubic = (3.0 / 128.0) * (h**4) * (omega**4)
        bound_lin = (1.0 / 8.0) * (h**2) * (omega**2)
        max_c = 0.0
        max_l = 0.0
        for k_idx in range(10, 60):
            ts = np.arange(k_idx - 1, k_idx + 3, dtype=np.float64) * h
            xs = np.sin(omega * ts)
            for mu in np.linspace(0.0, 1.0, 41):
                y_true = math.sin(omega * (k_idx + mu) * h)
                yc = farrow_cubic_4tap(xs[0], xs[1], xs[2], xs[3], float(mu))
                yl = xs[1] + float(mu) * (xs[2] - xs[1])
                max_c = max(max_c, abs(yc - y_true))
                max_l = max(max_l, abs(yl - y_true))
        rows.append(
            {
                "fs_hz": fs,
                "h_s": h,
                "farrow_max_err": max_c,
                "property2_bound": bound_cubic,
                "linear_max_err": max_l,
                "linear_bound": bound_lin,
            }
        )
    return rows


def eval_seed_robustness(bvp: np.ndarray, ecg: np.ndarray, n_seeds: int = 20) -> dict:
    """Repeat the main semi-synthetic conditions over independent impairment seeds.

    Reports per-method median and interquartile range of the SNR, and paired SNR differences
    of edgesync (FP64) against each ablation with a two-sided Wilcoxon signed-rank test.
    """
    from scipy.stats import wilcoxon

    conditions = {
        "ppg_150ppm_10pct_bursts": dict(signal_ref=bvp, fs=64.0, drift_ppm=150.0,
                                        offset_s=1.85, outlier_rate=0.10),
        "ppg_0ppm_10pct_bursts": dict(signal_ref=bvp, fs=64.0, drift_ppm=0.0,
                                      offset_s=1.85, outlier_rate=0.10),
        "ppg_120ppm_5pct_drop": dict(signal_ref=bvp, fs=64.0, drift_ppm=120.0, offset_s=1.50,
                                     outlier_rate=0.05, packet_drop_rate=0.05),
        "ecg_150ppm_10pct_bursts": dict(signal_ref=ecg, fs=250.0, drift_ppm=150.0,
                                        offset_s=-1.25, outlier_rate=0.10),
    }
    methods = ["fp64", "fp32", "hrls_linear", "std_rls_cubic", "unsync"]
    out = {}
    for name, kw in conditions.items():
        snr = {m: [] for m in methods}
        t_med = []
        for k in range(n_seeds):
            res = eval_stream_methods(seed=1000 + k, **kw)
            for m in methods:
                snr[m].append(res[m]["snr_db"])
            t_med.append(res["fp64"]["timing_med_ms"])
        row = {"n_seeds": n_seeds, "seeds": f"1000-{1000 + n_seeds - 1}"}
        for m in methods:
            a = np.asarray(snr[m])
            row[m] = {"median_db": float(np.median(a)), "q25_db": float(np.percentile(a, 25)),
                      "q75_db": float(np.percentile(a, 75))}
        row["fp64_timing_med_ms_median"] = float(np.median(t_med))
        for other in ["hrls_linear", "std_rls_cubic"]:
            d = np.asarray(snr["fp64"]) - np.asarray(snr[other])
            p_val = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            row[f"fp64_minus_{other}"] = {
                "median_db": float(np.median(d)), "q25_db": float(np.percentile(d, 25)),
                "q75_db": float(np.percentile(d, 75)), "n_positive": int(np.sum(d > 0)),
                "wilcoxon_p": p_val}
        out[name] = row
    return out


def main() -> None:
    """Run all experiments and serialize results_summary.json."""
    wesad = load_wesad_s10_slice()
    bvp_300 = wesad["bvp_64hz"][: 64 * 300]
    ecg_180 = wesad["ecg_250hz"][: 250 * 180]

    print("1/6 Running native C11 benchmark & binary audit...")
    audit = audit_zero_heap_symbols()
    sec_o3 = measure_object_sections("-O3")
    sec_os = measure_object_sections("-Os")
    native = run_native_bench()
    insns = count_objdump_instructions()
    parity = verify_c11_python_parity()
    sp_parity = verify_scipy_cubic_parity()
    f32_quant = verify_f32_quantization()

    print("2/6 Benchmarking Python/SciPy baselines...")
    baselines = benchmark_python_baselines()

    print("3/6 Sweeping crystal drift (0-200 ppm) on WESAD S10 PPG...")
    drift_sweep = []
    for ppm in [0.0, 50.0, 100.0, 150.0, 200.0]:
        res = eval_stream_methods(
            bvp_300, fs=64.0, drift_ppm=ppm, offset_s=1.85, outlier_rate=0.10, seed=42
        )
        drift_sweep.append({"drift_ppm": ppm, **res})

    print("4/6 Sweeping BLE burst outlier rate (0-20%) on WESAD S10 PPG & ECG...")
    outlier_sweep_ppg = []
    for rate in [0.0, 0.05, 0.10, 0.15, 0.20]:
        res = eval_stream_methods(
            bvp_300, fs=64.0, drift_ppm=150.0, offset_s=1.85, outlier_rate=rate, seed=77
        )
        outlier_sweep_ppg.append({"outlier_rate": rate, **res})

    ecg_bench = eval_stream_methods(
        ecg_180, fs=250.0, drift_ppm=150.0, offset_s=-1.25, outlier_rate=0.10, seed=55
    )

    print("5/6 Sweeping BLE packet drop rate (0-20%) on WESAD S10 PPG...")
    drop_sweep = []
    for dr in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
        res = eval_stream_methods(
            bvp_300,
            fs=64.0,
            drift_ppm=120.0,
            offset_s=1.50,
            outlier_rate=0.05,
            packet_drop_rate=dr,
            seed=99,
        )
        drop_sweep.append({"packet_drop_rate": dr, **res})

    print("6/6 Evaluating 5-subject WESAD cohort & Property 2 O(h^4) bound...")
    cohort_5 = eval_wesad_5subj_cohort()
    oh4 = eval_farrow_oh4_scaling()

    print("Repeating main conditions over 20 impairment seeds...")
    seeds = eval_seed_robustness(bvp_300, ecg_180)

    summary = {
        "audit": audit,
        "sections_O3": sec_o3,
        "sections_Os": sec_os,
        "native_bench": native,
        "objdump_insns": insns,
        "parity": {
            "c11_vs_python": parity,
            "scipy_lagrange": sp_parity,
            "fp32_quantization": f32_quant,
        },
        "baselines": baselines,
        "wesad_s10_ppg_drift_sweep": drift_sweep,
        "wesad_s10_ppg_outlier_sweep": outlier_sweep_ppg,
        "wesad_s10_ecg_250hz": ecg_bench,
        "wesad_s10_ppg_drop_sweep": drop_sweep,
        "wesad_5subj_cohort": cohort_5,
        "property2_oh4_scaling": oh4,
        "seed_robustness": seeds,
    }

    out_path = OUT_DIR / "results_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved complete benchmark results to {out_path.name}")


if __name__ == "__main__":
    main()
