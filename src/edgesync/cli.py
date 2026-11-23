"""Command-line diagnostics interface for the EdgeSync microkernel package.

Supports three subcommands:
- ``edgesync audit``: Inspect static C11 binary size and zero-heap symbol proof.
- ``edgesync parity``: Run C11-vs-Python (<1e-12) and FP32 quantization checks.
- ``edgesync demo``: Run a live streaming synchronization demo on a synthetic
  64 Hz PPG signal (or on WESAD S10 BVP with ``--wesad-dir``).
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

import numpy as np

from .datasets import (
    load_wesad_s10_slice,
    simulate_asynchronous_ble_stream,
    synthetic_ppg,
)
from .equiv import (
    audit_zero_heap_symbols,
    measure_object_sections,
    verify_c11_python_parity,
    verify_f32_quantization,
    verify_scipy_cubic_parity,
)
from .wrapper import EdgeSyncConfig, EdgeSyncKernel


def _cmd_audit() -> dict[str, object]:
    """Run static footprint and zero-heap binary audit."""
    sym = audit_zero_heap_symbols()
    sec_o3 = measure_object_sections("-O3")
    sec_os = measure_object_sections("-Os")
    return {
        "symbol_audit": sym,
        "sections_O3": sec_o3,
        "sections_Os": sec_os,
    }


def _cmd_parity() -> dict[str, object]:
    """Run C11-vs-Python, SciPy Lagrange, and FP32 quantization checks."""
    return {
        "c11_vs_python": verify_c11_python_parity(),
        "scipy_lagrange": verify_scipy_cubic_parity(),
        "fp32_quantization": verify_f32_quantization(),
    }


def _cmd_demo(
    drift_ppm: float, outlier_rate: float, wesad_dir: Optional[str] = None
) -> dict[str, object]:
    """Run streaming synchronization on a 120 s, 64 Hz PPG segment.

    Uses a synthetic PPG by default, or the WESAD S10 BVP excerpt when
    ``wesad_dir`` is given.
    """
    if wesad_dir is not None:
        data = load_wesad_s10_slice(wesad_dir)
        bvp = data["bvp_64hz"][: 64 * 120]  # 120 s segment
        fs = data["fs_bvp_hz"]
        source = "WESAD S10 BVP"
    else:
        fs = 64.0
        bvp = synthetic_ppg(duration_s=120.0, fs_hz=fs, seed=7)
        source = "synthetic PPG"
    sim = simulate_asynchronous_ble_stream(
        bvp,
        fs_hz=fs,
        drift_ppm=drift_ppm,
        offset_s=1.85,
        outlier_rate=outlier_rate,
        seed=7,
    )

    k = EdgeSyncKernel(EdgeSyncConfig(fs_in=fs))
    t_loc = sim["t_loc"]
    x_loc = sim["x_loc"]
    b_loc = sim["beacon_t_loc"]
    b_ref = sim["beacon_t_ref"]
    t_ref_grid = sim["t_ref_grid"]
    x_true = sim["x_ref_true"]

    b_ptr = 0
    q_ptr = 0
    preds = []
    trues = []
    delay_s = 0.25

    for i in range(t_loc.size):
        tl = float(t_loc[i])
        k.push_sample(tl, float(x_loc[i]))
        while b_ptr < b_loc.size and b_loc[b_ptr] <= tl:
            k.update_sync(float(b_loc[b_ptr]), float(b_ref[b_ptr]))
            b_ptr += 1
        cur_ref = k.loc_to_ref(tl) - delay_s
        while q_ptr < t_ref_grid.size and t_ref_grid[q_ptr] <= cur_ref:
            st, val = k.resample_at(float(t_ref_grid[q_ptr]))
            if st >= 0 and q_ptr >= int(15 * fs):
                preds.append(val)
                trues.append(float(x_true[q_ptr]))
            q_ptr += 1

    p_arr = np.asarray(preds, dtype=np.float64)
    t_arr = np.asarray(trues, dtype=np.float64)
    err = p_arr - t_arr
    sig_pow = float(np.mean(t_arr**2))
    err_pow = float(np.mean(err**2))
    snr_db = 10.0 * np.log10(sig_pow / max(err_pow, 1e-18))
    corr = float(np.corrcoef(p_arr, t_arr)[0, 1])

    return {
        "signal": source,
        "true_drift_ppm": drift_ppm,
        "estimated_drift_ppm": round(k.drift_ppm, 4),
        "drift_error_ppm": round(abs(k.drift_ppm - drift_ppm), 4),
        "n_eval_samples": int(p_arr.size),
        "snr_db": round(float(snr_db), 2),
        "pearson_r": round(corr, 6),
        "kernel_bytes": k.kernel_bytes,
        "stats": k.stats,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the ``edgesync`` command-line interface.

    Args:
        argv: Optional list of CLI argument strings.

    Returns:
        Process exit status code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        prog="edgesync",
        description="EdgeSync C11 static-memory streaming sync microkernel CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="Audit static footprint and zero-heap symbols")
    sub.add_parser("parity", help="Verify C11-vs-Python and FP32 numerical parity")
    demo_p = sub.add_parser("demo", help="Run a streaming sync demo on a PPG signal")
    demo_p.add_argument("--drift-ppm", type=float, default=120.0)
    demo_p.add_argument("--outlier-rate", type=float, default=0.10)
    demo_p.add_argument(
        "--wesad-dir",
        default=None,
        help="Directory with the WESAD excerpt files (default: synthetic PPG)",
    )

    args = parser.parse_args(argv)
    if args.command == "audit":
        out = _cmd_audit()
    elif args.command == "parity":
        out = _cmd_parity()
    else:
        out = _cmd_demo(args.drift_ppm, args.outlier_rate, args.wesad_dir)

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
