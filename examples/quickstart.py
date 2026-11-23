"""Minimal end-to-end quickstart demonstrating EdgeSync on a synthetic 64 Hz PPG.

To run it on the WESAD S10 BVP excerpt instead, replace ``synthetic_ppg(...)``
with ``load_wesad_s10_slice()["bvp_64hz"][: 64 * 90]`` after setting
``EDGESYNC_WESAD_DIR`` (WESAD is not bundled; see the README).
"""

from __future__ import annotations

import numpy as np

from edgesync import (
    EdgeSyncConfig,
    EdgeSyncKernel,
    audit_zero_heap_symbols,
    simulate_asynchronous_ble_stream,
    synthetic_ppg,
    verify_c11_python_parity,
)


def run_quickstart() -> None:
    """Execute static-memory audit, parity check, and a streaming demo."""
    audit = audit_zero_heap_symbols()
    parity = verify_c11_python_parity()
    print(f"Zero-heap verified: {audit['zero_heap_verified']}")
    print(f"C11-vs-Python max diff: {parity['max_abs_sample_diff']:.3e}")

    fs = 64.0
    bvp = synthetic_ppg(duration_s=90.0, fs_hz=fs, seed=42)
    sim = simulate_asynchronous_ble_stream(
        bvp,
        fs_hz=fs,
        drift_ppm=125.0,
        offset_s=1.40,
        outlier_rate=0.10,
        seed=42,
    )

    kernel = EdgeSyncKernel(EdgeSyncConfig(fs_in=fs))
    b_ptr = 0
    q_ptr = 0
    preds = []
    trues = []
    for tl, xl in zip(sim["t_loc"], sim["x_loc"]):
        kernel.push_sample(float(tl), float(xl))
        while b_ptr < sim["beacon_t_loc"].size and sim["beacon_t_loc"][b_ptr] <= tl:
            kernel.update_sync(
                float(sim["beacon_t_loc"][b_ptr]), float(sim["beacon_t_ref"][b_ptr])
            )
            b_ptr += 1
        cur_ref = kernel.loc_to_ref(float(tl)) - 0.25
        while q_ptr < sim["t_ref_grid"].size and sim["t_ref_grid"][q_ptr] <= cur_ref:
            st, y = kernel.resample_at(float(sim["t_ref_grid"][q_ptr]))
            if st >= 0 and q_ptr >= int(15 * fs):
                preds.append(y)
                trues.append(float(sim["x_ref_true"][q_ptr]))
            q_ptr += 1

    p = np.asarray(preds)
    t = np.asarray(trues)
    snr_db = 10.0 * np.log10(np.mean(t**2) / np.mean((p - t) ** 2))
    print(
        f"Static kernel: {kernel.kernel_bytes} B | "
        f"Estimated drift: {kernel.drift_ppm:.2f} ppm (true 125.00) | "
        f"SNR: {snr_db:.2f} dB"
    )


if __name__ == "__main__":
    run_quickstart()
