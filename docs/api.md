# API Reference

## C11 Freestanding Microkernel (`csrc/edgesync.h`)

- `es_sizeof_kernel()` / `es_sizeof_kernel_f32()`: Exact compile-time SRAM sizes.
- `es_default_config(double fs_in)`: Validated BLE wearable default parameters.
- `es_init(es_kernel_t *k, const es_config_t *cfg)`: Static zero-heap init.
- `es_update_sync(es_kernel_t *k, double t_loc, double t_ref)`: $O(1)$ Huber-RLS update.
- `es_push_sample(es_kernel_t *k, double t_loc, double x)`: $O(1)$ ring write.
- `es_farrow_cubic_4tap(xm1, x0, x1, x2, mu)`: Branch-free 4-tap Horner Farrow kernel.
- `es_resample_at(es_kernel_t *k, double t_ref, double *out_x)`: Streaming resample (cursor fast path, bounded binary search).
- `es_f32_*`: Single-precision storage counterparts (`4,344` bytes static SRAM; float32 timestamps relative to two double epoch bases).

## Python Package (`edgesync`)

- `EdgeSyncConfig`, `EdgeSyncKernel`, `EdgeSyncKernelF32`
- `ReferenceConfig`, `ReferenceEdgeSync`, `farrow_cubic_4tap_py`
- `audit_zero_heap_symbols`, `measure_object_sections`
- `verify_c11_python_parity`, `verify_scipy_cubic_parity`, `verify_f32_quantization`
- `synthetic_ppg`, `synthetic_ecg`: deterministic synthetic signals (bundled)
- `simulate_asynchronous_ble_stream`: drift, jitter, outlier and packet-loss injection
- `load_wesad_s10_slice`, `load_wesad_5subj_events`, `wesad_available`, `wesad_data_dir`: WESAD excerpt loaders (data not bundled; `data_dir=` or `EDGESYNC_WESAD_DIR`)
