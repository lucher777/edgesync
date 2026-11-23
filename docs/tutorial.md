# Tutorial

## 1. Compile and Audit the Freestanding C11 Microkernel

```bash
make all
make audit
make bench
```

The `make audit` target prints exact `.text`, `.data`, and `.bss` sizes via
`size` and confirms zero undefined external symbols via `nm -u`.

## 2. Streaming Wearable Synchronization in Python

```python
from edgesync import (
    EdgeSyncConfig,
    EdgeSyncKernel,
    simulate_asynchronous_ble_stream,
    synthetic_ppg,
)

fs = 64.0
ppg = synthetic_ppg(duration_s=60.0, fs_hz=fs, seed=1)
sim = simulate_asynchronous_ble_stream(
    ppg,
    fs_hz=fs,
    drift_ppm=150.0,
    offset_s=1.25,
    outlier_rate=0.10,
    seed=1,
)

kernel = EdgeSyncKernel(EdgeSyncConfig(fs_in=fs))
for t_loc, x in zip(sim["t_loc"][:100], sim["x_loc"][:100]):
    kernel.push_sample(float(t_loc), float(x))
```

### Using the WESAD excerpts

The paper's benchmarks use excerpts of the WESAD dataset (Schmidt et al.,
ICMI 2018). WESAD is licensed for scientific, non-commercial use only, so it is
not distributed with edgesync. After obtaining WESAD from its
[official page](https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/)
and preparing the two excerpt files (`wesad_s10_slice.npz`,
`wesad_5subj_events.json`), point edgesync at their directory:

```bash
export EDGESYNC_WESAD_DIR=/path/to/wesad_excerpts
edgesync demo --wesad-dir "$EDGESYNC_WESAD_DIR"
```

```python
from edgesync import load_wesad_s10_slice

wesad = load_wesad_s10_slice()          # or load_wesad_s10_slice("/path/to/dir")
bvp, fs = wesad["bvp_64hz"], wesad["fs_bvp_hz"]
```

## 3. Verifying C11-vs-Python Numerical Parity

```python
from edgesync import verify_c11_python_parity, audit_zero_heap_symbols

print(audit_zero_heap_symbols())
print(verify_c11_python_parity())
```
