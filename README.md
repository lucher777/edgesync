# edgesync

**A static-memory clock synchronization and four-tap Farrow resampling kernel for wearable sensors, in freestanding C11.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C Standard: C11](https://img.shields.io/badge/C-C11%20freestanding-brightgreen.svg)](csrc/edgesync.h)

A sensor node samples on its own crystal clock and receives occasional reference beacons
(for example over BLE). edgesync estimates the clock offset and drift with a Huber-weighted
recursive least-squares update and returns the signal at any requested reference time by
four-tap Lagrange (cubic Farrow) interpolation, falling back to linear interpolation across
packet gaps. All state lives in one statically sized structure.

## Properties

- **No heap, no libc.** `nm -u` on the freestanding object lists no undefined symbols.
- **Fixed memory.** With the default ring of 512 samples the state is 8,424 B (FP64
  `es_kernel_t`) or 4,344 B (FP32 `es_kernel_f32_t`). Code size on x86-64 is about 5.1 kB at `-Os`.
- **Bounded work per call.** Sample push and beacon update contain no loops; a resample
  request tests at most two cached intervals and then runs a binary search over the ring
  (at most log2(R) steps).
- **Robust to delayed beacons.** Beacons whose residual exceeds three times the Huber
  threshold are ignored after a three-beacon warm-up.
- **FP32 variant.** Samples are stored as float32 and timestamps as float32 offsets from
  two double-precision epoch bases, so precision does not degrade on long streams.
  Clock arithmetic stays in double precision (software floating point on a Cortex-M4F).
- **Checked against references.** The C kernel matches an independent Python reference
  bit for bit on the test streams, and the Lagrange stage matches SciPy to 4.4e-16.

Timings in the accompanying paper were measured on an x86-64 desktop only; the kernel has
not yet been measured on a microcontroller.

## Install

The Python wrapper compiles `csrc/edgesync.c` with `gcc` on first use, so install from a clone:

```bash
git clone https://github.com/lucher777/edgesync.git
cd edgesync
git checkout v0.1.0
pip install -e ".[dev]"
```

Set `EDGESYNC_CC` to point to a specific `gcc` if it is not on `PATH`.

## Build and test

```bash
make all      # freestanding object, shared library, benchmark binary
make audit    # section sizes and undefined symbols
make bench    # 10^6-sample timing run
pytest        # 49 tests; the 4 WESAD tests skip without EDGESYNC_WESAD_DIR

edgesync audit
edgesync parity
edgesync demo --drift-ppm 120 --outlier-rate 0.10
```

## C usage

```c
#include "edgesync.h"

static es_kernel_t g_sync; /* 8,424 B, statically allocated */

void init_node(void) {
    es_config_t cfg = es_default_config(64.0);
    es_init(&g_sync, &cfg);
}

void on_sensor_sample(double t_loc, double sample) {
    es_push_sample(&g_sync, t_loc, sample);
}

void on_beacon(double t_loc, double t_ref) {
    es_update_sync(&g_sync, t_loc, t_ref);
}

int32_t read_synced_sample(double t_ref, double *out) {
    return es_resample_at(&g_sync, t_ref, out); /* >= 0 on success */
}
```

## Reproducing the paper

```bash
make all
export EDGESYNC_WESAD_DIR=/path/to/wesad_excerpts   # see "Data" below
cd paper
PYTHONPATH=../src python scripts/run_experiments.py   # writes results/results_summary.json
python scripts/make_figures.py                        # writes figures/
```

The run takes about ten seconds. The accuracy experiments are semi-synthetic: real WESAD
signals are resampled onto simulated drifting clocks with simulated beacon jitter, burst
delays and packet loss.

## Data

The demo, quickstart and test suite run on synthetic PPG/ECG signals generated in code
(`synthetic_ppg`, `synthetic_ecg`), so no external data are needed to use edgesync.

The paper's benchmarks use excerpts of the WESAD dataset (P. Schmidt et al., "Introducing
WESAD, a multimodal dataset for wearable stress and affect detection," ICMI 2018,
doi:10.1145/3242969.3242985). WESAD may be used for scientific, non-commercial purposes
only, so it is not redistributed here. Obtain it from the
[official page](https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/), prepare the
two excerpt files (`wesad_s10_slice.npz`, `wesad_5subj_events.json`), and set
`EDGESYNC_WESAD_DIR` to their directory. Tests that need WESAD are skipped when it is absent.

## Citation

See `CITATION.cff`.

## License

MIT (code only). WESAD is not included and remains subject to its own terms of use.
