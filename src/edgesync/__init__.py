"""edgesync: static-memory clock synchronization and Farrow resampling kernel.

Provides thin ``ctypes`` bindings to the freestanding ANSI C11 ``edgesync``
microkernel (zero dynamic heap allocation, <16 KB static SRAM, O(1) Huber-RLS
clock-drift estimation, and branch-free 4-tap Lagrange cubic Farrow
fractional-delay resampling), together with an independent exact Python
reference implementation, binary symbol/footprint auditing utilities,
synthetic PPG/ECG signal generators, and WESAD benchmark loaders (data not
bundled; see :mod:`edgesync.datasets`).
"""

from .datasets import (
    WESAD_ENV_VAR,
    load_wesad_5subj_events,
    load_wesad_s10_slice,
    simulate_asynchronous_ble_stream,
    synthetic_ecg,
    synthetic_ppg,
    wesad_available,
    wesad_data_dir,
)
from .equiv import (
    audit_zero_heap_symbols,
    measure_object_sections,
    verify_c11_python_parity,
    verify_f32_quantization,
    verify_scipy_cubic_parity,
)
from .reference import (
    ReferenceConfig,
    ReferenceEdgeSync,
    farrow_cubic_4tap_py,
)
from .wrapper import (
    ES_ERR_NONMONOTONIC,
    ES_ERR_NULL,
    ES_ERR_OVERRUN,
    ES_ERR_PARAM,
    ES_ERR_UNDERRUN,
    ES_OK,
    ES_OK_EDGE_LINEAR,
    ES_OK_GAP_LINEAR,
    EdgeSyncConfig,
    EdgeSyncKernel,
    EdgeSyncKernelF32,
    compile_native_library,
    farrow_cubic_4tap,
    load_native_library,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ES_OK",
    "ES_OK_GAP_LINEAR",
    "ES_OK_EDGE_LINEAR",
    "ES_ERR_NULL",
    "ES_ERR_PARAM",
    "ES_ERR_UNDERRUN",
    "ES_ERR_OVERRUN",
    "ES_ERR_NONMONOTONIC",
    "EdgeSyncConfig",
    "EdgeSyncKernel",
    "EdgeSyncKernelF32",
    "ReferenceConfig",
    "ReferenceEdgeSync",
    "compile_native_library",
    "load_native_library",
    "farrow_cubic_4tap",
    "farrow_cubic_4tap_py",
    "audit_zero_heap_symbols",
    "measure_object_sections",
    "verify_c11_python_parity",
    "verify_f32_quantization",
    "verify_scipy_cubic_parity",
    "load_wesad_s10_slice",
    "load_wesad_5subj_events",
    "synthetic_ppg",
    "synthetic_ecg",
    "wesad_available",
    "wesad_data_dir",
    "WESAD_ENV_VAR",
    "simulate_asynchronous_ble_stream",
]
