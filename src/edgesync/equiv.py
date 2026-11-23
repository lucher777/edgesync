"""Numerical parity, quantization, and binary zero-heap auditing for EdgeSync.

Provides automated verification tools confirming:
1. Strictly zero dynamic heap allocator imports (``malloc``, ``calloc``,
   ``realloc``, ``free``) via ``nm -u`` on the compiled ``edgesync.o`` object.
2. Exact ``.text`` / ``.data`` / ``.bss`` footprints via ``size``.
3. Strict ``< 1e-12`` numerical parity between the compiled C11 microkernel and
   the independent Python reference implementation.
4. Agreement between the 4-tap Lagrange cubic Farrow kernel and exact degree-3
   SciPy/NumPy polynomial interpolation.
5. Quantization error (< 1e-5 relative) between FP64 and FP32 microkernels.
"""

from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Dict, List, Optional

import numpy as np

from .reference import ReferenceConfig, ReferenceEdgeSync
from .wrapper import (
    EdgeSyncConfig,
    EdgeSyncKernel,
    EdgeSyncKernelF32,
    _find_gcc_executable,
    _locate_csrc_dir,
    farrow_cubic_4tap,
)

FORBIDDEN_HEAP_SYMBOLS = frozenset(
    {
        "malloc",
        "calloc",
        "realloc",
        "free",
        "_malloc",
        "_calloc",
        "_realloc",
        "_free",
        "aligned_alloc",
        "posix_memalign",
    }
)


def _find_binutil(tool_name: str) -> Optional[str]:
    """Locate a binutils executable (``nm`` or ``size``) matching the compiler."""
    found = shutil.which(tool_name)
    if found:
        return found
    gcc = _find_gcc_executable()
    if gcc:
        ext = ".exe" if platform.system() == "Windows" else ""
        candidate = Path(gcc).parent / f"{tool_name}{ext}"
        if candidate.is_file():
            return str(candidate)
    return None


def _ensure_object_file(opt_flag: str = "-O3") -> Path:
    """Compile ``csrc/edgesync.c`` to an object file and return its path."""
    csrc = _locate_csrc_dir()
    obj_name = "edgesync.o" if opt_flag == "-O3" else "edgesync_os.o"
    obj_path = csrc / obj_name
    gcc = _find_gcc_executable()
    if gcc is None:
        if obj_path.is_file():
            return obj_path
        raise RuntimeError("gcc not found and prebuilt object file missing.")

    env = os.environ.copy()
    env["PATH"] = str(Path(gcc).parent) + os.pathsep + env.get("PATH", "")
    cmd = [
        gcc,
        "-std=c11",
        opt_flag,
        "-Wall",
        "-Wextra",
        "-Werror",
        "-ffreestanding",
        "-fno-builtin",
        "-c",
        "edgesync.c",
        "-o",
        obj_name,
    ]
    proc = subprocess.run(
        cmd, cwd=str(csrc), env=env, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Object compile failed: {proc.stderr.strip()}")
    return obj_path


def audit_zero_heap_symbols() -> Dict[str, object]:
    """Audit ``edgesync.o`` via ``nm -u`` to verify zero heap allocator imports.

    Returns:
        Dictionary containing ``undefined_symbols``, ``heap_symbols``,
        ``zero_heap_verified``, and ``freestanding_zero_undef``.
    """
    obj_path = _ensure_object_file("-O3")
    nm = _find_binutil("nm")
    if nm is None:
        raise RuntimeError("nm utility not found on system.")

    env = os.environ.copy()
    env["PATH"] = str(Path(nm).parent) + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(
        [nm, "-u", obj_path.name],
        cwd=str(obj_path.parent),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nm audit failed: {proc.stderr.strip()}")

    undef: List[str] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split()
        if parts:
            undef.append(parts[-1])

    heap_hits = [s for s in undef if s.lstrip("_") in FORBIDDEN_HEAP_SYMBOLS or s in FORBIDDEN_HEAP_SYMBOLS]
    return {
        "undefined_symbols": undef,
        "heap_symbols": heap_hits,
        "zero_heap_verified": len(heap_hits) == 0,
        "freestanding_zero_undef": len(undef) == 0,
    }


def measure_object_sections(opt_flag: str = "-O3") -> Dict[str, int]:
    """Measure exact ``.text``, ``.data``, and ``.bss`` byte sizes via ``size``.

    Args:
        opt_flag: GCC optimization level (e.g., ``"-O3"`` or ``"-Os"``).

    Returns:
        Dictionary with keys ``text_bytes``, ``data_bytes``, ``bss_bytes``,
        ``total_bytes``, ``sizeof_kernel_fp64_bytes``, and
        ``sizeof_kernel_fp32_bytes``.
    """
    obj_path = _ensure_object_file(opt_flag)
    size_bin = _find_binutil("size")
    if size_bin is None:
        raise RuntimeError("size utility not found on system.")

    env = os.environ.copy()
    env["PATH"] = str(Path(size_bin).parent) + os.pathsep + env.get("PATH", "")
    proc = subprocess.run(
        [size_bin, obj_path.name],
        cwd=str(obj_path.parent),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"size invocation failed: {proc.stderr.strip()}")

    if opt_flag != "-O3" and obj_path.is_file():
        obj_path.unlink(missing_ok=True)

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Unexpected size output: {proc.stdout}")
    cols = lines[1].split()
    k64 = EdgeSyncKernel()
    k32 = EdgeSyncKernelF32()
    return {
        "text_bytes": int(cols[0]),
        "data_bytes": int(cols[1]),
        "bss_bytes": int(cols[2]),
        "total_bytes": int(cols[3]),
        "sizeof_kernel_fp64_bytes": k64.kernel_bytes,
        "sizeof_kernel_fp32_bytes": k32.kernel_bytes,
    }


def verify_c11_python_parity(
    n_samples: int = 2000,
    drift_ppm: float = 135.0,
    offset_s: float = -2.45,
    outlier_fraction: float = 0.1,
    seed: int = 42,
) -> Dict[str, float]:
    """Verify strict ``< 1e-12`` parity between C11 and Python reference solvers.

    Streams identical drifted multi-tone samples and outlier-corrupted beacons
    through :class:`EdgeSyncKernel` (C11) and :class:`ReferenceEdgeSync`
    (Python) and reports maximum absolute discrepancies.

    Args:
        n_samples: Number of streaming samples to compare.
        drift_ppm: Injected relative crystal drift in ppm.
        offset_s: Injected startup offset in seconds.
        outlier_fraction: Fraction of sync beacons corrupted by burst spikes.
        seed: Deterministic RNG seed.

    Returns:
        Dictionary of maximum absolute differences across output waveform,
        estimated drift (ppm), estimated offset (s), and status mismatches.
    """
    rng = np.random.default_rng(seed)
    fs = 64.0
    h = 1.0 / fs
    alpha = drift_ppm * 1e-6

    cfg_c = EdgeSyncConfig(fs_in=fs)
    cfg_py = ReferenceConfig(fs_in=fs)
    kc = EdgeSyncKernel(cfg_c)
    kpy = ReferenceEdgeSync(cfg_py)

    max_val_err = 0.0
    status_mismatches = 0
    n_compared = 0

    for i in range(n_samples):
        t_loc = i * h
        t_ref_true = (1.0 + alpha) * t_loc + offset_s
        x = (
            math.sin(2.0 * math.pi * 1.15 * t_ref_true)
            + 0.35 * math.cos(2.0 * math.pi * 2.75 * t_ref_true)
        )
        assert kc.push_sample(t_loc, x) == kpy.push_sample(t_loc, x)

        if i % 32 == 0:
            jitter = float(rng.normal(0.0, 0.002))
            if i >= 128 and rng.random() < outlier_fraction:
                jitter += float(rng.choice([-1.0, 1.0]) * rng.uniform(0.08, 0.25))
            t_beacon_ref = t_ref_true + jitter
            assert kc.update_sync(t_loc, t_beacon_ref) == kpy.update_sync(
                t_loc, t_beacon_ref
            )

        if i >= 8:
            t_q = t_ref_true - 3.37 * h
            st_c, y_c = kc.resample_at(t_q)
            st_py, y_py = kpy.resample_at(t_q)
            if st_c != st_py:
                status_mismatches += 1
            if st_c >= 0 and st_py >= 0:
                err = abs(y_c - y_py)
                if err > max_val_err:
                    max_val_err = err
                n_compared += 1

    return {
        "n_compared": float(n_compared),
        "status_mismatches": float(status_mismatches),
        "max_abs_sample_diff": float(max_val_err),
        "drift_ppm_diff": abs(kc.drift_ppm - kpy.drift_ppm),
        "offset_s_diff": abs(kc.offset_s - kpy.offset_s),
    }


def verify_scipy_cubic_parity(n_points: int = 200) -> Dict[str, float]:
    """Verify 4-tap Lagrange cubic Farrow exactness against NumPy/SciPy.

    Checks exact recovery on arbitrary degree-3 polynomials (machine precision
    ``< 1e-13``) and comparison against ``scipy.interpolate.lagrange``.

    Args:
        n_points: Number of fractional delays ``mu`` in ``[0, 1]`` to test.

    Returns:
        Dictionary with ``max_cubic_poly_error`` and ``max_scipy_lagrange_diff``.
    """
    from scipy.interpolate import lagrange

    nodes = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    # Arbitrary degree-3 polynomial p(u) = 0.7 u^3 - 1.3 u^2 + 0.4 u + 2.1
    vals = 0.7 * nodes**3 - 1.3 * nodes**2 + 0.4 * nodes + 2.1
    poly = lagrange(nodes, vals)

    mus = np.linspace(0.0, 1.0, n_points)
    max_exact_err = 0.0
    max_scipy_err = 0.0
    for mu in mus:
        y_c = farrow_cubic_4tap(vals[0], vals[1], vals[2], vals[3], float(mu))
        y_exact = 0.7 * mu**3 - 1.3 * mu**2 + 0.4 * mu + 2.1
        y_sp = float(poly(mu))
        max_exact_err = max(max_exact_err, abs(y_c - y_exact))
        max_scipy_err = max(max_scipy_err, abs(y_c - y_sp))

    return {
        "max_cubic_poly_error": float(max_exact_err),
        "max_scipy_lagrange_diff": float(max_scipy_err),
    }


def verify_f32_quantization(n_samples: int = 2000) -> Dict[str, float]:
    """Measure quantization error between FP64 and FP32 C11 microkernels.

    Args:
        n_samples: Number of streaming samples to compare.

    Returns:
        Dictionary with ``max_abs_diff`` and ``rmse_diff`` across valid outputs.
    """
    fs = 64.0
    h = 1.0 / fs
    alpha = 110e-6
    offset_s = 1.25

    cfg = EdgeSyncConfig(fs_in=fs)
    k64 = EdgeSyncKernel(cfg)
    k32 = EdgeSyncKernelF32(cfg)

    diffs = []
    for i in range(n_samples):
        t_loc = i * h
        t_ref = (1.0 + alpha) * t_loc + offset_s
        x = math.sin(2.0 * math.pi * 1.2 * t_ref) + 0.25 * math.cos(
            2.0 * math.pi * 3.1 * t_ref
        )
        k64.push_sample(t_loc, x)
        k32.push_sample(t_loc, x)
        if i % 32 == 0:
            k64.update_sync(t_loc, t_ref)
            k32.update_sync(t_loc, t_ref)
        if i >= 8:
            t_q = t_ref - 3.25 * h
            s64, y64 = k64.resample_at(t_q)
            s32, y32 = k32.resample_at(t_q)
            if s64 == 0 and s32 == 0:
                diffs.append(abs(y64 - y32))

    arr = np.asarray(diffs, dtype=np.float64)
    return {
        "n_compared": float(arr.size),
        "max_abs_diff": float(np.max(arr)) if arr.size else 0.0,
        "rmse_diff": float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0,
    }


import math  # noqa: E402
