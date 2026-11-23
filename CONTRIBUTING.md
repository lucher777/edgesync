# Contributing to EdgeSync

Thank you for contributing to EdgeSync.

## Hard Engineering Invariants

All pull requests modifying `csrc/edgesync.h` or `csrc/edgesync.c` must preserve:

1. **Zero Heap & Freestanding Linkage**: `gcc -std=c11 -O3 -Wall -Wextra -Werror -ffreestanding -fno-builtin -c csrc/edgesync.c` must compile cleanly with zero warnings and zero undefined symbols in `nm -u edgesync.o`.
2. **Static Footprint Budget**: `sizeof(es_kernel_t) <= 16384` and `sizeof(es_kernel_f32_t) <= 8192` enforced via C11 `_Static_assert`.
3. **Bounded Work per Call**: No unbounded loops, no recursion, and no dynamic memory allocation in any microkernel path.
4. **Numerical Parity**: `edgesync.equiv.verify_c11_python_parity()` must report `< 1e-12` discrepancy and all `pytest` tests must pass.

## Running Checks Locally

```bash
make all && make audit
PYTHONPATH=src pytest
```
