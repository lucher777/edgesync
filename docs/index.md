# EdgeSync Overview

`edgesync` is a freestanding ANSI C11 static-memory microkernel and Python
verification toolkit for wearable clock synchronization and
fractional-delay resampling.

## Architecture

1. **Power-of-Two Static Ring Buffer**: Embedded inside `es_kernel_t`
   (8,424 bytes FP64; 4,344 bytes FP32) with branch-free bitmask indexing and
   zero heap allocation (`nm -u edgesync.o` has zero undefined symbols).
2. **Origin-Centered Redescending Huber-RLS Clock Estimator**: Tracks affine
   crystal frequency drift $\alpha$ and startup offset $\theta_0$ in $O(1)$
   time per beacon while rejecting asymmetric BLE packet congestion bursts
   without covariance wind-up via effective forgetting
   $\lambda_{\mathrm{eff}} = 1 - w(1 - \lambda)$.
3. **Branch-Free 4-Tap Lagrange Cubic Farrow Resampler**: Evaluates degree-3
   Horner polynomials with exact $\frac{3}{128} h^4 \|x^{(4)}\|_\infty$ error
   bounds and deterministic gap fallback under BLE packet drops.
