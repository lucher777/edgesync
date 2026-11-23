/**
 * @file edgesync.h
 * @brief Pure ANSI C11 zero-heap static-memory streaming clock synchronization
 *        and 4-tap cubic Farrow fractional-delay resampling microkernel.
 *
 * Designed for resource-constrained wearable and IoT microcontrollers
 * (e.g., ARM Cortex-M4/M33, ESP32-S3, RISC-V RV32IMFC) that need bounded
 * per-call work (no unbounded loops) and a static SRAM footprint (< 16 KB)
 * with no dynamic heap allocation. Timing has been measured on x86-64 only.
 *
 * @copyright MIT License
 */

#ifndef EDGESYNC_H
#define EDGESYNC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compile-time power-of-two capacity of the static ring buffer.
 *
 * Defaults to 512 samples (8.0 s at 64 Hz PPG, 2.048 s at 250 Hz ECG),
 * yielding an exact static footprint of ~8.4 KB for double-precision
 * ::es_kernel_t and ~4.3 KB for single-precision ::es_kernel_f32_t.
 */
#ifndef ES_RING_CAP
#define ES_RING_CAP 512U
#endif

/** @brief Bitmask for branch-free power-of-two ring indexing. */
#define ES_RING_MASK (ES_RING_CAP - 1U)

/** @brief Maximum compile-time binary-search depth (ceil(log2(ES_RING_CAP))). */
#define ES_MAX_BSEARCH_ITERS 16U

/**
 * @brief Return status codes for all EdgeSync microkernel operations.
 */
typedef enum es_status {
    ES_OK = 0,               /**< Success (4-tap cubic Farrow executed). */
    ES_OK_GAP_LINEAR = 1,    /**< Success via linear fallback across packet gap. */
    ES_OK_EDGE_LINEAR = 2,   /**< Success via linear fallback at ring boundary. */
    ES_ERR_NULL = -1,        /**< Null pointer argument supplied. */
    ES_ERR_PARAM = -2,       /**< Invalid or non-finite configuration/input. */
    ES_ERR_UNDERRUN = -3,    /**< Target time precedes oldest buffered sample. */
    ES_ERR_OVERRUN = -4,     /**< Target time exceeds newest buffered sample. */
    ES_ERR_NONMONOTONIC = -5 /**< Pushed sample timestamp is non-monotonic. */
} es_status_t;

/**
 * @brief Static configuration parameters for the EdgeSync microkernel.
 */
typedef struct es_config {
    double fs_in;          /**< Nominal input sampling rate in Hz (> 0). */
    double lambda;         /**< RLS forgetting factor in (0.5, 1.0]. */
    double huber_delta;    /**< Huber outlier residual threshold in seconds (> 0). */
    double max_drift_ppm;  /**< Physical crystal frequency drift clamp in ppm (> 0). */
    double init_p_offset;  /**< Initial RLS inverse-information offset variance. */
    double init_p_drift;   /**< Initial RLS inverse-information drift variance. */
    double gap_factor;     /**< Gap multiplier of nominal period h (> 1.0). */
} es_config_t;

/**
 * @brief Incremental Huber-weighted Recursive Least-Squares (H-RLS) clock state.
 *
 * Models the affine clock law:
 *   t_ref = r0 + delta0 + (1 + alpha) * (t_loc - t0)
 * in origin-centered coordinates to preserve full floating-point conditioning.
 */
typedef struct es_clock {
    double t_loc_origin;   /**< Local clock anchor t0 (seconds). */
    double t_ref_origin;   /**< Reference clock anchor r0 (seconds). */
    double delta0;         /**< Estimated residual offset at anchor (seconds). */
    double alpha;          /**< Estimated relative drift (dimensionless, 1e-6 = 1 ppm). */
    double p00;            /**< Covariance element P[0,0]. */
    double p01;            /**< Covariance element P[0,1] == P[1,0]. */
    double p11;            /**< Covariance element P[1,1]. */
    double last_residual;  /**< Most recent prediction residual (seconds). */
    double last_weight;    /**< Most recent Huber weight in (0, 1]. */
    uint64_t n_sync;       /**< Total valid synchronization updates processed. */
    uint64_t n_outliers;   /**< Count of Huber-downweighted sync updates. */
    uint8_t initialized;   /**< Non-zero once first sync anchor is recorded. */
    uint8_t _pad[7];       /**< Explicit alignment padding. */
} es_clock_t;

/**
 * @brief Deterministic telemetry counters maintained without heap allocation.
 */
typedef struct es_stats {
    uint64_t n_pushed;       /**< Total samples pushed into the ring buffer. */
    uint64_t n_resampled;    /**< Total output samples successfully produced. */
    uint64_t n_farrow_cubic; /**< Resamples using 4-tap Lagrange cubic Farrow. */
    uint64_t n_gap_linear;   /**< Resamples using bounded linear gap/edge fallback. */
    uint64_t n_underrun;     /**< Count of underrun resample queries. */
    uint64_t n_overrun;      /**< Count of overrun resample queries. */
    uint32_t ring_capacity;  /**< Compile-time ES_RING_CAP. */
    uint32_t kernel_bytes;   /**< Exact sizeof(kernel struct) in bytes. */
} es_stats_t;

/**
 * @brief Double-precision static-memory EdgeSync microkernel instance.
 *
 * Strictly zero dynamic allocation; all ring storage is statically embedded.
 */
typedef struct es_kernel {
    es_config_t cfg;              /**< Immutable runtime configuration. */
    es_clock_t clk;               /**< Online H-RLS clock-drift estimator. */
    es_stats_t stats;             /**< Runtime telemetry counters. */
    double ring_t[ES_RING_CAP];   /**< Static ring of local sample timestamps (s). */
    double ring_x[ES_RING_CAP];   /**< Static ring of sample amplitudes. */
    uint64_t write_seq;           /**< Monotonic count of pushed samples. */
    uint64_t read_seq;            /**< Persistent monotonic streaming read cursor. */
    double h_nom;                 /**< Nominal local sampling interval (1 / fs_in). */
} es_kernel_t;

/**
 * @brief Single-precision FPU static-memory EdgeSync microkernel (~4.3 KB SRAM).
 *
 * Stores sample amplitudes as IEEE-754 float32 and timestamps as float32 offsets from one of
 * two double-precision epoch bases. The base of epoch e = seq / ES_RING_CAP is the time of the
 * epoch's first sample; the ring never holds more than two epochs, so the bases alternate and
 * timestamp resolution does not degrade with stream length. Clock arithmetic is double.
 */
typedef struct es_kernel_f32 {
    es_config_t cfg;              /**< Immutable runtime configuration. */
    es_clock_t clk;               /**< Online H-RLS clock-drift estimator. */
    es_stats_t stats;             /**< Runtime telemetry counters. */
    double t_base[2];             /**< Epoch bases (s), indexed by (seq / ES_RING_CAP) & 1. */
    float ring_dt[ES_RING_CAP];   /**< Static ring of epoch-relative local timestamps (s). */
    float ring_x[ES_RING_CAP];    /**< Static ring of float32 sample amplitudes. */
    uint64_t write_seq;           /**< Monotonic count of pushed samples. */
    uint64_t read_seq;            /**< Persistent monotonic streaming read cursor. */
    double h_nom;                 /**< Nominal local sampling interval (1 / fs_in). */
} es_kernel_f32_t;

/** @brief Return exact compile-time byte size of ::es_kernel_t. */
size_t es_sizeof_kernel(void);

/** @brief Return exact compile-time byte size of ::es_kernel_f32_t. */
size_t es_sizeof_kernel_f32(void);

/** @brief Return compile-time ring capacity ::ES_RING_CAP. */
uint32_t es_ring_capacity(void);

/**
 * @brief Build a validated default configuration for nominal input rate @p fs_in.
 *
 * @param fs_in Nominal input sampling frequency in Hz (e.g., 64.0 or 250.0).
 * @return Default ::es_config_t suitable for BLE wearable nodes.
 */
es_config_t es_default_config(double fs_in);

/**
 * @brief Initialize a double-precision EdgeSync microkernel in static memory.
 *
 * @param k   Pointer to caller-allocated ::es_kernel_t.
 * @param cfg Configuration parameters (must satisfy fs_in > 0, lambda in (0.5, 1]).
 * @return ::ES_OK on success, or negative ::es_status_t on invalid arguments.
 */
int32_t es_init(es_kernel_t *k, const es_config_t *cfg);

/**
 * @brief Reset ring buffer, clock state, and telemetry while preserving config.
 *
 * @param k Pointer to initialized ::es_kernel_t.
 */
void es_reset(es_kernel_t *k);

/**
 * @brief Incorporate one asynchronous synchronization timestamp pair in O(1) time.
 *
 * Performs origin-centered Huber-weighted Recursive Least Squares (H-RLS) with
 * effective forgetting lambda_eff = 1 - w * (1 - lambda) to prevent covariance
 * wind-up during burst packet loss or BLE arrival spikes.
 *
 * @param k     Pointer to initialized ::es_kernel_t.
 * @param t_loc Local wearable node timestamp (seconds).
 * @param t_ref Host/reference timeline timestamp (seconds).
 * @return ::ES_OK on success, or negative ::es_status_t on invalid input.
 */
int32_t es_update_sync(es_kernel_t *k, double t_loc, double t_ref);

/**
 * @brief Map a local timestamp to the estimated reference timeline in O(1) time.
 *
 * @param k     Pointer to initialized ::es_kernel_t.
 * @param t_loc Local timestamp (seconds).
 * @return Estimated reference timestamp t_ref (seconds).
 */
double es_loc_to_ref(const es_kernel_t *k, double t_loc);

/**
 * @brief Map a target reference timestamp to the local clock in O(1) time.
 *
 * @param k     Pointer to initialized ::es_kernel_t.
 * @param t_ref Target reference timestamp (seconds).
 * @return Corresponding local timestamp t_loc (seconds).
 */
double es_ref_to_loc(const es_kernel_t *k, double t_ref);

/**
 * @brief Return the current estimated crystal frequency drift in parts per million.
 *
 * @param k Pointer to initialized ::es_kernel_t.
 * @return Drift estimate 1e6 * alpha (ppm), or 0.0 if null/uninitialized.
 */
double es_get_drift_ppm(const es_kernel_t *k);

/**
 * @brief Return the current estimated clock offset at t_loc = 0 (seconds).
 *
 * @param k Pointer to initialized ::es_kernel_t.
 * @return Offset theta0 = t_ref_origin + delta0 - (1 + alpha) * t_loc_origin.
 */
double es_get_offset_s(const es_kernel_t *k);

/**
 * @brief Push one incoming sensor sample into the static ring buffer.
 *
 * Strictly O(1) constant-time write with power-of-two bitmask wrap-around.
 *
 * @param k     Pointer to initialized ::es_kernel_t.
 * @param t_loc Monotonically increasing local sample timestamp (seconds).
 * @param x     Sensor sample value.
 * @return ::ES_OK on success, or negative ::es_status_t on invalid/backward time.
 */
int32_t es_push_sample(es_kernel_t *k, double t_loc, double x);

/**
 * @brief Branch-free 4-tap Lagrange cubic Farrow polynomial kernel (Horner form).
 *
 * Evaluates the unique cubic polynomial passing through (-1, xm1), (0, x0),
 * (1, x1), (2, x2) at fractional delay mu in [0, 1].
 *
 * @param xm1 Sample at index k - 1.
 * @param x0  Sample at index k.
 * @param x1  Sample at index k + 1.
 * @param x2  Sample at index k + 2.
 * @param mu  Fractional delay in [0, 1].
 * @return Interpolated value x(k + mu).
 */
double es_farrow_cubic_4tap(double xm1, double x0, double x1, double x2, double mu);

/**
 * @brief Resample the buffered stream at target reference timestamp @p t_ref.
 *
 * Uses a persistent monotonic read cursor (O(1) fast path in steady-state
 * streaming) with a compile-time bounded binary search (<= log2(ES_RING_CAP)
 * steps) so the work per call is bounded, followed by branch-free 4-tap
 * cubic Farrow interpolation (or linear fallback across packet-drop gaps).
 *
 * @param k     Pointer to initialized ::es_kernel_t.
 * @param t_ref Target reference timeline timestamp (seconds).
 * @param out_x Output pointer receiving the resampled amplitude.
 * @return ::ES_OK, ::ES_OK_GAP_LINEAR, ::ES_OK_EDGE_LINEAR, or negative error.
 */
int32_t es_resample_at(es_kernel_t *k, double t_ref, double *out_x);

/**
 * @brief Resample a contiguous array of target reference timestamps in-place.
 *
 * @param k              Pointer to initialized ::es_kernel_t.
 * @param t_ref_arr      Input array of @p n target reference timestamps.
 * @param n              Number of target timestamps.
 * @param out_x_arr      Output array of length @p n for resampled values.
 * @param out_status_arr Optional output array of length @p n for per-sample status (may be NULL).
 * @return Count of successfully resampled points (>= 0), or negative ::es_status_t.
 */
int32_t es_resample_batch(
    es_kernel_t *k,
    const double *t_ref_arr,
    size_t n,
    double *out_x_arr,
    int32_t *out_status_arr);

/**
 * @brief Initialize a single-precision FPU EdgeSync microkernel (~4.3 KB SRAM).
 *
 * @param k   Pointer to caller-allocated ::es_kernel_f32_t.
 * @param cfg Configuration parameters.
 * @return ::ES_OK on success, or negative ::es_status_t on invalid arguments.
 */
int32_t es_f32_init(es_kernel_f32_t *k, const es_config_t *cfg);

/**
 * @brief Update the H-RLS clock estimator on a float32 microkernel instance.
 *
 * @param k     Pointer to initialized ::es_kernel_f32_t.
 * @param t_loc Local node timestamp (seconds).
 * @param t_ref Host/reference timestamp (seconds).
 * @return ::ES_OK on success, or negative ::es_status_t.
 */
int32_t es_f32_update_sync(es_kernel_f32_t *k, double t_loc, double t_ref);

/**
 * @brief Push one float32 sensor sample into the static float32 ring buffer.
 *
 * @param k     Pointer to initialized ::es_kernel_f32_t.
 * @param t_loc Monotonically increasing local timestamp (seconds).
 * @param x     Float32 sensor amplitude.
 * @return ::ES_OK on success, or negative ::es_status_t.
 */
int32_t es_f32_push_sample(es_kernel_f32_t *k, double t_loc, float x);

/**
 * @brief Branch-free 4-tap Lagrange cubic Farrow kernel in IEEE-754 float32.
 *
 * @param xm1 Sample at index k - 1.
 * @param x0  Sample at index k.
 * @param x1  Sample at index k + 1.
 * @param x2  Sample at index k + 2.
 * @param mu  Fractional delay in [0, 1].
 * @return Interpolated float32 amplitude.
 */
float es_farrow_cubic_4tap_f32(float xm1, float x0, float x1, float x2, float mu);

/**
 * @brief Resample the float32 static ring buffer at reference timestamp @p t_ref.
 *
 * @param k     Pointer to initialized ::es_kernel_f32_t.
 * @param t_ref Target reference timestamp (seconds).
 * @param out_x Output pointer receiving the resampled float32 amplitude.
 * @return ::ES_OK, ::ES_OK_GAP_LINEAR, ::ES_OK_EDGE_LINEAR, or negative error.
 */
int32_t es_f32_resample_at(es_kernel_f32_t *k, double t_ref, float *out_x);

/**
 * @brief Batch resample a float32 microkernel over @p n target timestamps.
 *
 * @param k              Pointer to initialized ::es_kernel_f32_t.
 * @param t_ref_arr      Input array of @p n reference timestamps.
 * @param n              Number of target timestamps.
 * @param out_x_arr      Output array of length @p n for float32 values.
 * @param out_status_arr Optional output array of length @p n for status codes.
 * @return Count of successfully resampled points (>= 0), or negative error.
 */
int32_t es_f32_resample_batch(
    es_kernel_f32_t *k,
    const double *t_ref_arr,
    size_t n,
    float *out_x_arr,
    int32_t *out_status_arr);

#ifdef __cplusplus
}
#endif

#endif /* EDGESYNC_H */
