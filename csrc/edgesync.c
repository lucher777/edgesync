/**
 * @file edgesync.c
 * @brief Freestanding ANSI C11 implementation of the EdgeSync zero-heap
 *        streaming clock-synchronization and cubic Farrow microkernel.
 *
 * Strictly zero dynamic memory allocation (zero malloc/calloc/realloc/free)
 * and zero external libc/libm calls so the compiled object links cleanly on
 * bare-metal microcontrollers and freestanding shared libraries alike.
 *
 * @copyright MIT License
 */

#include "edgesync.h"

_Static_assert(
    (ES_RING_CAP >= 8U) && ((ES_RING_CAP & (ES_RING_CAP - 1U)) == 0U),
    "ES_RING_CAP must be a power of two >= 8");

_Static_assert(
    sizeof(es_kernel_t) <= 16384U,
    "es_kernel_t static footprint must remain strictly <= 16 KB");

_Static_assert(
    sizeof(es_kernel_f32_t) <= 8192U,
    "es_kernel_f32_t static footprint must remain strictly <= 8 KB");

/* ---- Freestanding scalar helpers (zero libm / libc dependency) ---- */

static inline double es_fabs(double x) {
    return (x < 0.0) ? -x : x;
}

static inline int es_isfinite_d(double x) {
    return (x == x) && ((x - x) == 0.0);
}

static inline int es_isfinite_f(float x) {
    return (x == x) && ((x - x) == 0.0f);
}

static inline double es_clamp_d(double x, double lo, double hi) {
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

static inline float es_clamp_f(float x, float lo, float hi) {
    if (x < lo) {
        return lo;
    }
    if (x > hi) {
        return hi;
    }
    return x;
}

static void es_zero_bytes(void *ptr, size_t n) {
    volatile uint8_t *p = (volatile uint8_t *)ptr;
    for (size_t i = 0U; i < n; ++i) {
        p[i] = 0U;
    }
}

static int es_validate_config(const es_config_t *cfg) {
    if (cfg == NULL) {
        return 0;
    }
    if (!es_isfinite_d(cfg->fs_in) || cfg->fs_in <= 0.0) {
        return 0;
    }
    if (!es_isfinite_d(cfg->lambda) || cfg->lambda <= 0.5 || cfg->lambda > 1.0) {
        return 0;
    }
    if (!es_isfinite_d(cfg->huber_delta) || cfg->huber_delta <= 0.0) {
        return 0;
    }
    if (!es_isfinite_d(cfg->max_drift_ppm) || cfg->max_drift_ppm <= 0.0) {
        return 0;
    }
    if (!es_isfinite_d(cfg->init_p_offset) || cfg->init_p_offset <= 0.0) {
        return 0;
    }
    if (!es_isfinite_d(cfg->init_p_drift) || cfg->init_p_drift <= 0.0) {
        return 0;
    }
    if (!es_isfinite_d(cfg->gap_factor) || cfg->gap_factor <= 1.0) {
        return 0;
    }
    return 1;
}

/* ---- Shared O(1) Huber-Weighted RLS Clock Estimator ---- */

static void es_clock_reset(es_clock_t *clk, const es_config_t *cfg) {
    clk->t_loc_origin = 0.0;
    clk->t_ref_origin = 0.0;
    clk->delta0 = 0.0;
    clk->alpha = 0.0;
    clk->p00 = cfg->init_p_offset;
    clk->p01 = 0.0;
    clk->p11 = cfg->init_p_drift;
    clk->last_residual = 0.0;
    clk->last_weight = 1.0;
    clk->n_sync = 0U;
    clk->n_outliers = 0U;
    clk->initialized = 0U;
}

static int32_t es_clock_update(
    es_clock_t *clk,
    const es_config_t *cfg,
    double t_loc,
    double t_ref) {
    if (!es_isfinite_d(t_loc) || !es_isfinite_d(t_ref)) {
        return ES_ERR_PARAM;
    }

    if (!clk->initialized) {
        clk->t_loc_origin = t_loc;
        clk->t_ref_origin = t_ref;
        clk->delta0 = 0.0;
        clk->alpha = 0.0;
        clk->p00 = cfg->init_p_offset;
        clk->p01 = 0.0;
        clk->p11 = cfg->init_p_drift;
        clk->last_residual = 0.0;
        clk->last_weight = 1.0;
        clk->n_sync = 1U;
        clk->n_outliers = 0U;
        clk->initialized = 1U;
        return ES_OK;
    }

    /* Origin-centered regressor phi = [1, u]^T and target y = (t_ref - r0) - u */
    const double u = t_loc - clk->t_loc_origin;
    const double y = (t_ref - clk->t_ref_origin) - u;
    const double y_hat = clk->delta0 + clk->alpha * u;
    const double err = y - y_hat;
    const double abs_err = es_fabs(err);

    /* Redescending Huber M-estimator weight (active after 3 warm-up beacons) */
    double w = 1.0;
    if ((clk->n_sync >= 3U) && (abs_err > cfg->huber_delta)) {
        clk->n_outliers += 1U;
        if (abs_err > (3.0 * cfg->huber_delta)) {
            /* Severe BLE burst outlier: redescending rejection, zero covariance drift */
            clk->last_residual = err;
            clk->last_weight = 0.0;
            clk->n_sync += 1U;
            return ES_OK;
        }
        w = cfg->huber_delta / abs_err;
    }

    /* Effective forgetting prevents covariance wind-up during outlier bursts */
    const double lam_eff = 1.0 - w * (1.0 - cfg->lambda);
    const double inv_lam = 1.0 / lam_eff;

    /* Exact 2x2 closed-form RLS update: v = P * phi */
    const double v0 = clk->p00 + clk->p01 * u;
    const double v1 = clk->p01 + clk->p11 * u;
    const double denom = lam_eff + w * (v0 + v1 * u);

    if (denom <= 1e-18 || !es_isfinite_d(denom)) {
        return ES_ERR_PARAM;
    }

    const double gain_scale = w / denom;
    const double k0 = gain_scale * v0;
    const double k1 = gain_scale * v1;

    /* State update with physical crystal drift clamp preserving contraction */
    const double max_alpha = cfg->max_drift_ppm * 1e-6;
    const double alpha_raw = clk->alpha + k1 * err;
    const double alpha_clamped = es_clamp_d(alpha_raw, -max_alpha, max_alpha);
    if (alpha_clamped == alpha_raw) {
        clk->delta0 += k0 * err;
        clk->alpha = alpha_raw;
    } else {
        const double d_alpha = alpha_clamped - clk->alpha;
        const double step_y = (k0 + k1 * u) * err;
        clk->delta0 += step_y - d_alpha * u;
        clk->alpha = alpha_clamped;
    }

    /* Symmetric covariance update with bounded positive-definiteness guard */
    double np00 = (clk->p00 - k0 * v0) * inv_lam;
    double np01 = (clk->p01 - k0 * v1) * inv_lam;
    double np11 = (clk->p11 - k1 * v1) * inv_lam;

    const double p00_max = cfg->init_p_offset * 10.0;
    const double p11_max = cfg->init_p_drift * 10.0;
    if (np00 < 1e-15) {
        np00 = 1e-15;
    } else if (np00 > p00_max) {
        np00 = p00_max;
    }
    if (np11 < 1e-18) {
        np11 = 1e-18;
    } else if (np11 > p11_max) {
        np11 = p11_max;
    }

    /* Enforce Cauchy-Schwarz bound p01^2 <= 0.999999 * p00 * p11 */
    const double max_p01_sq = 0.999999 * np00 * np11;
    if ((np01 * np01) > max_p01_sq) {
        /* Scale off-diagonal gently while remaining libm-free */
        np01 *= 0.95;
    }

    clk->p00 = np00;
    clk->p01 = np01;
    clk->p11 = np11;
    clk->last_residual = err;
    clk->last_weight = w;
    clk->n_sync += 1U;
    return ES_OK;
}

static inline double es_clock_loc_to_ref(const es_clock_t *clk, double t_loc) {
    if (!clk->initialized) {
        return t_loc;
    }
    return clk->t_ref_origin + clk->delta0 + (1.0 + clk->alpha) * (t_loc - clk->t_loc_origin);
}

static inline double es_clock_ref_to_loc(const es_clock_t *clk, double t_ref) {
    if (!clk->initialized) {
        return t_ref;
    }
    return clk->t_loc_origin + (t_ref - clk->t_ref_origin - clk->delta0) / (1.0 + clk->alpha);
}

/* ---- Introspection & Defaults ---- */

size_t es_sizeof_kernel(void) {
    return sizeof(es_kernel_t);
}

size_t es_sizeof_kernel_f32(void) {
    return sizeof(es_kernel_f32_t);
}

uint32_t es_ring_capacity(void) {
    return (uint32_t)ES_RING_CAP;
}

es_config_t es_default_config(double fs_in) {
    es_config_t cfg;
    cfg.fs_in = (es_isfinite_d(fs_in) && fs_in > 0.0) ? fs_in : 64.0;
    cfg.lambda = 0.995;
    cfg.huber_delta = 0.015;
    cfg.max_drift_ppm = 500.0;
    cfg.init_p_offset = 100.0;
    cfg.init_p_drift = 0.05;
    cfg.gap_factor = 1.5;
    return cfg;
}

/* ---- Double-Precision Microkernel API ---- */

int32_t es_init(es_kernel_t *k, const es_config_t *cfg) {
    if (k == NULL || cfg == NULL) {
        return ES_ERR_NULL;
    }
    if (!es_validate_config(cfg)) {
        return ES_ERR_PARAM;
    }
    es_zero_bytes(k, sizeof(es_kernel_t));
    k->cfg = *cfg;
    k->h_nom = 1.0 / cfg->fs_in;
    es_clock_reset(&k->clk, &k->cfg);
    k->stats.ring_capacity = (uint32_t)ES_RING_CAP;
    k->stats.kernel_bytes = (uint32_t)sizeof(es_kernel_t);
    return ES_OK;
}

void es_reset(es_kernel_t *k) {
    if (k == NULL) {
        return;
    }
    es_config_t saved = k->cfg;
    (void)es_init(k, &saved);
}

int32_t es_update_sync(es_kernel_t *k, double t_loc, double t_ref) {
    if (k == NULL) {
        return ES_ERR_NULL;
    }
    return es_clock_update(&k->clk, &k->cfg, t_loc, t_ref);
}

double es_loc_to_ref(const es_kernel_t *k, double t_loc) {
    if (k == NULL) {
        return t_loc;
    }
    return es_clock_loc_to_ref(&k->clk, t_loc);
}

double es_ref_to_loc(const es_kernel_t *k, double t_ref) {
    if (k == NULL) {
        return t_ref;
    }
    return es_clock_ref_to_loc(&k->clk, t_ref);
}

double es_get_drift_ppm(const es_kernel_t *k) {
    if (k == NULL || !k->clk.initialized) {
        return 0.0;
    }
    return k->clk.alpha * 1e6;
}

double es_get_offset_s(const es_kernel_t *k) {
    if (k == NULL || !k->clk.initialized) {
        return 0.0;
    }
    return k->clk.t_ref_origin + k->clk.delta0 - (1.0 + k->clk.alpha) * k->clk.t_loc_origin;
}

int32_t es_push_sample(es_kernel_t *k, double t_loc, double x) {
    if (k == NULL) {
        return ES_ERR_NULL;
    }
    if (!es_isfinite_d(t_loc) || !es_isfinite_d(x)) {
        return ES_ERR_PARAM;
    }
    if (k->write_seq > 0U) {
        const uint64_t prev_slot = (k->write_seq - 1U) & ES_RING_MASK;
        if (t_loc <= k->ring_t[prev_slot]) {
            return ES_ERR_NONMONOTONIC;
        }
    }
    const uint64_t slot = k->write_seq & ES_RING_MASK;
    k->ring_t[slot] = t_loc;
    k->ring_x[slot] = x;
    k->write_seq += 1U;
    k->stats.n_pushed += 1U;
    return ES_OK;
}

double es_farrow_cubic_4tap(double xm1, double x0, double x1, double x2, double mu) {
    /*
     * Exact 4-point Lagrange cubic Farrow filter on nodes {-1, 0, 1, 2}:
     *   c0 = x0
     *   c2 = 0.5*(xm1 + x1) - x0
     *   c3 = (1/6)*(x2 - xm1) + 0.5*(x0 - x1)
     *   c1 = (x1 - x0) - c2 - c3
     *   y  = ((c3 * mu + c2) * mu + c1) * mu + c0
     * Branch-free, zero loops, exact O(h^4 ||x^(4)||_inf) remainder (3/128).
     */
    const double d01 = x1 - x0;
    const double c2 = 0.5 * (xm1 + x1) - x0;
    const double c3 = (1.0 / 6.0) * (x2 - xm1) - 0.5 * d01;
    const double c1 = d01 - c2 - c3;
    return ((c3 * mu + c2) * mu + c1) * mu + x0;
}

static uint64_t es_find_bracket_d(
    const es_kernel_t *k,
    uint64_t oldest,
    uint64_t newest,
    double t_loc) {
    /*
     * O(1) steady-state fast path: test persistent read_seq and read_seq + 1.
     * Fallback: bounded binary search (<= log2(ES_RING_CAP) iters).
     */
    uint64_t cur = k->read_seq;
    if (cur < oldest) {
        cur = oldest;
    } else if (cur >= newest) {
        cur = newest - 1U;
    }

    if ((t_loc >= k->ring_t[cur & ES_RING_MASK]) &&
        (t_loc <= k->ring_t[(cur + 1U) & ES_RING_MASK])) {
        return cur;
    }

    if ((cur + 2U) <= newest) {
        const uint64_t nxt = cur + 1U;
        if ((t_loc >= k->ring_t[nxt & ES_RING_MASK]) &&
            (t_loc <= k->ring_t[(nxt + 1U) & ES_RING_MASK])) {
            return nxt;
        }
    }

    /* Strictly bounded binary search: max ceil(log2(ES_RING_CAP)) iterations */
    uint64_t lo = oldest;
    uint64_t hi = newest - 1U;
    for (uint32_t iter = 0U; iter < ES_MAX_BSEARCH_ITERS; ++iter) {
        if (lo >= hi) {
            break;
        }
        const uint64_t mid = lo + ((hi - lo) >> 1U);
        if (k->ring_t[(mid + 1U) & ES_RING_MASK] < t_loc) {
            lo = mid + 1U;
        } else if (k->ring_t[mid & ES_RING_MASK] > t_loc) {
            hi = mid;
        } else {
            return mid;
        }
    }
    return lo;
}

int32_t es_resample_at(es_kernel_t *k, double t_ref, double *out_x) {
    if (k == NULL || out_x == NULL) {
        return ES_ERR_NULL;
    }
    if (!es_isfinite_d(t_ref)) {
        return ES_ERR_PARAM;
    }
    if (k->write_seq < 2U) {
        k->stats.n_underrun += 1U;
        return ES_ERR_UNDERRUN;
    }

    const double t_loc = es_clock_ref_to_loc(&k->clk, t_ref);
    const uint64_t newest = k->write_seq - 1U;
    const uint64_t oldest =
        (k->write_seq > (uint64_t)ES_RING_CAP)
            ? (k->write_seq - (uint64_t)ES_RING_CAP)
            : 0U;

    const double t_oldest = k->ring_t[oldest & ES_RING_MASK];
    const double t_newest = k->ring_t[newest & ES_RING_MASK];

    if (t_loc < t_oldest) {
        k->stats.n_underrun += 1U;
        return ES_ERR_UNDERRUN;
    }
    if (t_loc > t_newest) {
        k->stats.n_overrun += 1U;
        return ES_ERR_OVERRUN;
    }

    const uint64_t idx = es_find_bracket_d(k, oldest, newest, t_loc);
    k->read_seq = idx;

    const uint64_t s0 = idx & ES_RING_MASK;
    const uint64_t s1 = (idx + 1U) & ES_RING_MASK;
    const double t0 = k->ring_t[s0];
    const double t1 = k->ring_t[s1];
    const double dt = t1 - t0;
    const double mu = (dt > 0.0) ? es_clamp_d((t_loc - t0) / dt, 0.0, 1.0) : 0.0;

    const double gap_lim = k->cfg.gap_factor * k->h_nom;

    /* Check if 4 contiguous uniform-rate taps [idx-1, idx, idx+1, idx+2] exist */
    if ((idx > oldest) && ((idx + 2U) <= newest)) {
        const uint64_t sm1 = (idx - 1U) & ES_RING_MASK;
        const uint64_t s2 = (idx + 2U) & ES_RING_MASK;
        const double dt_left = t0 - k->ring_t[sm1];
        const double dt_right = k->ring_t[s2] - t1;

        if ((dt <= gap_lim) && (dt_left <= gap_lim) && (dt_right <= gap_lim)) {
            *out_x = es_farrow_cubic_4tap(
                k->ring_x[sm1], k->ring_x[s0], k->ring_x[s1], k->ring_x[s2], mu);
            k->stats.n_resampled += 1U;
            k->stats.n_farrow_cubic += 1U;
            return ES_OK;
        }

        /* Gap detected (e.g., BLE packet burst drop): bounded linear fallback */
        *out_x = k->ring_x[s0] + mu * (k->ring_x[s1] - k->ring_x[s0]);
        k->stats.n_resampled += 1U;
        k->stats.n_gap_linear += 1U;
        return ES_OK_GAP_LINEAR;
    }

    /* Ring head/tail boundary (< 4 contiguous taps): bounded linear fallback */
    *out_x = k->ring_x[s0] + mu * (k->ring_x[s1] - k->ring_x[s0]);
    k->stats.n_resampled += 1U;
    k->stats.n_gap_linear += 1U;
    return ES_OK_EDGE_LINEAR;
}

int32_t es_resample_batch(
    es_kernel_t *k,
    const double *t_ref_arr,
    size_t n,
    double *out_x_arr,
    int32_t *out_status_arr) {
    if (k == NULL || t_ref_arr == NULL || out_x_arr == NULL) {
        return ES_ERR_NULL;
    }
    int32_t ok_count = 0;
    for (size_t i = 0U; i < n; ++i) {
        double val = 0.0;
        const int32_t st = es_resample_at(k, t_ref_arr[i], &val);
        out_x_arr[i] = val;
        if (out_status_arr != NULL) {
            out_status_arr[i] = st;
        }
        if (st >= 0) {
            ok_count += 1;
        }
    }
    return ok_count;
}

/* ---- Single-Precision FPU Microkernel API (~4.3 KB SRAM) ---- */

int32_t es_f32_init(es_kernel_f32_t *k, const es_config_t *cfg) {
    if (k == NULL || cfg == NULL) {
        return ES_ERR_NULL;
    }
    if (!es_validate_config(cfg)) {
        return ES_ERR_PARAM;
    }
    es_zero_bytes(k, sizeof(es_kernel_f32_t));
    k->cfg = *cfg;
    k->h_nom = 1.0 / cfg->fs_in;
    es_clock_reset(&k->clk, &k->cfg);
    k->stats.ring_capacity = (uint32_t)ES_RING_CAP;
    k->stats.kernel_bytes = (uint32_t)sizeof(es_kernel_f32_t);
    return ES_OK;
}

int32_t es_f32_update_sync(es_kernel_f32_t *k, double t_loc, double t_ref) {
    if (k == NULL) {
        return ES_ERR_NULL;
    }
    return es_clock_update(&k->clk, &k->cfg, t_loc, t_ref);
}

/* Absolute local time of ring sequence number seq (epoch base + float32 offset). */
static double es_f32_time(const es_kernel_f32_t *k, uint64_t seq) {
    return k->t_base[(seq / (uint64_t)ES_RING_CAP) & 1U] +
           (double)k->ring_dt[seq & ES_RING_MASK];
}

int32_t es_f32_push_sample(es_kernel_f32_t *k, double t_loc, float x) {
    if (k == NULL) {
        return ES_ERR_NULL;
    }
    if (!es_isfinite_d(t_loc) || !es_isfinite_f(x)) {
        return ES_ERR_PARAM;
    }
    const uint64_t seq = k->write_seq;
    if ((seq > 0U) && (t_loc <= es_f32_time(k, seq - 1U))) {
        return ES_ERR_NONMONOTONIC;
    }
    const uint64_t epoch_slot = (seq / (uint64_t)ES_RING_CAP) & 1U;
    if ((seq & ES_RING_MASK) == 0U) {
        /* First sample of a new epoch: re-anchor the base two epochs old (no longer in ring). */
        k->t_base[epoch_slot] = t_loc;
    }
    const uint64_t slot = seq & ES_RING_MASK;
    k->ring_dt[slot] = (float)(t_loc - k->t_base[epoch_slot]);
    k->ring_x[slot] = x;
    k->write_seq += 1U;
    k->stats.n_pushed += 1U;
    return ES_OK;
}

float es_farrow_cubic_4tap_f32(float xm1, float x0, float x1, float x2, float mu) {
    const float d01 = x1 - x0;
    const float c2 = 0.5f * (xm1 + x1) - x0;
    const float c3 = (1.0f / 6.0f) * (x2 - xm1) - 0.5f * d01;
    const float c1 = d01 - c2 - c3;
    return ((c3 * mu + c2) * mu + c1) * mu + x0;
}

static uint64_t es_find_bracket_f32(
    const es_kernel_f32_t *k,
    uint64_t oldest,
    uint64_t newest,
    double t_loc) {
    uint64_t cur = k->read_seq;
    if (cur < oldest) {
        cur = oldest;
    } else if (cur >= newest) {
        cur = newest - 1U;
    }

    if ((t_loc >= es_f32_time(k, cur)) && (t_loc <= es_f32_time(k, cur + 1U))) {
        return cur;
    }

    if ((cur + 2U) <= newest) {
        const uint64_t nxt = cur + 1U;
        if ((t_loc >= es_f32_time(k, nxt)) && (t_loc <= es_f32_time(k, nxt + 1U))) {
            return nxt;
        }
    }

    uint64_t lo = oldest;
    uint64_t hi = newest - 1U;
    for (uint32_t iter = 0U; iter < ES_MAX_BSEARCH_ITERS; ++iter) {
        if (lo >= hi) {
            break;
        }
        const uint64_t mid = lo + ((hi - lo) >> 1U);
        if (es_f32_time(k, mid + 1U) < t_loc) {
            lo = mid + 1U;
        } else if (es_f32_time(k, mid) > t_loc) {
            hi = mid;
        } else {
            return mid;
        }
    }
    return lo;
}

int32_t es_f32_resample_at(es_kernel_f32_t *k, double t_ref, float *out_x) {
    if (k == NULL || out_x == NULL) {
        return ES_ERR_NULL;
    }
    if (!es_isfinite_d(t_ref)) {
        return ES_ERR_PARAM;
    }
    if (k->write_seq < 2U) {
        k->stats.n_underrun += 1U;
        return ES_ERR_UNDERRUN;
    }

    const double t_loc = es_clock_ref_to_loc(&k->clk, t_ref);
    const uint64_t newest = k->write_seq - 1U;
    const uint64_t oldest =
        (k->write_seq > (uint64_t)ES_RING_CAP)
            ? (k->write_seq - (uint64_t)ES_RING_CAP)
            : 0U;

    if (t_loc < es_f32_time(k, oldest)) {
        k->stats.n_underrun += 1U;
        return ES_ERR_UNDERRUN;
    }
    if (t_loc > es_f32_time(k, newest)) {
        k->stats.n_overrun += 1U;
        return ES_ERR_OVERRUN;
    }

    const uint64_t idx = es_find_bracket_f32(k, oldest, newest, t_loc);
    k->read_seq = idx;

    const uint64_t s0 = idx & ES_RING_MASK;
    const uint64_t s1 = (idx + 1U) & ES_RING_MASK;
    const double t0 = es_f32_time(k, idx);
    const double t1 = es_f32_time(k, idx + 1U);
    const double dt = t1 - t0;
    const float mu = (dt > 0.0) ? (float)es_clamp_d((t_loc - t0) / dt, 0.0, 1.0) : 0.0f;

    const double gap_lim = k->cfg.gap_factor * k->h_nom;

    if ((idx > oldest) && ((idx + 2U) <= newest)) {
        const uint64_t sm1 = (idx - 1U) & ES_RING_MASK;
        const uint64_t s2 = (idx + 2U) & ES_RING_MASK;
        const double dt_left = t0 - es_f32_time(k, idx - 1U);
        const double dt_right = es_f32_time(k, idx + 2U) - t1;

        if ((dt <= gap_lim) && (dt_left <= gap_lim) && (dt_right <= gap_lim)) {
            *out_x = es_farrow_cubic_4tap_f32(
                k->ring_x[sm1], k->ring_x[s0], k->ring_x[s1], k->ring_x[s2], mu);
            k->stats.n_resampled += 1U;
            k->stats.n_farrow_cubic += 1U;
            return ES_OK;
        }

        *out_x = k->ring_x[s0] + mu * (k->ring_x[s1] - k->ring_x[s0]);
        k->stats.n_resampled += 1U;
        k->stats.n_gap_linear += 1U;
        return ES_OK_GAP_LINEAR;
    }

    *out_x = k->ring_x[s0] + mu * (k->ring_x[s1] - k->ring_x[s0]);
    k->stats.n_resampled += 1U;
    k->stats.n_gap_linear += 1U;
    return ES_OK_EDGE_LINEAR;
}

int32_t es_f32_resample_batch(
    es_kernel_f32_t *k,
    const double *t_ref_arr,
    size_t n,
    float *out_x_arr,
    int32_t *out_status_arr) {
    if (k == NULL || t_ref_arr == NULL || out_x_arr == NULL) {
        return ES_ERR_NULL;
    }
    int32_t ok_count = 0;
    for (size_t i = 0U; i < n; ++i) {
        float val = 0.0f;
        const int32_t st = es_f32_resample_at(k, t_ref_arr[i], &val);
        out_x_arr[i] = val;
        if (out_status_arr != NULL) {
            out_status_arr[i] = st;
        }
        if (st >= 0) {
            ok_count += 1;
        }
    }
    return ok_count;
}
