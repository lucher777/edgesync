/**
 * @file bench_compare.c
 * @brief Host latency comparison of edgesync against two plain-C baselines.
 *
 * All implementations run the same streaming workload (250 Hz input, +125 ppm
 * drift, one clock beacon every 256 samples, one resample query per pushed
 * sample, query delay 4 samples) under the same timing harness:
 *
 *   EDGESYNC_F64  edgesync double-precision kernel (Huber-RLS + cubic Farrow)
 *   EDGESYNC_F32  edgesync single-precision ring kernel
 *   C_RLS_LIN     static ring of the same capacity, unweighted origin-centered
 *                 RLS, forward-scanning cursor, linear interpolation
 *   C_HEAP_LIN    heap arrays grown by realloc doubling (whole history kept),
 *                 the same RLS, binary search, linear interpolation
 *
 * For each implementation and repetition two passes are made: an
 * uninstrumented pass (mean time per sample from the wall clock) and an
 * instrumented pass (per-sample time stamp counter deltas for percentiles).
 * The JSON output reports, for every statistic, the median over repetitions.
 *
 * @copyright MIT License
 */

#define _POSIX_C_SOURCE 200809L

#include "edgesync.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_SAMPLES 1000000U
#define N_WARMUP 10000U
#define N_REPS 7
#define RING_CAP 512U
#define RING_MASK (RING_CAP - 1U)

/* ------------------------------------------------------------------ timing */

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

#if defined(__x86_64__) || defined(__i386__)
static inline uint64_t ticks(void) {
    uint32_t lo, hi;
    __asm__ __volatile__("lfence\n\trdtsc\n\tlfence" : "=a"(lo), "=d"(hi)::"memory");
    return ((uint64_t)hi << 32U) | (uint64_t)lo;
}
#else
static inline uint64_t ticks(void) { return now_ns(); }
#endif

static double ns_per_tick(void) {
    const uint64_t t0 = now_ns(), c0 = ticks();
    while (now_ns() - t0 < 100000000ULL) {
    }
    const uint64_t t1 = now_ns(), c1 = ticks();
    return (double)(t1 - t0) / (double)(c1 - c0);
}

/* ------------------------------------------------- shared baseline clock */

typedef struct {
    double t0, r0, d0, a, p00, p01, p11, lam;
    int init;
} rls_t;

static void rls_init(rls_t *c) {
    memset(c, 0, sizeof(*c));
    c->p00 = 1.0;
    c->p11 = 1e-4;
    c->lam = 0.995;
}

static void rls_update(rls_t *c, double tl, double tr) {
    if (!c->init) {
        c->t0 = tl;
        c->r0 = tr;
        c->init = 1;
        return;
    }
    const double u = tl - c->t0, y = (tr - c->r0) - u;
    const double e = y - (c->d0 + c->a * u);
    const double g0 = c->p00 + c->p01 * u, g1 = c->p01 + c->p11 * u;
    const double d = c->lam + g0 + g1 * u;
    const double k0 = g0 / d, k1 = g1 / d;
    c->d0 += k0 * e;
    c->a += k1 * e;
    c->p00 = (c->p00 - k0 * g0) / c->lam;
    c->p01 = (c->p01 - k0 * g1) / c->lam;
    c->p11 = (c->p11 - k1 * g1) / c->lam;
}

static double rls_ref_to_loc(const rls_t *c, double tr) {
    if (!c->init) {
        return tr;
    }
    return c->t0 + (tr - c->r0 - c->d0) / (1.0 + c->a);
}

/* ------------------------------------------ baseline 1: static ring, linear */

typedef struct {
    rls_t clk;
    double t[RING_CAP], x[RING_CAP];
    uint64_t w, r;
} b_ring_t;

static b_ring_t g_ring;

static void ring_init(b_ring_t *b) {
    memset(b, 0, sizeof(*b));
    rls_init(&b->clk);
}

static void ring_push(b_ring_t *b, double t, double x) {
    b->t[b->w & RING_MASK] = t;
    b->x[b->w & RING_MASK] = x;
    b->w++;
}

static int ring_resample(b_ring_t *b, double tr, double *y) {
    const double tl = rls_ref_to_loc(&b->clk, tr);
    const uint64_t oldest = b->w > RING_CAP ? b->w - RING_CAP : 0U;
    if (b->w < 2U) {
        return -1;
    }
    if (b->r < oldest) {
        b->r = oldest;
    }
    while (b->r + 2U < b->w && b->t[(b->r + 1U) & RING_MASK] <= tl) {
        b->r++;
    }
    const double ta = b->t[b->r & RING_MASK], tb = b->t[(b->r + 1U) & RING_MASK];
    const double xa = b->x[b->r & RING_MASK], xb = b->x[(b->r + 1U) & RING_MASK];
    const double mu = (tl - ta) / (tb - ta);
    *y = xa + mu * (xb - xa);
    return 0;
}

/* --------------------------------------- baseline 2: heap history, linear */

typedef struct {
    rls_t clk;
    double *t, *x;
    size_t n, cap;
} b_heap_t;

static int heap_push(b_heap_t *b, double t, double x) {
    if (b->n == b->cap) {
        const size_t nc = b->cap ? 2U * b->cap : 256U;
        double *nt = (double *)realloc(b->t, nc * sizeof(double));
        if (nt == NULL) {
            return -1;
        }
        b->t = nt;
        double *nx = (double *)realloc(b->x, nc * sizeof(double));
        if (nx == NULL) {
            return -1;
        }
        b->x = nx;
        b->cap = nc;
    }
    b->t[b->n] = t;
    b->x[b->n] = x;
    b->n++;
    return 0;
}

static int heap_resample(b_heap_t *b, double tr, double *y) {
    if (b->n < 2U) {
        return -1;
    }
    const double tl = rls_ref_to_loc(&b->clk, tr);
    size_t lo = 0U, hi = b->n - 1U;
    while (hi - lo > 1U) {
        const size_t mid = lo + (hi - lo) / 2U;
        if (b->t[mid] <= tl) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const double mu = (tl - b->t[lo]) / (b->t[hi] - b->t[lo]);
    *y = b->x[lo] + mu * (b->x[hi] - b->x[lo]);
    return 0;
}

/* --------------------------------------------------------------- harness */

enum { M_F64 = 0, M_F32, M_RING, M_HEAP, N_METHODS };
static const char *NAMES[N_METHODS] = {"edgesync_fp64", "edgesync_fp32", "c_rls_linear_static",
                                       "c_heap_linear_dynamic"};

static es_kernel_t g_k64;
static es_kernel_f32_t g_k32;
static b_heap_t g_heap;

static void reset_all(void) {
    es_config_t cfg = es_default_config(250.0);
    (void)es_init(&g_k64, &cfg);
    (void)es_f32_init(&g_k32, &cfg);
    ring_init(&g_ring);
    free(g_heap.t);
    free(g_heap.x);
    memset(&g_heap, 0, sizeof(g_heap));
    rls_init(&g_heap.clk);
}

/* One streaming step: push, occasional beacon, one resample query. */
static inline double step(int m, uint32_t idx) {
    const double h = 1.0 / 250.0, alpha = 125e-6, off = 1.375;
    const double tl = (double)idx * h;
    const double tr = (1.0 + alpha) * tl + off;
    const double x = 0.5 * (double)((int32_t)(idx & 15U) - 8);
    const double tq = tr - 4.0 * h;
    const int beacon = (idx & 255U) == 0U;
    double y = 0.0;
    float yf = 0.0f;
    switch (m) {
    case M_F64:
        (void)es_push_sample(&g_k64, tl, x);
        if (beacon) (void)es_update_sync(&g_k64, tl, tr);
        (void)es_resample_at(&g_k64, tq, &y);
        return y;
    case M_F32:
        (void)es_f32_push_sample(&g_k32, tl, (float)x);
        if (beacon) (void)es_f32_update_sync(&g_k32, tl, tr);
        (void)es_f32_resample_at(&g_k32, tq, &yf);
        return (double)yf;
    case M_RING:
        ring_push(&g_ring, tl, x);
        if (beacon) rls_update(&g_ring.clk, tl, tr);
        (void)ring_resample(&g_ring, tq, &y);
        return y;
    default:
        (void)heap_push(&g_heap, tl, x);
        if (beacon) rls_update(&g_heap.clk, tl, tr);
        (void)heap_resample(&g_heap, tq, &y);
        return y;
    }
}

static int cmp_u64(const void *a, const void *b) {
    const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static int cmp_d(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

#define N_STATS 6 /* mean_ns, p50, p99, p999, max, peak_bytes */

int main(void) {
    const double npt = ns_per_tick();
    uint64_t *lat = (uint64_t *)malloc(N_SAMPLES * sizeof(uint64_t));
    static double res[N_METHODS][N_STATS][N_REPS];
    volatile double sink = 0.0;
    if (lat == NULL) {
        return 1;
    }
    for (int rep = 0; rep < N_REPS; ++rep) {
        for (int m = 0; m < N_METHODS; ++m) {
            /* pass 1: uninstrumented throughput */
            reset_all();
            for (uint32_t i = 0U; i < N_WARMUP; ++i) sink += step(m, i);
            const uint64_t w0 = now_ns();
            for (uint32_t i = 0U; i < N_SAMPLES; ++i) sink += step(m, N_WARMUP + i);
            const uint64_t w1 = now_ns();
            res[m][0][rep] = (double)(w1 - w0) / (double)N_SAMPLES;
            /* pass 2: per-sample latency */
            reset_all();
            for (uint32_t i = 0U; i < N_WARMUP; ++i) sink += step(m, i);
            for (uint32_t i = 0U; i < N_SAMPLES; ++i) {
                const uint64_t c0 = ticks();
                sink += step(m, N_WARMUP + i);
                lat[i] = ticks() - c0;
            }
            qsort(lat, N_SAMPLES, sizeof(uint64_t), cmp_u64);
            res[m][1][rep] = (double)lat[N_SAMPLES / 2U] * npt;
            res[m][2][rep] = (double)lat[(size_t)(0.99 * (N_SAMPLES - 1U))] * npt;
            res[m][3][rep] = (double)lat[(size_t)(0.999 * (N_SAMPLES - 1U))] * npt;
            res[m][4][rep] = (double)lat[N_SAMPLES - 1U] * npt;
            if (m == M_F64) res[m][5][rep] = (double)sizeof(es_kernel_t);
            else if (m == M_F32) res[m][5][rep] = (double)sizeof(es_kernel_f32_t);
            else if (m == M_RING) res[m][5][rep] = (double)sizeof(b_ring_t);
            else res[m][5][rep] = (double)(sizeof(b_heap_t) + 2U * g_heap.cap * sizeof(double));
        }
    }
    /* timer floor: back-to-back time stamps */
    for (uint32_t i = 0U; i < N_SAMPLES; ++i) {
        const uint64_t c0 = ticks();
        lat[i] = ticks() - c0;
    }
    qsort(lat, N_SAMPLES, sizeof(uint64_t), cmp_u64);

    static const char *SN[N_STATS] = {"mean_ns", "p50_ns", "p99_ns", "p999_ns", "max_ns", "state_bytes"};
    printf("{\n  \"n_samples\": %u,\n  \"n_reps\": %d,\n  \"ns_per_tick\": %.6f,\n", N_SAMPLES, N_REPS, npt);
    printf("  \"timer_floor_p50_ns\": %.2f,\n", (double)lat[N_SAMPLES / 2U] * npt);
    for (int m = 0; m < N_METHODS; ++m) {
        printf("  \"%s\": {", NAMES[m]);
        for (int s = 0; s < N_STATS; ++s) {
            qsort(res[m][s], N_REPS, sizeof(double), cmp_d);
            printf("\"%s\": %.3f%s", SN[s], res[m][s][N_REPS / 2], s + 1 < N_STATS ? ", " : "");
        }
        printf("}%s\n", m + 1 < N_METHODS ? "," : "");
    }
    printf("}\n");
    free(lat);
    free(g_heap.t);
    free(g_heap.x);
    return (sink == 12345.678) ? 2 : 0;
}
