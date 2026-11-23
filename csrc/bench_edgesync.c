/**
 * @file bench_edgesync.c
 * @brief Native C11 streaming latency, throughput, and static-footprint benchmark
 *        for the EdgeSync microkernel (1,000,000 streaming samples).
 *
 * Reports exact compile-time sizeof(es_kernel_t), sizeof(es_kernel_f32_t),
 * per-sample latency percentiles (P50, P95, P99, P99.9, Max) in nanoseconds,
 * and sustained streaming throughput in MSamples/s for both FP64 and FP32 kernels.
 *
 * @copyright MIT License
 */

#define _POSIX_C_SOURCE 200809L

#include "edgesync.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BENCH_N_SAMPLES 1000000U
#define BENCH_WARMUP 10000U

static es_kernel_t g_static_kernel;
static es_kernel_f32_t g_static_kernel_f32;

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
static inline uint64_t read_tsc(void) {
    uint32_t lo = 0U;
    uint32_t hi = 0U;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32U) | (uint64_t)lo;
}
#define ES_HAS_RDTSC 1
#else
#define ES_HAS_RDTSC 0
#endif

static int cmp_u64(const void *a, const void *b) {
    const uint64_t va = *(const uint64_t *)a;
    const uint64_t vb = *(const uint64_t *)b;
    if (va < vb) {
        return -1;
    }
    if (va > vb) {
        return 1;
    }
    return 0;
}

static double calibrate_ns_per_cycle(void) {
#if ES_HAS_RDTSC
    const uint64_t t0 = now_ns();
    const uint64_t c0 = read_tsc();
    volatile uint64_t spin = 0U;
    while ((now_ns() - t0) < 50000000ULL) {
        spin += 1U;
    }
    const uint64_t t1 = now_ns();
    const uint64_t c1 = read_tsc();
    (void)spin;
    if (c1 > c0 && t1 > t0) {
        return (double)(t1 - t0) / (double)(c1 - c0);
    }
#endif
    return 1.0;
}

int main(void) {
    const double ns_per_tick = calibrate_ns_per_cycle();
    const double fs_in = 250.0;
    const double h = 1.0 / fs_in;
    const double true_alpha = 125e-6; /* +125 ppm crystal drift */
    const double true_offset = 1.375; /* +1.375 s startup offset */

    es_config_t cfg = es_default_config(fs_in);
    if (es_init(&g_static_kernel, &cfg) != ES_OK) {
        fprintf(stderr, "es_init failed\n");
        return 1;
    }
    if (es_f32_init(&g_static_kernel_f32, &cfg) != ES_OK) {
        fprintf(stderr, "es_f32_init failed\n");
        return 1;
    }

    uint64_t *lat_fp64 = (uint64_t *)malloc(BENCH_N_SAMPLES * sizeof(uint64_t));
    uint64_t *lat_fp32 = (uint64_t *)malloc(BENCH_N_SAMPLES * sizeof(uint64_t));
    if (lat_fp64 == NULL || lat_fp32 == NULL) {
        fprintf(stderr, "benchmark histogram allocation failed\n");
        free(lat_fp64);
        free(lat_fp32);
        return 1;
    }

    /* Warmup both kernels */
    for (uint32_t i = 0U; i < BENCH_WARMUP; ++i) {
        const double t_loc = (double)i * h;
        const double t_ref_true = (1.0 + true_alpha) * t_loc + true_offset;
        const double x = sin(2.0 * 3.14159265358979323846 * 1.25 * t_ref_true);
        (void)es_push_sample(&g_static_kernel, t_loc, x);
        (void)es_f32_push_sample(&g_static_kernel_f32, t_loc, (float)x);
        if ((i % 250U) == 0U) {
            (void)es_update_sync(&g_static_kernel, t_loc, t_ref_true);
            (void)es_f32_update_sync(&g_static_kernel_f32, t_loc, t_ref_true);
        }
        if (i >= 8U) {
            double y64 = 0.0;
            float y32 = 0.0f;
            const double t_q = t_ref_true - 4.0 * h;
            (void)es_resample_at(&g_static_kernel, t_q, &y64);
            (void)es_f32_resample_at(&g_static_kernel_f32, t_q, &y32);
        }
    }

    /* Measure FP64 streaming push + resample over 1,000,000 samples */
    double acc64 = 0.0;
    const uint64_t wall_t0 = now_ns();
    for (uint32_t i = 0U; i < BENCH_N_SAMPLES; ++i) {
        const uint32_t idx = BENCH_WARMUP + i;
        const double t_loc = (double)idx * h;
        const double t_ref_true = (1.0 + true_alpha) * t_loc + true_offset;
        const double x = 0.5 * (double)((int32_t)(idx & 15U) - 8);
        const double t_q = t_ref_true - 4.0 * h;
        double y = 0.0;

#if ES_HAS_RDTSC
        const uint64_t c0 = read_tsc();
#else
        const uint64_t c0 = now_ns();
#endif
        (void)es_push_sample(&g_static_kernel, t_loc, x);
        if ((idx & 255U) == 0U) {
            (void)es_update_sync(&g_static_kernel, t_loc, t_ref_true);
        }
        (void)es_resample_at(&g_static_kernel, t_q, &y);
#if ES_HAS_RDTSC
        const uint64_t c1 = read_tsc();
#else
        const uint64_t c1 = now_ns();
#endif
        lat_fp64[i] = c1 - c0;
        acc64 += y;
    }
    const uint64_t wall_t1 = now_ns();
    const double fp64_total_s = (double)(wall_t1 - wall_t0) * 1e-9;

    /* Measure FP32 streaming push + resample over 1,000,000 samples */
    double acc32 = 0.0;
    const uint64_t wall32_t0 = now_ns();
    for (uint32_t i = 0U; i < BENCH_N_SAMPLES; ++i) {
        const uint32_t idx = BENCH_WARMUP + i;
        const double t_loc = (double)idx * h;
        const double t_ref_true = (1.0 + true_alpha) * t_loc + true_offset;
        const float x = 0.5f * (float)((int32_t)(idx & 15U) - 8);
        const double t_q = t_ref_true - 4.0 * h;
        float y = 0.0f;

#if ES_HAS_RDTSC
        const uint64_t c0 = read_tsc();
#else
        const uint64_t c0 = now_ns();
#endif
        (void)es_f32_push_sample(&g_static_kernel_f32, t_loc, x);
        if ((idx & 255U) == 0U) {
            (void)es_f32_update_sync(&g_static_kernel_f32, t_loc, t_ref_true);
        }
        (void)es_f32_resample_at(&g_static_kernel_f32, t_q, &y);
#if ES_HAS_RDTSC
        const uint64_t c1 = read_tsc();
#else
        const uint64_t c1 = now_ns();
#endif
        lat_fp32[i] = c1 - c0;
        acc32 += (double)y;
    }
    const uint64_t wall32_t1 = now_ns();
    const double fp32_total_s = (double)(wall32_t1 - wall32_t0) * 1e-9;

    qsort(lat_fp64, BENCH_N_SAMPLES, sizeof(uint64_t), cmp_u64);
    qsort(lat_fp32, BENCH_N_SAMPLES, sizeof(uint64_t), cmp_u64);

    const size_t i50 = (size_t)(0.50 * (double)(BENCH_N_SAMPLES - 1U));
    const size_t i95 = (size_t)(0.95 * (double)(BENCH_N_SAMPLES - 1U));
    const size_t i99 = (size_t)(0.99 * (double)(BENCH_N_SAMPLES - 1U));
    const size_t i999 = (size_t)(0.999 * (double)(BENCH_N_SAMPLES - 1U));
    const size_t imax = BENCH_N_SAMPLES - 1U;

    printf("{\n");
    printf("  \"n_samples\": %u,\n", BENCH_N_SAMPLES);
    printf("  \"ring_capacity\": %u,\n", es_ring_capacity());
    printf("  \"sizeof_kernel_fp64_bytes\": %zu,\n", es_sizeof_kernel());
    printf("  \"sizeof_kernel_fp32_bytes\": %zu,\n", es_sizeof_kernel_f32());
    printf("  \"ns_per_tick\": %.6f,\n", ns_per_tick);
    printf("  \"fp64\": {\n");
    printf("    \"p50_ns\": %.2f,\n", (double)lat_fp64[i50] * ns_per_tick);
    printf("    \"p95_ns\": %.2f,\n", (double)lat_fp64[i95] * ns_per_tick);
    printf("    \"p99_ns\": %.2f,\n", (double)lat_fp64[i99] * ns_per_tick);
    printf("    \"p999_ns\": %.2f,\n", (double)lat_fp64[i999] * ns_per_tick);
    printf("    \"max_ns\": %.2f,\n", (double)lat_fp64[imax] * ns_per_tick);
    printf("    \"throughput_msps\": %.3f,\n", ((double)BENCH_N_SAMPLES / fp64_total_s) * 1e-6);
    printf("    \"farrow_cubic_count\": %llu,\n", (unsigned long long)g_static_kernel.stats.n_farrow_cubic);
    printf("    \"drift_ppm\": %.6f,\n", es_get_drift_ppm(&g_static_kernel));
    printf("    \"checksum\": %.6f\n", acc64);
    printf("  },\n");
    printf("  \"fp32\": {\n");
    printf("    \"p50_ns\": %.2f,\n", (double)lat_fp32[i50] * ns_per_tick);
    printf("    \"p95_ns\": %.2f,\n", (double)lat_fp32[i95] * ns_per_tick);
    printf("    \"p99_ns\": %.2f,\n", (double)lat_fp32[i99] * ns_per_tick);
    printf("    \"p999_ns\": %.2f,\n", (double)lat_fp32[i999] * ns_per_tick);
    printf("    \"max_ns\": %.2f,\n", (double)lat_fp32[imax] * ns_per_tick);
    printf("    \"throughput_msps\": %.3f,\n", ((double)BENCH_N_SAMPLES / fp32_total_s) * 1e-6);
    printf("    \"farrow_cubic_count\": %llu,\n", (unsigned long long)g_static_kernel_f32.stats.n_farrow_cubic);
    printf("    \"checksum\": %.6f\n", acc32);
    printf("  }\n");
    printf("}\n");

    free(lat_fp64);
    free(lat_fp32);
    return 0;
}
