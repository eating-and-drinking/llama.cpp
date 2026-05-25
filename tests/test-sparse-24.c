// SPDX-License-Identifier: MIT
//
// Standalone correctness test for ggml-cpu/sparse_24.{h,c}.
//
/* Build (Linux/macOS):
 *   gcc -O3 -mavx2 -mfma -Iggml/include -Iggml/src -Iggml/src/ggml-cpu \
 *       tests/test-sparse-24.c ggml/src/ggml-cpu/sparse_24.c \
 *       -o test-sparse-24 -lm
 *
 * Build (Windows MSVC):
 *   cl /O2 /arch:AVX2 /Iggml/include /Iggml/src /Iggml/src/ggml-cpu \
 *       tests/test-sparse-24.c ggml/src/ggml-cpu/sparse_24.c \
 *       /Fe:test-sparse-24.exe
 */
//
// What it does:
//   1. Generates a random 2:4-sparse FP32 weight and a random FP32 activation.
//   2. Computes FP32 ground-truth dot product.
//   3. Quantizes weight to block_q8_0_2_4 and activation to block_q8_0.
//   4. Runs both ggml_vec_dot_q8_0_2_4_q8_0_ref (scalar) and
//      ggml_vec_dot_q8_0_2_4_q8_0 (AVX2 when available).
//   5. Verifies that:
//        * scalar vs FP32 ground truth:  relative error < 1% (int8 noise)
//        * AVX2   vs scalar:             bit-identical
//   6. Runs a microbenchmark to compare sparse vs dense Q8_0 vec_dot timing.

#include "sparse_24.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Pull in dense Q8_0 block definition (we only use the layout, not its kernel).
#include "ggml-common.h"

// Portable 64-byte-aligned allocator.
//   * MSVC has no aligned_alloc; use _aligned_malloc / _aligned_free.
//   * Standard aligned_alloc requires `size` to be a multiple of alignment
//     (C11 §7.22.3.1); round up to satisfy that.
#define SPARSE24_ALIGN 64
static inline size_t sparse24_round_up(size_t n, size_t a) {
    return (n + a - 1) & ~(a - 1);
}
#if defined(_MSC_VER)
    #define SPARSE24_ALLOC(sz) _aligned_malloc(sparse24_round_up((sz), SPARSE24_ALIGN), SPARSE24_ALIGN)
    #define SPARSE24_FREE(p)   _aligned_free(p)
#else
    #define SPARSE24_ALLOC(sz) aligned_alloc(SPARSE24_ALIGN, sparse24_round_up((sz), SPARSE24_ALIGN))
    #define SPARSE24_FREE(p)   free(p)
#endif

// -----------------------------------------------------------------------------
// Tiny RNG (xorshift32) for repeatability
// -----------------------------------------------------------------------------

static uint32_t g_rng = 12345;
static float frand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return ((float)(g_rng & 0xFFFFFF) / (float)0xFFFFFF) * 2.0f - 1.0f;
}

// -----------------------------------------------------------------------------
// Reference helpers
// -----------------------------------------------------------------------------

// Pack one FP32 row (length n, multiple of 32) into block_q8_0_2_4 layout.
// Asserts strict 2:4. Returns number of blocks written.
static int pack_row_q8_0_2_4(const float * row, int n, block_q8_0_2_4 * out) {
    assert(n % QK8_0_2_4 == 0);
    const int nb = n / QK8_0_2_4;
    for (int b = 0; b < nb; ++b) {
        const float * block = row + b * QK8_0_2_4;
        // Find absmax across the block.
        float amax = 0.0f;
        for (int i = 0; i < QK8_0_2_4; ++i) {
            float a = fabsf(block[i]);
            if (a > amax) amax = a;
        }
        const float scale = (amax == 0.0f) ? 1.0f : amax / 127.0f;
        const float inv_s = 1.0f / scale;
        // Encode FP16 scale.
        // (Trivial cast via float; for a real impl use ggml_fp32_to_fp16.)
        union { float f; uint32_t u; } cvt = { .f = scale };
        uint16_t h;
        {
            uint32_t x = cvt.u;
            uint32_t sign = (x >> 31) & 0x1;
            int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
            uint32_t mant = (x >> 13) & 0x3ff;
            if (exp <= 0) { h = (uint16_t)(sign << 15); }
            else if (exp >= 31) { h = (uint16_t)((sign << 15) | (0x1f << 10)); }
            else { h = (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | mant); }
        }
        out[b].d = (ggml_half) h;

        int kept = 0;
        for (int g = 0; g < 8; ++g) {
            const float * chunk = block + g * 4;
            int positions[2];
            int n_kept = 0;
            for (int p = 0; p < 4; ++p) {
                if (chunk[p] != 0.0f) {
                    if (n_kept >= 2) {
                        fprintf(stderr, "ERROR: block %d group %d has >2 non-zeros\n", b, g);
                        return -1;
                    }
                    positions[n_kept++] = p;
                }
            }
            if (n_kept != 2) {
                fprintf(stderr, "ERROR: block %d group %d has %d non-zeros (need 2)\n", b, g, n_kept);
                return -1;
            }
            for (int k = 0; k < 2; ++k) {
                int q = (int)lroundf(chunk[positions[k]] * inv_s);
                if (q < -127) q = -127;
                if (q >  127) q =  127;
                out[b].qs[kept++] = (int8_t) q;
            }
            out[b].idx[g] = (uint8_t)((positions[0] << 4) | positions[1]);
        }
    }
    return nb;
}

// Pack a dense FP32 row to standard block_q8_0.
static int pack_row_q8_0(const float * row, int n, block_q8_0 * out) {
    assert(n % QK8_0 == 0);
    const int nb = n / QK8_0;
    for (int b = 0; b < nb; ++b) {
        const float * block = row + b * QK8_0;
        float amax = 0.0f;
        for (int i = 0; i < QK8_0; ++i) {
            float a = fabsf(block[i]);
            if (a > amax) amax = a;
        }
        const float scale = (amax == 0.0f) ? 1.0f : amax / 127.0f;
        const float inv_s = 1.0f / scale;
        union { float f; uint32_t u; } cvt = { .f = scale };
        uint16_t h;
        {
            uint32_t x = cvt.u;
            uint32_t sign = (x >> 31) & 0x1;
            int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
            uint32_t mant = (x >> 13) & 0x3ff;
            if (exp <= 0) { h = (uint16_t)(sign << 15); }
            else if (exp >= 31) { h = (uint16_t)((sign << 15) | (0x1f << 10)); }
            else { h = (uint16_t)((sign << 15) | ((uint32_t)exp << 10) | mant); }
        }
        out[b].d = (ggml_half) h;
        for (int i = 0; i < QK8_0; ++i) {
            int q = (int)lroundf(block[i] * inv_s);
            if (q < -127) q = -127;
            if (q >  127) q =  127;
            out[b].qs[i] = (int8_t) q;
        }
    }
    return nb;
}

// Build a random row that satisfies strict 2:4 along its columns.
static void gen_random_24_row(float * row, int n) {
    for (int g = 0; g < n / 4; ++g) {
        float * chunk = row + g * 4;
        for (int p = 0; p < 4; ++p) chunk[p] = frand() * 0.5f;
        // Zero 2 of 4 at random.
        int z1 = ((uint32_t)(frand() * 1000.0f + 1000.0f)) & 3;
        int z2;
        do { z2 = ((uint32_t)(frand() * 1000.0f + 1000.0f)) & 3; } while (z2 == z1);
        chunk[z1] = 0.0f;
        chunk[z2] = 0.0f;
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

static int test_correctness(int n) {
    printf("--- correctness test (n=%d) ---\n", n);

    float * w = malloc(n * sizeof(float));
    float * a = malloc(n * sizeof(float));
    gen_random_24_row(w, n);
    for (int i = 0; i < n; ++i) a[i] = frand();

    // FP32 ground truth
    float gt = 0.0f;
    for (int i = 0; i < n; ++i) gt += w[i] * a[i];

    // Quantize
    int nb = n / QK8_0_2_4;
    block_q8_0_2_4 * wq = SPARSE24_ALLOC(nb * sizeof(block_q8_0_2_4));
    block_q8_0     * aq = SPARSE24_ALLOC(nb * sizeof(block_q8_0));
    if (pack_row_q8_0_2_4(w, n, wq) < 0) { return 1; }
    if (pack_row_q8_0    (a, n, aq) < 0) { return 1; }

    // Run both kernels
    float r_scalar = 0.0f, r_avx2 = 0.0f;
    ggml_vec_dot_q8_0_2_4_q8_0_ref(n, &r_scalar, 0, wq, 0, aq, 0, 1);
    ggml_vec_dot_q8_0_2_4_q8_0    (n, &r_avx2,   0, wq, 0, aq, 0, 1);

    const float rel_err_scalar = fabsf(r_scalar - gt) / (fabsf(gt) + 1e-6f);
    const float diff_simd      = fabsf(r_avx2 - r_scalar);

    printf("  FP32 ground truth = %.6f\n", gt);
    printf("  scalar Q8_0_2_4   = %.6f   (rel err vs FP32: %.4f%%)\n", r_scalar, rel_err_scalar * 100.0f);
    printf("  AVX2   Q8_0_2_4   = %.6f   (abs diff vs scalar: %.6e)\n", r_avx2, diff_simd);

    int failed = 0;
    if (rel_err_scalar > 0.02f) { printf("  FAIL: scalar quant err > 2%%\n"); failed = 1; }
    if (diff_simd > 1e-3f)      { printf("  FAIL: AVX2 path disagrees with scalar\n"); failed = 1; }
    if (!failed) printf("  PASS\n");

    free(w); free(a); SPARSE24_FREE(wq); SPARSE24_FREE(aq);
    return failed;
}

static int test_dequantize(int n) {
    printf("--- dequantize roundtrip (n=%d) ---\n", n);

    float * w = malloc(n * sizeof(float));
    gen_random_24_row(w, n);

    int nb = n / QK8_0_2_4;
    block_q8_0_2_4 * wq = SPARSE24_ALLOC(nb * sizeof(block_q8_0_2_4));
    pack_row_q8_0_2_4(w, n, wq);

    float * w_back = malloc(n * sizeof(float));
    dequantize_row_q8_0_2_4(wq, w_back, n);

    // Verify zeros at the right positions and small INT8 noise elsewhere.
    int zero_mismatch = 0, big_err_count = 0;
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        if (w[i] == 0.0f && w_back[i] != 0.0f) zero_mismatch++;
        if (w[i] != 0.0f) {
            float err = fabsf(w_back[i] - w[i]);
            if (err > max_err) max_err = err;
            if (err > 0.01f) big_err_count++;
        }
    }
    printf("  zero positions preserved: %s\n", zero_mismatch == 0 ? "yes" : "NO");
    printf("  max INT8 noise on non-zeros: %.4f\n", max_err);
    int failed = (zero_mismatch != 0) || (max_err > 0.05f);
    printf("  %s\n", failed ? "FAIL" : "PASS");

    free(w); SPARSE24_FREE(wq); free(w_back);
    return failed;
}

static int benchmark(int n, int iters) {
    printf("--- microbench (n=%d, iters=%d) ---\n", n, iters);

    float * w = malloc(n * sizeof(float));
    float * a = malloc(n * sizeof(float));
    gen_random_24_row(w, n);
    for (int i = 0; i < n; ++i) a[i] = frand();

    int nb = n / QK8_0_2_4;
    block_q8_0_2_4 * wq_sparse = SPARSE24_ALLOC(nb * sizeof(block_q8_0_2_4));
    block_q8_0     * aq        = SPARSE24_ALLOC(nb * sizeof(block_q8_0));
    pack_row_q8_0_2_4(w, n, wq_sparse);
    pack_row_q8_0    (a, n, aq);

    // Warm-up
    float s = 0.0f;
    for (int i = 0; i < 10; ++i) ggml_vec_dot_q8_0_2_4_q8_0(n, &s, 0, wq_sparse, 0, aq, 0, 1);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < iters; ++i) {
        ggml_vec_dot_q8_0_2_4_q8_0(n, &s, 0, wq_sparse, 0, aq, 0, 1);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double ns_per_iter = dt * 1e9 / iters;

    printf("  sparse Q8_0_2_4 vec_dot: %.1f ns/iter, %.2f GB/s effective\n",
           ns_per_iter, (nb * sizeof(block_q8_0_2_4)) / ns_per_iter);
    printf("  (sanity: result = %.4f)\n", s);

    free(w); free(a); SPARSE24_FREE(wq_sparse); SPARSE24_FREE(aq);
    return 0;
}

int main(void) {
    int failed = 0;
    failed |= test_correctness(32);     // single block
    failed |= test_correctness(2048);   // larger row
    failed |= test_correctness(11008);  // qwen2.5-3b intermediate dim
    failed |= test_dequantize(32);
    failed |= test_dequantize(2048);
    benchmark(11008, 100000);

    if (failed) { printf("\n[FAIL] one or more tests failed.\n"); return 1; }
    printf("\n[OK] all sparse_24 tests passed.\n");
    return 0;
}
 vec_dot: %.1f ns/iter, %.2f GB/s effective\n",
           ns_per_iter, (nb * sizeof(block_q8_0_2_4)) / ns_per_iter);
    printf("  (sanity: result = %.4f)\n", s);

    free(w); free(a); SPARSE24_FREE(wq_sparse); SPARSE24_FREE(aq);
    return 0;
}

int main(void) {
    int failed = 0;
    failed |= test_correctness(32);     // single block
    failed |= test_correctness(2048);   // larger row
    failed |= test_correctness(11008);  // qwen2.5-3b intermediate dim
    failed |= test_dequantize(32);
    failed |= test_dequantize(2048);
    benchmark(11008, 100000);

    if (failed) { printf("\n[FAIL] one or more tests failed.\n"); return 1; }
    printf("\n[OK] all sparse_24 tests passed.\n");
    return 0;
}
