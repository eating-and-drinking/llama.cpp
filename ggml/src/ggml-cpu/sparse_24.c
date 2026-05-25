// SPDX-License-Identifier: MIT
//
// Scalar reference + SIMD fast paths for GGML_TYPE_Q8_0_2_4.
//
// Dispatch (compile-time, picked top-down):
//   1. AVX-512 BW + VL + VBMI (Ice Lake+)        -> 2-block parallel using VPERMB
//        + with VNNI (Ice Lake+): VPDPBUSD for the dot product
//        + without VNNI: fall back to MADDUBS + MADD
//   2. AVX2 (Haswell+ / Excavator+)              -> 1-block using PSHUFB gather
//   3. AArch64 NEON (Apple Silicon, ARMv8 phone) -> 1-block using VQTBL2Q
//        + with DOTPROD (ARMv8.4+): SDOT for the dot product
//        + without DOTPROD: widening mull + pairwise add
//   4. otherwise                                  -> delegate to scalar reference

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "sparse_24.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "ggml-common.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// Portable FP16 -> FP32.
#if defined(__F16C__)
static inline float sparse24_fp16_to_fp32(ggml_half h) {
    return _cvtsh_ss((unsigned short) h);
}
#elif defined(__ARM_NEON) && defined(__aarch64__)
static inline float sparse24_fp16_to_fp32(ggml_half h) {
    __fp16 v;
    memcpy(&v, &h, sizeof(v));
    return (float) v;
}
#else
static inline float sparse24_fp16_to_fp32(ggml_half h) {
    uint32_t x = (uint32_t) h;
    uint32_t sign = (x & 0x8000) << 16;
    uint32_t exp  = (x & 0x7c00) >> 10;
    uint32_t mant = (x & 0x03ff);
    uint32_t f;
    if (exp == 0) { f = sign; }
    else if (exp == 31) { f = sign | 0x7f800000u | (mant << 13); }
    else { f = sign | ((exp + 112) << 23) | (mant << 13); }
    union { uint32_t u; float f; } cvt; cvt.u = f; return cvt.f;
}
#endif
#undef  GGML_CPU_FP16_TO_FP32
#define GGML_CPU_FP16_TO_FP32(x) sparse24_fp16_to_fp32(x)


// =========================================================================
// Scalar reference (always compiled, always exported)
// =========================================================================

void ggml_vec_dot_q8_0_2_4_q8_0_ref(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc) {

    assert(n % QK8_0_2_4 == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_q8_0_2_4 * GGML_RESTRICT x = (const block_q8_0_2_4 *) vx;
    const block_q8_0      * GGML_RESTRICT y = (const block_q8_0 *)      vy;

    const int nb = n / QK8_0_2_4;
    float sumf = 0.0f;

    for (int b = 0; b < nb; ++b) {
        int32_t acc_i = 0;
        int kept = 0;
        for (int g = 0; g < 8; ++g) {
            const uint8_t idx_byte = x[b].idx[g];
            const int p1 = (idx_byte >> 4) & 0x3;
            const int p2 = (idx_byte     ) & 0x3;
            // 2:4 invariant: the two kept positions in a group must be distinct.
            // Asserted in debug builds; release builds trust the packer.
            assert(p1 != p2);
            const int base = g * 4;
            acc_i += (int32_t)x[b].qs[kept]     * (int32_t)y[b].qs[base + p1];
            acc_i += (int32_t)x[b].qs[kept + 1] * (int32_t)y[b].qs[base + p2];
            kept += 2;
        }
        const float d = GGML_CPU_FP16_TO_FP32(x[b].d) * GGML_CPU_FP16_TO_FP32(y[b].d);
        sumf += (float)acc_i * d;
    }

    *s = sumf;
}

void dequantize_row_q8_0_2_4(
    const block_q8_0_2_4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {

    assert(k % QK8_0_2_4 == 0);
    const int64_t nb = k / QK8_0_2_4;

    for (int64_t b = 0; b < nb; ++b) {
        const float d = GGML_CPU_FP16_TO_FP32(x[b].d);
        float * GGML_RESTRICT yb = y + b * QK8_0_2_4;
        memset(yb, 0, QK8_0_2_4 * sizeof(float));
        int kept = 0;
        for (int g = 0; g < 8; ++g) {
            const uint8_t idx_byte = x[b].idx[g];
            const int p1 = (idx_byte >> 4) & 0x3;
            const int p2 = (idx_byte     ) & 0x3;
            // 2:4 invariant: the two kept positions in a group must be distinct.
            // Asserted in debug builds; release builds trust the packer.
            assert(p1 != p2);
            const int base = g * 4;
            yb[base + p1] = (float)x[b].qs[kept]     * d;
            yb[base + p2] = (float)x[b].qs[kept + 1] * d;
            kept += 2;
        }
    }
}


// =========================================================================
// Fast-path: AVX-512 (BW + VL + VBMI)
// =========================================================================

#if defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VBMI__)

static inline float sparse24_hsum_ps_256(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// SIMD construction of the 16 absolute gather indices for one block.
//   Input  : 8 packed bytes, each = (p1 << 4) | p2 with p1, p2 in 0..3.
//   Output : 16 bytes in [0..31], pattern [g*4+p1, g*4+p2] for g = 0..7.
// Cost: 4 cheap intrinsics; replaces a scalar loop of 16 byte stores.
static inline __m128i sparse24_build_block_idx(const uint8_t * GGML_RESTRICT idx) {
    const __m128i base = _mm_setr_epi8(
        0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28);
    // Mask to 0..3 (the valid range of a 2:4 position), matching the scalar
    // reference's `(idx_byte >> 4) & 0x3` / `idx_byte & 0x3`. Equivalent to 0x0F
    // on well-formed inputs (high nibble is always 0..3); 0x03 makes us bit-
    // identical to scalar on malformed inputs as well.
    const __m128i nibble_mask = _mm_set1_epi8(0x03);

    // Load 8 idx bytes into the low 64 bits; upper 64 bits zeroed.
    const __m128i idx8 = _mm_loadl_epi64((const __m128i *) idx);
    // Shift each 16-bit lane right by 4, then mask to 0x03:
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(idx8, 4), nibble_mask);
    // Low nibble masked to 0x03 (top 2 bits of each nibble discarded).
    const __m128i lo = _mm_and_si128(idx8, nibble_mask);
    // Interleave: result byte 2g = hi[g] (= p1), byte 2g+1 = lo[g] (= p2).
    const __m128i interleaved = _mm_unpacklo_epi8(hi, lo);
    return _mm_add_epi8(interleaved, base);
}

void ggml_vec_dot_q8_0_2_4_q8_0(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc) {

    assert(n % QK8_0_2_4 == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_q8_0_2_4 * GGML_RESTRICT x = (const block_q8_0_2_4 *) vx;
    const block_q8_0      * GGML_RESTRICT y = (const block_q8_0 *)      vy;

    const int nb = n / QK8_0_2_4;
    __m256 acc = _mm256_setzero_ps();
#if !defined(__AVX512VNNI__)
    const __m256i ones = _mm256_set1_epi16(1);
#endif

    int b = 0;
    for (; b + 1 < nb; b += 2) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[b].d)   * GGML_CPU_FP16_TO_FP32(y[b].d);
        const float d1 = GGML_CPU_FP16_TO_FP32(x[b+1].d) * GGML_CPU_FP16_TO_FP32(y[b+1].d);

        // 32 weight bytes = 2 blocks concatenated
        const __m128i w0  = _mm_loadu_si128((const __m128i *) x[b].qs);
        const __m128i w1  = _mm_loadu_si128((const __m128i *) x[b+1].qs);
        const __m256i wq  = _mm256_set_m128i(w1, w0);

        // 64 activation bytes = 2 blocks concatenated, into a zmm
        const __m256i a0  = _mm256_loadu_si256((const __m256i *) y[b].qs);
        const __m256i a1  = _mm256_loadu_si256((const __m256i *) y[b+1].qs);
        const __m512i ak  = _mm512_inserti64x4(_mm512_castsi256_si512(a0), a1, 1);

        // 32 absolute gather indices, built entirely in SIMD.
        const __m128i idx_b  = sparse24_build_block_idx(x[b].idx);
        const __m128i idx_b1 = _mm_add_epi8(sparse24_build_block_idx(x[b+1].idx),
                                            _mm_set1_epi8(32));
        const __m256i idx_ymm = _mm256_set_m128i(idx_b1, idx_b);
        const __m512i idx_zmm = _mm512_castsi256_si512(idx_ymm);

        // Cross-lane byte permute (VPERMB / AVX512VBMI).
        const __m512i sel_zmm = _mm512_permutexvar_epi8(idx_zmm, ak);
        const __m256i aq      = _mm512_castsi512_si256(sel_zmm);

        // Flip-trick + VPDPBUSD (or MADDUBS+MADD fallback).
        const __m256i ax = _mm256_sign_epi8(wq, wq);
        const __m256i sy = _mm256_sign_epi8(aq, wq);
#if defined(__AVX512VNNI__)
        const __m256i s32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ax, sy);
#else
        const __m256i dot = _mm256_maddubs_epi16(ax, sy);
        const __m256i s32 = _mm256_madd_epi16(ones, dot);
#endif

        // Lanes 0..3 belong to block b (use d0), 4..7 to block b+1 (d1).
        const __m256 d_v   = _mm256_set_ps(d1, d1, d1, d1, d0, d0, d0, d0);
        const __m256 sum_f = _mm256_cvtepi32_ps(s32);
        acc = _mm256_fmadd_ps(d_v, sum_f, acc);
    }

    float result = sparse24_hsum_ps_256(acc);

    // Tail: 0 or 1 block remaining.
    for (; b < nb; ++b) {
        float tail = 0.0f;
        ggml_vec_dot_q8_0_2_4_q8_0_ref(QK8_0_2_4, &tail, 0, x + b, 0, y + b, 0, 1);
        result += tail;
    }

    *s = result;
}


// =========================================================================
// Fast-path: AVX2  -- 2 blocks per iteration via lane-local PSHUFB gather
// =========================================================================
//
// Each 128-bit AVX2 lane gathers its own block's activations using PSHUFB
// (PSHUFB is per-lane only on AVX2, so cross-lane VPERMB isn't available).
// We pack two blocks side by side and run the dot-product across both halves
// of the 256-bit accumulator, exactly doubling throughput vs the old
// 1-block/iter path.

#elif defined(__AVX2__)

static inline float sparse24_hsum_ps_256(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// Per-block index builder, shared with AVX-512 in spirit (separate compile
// unit each side of the #if/#elif). See the AVX-512 path for the algorithm.
static inline __m128i sparse24_build_block_idx_avx2(const uint8_t * GGML_RESTRICT idx) {
    const __m128i base = _mm_setr_epi8(
        0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28);
    // Mask to 0..3 (the valid range of a 2:4 position), matching the scalar
    // reference's `(idx_byte >> 4) & 0x3` / `idx_byte & 0x3`. Equivalent to 0x0F
    // on well-formed inputs (high nibble is always 0..3); 0x03 makes us bit-
    // identical to scalar on malformed inputs as well.
    const __m128i nibble_mask = _mm_set1_epi8(0x03);
    const __m128i idx8 = _mm_loadl_epi64((const __m128i *) idx);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(idx8, 4), nibble_mask);
    const __m128i lo = _mm_and_si128(idx8, nibble_mask);
    return _mm_add_epi8(_mm_unpacklo_epi8(hi, lo), base);
}

// 256-bit signed i8 dot product packed into 8 int32 lanes.
// Same flip-trick as AVX-512: ax = |x|, sy = sign(x)*y.
static inline __m256i sparse24_mul_sum_i8_pairs_256(__m256i x, __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_madd_epi16(ones, dot);
}

void ggml_vec_dot_q8_0_2_4_q8_0(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc) {

    assert(n % QK8_0_2_4 == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_q8_0_2_4 * GGML_RESTRICT x = (const block_q8_0_2_4 *) vx;
    const block_q8_0      * GGML_RESTRICT y = (const block_q8_0 *)      vy;

    const int nb = n / QK8_0_2_4;
    __m256 acc = _mm256_setzero_ps();

    // Hoisted constants for the OR-trick gather. PSHUFB clears any output byte
    // whose index has bit 7 set, so we precondition the indices so that:
    //   sel_lo = PSHUFB(a_lo, idx + 112)   - idx in 0..15 stays in 0x70..0x7F
    //                                        (bit 7 clear -> gathers byte from a_lo);
    //                                        idx in 16..31 becomes >=128
    //                                        (bit 7 set -> PSHUFB zeros it).
    //   sel_hi = PSHUFB(a_hi, idx - 16)    - idx in 16..31 becomes 0..15
    //                                        (gathers byte from a_hi);
    //                                        idx in 0..15 underflows to 0xF0..0xFF
    //                                        (bit 7 set -> PSHUFB zeros it).
    //   aq    = sel_lo | sel_hi            - exactly one is non-zero per lane.
    //
    // This replaces { cmpgt, and, andnot, sub, pshufb*2, blendv } = 7 ops
    // with         { add,  sub, pshufb*2, or }                    = 5 ops,
    // and eliminates the high-latency port-5-only blendv_epi8.
    const __m256i v112 = _mm256_set1_epi8(112);
    const __m256i v16  = _mm256_set1_epi8(16);

    // Lookahead distance (in blocks) for software prefetch. PF=32 blocks
    // ahead == ~832 B of weight (13 cache lines) and ~1088 B of activation
    // (17 lines) — far enough that the line is in L1 by the time we need it
    // on a DDR4-2667 system (~200-cycle DRAM, ~2 cycles/iter at peak), and
    // close enough that we don't pollute L1 with lines we won't reach.
    // _MM_HINT_T0 targets L1d (-> all higher caches as a side effect).
    const int PF = 32;

    int b = 0;
    for (; b + 1 < nb; b += 2) {
        if (b + PF < nb) {
            _mm_prefetch((const char *) &x[b + PF], _MM_HINT_T0);
            _mm_prefetch((const char *) &y[b + PF], _MM_HINT_T0);
        }
        const float d0 = GGML_CPU_FP16_TO_FP32(x[b].d)   * GGML_CPU_FP16_TO_FP32(y[b].d);
        const float d1 = GGML_CPU_FP16_TO_FP32(x[b+1].d) * GGML_CPU_FP16_TO_FP32(y[b+1].d);

        // Weights: 16 bytes from each block -> __m256i (block b low, b+1 high)
        const __m128i w0 = _mm_loadu_si128((const __m128i *) x[b].qs);
        const __m128i w1 = _mm_loadu_si128((const __m128i *) x[b+1].qs);
        const __m256i wq = _mm256_set_m128i(w1, w0);

        // Activations: each block's 32 bytes split into per-lane lo/hi halves.
        // (Tried VPERM2I128 deinterleave, but on this uarch it's slower than
        // 4 movdqu + 2 vinserti128 -- non-aligned 256-bit loads + VPERM2I128's
        // higher latency on Zen / AVX2-era cores tip the balance the other way.)
        const __m128i a0_lo = _mm_loadu_si128((const __m128i *)  y[b].qs);
        const __m128i a0_hi = _mm_loadu_si128((const __m128i *) (y[b].qs + 16));
        const __m128i a1_lo = _mm_loadu_si128((const __m128i *)  y[b+1].qs);
        const __m128i a1_hi = _mm_loadu_si128((const __m128i *) (y[b+1].qs + 16));
        const __m256i a_lo  = _mm256_set_m128i(a1_lo, a0_lo);
        const __m256i a_hi  = _mm256_set_m128i(a1_hi, a0_hi);

        // Indices in 0..31, per-lane addressing (each lane handles its block).
        const __m128i idx_b  = sparse24_build_block_idx_avx2(x[b].idx);
        const __m128i idx_b1 = sparse24_build_block_idx_avx2(x[b+1].idx);
        const __m256i idx_vec = _mm256_set_m128i(idx_b1, idx_b);

        // Lane-local PSHUFB gather via OR-trick (see comment above the loop).
        const __m256i sel_l = _mm256_shuffle_epi8(a_lo, _mm256_add_epi8(idx_vec, v112));
        const __m256i sel_h = _mm256_shuffle_epi8(a_hi, _mm256_sub_epi8(idx_vec, v16));
        const __m256i aq    = _mm256_or_si256(sel_l, sel_h);

        // Dot product: full 256-bit, 8 int32 lanes (lanes 0..3 = b, 4..7 = b+1).
        const __m256i s32   = sparse24_mul_sum_i8_pairs_256(wq, aq);
        const __m256  d_v   = _mm256_set_ps(d1, d1, d1, d1, d0, d0, d0, d0);
        const __m256  sum_f = _mm256_cvtepi32_ps(s32);
        acc = _mm256_fmadd_ps(d_v, sum_f, acc);
    }

    float result = sparse24_hsum_ps_256(acc);

    // Tail: 0 or 1 block remaining -- delegate to scalar reference.
    for (; b < nb; ++b) {
        float tail = 0.0f;
        ggml_vec_dot_q8_0_2_4_q8_0_ref(QK8_0_2_4, &tail, 0, x + b, 0, y + b, 0, 1);
        result += tail;
    }

    *s = result;
}


// =========================================================================
// Fast-path: AArch64 NEON
// =========================================================================

#elif defined(__ARM_NEON) && defined(__aarch64__)

static inline void sparse24_build_gather_neon(const uint8_t * GGML_RESTRICT idx,
                                              uint8_t * GGML_RESTRICT out) {
    for (int g = 0; g < 8; ++g) {
        const uint8_t b = idx[g];
        out[g * 2    ] = (uint8_t)(g * 4 + ((b >> 4) & 0x3));
        out[g * 2 + 1] = (uint8_t)(g * 4 + ((b     ) & 0x3));
    }
}

void ggml_vec_dot_q8_0_2_4_q8_0(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc) {

    assert(n % QK8_0_2_4 == 0);
    assert(nrc == 1);
    (void)bs; (void)bx; (void)by; (void)nrc;

    const block_q8_0_2_4 * GGML_RESTRICT x = (const block_q8_0_2_4 *) vx;
    const block_q8_0      * GGML_RESTRICT y = (const block_q8_0 *)      vy;

    const int nb = n / QK8_0_2_4;
    float32x4_t acc = vdupq_n_f32(0.0f);

    for (int b = 0; b < nb; ++b) {
        const float d = GGML_CPU_FP16_TO_FP32(x[b].d) * GGML_CPU_FP16_TO_FP32(y[b].d);

        const int8x16_t wq = vld1q_s8(x[b].qs);

        int8x16x2_t a_table;
        a_table.val[0] = vld1q_s8(y[b].qs);
        a_table.val[1] = vld1q_s8(y[b].qs + 16);

        uint8_t abs_idx[16];
        sparse24_build_gather_neon(x[b].idx, abs_idx);
        const uint8x16_t idx_vec = vld1q_u8(abs_idx);

        const int8x16_t aq = vqtbl2q_s8(a_table, idx_vec);

#if defined(__ARM_FEATURE_DOTPROD)
        // SDOT: one instruction does 4 lanes of (i8 * i8) accumulation.
        int32x4_t sum_i32 = vdotq_s32(vdupq_n_s32(0), wq, aq);
#else
        const int16x8_t prod_lo = vmull_s8(vget_low_s8(wq),  vget_low_s8(aq));
        const int16x8_t prod_hi = vmull_s8(vget_high_s8(wq), vget_high_s8(aq));
        int32x4_t sum_i32 = vpaddlq_s16(prod_lo);
        sum_i32           = vpadalq_s16(sum_i32, prod_hi);
#endif

        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(sum_i32), d);
    }

    *s = vaddvq_f32(acc);
}


// =========================================================================
// Scalar fallback: no SIMD available -- delegate to the reference.
// =========================================================================

#else

void ggml_vec_dot_q8_0_2_4_q8_0(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc) {
    ggml_vec_dot_q8_0_2_4_q8_0_ref(n, s, bs, vx, bx, vy, by, nrc);
}

#endif
