// SPDX-License-Identifier: MIT
//
// block_q8_0_2_4 — 2:4 structured-sparse INT8 quantization for ggml.
//
// Layout (26 bytes per 32 original columns):
//     ggml_half  d;        // scale (FP16)
//     int8_t     qs[16];   // 16 non-zero INT8 values, in original column order
//     uint8_t    idx[8];   // 8 group bytes; high|low nibble = the two kept
//                          // positions (each 0..3) within that 4-column group
//
// Saving vs dense block_q8_0 (34 bytes/32 cols): 23.5% smaller.
// On memory-bandwidth-bound CPUs (e.g. i9-9880H @ DDR4-2667), this directly
// translates to a similar speedup on weight-loading time during decoding.

#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-impl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QK8_0_2_4 32

typedef struct {
    ggml_half d;
    int8_t  qs[16];
    uint8_t idx[8];
} block_q8_0_2_4;

static_assert(sizeof(block_q8_0_2_4) == 26, "wrong block_q8_0_2_4 size");

// Scalar reference (correctness baseline). Used on platforms without SIMD
// implementations and as the testing oracle.
//
//   n   : total number of original (pre-sparsification) elements processed.
//         Must be a multiple of QK8_0_2_4.
//   vx  : pointer to a sequence of block_q8_0_2_4   (the sparse weight)
//   vy  : pointer to a sequence of block_q8_0       (the dense activation)
//   *s  : accumulator (single scalar dot-product)
void ggml_vec_dot_q8_0_2_4_q8_0_ref(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc);

// SIMD-optimised vec_dot. Compile-time dispatch picks (in order):
//   * AVX-512 (BW+VL+VBMI, Ice Lake+)        — 2 blocks/iter via VPERMB
//   * AVX2 (Haswell+ / Excavator+)           — 2 blocks/iter via PSHUFB
//                                              (lane-local gather + OR-trick;
//                                               software prefetch lookahead)
//   * AArch64 NEON (Apple Silicon, ARMv8)    — 1 block/iter via VQTBL2Q
//   * otherwise                              — delegates to the scalar reference.
//
// All paths require nrc == 1. The dispatcher's nrc==2 path (a 2x2 output tile,
// see the ARM i8mm Q8_0 reference) is not implemented for this type — the
// type_traits_cpu entry sets .nrows = 1 so ggml never asks for it.
void ggml_vec_dot_q8_0_2_4_q8_0(
    int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx, size_t bx,
    const void * GGML_RESTRICT vy, size_t by, int nrc);

// Block-level dequantization (inverse of pack_q8_0_2_4 in the Python tool).
// Useful for `ggml_get_type_traits().to_float` registration.
void dequantize_row_q8_0_2_4(
    const block_q8_0_2_4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

#ifdef __cplusplus
}
#endif
