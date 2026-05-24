# SPDX-License-Identifier: Apache-2.0
"""block_q8_0_2_4 packing primitives.

Encodes a (2:4 already-sparse) 2-D weight matrix into the binary layout
expected by llama.cpp's GGML_TYPE_Q8_0_2_4 kernel:

    per 32 consecutive input-columns of one row:
        ggml_half  d;        // 2 bytes — scale (FP16)
        int8_t     qs[16];   // 16 non-zero INT8 values (in original column order)
        uint8_t    idx[8];   // 8 group bytes, each encoding the 2 kept positions
                             // within that group of 4 columns:
                             //   high nibble = first kept position (0..3)
                             //   low  nibble = second kept position (0..3)
    => 26 bytes per 32-column block

Constraints on the input matrix:
    * shape must be (out_dim, in_dim) and in_dim % 32 == 0
    * each row must satisfy 2:4 along columns: in every 4 consecutive
      columns, exactly 2 values are zero.

The packer is dtype-agnostic on the input side (FP16 / BF16 / FP32 / FP64)
and always emits the same 26-byte block.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

BLOCK_N = 32          # original columns per block
NONZ_PER_BLOCK = 16   # non-zero positions per block (= BLOCK_N / 2)
GROUPS_PER_BLOCK = 8  # number of 4-column groups per block
BYTES_PER_BLOCK = 2 + 16 + 8  # 26


def _half_from_float(x: float) -> int:
    """Convert a Python float to its IEEE-754 binary16 representation (uint16)."""
    return int(np.array([x], dtype=np.float32).astype(np.float16).view(np.uint16)[0])


def pack_q8_0_2_4_row(row: np.ndarray) -> bytes:
    """Pack one row of a 2:4-sparse matrix into block_q8_0_2_4 binary.

    Parameters
    ----------
    row : 1-D ndarray of length divisible by 32.
        Must satisfy strict 2:4: every consecutive 4 values has exactly 2 zeros.

    Returns
    -------
    bytes of length ``(len(row) / 32) * 26``.
    """
    if row.ndim != 1:
        raise ValueError("expected 1-D row")
    n = row.size
    if n % BLOCK_N != 0:
        raise ValueError(f"row length {n} not divisible by {BLOCK_N}")

    fr = row.astype(np.float32, copy=False)
    n_blocks = n // BLOCK_N
    out = bytearray(n_blocks * BYTES_PER_BLOCK)

    cursor = 0
    for b in range(n_blocks):
        block = fr[b * BLOCK_N : (b + 1) * BLOCK_N]
        # Per-block scale uses ABSMAX of non-zero values to map to int8 range.
        absmax = float(np.abs(block).max())
        if absmax == 0.0:
            scale = 1.0  # all-zero block; arbitrary, will quantize to 0
        else:
            scale = absmax / 127.0
        inv_scale = 1.0 / scale

        # Walk the 8 groups of 4 columns.
        qs = np.zeros(NONZ_PER_BLOCK, dtype=np.int8)
        idx_bytes = bytearray(GROUPS_PER_BLOCK)
        kept_count = 0

        for g in range(GROUPS_PER_BLOCK):
            chunk = block[g * 4 : (g + 1) * 4]
            kept_positions = [p for p in range(4) if chunk[p] != 0.0]
            if len(kept_positions) != 2:
                raise ValueError(
                    f"block {b} group {g}: expected exactly 2 non-zeros, "
                    f"got {len(kept_positions)} (chunk = {chunk.tolist()})"
                )
            p1, p2 = kept_positions
            # Quantize the two kept values
            for p in (p1, p2):
                q = int(round(chunk[p] * inv_scale))
                q = max(-127, min(127, q))   # symmetric range, matches Q8_0
                qs[kept_count] = q
                kept_count += 1
            # Encode group's 2 kept positions into one byte:  high|low nibble
            idx_bytes[g] = (p1 << 4) | p2

        # Emit block: 2 bytes scale | 16 bytes qs | 8 bytes idx
        out[cursor : cursor + 2] = _half_from_float(scale).to_bytes(2, "little")
        out[cursor + 2 : cursor + 18] = qs.tobytes()
        out[cursor + 18 : cursor + 26] = bytes(idx_bytes)
        cursor += BYTES_PER_BLOCK

    return bytes(out)


def pack_q8_0_2_4(W: np.ndarray) -> Tuple[bytes, Tuple[int, ...]]:
    """Pack a 2-D 2:4-sparse weight matrix.

    Returns
    -------
    (packed_bytes, byte_shape)
        ``packed_bytes`` is a flat bytes object containing ``out_dim`` rows of
        packed blocks concatenated.
        ``byte_shape`` is the shape gguf-py expects when ``raw_dtype`` is used:
        ``(out_dim, n_blocks_per_row * 26)``.
    """
    if W.ndim != 2:
        raise ValueError(f"expected 2-D weight, got {W.ndim}-D")
    out_dim, in_dim = W.shape
    if in_dim % BLOCK_N != 0:
        raise ValueError(f"in_dim {in_dim} not divisible by {BLOCK_N}")

    n_blocks = in_dim // BLOCK_N
    row_bytes = n_blocks * BYTES_PER_BLOCK
    packed = bytearray(out_dim * row_bytes)
    for r in range(out_dim):
        packed[r * row_bytes : (r + 1) * row_bytes] = pack_q8_0_2_4_row(W[r])

    return bytes(packed), (out_dim, row_bytes)


def verify_2_4(W: np.ndarray) -> bool:
    """Return True iff W satisfies strict 2:4 on its last axis."""
    if W.shape[-1] % 4 != 0:
        return False
    last = W.shape[-1]
    groups = W.reshape(*W.shape[:-1], last // 4, 4)
    zeros_per_group = (groups == 0).sum(axis=-1)
    return bool(np.all(zeros_per_group == 2))


# ----------------------- self test -----------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Build a canonical 2:4 matrix
    H = 64
    W = rng.standard_normal((H, 64), dtype=np.float32)
    for r in range(H):
        for g in range(16):
            zero_pos = rng.choice(4, size=2, replace=False) + g * 4
            W[r, zero_pos] = 0.0
    assert verify_2_4(W), "test matrix is not 2:4"

    packed, byte_shape = pack_q8_0_2_4(W)
    assert byte_shape == (H, 2 * 26), f"unexpected byte shape {byte_shape}"
    print(f"OK packed {W.shape} float32 -> {len(packed)} bytes "
          f"({len(packed) * 8 / W.size:.2f} bits per original element)")
    print(f"   compression ratio vs FP16: {W.size * 2 / len(packed):.2f}x")
    print(f"   compression ratio vs Q8_0: {W.size * 34 / 32 / len(packed):.2f}x")
