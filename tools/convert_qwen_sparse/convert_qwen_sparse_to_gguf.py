#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert a qwen-compress (Qwen2.5 + 2:4 sparse) checkpoint to sparse GGUF.

Input layout (produced by `qwen-compress prune --config .../sparsegpt_perm_2_4.yaml`):

    <input_dir>/
        config.json                    # standard HF config
        tokenizer.json / tokenizer.model
        model.safetensors              # BF16/FP16 weights, some tensors already 2:4 sparse
        compression_meta.json          # records {stage, sparsity_type, permutation, ...}

Output: a single GGUF file with:
    - Standard Qwen2 architecture metadata
    - 2:4 sparse linear weights packed into GGML_TYPE_Q8_0_2_4 (block_q8_0_2_4)
    - Dense fallback weights (embedding, norms, lm_head) in F16 or Q8_0
    - general.sparsity_layout = "2:4"

Run:
    python convert_qwen_sparse_to_gguf.py \\
        /path/to/qwen-compress/checkpoints/qat/export \\
        --outfile qwen2.5-3b-q8-2-4.gguf \\
        --dense-dtype f16

The packed Q8_0_2_4 layout is decoded by the AVX2 vec_dot kernel in
``ggml/src/ggml-cpu/sparse_24.c``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# Allow running without installing the convert tool — patch sys.path so we can
# import gguf and our packer.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "gguf-py"))
sys.path.insert(0, str(HERE))

import gguf  # noqa: E402
from pack_q8_0_2_4 import (  # noqa: E402
    BYTES_PER_BLOCK,
    pack_q8_0_2_4,
    verify_2_4,
)


# --- Tensor classification --------------------------------------------------


# Linears in a Qwen2 decoder block that are produced as 2:4 sparse by
# qwen-compress (when configured with permutation.enabled=true). The transformer
# blocks are named "model.layers.{i}.<part>" in safetensors.
SPARSE_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


def is_sparse_candidate(tensor_name: str) -> bool:
    return any(tensor_name.endswith(s) for s in SPARSE_SUFFIXES)


# --- Name translation (HF -> GGUF) ------------------------------------------

# Qwen2 mapping. Borrowed from llama.cpp/convert_hf_to_gguf.py logic, simplified.
def hf_to_gguf_name(hf_name: str) -> str:
    n = hf_name
    if n == "model.embed_tokens.weight":
        return "token_embd.weight"
    if n == "model.norm.weight":
        return "output_norm.weight"
    if n == "lm_head.weight":
        return "output.weight"
    # Decoder layers
    if n.startswith("model.layers."):
        n = n.replace("model.layers.", "blk.", 1)
        n = n.replace(".self_attn.q_proj", ".attn_q")
        n = n.replace(".self_attn.k_proj", ".attn_k")
        n = n.replace(".self_attn.v_proj", ".attn_v")
        n = n.replace(".self_attn.o_proj", ".attn_output")
        n = n.replace(".input_layernorm", ".attn_norm")
        n = n.replace(".post_attention_layernorm", ".ffn_norm")
        n = n.replace(".mlp.gate_proj", ".ffn_gate")
        n = n.replace(".mlp.up_proj", ".ffn_up")
        n = n.replace(".mlp.down_proj", ".ffn_down")
        return n
    return n


# --- Dense quantization helpers ---------------------------------------------


def quantize_q8_0(W: np.ndarray) -> bytes:
    """Pack a dense FP32 matrix as standard Q8_0 (32 elements, 2 + 32 = 34 bytes).

    Only used for dense tensors we don't want to keep in FP16.
    """
    assert W.shape[-1] % 32 == 0
    flat = W.reshape(-1, 32).astype(np.float32, copy=False)
    nb = flat.shape[0]
    out = bytearray(nb * 34)
    for b in range(nb):
        block = flat[b]
        amax = float(np.abs(block).max())
        scale = 1.0 if amax == 0 else amax / 127.0
        inv = 1.0 / scale
        q = np.clip(np.round(block * inv), -127, 127).astype(np.int8)
        # FP16 scale
        h = int(np.array([scale], dtype=np.float16).view(np.uint16)[0])
        out[b * 34 : b * 34 + 2] = h.to_bytes(2, "little")
        out[b * 34 + 2 : (b + 1) * 34] = q.tobytes()
    return bytes(out)


# --- Main --------------------------------------------------------------------


def load_safetensors(path: Path) -> Dict[str, np.ndarray]:
    try:
        from safetensors.numpy import load_file
    except ImportError as e:
        raise SystemExit("Please `pip install safetensors`") from e
    return load_file(str(path))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, help="qwen-compress export directory")
    ap.add_argument("--outfile", type=Path, required=True, help="Output .gguf path")
    ap.add_argument("--dense-dtype", choices=["f16", "q8_0"], default="f16",
                    help="Format for tensors that are NOT 2:4 sparse")
    ap.add_argument("--force-sparse", action="store_true",
                    help="Treat ALL linears as 2:4 (fail loudly if any row is non-2:4)")
    args = ap.parse_args()

    in_dir = args.input_dir
    if not in_dir.is_dir():
        raise SystemExit(f"{in_dir} is not a directory")

    # --- 1. Read HF config + safetensors ---
    cfg = json.loads((in_dir / "config.json").read_text(encoding="utf-8"))
    weights = load_safetensors(in_dir / "model.safetensors")
    meta_path = in_dir / "compression_meta.json"
    comp_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    print(f"Loaded {len(weights)} tensors from {in_dir}")
    print(f"Compression metadata: {comp_meta.get('stage', 'unknown')}, "
          f"sparsity={comp_meta.get('sparsity_type', 'unknown')}")

    # --- 2. Open GGUF writer ---
    writer = gguf.GGUFWriter(str(args.outfile), arch="qwen2")

    # --- 3. Architecture metadata ---
    writer.add_name(in_dir.name)
    writer.add_context_length(cfg.get("max_position_embeddings", 32768))
    writer.add_embedding_length(cfg["hidden_size"])
    writer.add_block_count(cfg["num_hidden_layers"])
    writer.add_feed_forward_length(cfg["intermediate_size"])
    writer.add_head_count(cfg["num_attention_heads"])
    writer.add_head_count_kv(cfg.get("num_key_value_heads", cfg["num_attention_heads"]))
    writer.add_rope_freq_base(cfg.get("rope_theta", 10000.0))
    writer.add_layer_norm_rms_eps(cfg.get("rms_norm_eps", 1e-6))
    writer.add_vocab_size(cfg["vocab_size"])

    # Our 2:4 marker — this is the field added in gguf-py/constants.py
    writer.add_string(gguf.Keys.General.SPARSITY_LAYOUT, "2:4")
    # Reuse the standard file_type field as best-effort hint.
    writer.add_file_type(41)  # LLAMA_FTYPE_MOSTLY_Q8_0_2_4

    # --- 4. Tokenizer (delegate to the standard HF -> GGUF path) ---
    # For brevity we don't reimplement BPE merging here. Users should run the
    # tokenizer through llama.cpp's existing convert script, or accept that
    # the tokenizer needs a second tool. This converter focuses on the weight
    # path which is the novel part.
    print("\nNOTE: This converter writes weights only. Run "
          "convert_hf_to_gguf.py --vocab-only to merge tokenizer metadata "
          "into the same .gguf if needed.\n")

    # --- 5. Walk tensors and quantize/pack ---
    n_sparse = n_dense_f16 = n_dense_q8 = 0
    sparse_bytes = dense_bytes = 0

    for hf_name, arr in weights.items():
        gguf_name = hf_to_gguf_name(hf_name)
        W = arr.astype(np.float32) if arr.dtype not in (np.float32,) else arr

        # ---- Sparse path: 2:4-aware linears ----
        if is_sparse_candidate(hf_name) and W.ndim == 2:
            in_dim = W.shape[1]
            if in_dim % 32 == 0 and verify_2_4(W):
                packed, byte_shape = pack_q8_0_2_4(W)
                # Reshape for gguf: present the bytes as a (out_dim, n_blocks*26)
                # uint8 ndarray so the writer records the correct nbytes.
                arr_bytes = np.frombuffer(packed, dtype=np.uint8).reshape(byte_shape)
                writer.add_tensor(
                    gguf_name, arr_bytes,
                    raw_dtype=gguf.GGMLQuantizationType.Q8_0_2_4,
                )
                n_sparse += 1
                sparse_bytes += len(packed)
                print(f"  [sparse 2:4] {gguf_name:40s} shape={W.shape} -> {len(packed)} bytes")
                continue
            elif args.force_sparse:
                raise SystemExit(
                    f"--force-sparse: tensor {hf_name} (shape={W.shape}) is "
                    f"not strictly 2:4 along its last axis."
                )
            # else: fall through to dense

        # ---- Dense path ----
        if args.dense_dtype == "f16" or W.ndim != 2 or W.shape[-1] % 32 != 0:
            arr_f16 = W.astype(np.float16)
            writer.add_tensor(gguf_name, arr_f16)
            n_dense_f16 += 1
            dense_bytes += arr_f16.nbytes
            print(f"  [dense f16] {gguf_name:40s} shape={W.shape}")
        else:
            packed = quantize_q8_0(W)
            byte_shape = (W.shape[0], (W.shape[1] // 32) * 34)
            arr_bytes = np.frombuffer(packed, dtype=np.uint8).reshape(byte_shape)
            writer.add_tensor(
                gguf_name, arr_bytes,
                raw_dtype=gguf.GGMLQuantizationType.Q8_0,
            )
            n_dense_q8 += 1
            dense_bytes += len(packed)
            print(f"  [dense q8_0] {gguf_name:40s} shape={W.shape} -> {len(packed)} bytes")

    # --- 6. Write file ---
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    total = sparse_bytes + dense_bytes
    print(f"\n=== Conversion done ===")
    print(f"Output:        {args.outfile} ({total / 1e6:.1f} MB)")
    print(f"  Sparse 2:4:  {n_sparse:3d} tensors, {sparse_bytes / 1e6:.1f} MB")
    print(f"  Dense f16:   {n_dense_f16:3d} tensors")
    print(f"  Dense q8_0:  {n_dense_q8:3d} tensors")
    print(f"  Dense total: {dense_bytes / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
