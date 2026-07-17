"""Calculation fixture for the pilot reference model ``R1-tiny``.

This is the *predicted* side of the pilot's "predict-then-measure" loop and the
single source of truth for R1-tiny's parameter / FLOP / memory numbers. Chapters
and golden tests import ``r1_tiny_report()`` so a producer chapter and a consumer
chapter can never disagree on a number (plan §8: scenario records + calc fixtures).

Pure stdlib — no torch — so it runs anywhere and states every counting convention
explicitly (Codex flagged that unlabelled units are how composing-numbers chapters
silently diverge).

Config is frozen in ``reference-model.md`` §3 (R1-tiny).
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class R1TinyConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2          # GQA
    head_dim: int = 32
    d_ff: int = 704              # SwiGLU inner (≈ 8/3 · d_model, rounded to 11·64)
    vocab_size: int = 8192
    context_len: int = 512
    tie_embeddings: bool = True
    # training scenario
    global_batch_seqs: int = 32
    # precision accounting (mixed-precision AdamW), bytes per parameter:
    #   bf16 weight (2) + bf16 grad (2) + fp32 master (4) + Adam m (4) + Adam v (4)
    bytes_per_param_train: int = 16   # the canonical "16 bytes/param" (bf16 grad); 18 if fp32 grad


def parameter_counts(c: R1TinyConfig) -> dict:
    """Exact parameter counts, broken down. No biases; RoPE and RMSNorm-as-scale only."""
    q = c.d_model * (c.n_heads * c.head_dim)
    k = c.d_model * (c.n_kv_heads * c.head_dim)
    v = c.d_model * (c.n_kv_heads * c.head_dim)
    o = (c.n_heads * c.head_dim) * c.d_model
    attn = q + k + v + o
    ffn = 3 * c.d_model * c.d_ff                      # gate + up + down (SwiGLU)
    norms = 2 * c.d_model                             # pre-attn + pre-ffn RMSNorm scales
    per_layer = attn + ffn + norms
    blocks = c.n_layers * per_layer
    embedding = c.vocab_size * c.d_model              # tied -> also the output projection
    final_norm = c.d_model
    total = blocks + embedding + final_norm
    non_embedding = blocks + final_norm               # Kaplan's N for 6ND
    # params that participate in matmuls (norms are elementwise, not matmuls):
    matmul_params = c.n_layers * (attn + ffn) + embedding  # + tied output projection
    return {
        "attn_per_layer": attn, "ffn_per_layer": ffn, "norm_per_layer": norms,
        "per_layer": per_layer, "blocks": blocks, "embedding": embedding,
        "total": total, "non_embedding": non_embedding, "matmul_params": matmul_params,
    }


def flops_per_token(c: R1TinyConfig, p: dict) -> dict:
    """Training FLOPs/token. Convention: 1 MAC = 2 FLOPs; backward ≈ 2× forward (→ ×3 total).

    Dense (matmul) term uses the 6·N_matmul rule. The attention score/context term is
    reported separately because it scales with context length (the O(L) per-token, O(L²)
    per-sequence term the dense rule omits).
    """
    dense_fwd = 2 * p["matmul_params"]                      # 2 FLOPs/param/token, forward
    dense_train = 3 * dense_fwd                             # fwd + bwd ≈ ×3  == 6·N_matmul
    d_head_total = c.n_heads * c.head_dim
    # QK^T then A·V, each ~2·ctx·d_head_total FLOPs/token/layer, forward:
    attn_fwd = c.n_layers * (2 * (2 * c.context_len * d_head_total))
    attn_train = 3 * attn_fwd
    return {
        "dense_fwd": dense_fwd, "dense_train_6N": dense_train,
        "attn_fwd": attn_fwd, "attn_train": attn_train,
        "total_train": dense_train + attn_train,
    }


def memory_bytes(c: R1TinyConfig, p: dict) -> dict:
    """Training-state memory (weights+grads+optimizer) and a rough activation estimate."""
    state = p["total"] * c.bytes_per_param_train
    # activations (very rough): ~ layers · batch · seq · d_model · (a few bf16 tensors)
    toks = c.global_batch_seqs * c.context_len
    act_bytes = c.n_layers * toks * c.d_model * 2 * 8       # ~8 bf16-sized live tensors/layer
    return {"train_state": state, "activations_est": act_bytes, "tokens_per_step": toks}


def r1_tiny_report(c: R1TinyConfig | None = None) -> dict:
    c = c or R1TinyConfig()
    p = parameter_counts(c)
    f = flops_per_token(c, p)
    m = memory_bytes(c, p)
    return {"config": c, "params": p, "flops_per_token": f, "memory": m}


def _fmt(n: float) -> str:
    for unit in ["", "K", "M", "G", "T"]:
        if abs(n) < 1000:
            return f"{n:,.2f}{unit}"
        n /= 1000
    return f"{n:,.2f}P"


if __name__ == "__main__":
    r = r1_tiny_report()
    c, p, f, m = r["config"], r["params"], r["flops_per_token"], r["memory"]
    print("=== R1-tiny resource accounting (predicted; hand-checkable) ===")
    print(f"params: total={_fmt(p['total'])}  non-embedding={_fmt(p['non_embedding'])}  "
          f"embedding(tied)={_fmt(p['embedding'])}")
    print(f"  per-layer: attn={_fmt(p['attn_per_layer'])} ffn={_fmt(p['ffn_per_layer'])} "
          f"norm={p['norm_per_layer']}  ×{c.n_layers} = {_fmt(p['blocks'])}")
    print(f"FLOPs/token (train): dense(6·N_matmul)={_fmt(f['dense_train_6N'])}  "
          f"attn(ctx={c.context_len})={_fmt(f['attn_train'])}  total={_fmt(f['total_train'])}")
    print(f"FLOPs/step (global batch {c.global_batch_seqs}×{c.context_len}="
          f"{_fmt(m['tokens_per_step'])} tok): {_fmt(f['total_train'] * m['tokens_per_step'])}")
    print(f"train-state memory ({c.bytes_per_param_train} B/param): {_fmt(m['train_state'])}B  "
          f"| activations(rough): {_fmt(m['activations_est'])}B  "
          f"| fits 8 GB (RTX 2070 SUPER): {(m['train_state']+m['activations_est']) < 8e9}")
