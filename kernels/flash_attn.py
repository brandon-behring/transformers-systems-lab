"""M2 — a FlashAttention *forward* kernel in Triton (online-softmax, tiled, causal).

Tier-0 note (reference-model.md §7): on the RTX 2070 SUPER (Turing sm_75) this is
**correctness-only** — Triton can't emit tensor-core code on sm_75, so it validates the
IO-aware online-softmax algorithm (parity vs a reference), not FA2/3 speed. Real perf/FA2-3/FP8
are Tier-1. The kernel is the versioned lab; the *invariant* is the online-softmax identity
(a tiled softmax equals the full softmax) — that's what the parity test pins.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _fa_fwd(
    Q, K, V, O, sm_scale,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
):
    m_block = tl.program_id(0)
    bh = tl.program_id(1)  # flattened (batch*head); base pointers offset by bh * (N*D)
    q_base = Q + bh * stride_qh
    k_base = K + bh * stride_kh
    v_base = V + bh * stride_vh
    o_base = O + bh * stride_oh

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    q = tl.load(q_base + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                mask=offs_m[:, None] < N, other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    n_end = (m_block + 1) * BLOCK_M if CAUSAL else N
    for start_n in range(0, n_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                    mask=offs_n[:, None] < N, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
        qk = tl.where(offs_n[None, :] < N, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                    mask=offs_n[:, None] < N, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(o_base + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od, acc,
             mask=offs_m[:, None] < N)


def flash_attention(q, k, v, causal=True, block_m=64, block_n=64):
    """q,k,v: (B, H, N, D), fp32/fp16 on CUDA. Returns (B, H, N, D)."""
    B, H, N, D = q.shape
    assert q.is_cuda and D in (16, 32, 64, 128), "Tier-0 pilot: power-of-two head_dim"
    q, k, v = (x.contiguous() for x in (q, k, v))
    o = torch.empty_like(q)
    sm_scale = 1.0 / (D ** 0.5)
    qf, kf, vf, of = (x.reshape(B * H, N, D) for x in (q, k, v, o))
    grid = (triton.cdiv(N, block_m), B * H, 1)
    _fa_fwd[grid](
        qf, kf, vf, of, sm_scale,
        qf.stride(0), qf.stride(0), qf.stride(1), qf.stride(2),
        kf.stride(0), kf.stride(0), kf.stride(1), kf.stride(2),
        vf.stride(0), vf.stride(0), vf.stride(1), vf.stride(2),
        of.stride(0), of.stride(0), of.stride(1), of.stride(2),
        N, BLOCK_M=block_m, BLOCK_N=block_n, D=D, CAUSAL=causal,
    )
    return of.reshape(B, H, N, D)
