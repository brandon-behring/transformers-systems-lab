"""M2 parity test: the Triton FlashAttention forward equals reference attention.

Pins the online-softmax INVARIANT (a tiled/streamed softmax attention equals the full softmax
attention) by comparing the kernel to torch.nn.functional.scaled_dot_product_attention on the
Tier-0 GPU. Correctness-only on sm_75 (no tensor-core perf claim). Exit 0 = pass.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kernels.flash_attn import flash_attention


def main() -> int:
    if not torch.cuda.is_available():
        print("[skip] no CUDA (Tier-0 kernel test needs the GPU)")
        return 0
    torch.manual_seed(0)
    dev = "cuda"
    for (B, H, N, D, causal) in [(2, 4, 128, 32, True), (1, 8, 200, 64, True), (2, 2, 96, 32, False)]:
        q = torch.randn(B, H, N, D, device=dev)
        k = torch.randn(B, H, N, D, device=dev)
        v = torch.randn(B, H, N, D, device=dev)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = flash_attention(q, k, v, causal=causal)
        max_err = (out - ref).abs().max().item()
        ok = torch.allclose(out, ref, atol=2e-2, rtol=2e-2)
        print(f"[{'ok' if ok else 'FAIL'}] B{B} H{H} N{N} D{D} causal={causal}: max_err={max_err:.2e}")
        if not ok:
            return 1
    print("\nM2 Triton FlashAttention forward: PARITY OK (correctness-only on sm_75).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
