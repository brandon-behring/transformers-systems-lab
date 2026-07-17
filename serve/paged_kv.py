"""M4 — minimal paged-KV serving (prefill/decode split) for R1-tiny.

A PagedAttention-style KV cache: K/V live in fixed-size physical blocks, a per-sequence block
table maps logical positions to (possibly non-contiguous) physical blocks. Prefill fills the
cache for the whole prompt in one pass; decode adds one token at a time, attending the cache.
The invariant pinned by the test: cached incremental attention == full-recompute generation
(bit-identical greedy tokens). The prefill/decode timing split shows decode's memory-bound
per-token cost vs prefill's batched compute (reference-model §7).

Run: ``.venv/bin/python serve/paged_kv.py`` (exit 0 = parity holds).
"""
from __future__ import annotations
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.r1 import R1, _rotate_half
from fixtures.r1_tiny import R1TinyConfig


class PagedKVCache:
    def __init__(self, cfg, block_size=16, num_blocks=1024, device="cpu", dtype=torch.float32):
        self.bs = block_size
        L, Hkv, D = cfg.n_layers, cfg.n_kv_heads, cfg.head_dim
        self.kf = torch.zeros(L, num_blocks * block_size, Hkv, D, device=device, dtype=dtype)
        self.vf = torch.zeros_like(self.kf)
        self.free = list(range(num_blocks))
        self.table: list[int] = []      # physical block ids held by this sequence
        self.device = device

    def reserve(self, upto: int):
        need = (upto + self.bs - 1) // self.bs
        while len(self.table) < need:
            self.table.append(self.free.pop(0))     # allocate a fresh physical block

    def _phys(self, positions):
        return torch.tensor([self.table[p // self.bs] * self.bs + (p % self.bs) for p in positions],
                            device=self.device)

    def write(self, layer, start, k, v):            # k,v: (T, Hkv, D)
        idx = self._phys(range(start, start + k.shape[0]))
        self.kf[layer].index_copy_(0, idx, k)
        self.vf[layer].index_copy_(0, idx, v)

    def read(self, layer, length):                  # -> (length, Hkv, D)
        idx = self._phys(range(length))
        return self.kf[layer].index_select(0, idx), self.vf[layer].index_select(0, idx)

    def blocks_used(self):
        return len(self.table)


def _rope_at(x, cos, sin, pos):                     # x: (T, H, D); pos: (T,)
    c = cos[pos].unsqueeze(1); s = sin[pos].unsqueeze(1)
    return x * c + _rotate_half(x) * s


@torch.no_grad()
def _block_step(model, blk, x, cache, layer, start):
    c = model.c
    h = blk.attn_norm(x)
    T = h.shape[0]
    q = blk.attn.q(h).view(T, c.n_heads, c.head_dim)
    k = blk.attn.k(h).view(T, c.n_kv_heads, c.head_dim)
    v = blk.attn.v(h).view(T, c.n_kv_heads, c.head_dim)
    cos, sin = model.rope_cos.to(x.device), model.rope_sin.to(x.device)
    pos = torch.arange(start, start + T, device=x.device)
    q, k = _rope_at(q, cos, sin, pos), _rope_at(k, cos, sin, pos)
    cache.write(layer, start, k, v)
    Kall, Vall = cache.read(layer, start + T)       # (start+T, Hkv, D)
    rep = c.n_heads // c.n_kv_heads
    Kall, Vall = Kall.repeat_interleave(rep, 1), Vall.repeat_interleave(rep, 1)
    qh = q.transpose(0, 1).unsqueeze(0)             # (1, H, T, D)
    kh, vh = Kall.transpose(0, 1).unsqueeze(0), Vall.transpose(0, 1).unsqueeze(0)
    out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=(T > 1))
    out = out.squeeze(0).transpose(0, 1).reshape(T, c.n_heads * c.head_dim)
    x = x + blk.attn.o(out)
    return x + blk.ffn(blk.ffn_norm(x))


@torch.no_grad()
def serve(model, prompt_ids, max_new=32, block_size=16):
    dev = next(model.parameters()).device
    cache = PagedKVCache(model.c, block_size=block_size, device=dev)
    ids = prompt_ids.to(dev)
    cuda = dev.type == "cuda"

    if cuda: torch.cuda.synchronize()
    t0 = time.time()
    cache.reserve(len(ids))
    x = model.embed(ids)
    for i, blk in enumerate(model.blocks):
        x = _block_step(model, blk, x, cache, i, 0)
    logits = F.linear(model.norm(x[-1:]), model.embed.weight)
    nxt = logits.argmax(-1)
    if cuda: torch.cuda.synchronize()
    prefill_t = time.time() - t0

    out = [int(nxt)]
    t0 = time.time()
    for step in range(max_new - 1):
        pos = len(ids) + step
        cache.reserve(pos + 1)
        x = model.embed(nxt)
        for i, blk in enumerate(model.blocks):
            x = _block_step(model, blk, x, cache, i, pos)
        nxt = F.linear(model.norm(x), model.embed.weight).argmax(-1)
        out.append(int(nxt))
    if cuda: torch.cuda.synchronize()
    decode_t = time.time() - t0
    return out, prefill_t, decode_t, cache


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = R1(R1TinyConfig()).to(dev).eval()
    prompt = torch.randint(0, m.c.vocab_size, (24,))

    served, pref_t, dec_t, cache = serve(m, prompt, max_new=32)

    # parity: greedy full-recompute generation must match the paged-KV serve
    ref = m.generate(prompt.unsqueeze(0).to(dev), max_new_tokens=32)[0, 24:].tolist()
    parity = served[: len(ref)] == ref
    n_dec = 31
    print(f"prefill {len(prompt)} tok in {pref_t*1e3:.1f} ms | decode {n_dec} tok in {dec_t*1e3:.1f} ms "
          f"({dec_t/n_dec*1e3:.2f} ms/tok) | paged blocks used: {cache.blocks_used()}")
    print(f"parity vs full-recompute generate: {parity}")
    print("\nM4 paged-KV serving:", "PARITY OK" if parity else "FAILED")
    return 0 if parity else 1


if __name__ == "__main__":
    raise SystemExit(main())
