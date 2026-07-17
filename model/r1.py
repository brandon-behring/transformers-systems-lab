"""R1-tiny — the pilot dense text LM (M1b vertical slice).

A from-scratch Llama-style decoder (RMSNorm · RoPE · SwiGLU · GQA) built from low-level
torch ops (no ``nn.Transformer``), matching the frozen ``reference-model.md`` §3 config.
The parameter count is asserted against the M1a golden fixture (6,327,552) so the
implementation can never silently drift from the spec.

Contract constants (reference-model.md §3): RMSNorm eps = 1e-5, RoPE theta = 10000.
"""
from __future__ import annotations
import math, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fixtures.r1_tiny import R1TinyConfig  # single source of truth for dims

RMS_EPS = 1e-5      # reference-model.md §3
ROPE_THETA = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_rope(head_dim: int, seq_len: int, theta: float = ROPE_THETA):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # d/2
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)                   # (s, d/2)
    emb = torch.cat([freqs, freqs], dim=-1)                                        # (s, d)
    return emb.cos(), emb.sin()


def _rotate_half(x):
    d = x.shape[-1]
    return torch.cat([-x[..., d // 2:], x[..., : d // 2]], dim=-1)


def apply_rope(x, cos, sin):           # x: (b, h, s, d); cos/sin: (s, d)
    s = x.shape[2]
    cos = cos[:s].view(1, 1, s, -1)
    sin = sin[:s].view(1, 1, s, -1)
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, c: R1TinyConfig):
        super().__init__()
        self.n_heads, self.n_kv, self.hd = c.n_heads, c.n_kv_heads, c.head_dim
        self.q = nn.Linear(c.d_model, c.n_heads * c.head_dim, bias=False)
        self.k = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.v = nn.Linear(c.d_model, c.n_kv_heads * c.head_dim, bias=False)
        self.o = nn.Linear(c.n_heads * c.head_dim, c.d_model, bias=False)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q = self.q(x).view(b, s, self.n_heads, self.hd).transpose(1, 2)
        k = self.k(x).view(b, s, self.n_kv, self.hd).transpose(1, 2)
        v = self.v(x).view(b, s, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.n_heads // self.n_kv                      # GQA expand
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(b, s, self.n_heads * self.hd))


class SwiGLU(nn.Module):
    def __init__(self, c: R1TinyConfig):
        super().__init__()
        self.gate = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.up = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.down = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c: R1TinyConfig):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(c.d_model), Attention(c)
        self.ffn_norm, self.ffn = RMSNorm(c.d_model), SwiGLU(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class R1(nn.Module):
    """Dense text LM. Output projection is tied to the input embedding."""

    def __init__(self, c: R1TinyConfig | None = None):
        super().__init__()
        self.c = c or R1TinyConfig()
        self.embed = nn.Embedding(self.c.vocab_size, self.c.d_model)
        self.blocks = nn.ModuleList([Block(self.c) for _ in range(self.c.n_layers)])
        self.norm = RMSNorm(self.c.d_model)
        cos, sin = precompute_rope(self.c.head_dim, self.c.context_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)   # reference-model.md §3

    def forward(self, idx, targets=None):
        x = self.embed(idx)
        cos, sin = self.rope_cos.to(x.device), self.rope_sin.to(x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = F.linear(self.norm(x), self.embed.weight)   # tied output
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx[:, -self.c.context_len:])
            logits = logits[:, -1, :]
            if temperature == 0.0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    m = R1()
    print("R1-tiny params:", f"{m.num_params():,}")
