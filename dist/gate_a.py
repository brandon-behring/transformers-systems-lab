"""M3 Gate A — bitwise same-topology replay (the exact-resume invariant).

An interrupted-and-resumed run must reproduce an uninterrupted run BIT-FOR-BIT, which is only
true if the checkpoint captures the FULL training state — not just weights, but optimizer
moments, every RNG stream, and the dataloader cursor. Run on CPU (deterministic) so bit-identity
is unambiguous; GPU kernel non-determinism is a separate Tier-1 concern (reference-model §6).

Run: ``.venv/bin/python dist/gate_a.py`` (exit 0 = Gate A holds).
"""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.r1 import R1
from fixtures.r1_tiny import R1TinyConfig


def snapshot(model, opt, cursor):
    """Full training state (the thing that makes resume exact)."""
    return {
        "model": {k: v.clone() for k, v in model.state_dict().items()},
        "opt": opt.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "py_rng": random.getstate(),
        "np_rng": np.random.get_state(),
        "cursor": cursor,
    }


def restore(model, opt, snap):
    model.load_state_dict(snap["model"])
    opt.load_state_dict(snap["opt"])
    torch.set_rng_state(snap["torch_rng"])
    random.setstate(snap["py_rng"])
    np.random.set_state(snap["np_rng"])
    return snap["cursor"]


def make_model():
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    c = R1TinyConfig()
    m = R1(c)  # CPU
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    return m, opt


def step(model, opt, cursor, data):
    """One deterministic training step; the 'dataloader cursor' walks fixed data."""
    b = data[cursor % data.shape[0]].unsqueeze(0)
    x, y = b[:, :-1], b[:, 1:]
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()
    return cursor + 1, loss.item()


def run(steps, data, resume=None):
    m, opt = make_model()
    cursor = restore(m, opt, resume) if resume else 0
    losses = []
    for _ in range(steps):
        cursor, l = step(m, opt, cursor, data)
        losses.append(l)
    return m, opt, cursor, losses


def flat_params(m):
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()])


def main() -> int:
    torch.use_deterministic_algorithms(True)
    g = torch.Generator().manual_seed(123)
    data = torch.randint(0, R1TinyConfig().vocab_size, (16, 129), generator=g)  # tiny fixed corpus

    # (a) uninterrupted 20 steps
    m1, _, _, loss1 = run(20, data)

    # (b) interrupt at 10, snapshot full state, fresh model, restore, continue 10
    m_a, opt_a, cur_a, _ = run(10, data)
    snap = snapshot(m_a, opt_a, cur_a)
    m2, _, _, loss2 = run(10, data, resume=snap)

    p1, p2 = flat_params(m1), flat_params(m2)
    bit_identical = torch.equal(p1, p2)
    tail_identical = loss1[10:] == loss2
    print(f"final params bit-identical: {bit_identical}  (max|Δ|={(p1 - p2).abs().max().item():.2e})")
    print(f"resumed loss trace identical to uninterrupted tail: {tail_identical}")
    ok = bit_identical and tail_identical
    print("\nM3 Gate A (bitwise same-topology replay):", "HOLDS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
