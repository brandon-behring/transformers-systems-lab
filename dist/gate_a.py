"""M3 Gate A — bitwise same-topology replay (the exact-resume invariant).

An interrupted-and-resumed run must reproduce an uninterrupted run BIT-FOR-BIT, which is only
true if the checkpoint captures the FULL training state — optimizer moments, every RNG stream,
and the dataloader cursor — and round-trips through real serialization. To make RNG restoration
*load-bearing* (Codex #7), each step samples its batch via the torch RNG: with RNG restored the
resume is bit-identical; WITHOUT it the run provably diverges (also asserted). CPU/deterministic
so bit-identity is unambiguous; GPU kernel non-determinism is a separate Tier-1 concern.

Run: ``.venv/bin/python dist/gate_a.py`` (exit 0 = Gate A holds).
"""
import os
import random
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.r1 import R1
from fixtures.r1_tiny import R1TinyConfig

DATA = None


def data():
    global DATA
    if DATA is None:
        DATA = torch.randint(0, R1TinyConfig().vocab_size, (16, 129),
                             generator=torch.Generator().manual_seed(123))
    return DATA


def make():
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    m = R1(R1TinyConfig())
    return m, torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95))


def snapshot(m, opt, cursor):
    return {"model": m.state_dict(), "opt": opt.state_dict(), "cursor": cursor,
            "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
            "py_rng": random.getstate()}


def restore(m, opt, snap, rng=True):
    m.load_state_dict(snap["model"]); opt.load_state_dict(snap["opt"])
    if rng:
        torch.set_rng_state(snap["torch_rng"]); np.random.set_state(snap["np_rng"])
        random.setstate(snap["py_rng"])
    return snap["cursor"]


def step(m, opt):
    d = data()
    i = torch.randint(0, d.shape[0], (1,)).item()          # torch RNG — load-bearing
    _ = np.random.rand(); _ = random.random()               # exercise np + python RNG too
    b = d[i].unsqueeze(0)
    _, loss = m(b[:, :-1], b[:, 1:])
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def run(steps, resume=None, rng=True):
    m, opt = make()
    if resume is not None:
        restore(m, opt, resume, rng=rng)
    losses = [step(m, opt) for _ in range(steps)]
    return m, losses


def flat(m):
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()])


def main() -> int:
    torch.use_deterministic_algorithms(True)

    m1, loss1 = run(20)                                     # uninterrupted

    # interrupt after 10 steps; snapshot round-trips through real serialization
    m_a, opt_a = make()
    for _ in range(10):
        step(m_a, opt_a)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        torch.save(snapshot(m_a, opt_a, 10), path)
        snap = torch.load(path, weights_only=False)         # trusted own ckpt (has RNG/numpy state)

    m2, loss2 = run(10, resume=snap, rng=True)              # resume WITH rng
    m3, _ = run(10, resume=snap, rng=False)                # resume WITHOUT rng (should diverge)

    bit_identical = torch.equal(flat(m1), flat(m2))
    tail_ok = loss1[10:] == loss2
    rng_matters = not torch.equal(flat(m1), flat(m3))
    print(f"resume+RNG bit-identical to uninterrupted: {bit_identical} (max|Δ|={ (flat(m1)-flat(m2)).abs().max():.2e})")
    print(f"resumed loss tail identical: {tail_ok}")
    print(f"RNG restoration is load-bearing (no-RNG resume diverges): {rng_matters}")
    ok = bit_identical and tail_ok and rng_matters
    print("\nM3 Gate A (bitwise same-topology replay):", "HOLDS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
