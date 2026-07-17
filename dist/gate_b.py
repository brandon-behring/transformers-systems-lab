"""M3 Gate B — cross-world-size reshard via Distributed Checkpoint (DCP).

Distinct from Gate A: a DCP checkpoint written under one world size must LOAD under a different
world size with (i) exact state integrity BEFORE any new step and (ii) bounded divergence after.
Uses real gloo multi-rank + torch.distributed.checkpoint. On Tier-0 the model is replicated
across gloo/CPU ranks (single GPU); FSDP-sharded reshard is the Tier-1 extension. The invariant
this pins: the checkpoint is world-size-agnostic.

Driver (dist/run_gate_b.sh): torchrun WS=2 save  →  torchrun WS=1 load+verify.
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.r1 import R1
from fixtures.r1_tiny import R1TinyConfig

CKPT = os.environ.get("GATE_B_CKPT", "/home/brandon_behring/.claude/jobs/0de54cce/tmp/gate_b_ckpt")
REF = CKPT + "_ref.pt"


def build():
    torch.manual_seed(0)
    return R1(R1TinyConfig())  # CPU, identical on every rank (replicated)


def fixed_data():
    g = torch.Generator().manual_seed(123)
    return torch.randint(0, R1TinyConfig().vocab_size, (16, 129), generator=g)


def train(m, steps, data, start=0):
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95))
    for i in range(steps):
        b = data[(start + i) % data.shape[0]].unsqueeze(0)
        _, loss = m(b[:, :-1], b[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def flat(m):
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()])


def main(mode: str) -> int:
    dist.init_process_group("gloo")
    rank, ws = dist.get_rank(), dist.get_world_size()
    data = fixed_data()

    if mode == "save":
        m = train(build(), 10, data)
        dcp.save({"model": m.state_dict()}, checkpoint_id=CKPT)
        if rank == 0:
            torch.save(m.state_dict(), REF)   # plain reference for the integrity check
            print(f"[save WS={ws}] DCP checkpoint written ({sum(p.numel() for p in m.parameters()):,} params)")
    else:  # load at a (possibly different) world size
        m = build()
        sd = m.state_dict()
        dcp.load({"model": sd}, checkpoint_id=CKPT)
        m.load_state_dict(sd)
        if rank == 0:
            ref_sd = torch.load(REF)
            integrity = all(torch.equal(sd[k], ref_sd[k]) for k in sd)
            # bounded divergence: K steps from the reshard-loaded model vs a ref-loaded model
            ref_m = build(); ref_m.load_state_dict(ref_sd)
            train(m, 5, data, start=10); train(ref_m, 5, data, start=10)
            div = (flat(m) - flat(ref_m)).abs().max().item()
            ok = integrity and div <= 1e-6
            print(f"[load WS={ws}] state integrity: {integrity} | post-reshard divergence: {div:.2e} "
                  f"(budget 1e-6) -> Gate B {'HOLDS' if ok else 'FAILED'}")
            with open(CKPT + "_result", "w") as fh:
                fh.write("PASS" if ok else "FAIL")
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "save"))
