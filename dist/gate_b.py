"""M3 Gate B — cross-world-size Distributed-Checkpoint load (model + optimizer).

A DCP checkpoint written under one world size must LOAD under a different world size with exact
state integrity for BOTH model weights and optimizer moments (Codex #5: weights alone is not
enough). The result is broadcast so every rank exits nonzero on failure (Codex #3).

Honest scope (Codex #4): on Tier-0 the model is **replicated** across gloo/CPU ranks (single
GPU), so this pins DCP's world-size-agnostic *load* of replicated state + optimizer. Genuinely
*sharded* FSDP2 resharding (partition metadata redistribution) is the Tier-1 extension.

Driver: dist/run_gate_b.sh  (torchrun WS=2 save → torchrun WS=1 load, both exit-checked).
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.r1 import R1
from fixtures.r1_tiny import R1TinyConfig

CKPT = os.environ.get("GATE_B_CKPT", "/home/brandon_behring/.claude/jobs/0de54cce/tmp/gate_b_ckpt")
REF = CKPT + "_ref.pt"


def build():
    torch.manual_seed(0)
    m = R1(R1TinyConfig())
    return m, torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95))


def train(m, opt, steps):
    g = torch.Generator().manual_seed(123)
    d = torch.randint(0, R1TinyConfig().vocab_size, (16, 129), generator=g)
    for i in range(steps):
        b = d[i % d.shape[0]].unsqueeze(0)
        _, loss = m(b[:, :-1], b[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()


def opt_moments(osd):
    """Extract Adam m/v tensors from an optimizer state_dict for comparison."""
    out = {}
    for pid, st in osd.get("state", {}).items():
        for k in ("exp_avg", "exp_avg_sq"):
            if k in st:
                out[f"{pid}.{k}"] = st[k]
    return out


def main(mode: str) -> int:
    dist.init_process_group("gloo")
    rank, ws = dist.get_rank(), dist.get_world_size()
    m, opt = build()

    if mode == "save":
        train(m, opt, 10)
        msd, osd = get_state_dict(m, opt)
        dcp.save({"model": msd, "optim": osd}, checkpoint_id=CKPT)
        if rank == 0:
            torch.save({"model": m.state_dict(), "optim": opt.state_dict()}, REF)
            print(f"[save WS={ws}] DCP checkpoint (model+optim) written")
        ok = True
    else:  # load — possibly at a different world size
        msd, osd = get_state_dict(m, opt)
        dcp.load({"model": msd, "optim": osd}, checkpoint_id=CKPT)
        set_state_dict(m, opt, model_state_dict=msd, optim_state_dict=osd)
        ok = True
        if rank == 0:
            ref = torch.load(REF)
            model_ok = all(torch.equal(m.state_dict()[k], ref["model"][k]) for k in ref["model"])
            got, want = opt_moments(opt.state_dict()), opt_moments(ref["optim"])
            optim_ok = got.keys() == want.keys() and all(torch.equal(got[k], want[k]) for k in got) and len(got) > 0
            ok = model_ok and optim_ok
            print(f"[load WS={ws}] model integrity: {model_ok} | optimizer-moment integrity: {optim_ok} "
                  f"({len(got)} moment tensors) -> Gate B {'HOLDS' if ok else 'FAILED'}")

    flag = torch.tensor([1 if ok else 0])
    dist.broadcast(flag, src=0)          # every rank learns the rank-0 verdict
    dist.barrier(); dist.destroy_process_group()
    return 0 if bool(flag.item()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "save"))
