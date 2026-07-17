"""M1a-real (the G0b gate): train R1-tiny on the pinned R0 tokens; emit a validated report.

The end-to-end slice that unlocks scale: R0 corpus/tokenizer/batches → R1-tiny training →
a benchmark report carrying the frozen §7 denominators + measured numbers → the numbers-contract
validator passes. Retires the synthetic tokens of M1b (real tokenized FineWeb-Edu now).

Run: ``.venv/bin/python m1a_real.py``  (exit 0 = G0b PASS).
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model.r1 import R1
from fixtures.r1_tiny import R1TinyConfig, r1_tiny_report
from bench.contract import validate, load_yaml, numbers_hash

HERE = os.path.dirname(__file__)


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = R1TinyConfig()

    batches = np.load(os.path.join(HERE, "r0/data/train_batches.npy"))  # (256,512) uint16
    data = torch.from_numpy(batches.astype(np.int64))
    assert int(data.max()) < c.vocab_size, "R0 token id exceeds R1-tiny vocab"

    m = R1(c).to(dev)
    assert m.num_params() == 6_327_552, m.num_params()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    warmup, steps, bs = 10, 200, 8
    losses, t0, toks = [], None, 0
    for step in range(steps):
        batch = data[torch.randint(0, data.shape[0], (bs,))].to(dev)
        x, y = batch[:, :-1], batch[:, 1:]
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        losses.append(loss.item())
        if step == warmup:
            if dev == "cuda": torch.cuda.synchronize()
            t0 = time.time()
        elif step > warmup:
            toks += x.numel()
    if dev == "cuda": torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() if dev == "cuda" else 0
    tok_per_s = toks / elapsed if elapsed else 0.0

    # reference greedy generation (fixed prompt) for the golden record
    ref_prompt = data[0, :8].unsqueeze(0).to(dev)
    ref_gen = m.generate(ref_prompt, max_new_tokens=16)[0].tolist()

    report = {
        "scenario_id": "r1-tiny-train",
        "scenario_numbers_hash": numbers_hash(r1_tiny_report()),
        "measured": {"params_total": m.num_params()},
        "denominators": {
            "hw_sw_manifest": f"{torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}; "
                              f"torch {torch.__version__}; cuda {torch.version.cuda}; triton 3.7.1",
            "workload_shape": f"R1-tiny fwd+bwd, bs={bs}, ctx={c.context_len}, bf16=off(fp32 pilot)",
            "warmup_steps": warmup,
            "measurement_window": f"{steps - warmup} steps",
            "utilization_metric": f"{tok_per_s:.0f} tok/s throughput (MFU n/a — correctness pilot)",
            "price_timestamp": "n/a (Tier-0 local, owned)",
            "allocation_method": "local-owned single-GPU",
            "uncertainty": "single run, no CI",
            "measured_on": "Tier-0 RTX 2070 SUPER (sm_75)",
        },
        "informational": {
            "peak_mem_bytes": int(peak), "tok_per_s": round(tok_per_s, 1),
            "loss_first": round(losses[0], 3), "loss_last": round(sum(losses[-10:]) / 10, 3),
            "ref_generation_ids": ref_gen,
        },
    }

    scen = load_yaml(os.path.join(HERE, "scenarios/r1-tiny-train.yml"))
    problems = validate(scen, report)
    os.makedirs(os.path.join(HERE, "bench/reports"), exist_ok=True)
    with open(os.path.join(HERE, "bench/reports/r1-tiny-train.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"trained on real R0 tokens: loss {losses[0]:.3f} -> {report['informational']['loss_last']:.3f} "
          f"| {tok_per_s:.0f} tok/s | peak {peak/1e6:.0f} MB")
    print("NUMBERS-CONTRACT VALIDATOR:", "PASS  → G0b unlocked" if not problems else f"FAIL {problems}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
