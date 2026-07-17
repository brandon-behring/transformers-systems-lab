# transformers-systems-lab

Runnable, **version-pinned** labs for the *Transformer Systems Engineering* guide — the
code half of the book. The book carries prose + inline listings; this repo holds the
executable, version-quarantined implementations so fast-churning framework code can't
rot the timeless chapters (plan: invariant chapters vs versioned labs).

## Tier-0 (this machine)
Single **RTX 2070 SUPER, 8 GB, Turing sm_75** (~4 GB usable after the desktop) · torch
2.13.0+cu130 · triton 3.7.1 · Python 3.12. Real multi-GPU / FA2-3 / 7B-VLM numbers are
**Tier-1 (the scale-tier machine), deferred**. See the frozen `reference-model.md` in the book repo.

## Layout
- `fixtures/` — calculation fixtures (pure-Python resource accounting; the *predicted* side of predict-then-measure).
- `model/` — from-scratch reference models (`r1.py` = R1-tiny dense LM, matches `reference-model.md` §3).
- `tests/` — golden + conformance tests.

## Run
```bash
uv venv --python 3.12 .venv
uv pip install torch==2.13.0 triton==3.7.1 numpy pytest
.venv/bin/python tests/test_r1_tiny_fixture.py   # M1a golden — pure stdlib
.venv/bin/python tests/test_r1_module.py         # M1b vertical slice — GPU
```

## Pilot milestones
- **M1a** ✓ R1-tiny calc fixture + golden (params 6,327,552 / 47.4M FLOPs·tok⁻¹).
- **M1b** ✓ R1-tiny torch module (forward/train/generate/checkpoint) — verified on the 2070.
- **M2** Triton FlashAttention (correctness-parity on sm_75) + one compiled/fused op.
- **M3** gloo multi-rank FSDP2 + DCP — Gate A (bitwise same-topology replay) / Gate B (bounded-divergence reshard).
- **M4** minimal paged-KV server (prefill/decode split).
