"""M1b conformance test for the R1-tiny torch module.

Ties the implementation to the M1a fixture: the module's parameter count MUST equal
the golden 6,327,552, so code and spec can't diverge. Then exercises the full slice —
forward, a tiny overfit (loss must drop), generation, and an exact checkpoint round-trip.
Run: ``.venv/bin/python tests/test_r1_module.py`` (exit 0 = pass).
"""
import os, sys, tempfile
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.r1 import R1
from fixtures.r1_tiny import R1TinyConfig, r1_tiny_report

GOLDEN_PARAMS = 6_327_552


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    c = R1TinyConfig()
    m = R1(c).to(dev)

    # 1. parameter conformance vs the M1a fixture (single source of truth)
    fixture_total = r1_tiny_report()["params"]["total"]
    assert m.num_params() == GOLDEN_PARAMS == fixture_total, (m.num_params(), fixture_total)
    print(f"[ok] params == {m.num_params():,}  (matches M1a golden + calc fixture)")

    # 2. forward shape
    b, s = 2, 64
    idx = torch.randint(0, c.vocab_size, (b, s), device=dev)
    logits, _ = m(idx)
    assert logits.shape == (b, s, c.vocab_size), logits.shape
    print(f"[ok] forward -> logits {tuple(logits.shape)} on {dev} ({torch.cuda.get_device_name(0) if dev=='cuda' else 'cpu'})")

    # 3. overfit a fixed tiny batch — loss must fall (the training loop actually learns)
    x = torch.randint(0, c.vocab_size, (4, 32), device=dev)
    y = torch.randint(0, c.vocab_size, (4, 32), device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    first = last = None
    for step in range(60):
        _, loss = m(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    print(f"[ok] overfit loss {first:.3f} -> {last:.3f}")
    assert last < first * 0.5, (first, last)

    # 4. generate (greedy)
    out = m.generate(idx[:, :4], max_new_tokens=8)
    assert out.shape == (b, 12), out.shape
    print(f"[ok] generate -> {tuple(out.shape)}")

    # 5. checkpoint save/load is exact (foundation for M3 Gate A)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "r1_tiny.pt")
        torch.save({"model": m.state_dict(), "config": vars(c), "schema_version": 1}, path)
        m2 = R1(c).to(dev)
        m2.load_state_dict(torch.load(path, map_location=dev)["model"])
        with torch.no_grad():
            l1, _ = m(idx)
            l2, _ = m2(idx)
        assert torch.allclose(l1, l2), "checkpoint reload changed logits"
    print("[ok] checkpoint save/load reproduces logits exactly")

    print("\nM1b R1-tiny vertical slice: ALL CHECKS PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
