"""Golden test for the R1-tiny calculation fixture (M1a conformance).

Locks the *predicted* numbers so any change to the frozen R1-tiny config (or an
accidental edit to the accounting) trips CI rather than silently re-canonizing.
Pure stdlib; run with ``python3 tests/test_r1_tiny_fixture.py`` (exit 0 = pass).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fixtures.r1_tiny import r1_tiny_report  # noqa: E402

GOLDEN = {
    "params.total": 6_327_552,
    "params.non_embedding": 4_230_400,
    "params.embedding": 2_097_152,
    "params.per_layer": 705_024,
    "flops.dense_train_6N": 37_945_344,   # 6 · N_matmul, per token
    "flops.attn_train": 9_437_184,        # context-dependent term, per token
    "flops.total_train": 47_382_528,
    "mem.train_state_bytes": 101_240_832, # 16 B/param
}


def _actual():
    r = r1_tiny_report()
    p, f, m = r["params"], r["flops_per_token"], r["memory"]
    return {
        "params.total": p["total"],
        "params.non_embedding": p["non_embedding"],
        "params.embedding": p["embedding"],
        "params.per_layer": p["per_layer"],
        "flops.dense_train_6N": f["dense_train_6N"],
        "flops.attn_train": f["attn_train"],
        "flops.total_train": f["total_train"],
        "mem.train_state_bytes": m["train_state"],
    }


def main() -> int:
    actual, failures = _actual(), []
    for key, want in GOLDEN.items():
        got = actual[key]
        status = "ok" if got == want else "FAIL"
        if got != want:
            failures.append((key, want, got))
        print(f"[{status}] {key}: {got:,}")
    if failures:
        print(f"\n{len(failures)} golden mismatch(es):")
        for key, want, got in failures:
            print(f"  {key}: expected {want:,}, got {got:,}")
        return 1
    print("\nAll golden values match. R1-tiny spec conformance OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
