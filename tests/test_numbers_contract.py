"""Negative-tested validator for the numbers contract (Phase −1 governance gate).

Confirms the validator PASSES a well-formed report and FAILS on each way a report can lie:
missing denominators, out-of-tolerance measurement, drifted hash, wrong scenario id, and —
crucially — an empty or partial `measured` set (Codex #1). Exit 0 = pass.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.contract import validate, load_yaml, numbers_hash, REQUIRED_DENOMINATORS
from fixtures.r1_tiny import r1_tiny_report

SCEN = os.path.join(os.path.dirname(__file__), "..", "scenarios", "r1-tiny-train.yml")


def _good(live: str) -> dict:
    return {
        "scenario_id": "r1-tiny-train",
        "scenario_numbers_hash": live,
        "denominators": {k: "recorded" for k in REQUIRED_DENOMINATORS},
        "measured": {"params_total": 6_327_552, "train_state_bytes": 101_240_832},
    }


def main() -> int:
    scen = load_yaml(SCEN)
    live = numbers_hash(r1_tiny_report())

    assert validate(scen, _good(live)) == [], f"valid report should pass: {validate(scen, _good(live))}"
    print("[ok] well-formed report passes")

    checks = [
        ("empty measured", lambda r: r.update(measured={})),
        ("partial measured (missing train_state_bytes)", lambda r: r.update(measured={"params_total": 6_327_552})),
        ("extra un-contracted metric", lambda r: r["measured"].update(bogus=1)),
        ("missing denominator", lambda r: r["denominators"].pop("measured_on")),
        ("out-of-tolerance measurement", lambda r: r["measured"].update(train_state_bytes=101_240_832 * 2)),
        ("wrong scenario_id", lambda r: r.update(scenario_id="nope")),
    ]
    for label, mutate in checks:
        r = _good(live); mutate(r)
        problems = validate(scen, r)
        assert problems, f"{label!r} must fail"
        print(f"[ok] {label} fails")

    # drift: a stale pinned hash
    stale = dict(scen); stale["numbers_hash"] = "deadbeef" * 8
    assert any("drift" in p for p in validate(stale, _good(live))), "stale hash must fail"
    print("[ok] scenario-hash drift fails")

    print("\nNumbers contract: ALL CHECKS PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
