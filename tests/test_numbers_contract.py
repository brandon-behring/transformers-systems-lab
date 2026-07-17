"""Negative-tested validator for the numbers contract (Phase −1 governance gate).

Confirms the validator PASSES a well-formed report and FAILS on each way a report can lie:
missing frozen denominators, an out-of-tolerance measurement, and a drifted scenario hash.
Run: ``.venv/bin/python tests/test_numbers_contract.py`` (exit 0 = pass).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.contract import validate, load_yaml, numbers_hash, REQUIRED_DENOMINATORS
from fixtures.r1_tiny import r1_tiny_report

SCEN = os.path.join(os.path.dirname(__file__), "..", "scenarios", "r1-tiny-train.yml")


def _good_report(live_hash: str) -> dict:
    return {
        "scenario_numbers_hash": live_hash,
        "denominators": {k: "recorded" for k in REQUIRED_DENOMINATORS},
        "measured": {"params_total": 6_327_552, "train_state_bytes": 101_240_832},
    }


def main() -> int:
    scen = load_yaml(SCEN)
    live = numbers_hash(r1_tiny_report())

    assert validate(scen, _good_report(live)) == [], "valid report should pass"
    print("[ok] well-formed report passes")

    r = _good_report(live); del r["denominators"]["measured_on"]
    assert any("measured_on" in p for p in validate(scen, r)), "missing denominator must fail"
    print("[ok] missing frozen denominator fails")

    r = _good_report(live); r["measured"]["train_state_bytes"] = 101_240_832 * 2
    assert any("train_state_bytes" in p for p in validate(scen, r)), "bad measurement must fail"
    print("[ok] out-of-tolerance measurement fails")

    stale = dict(scen); stale["numbers_hash"] = "deadbeef" * 8
    assert any("drift" in p for p in validate(stale, _good_report(live))), "stale hash must fail"
    print("[ok] scenario-hash drift fails")

    print("\nNumbers contract: ALL CHECKS PASS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
