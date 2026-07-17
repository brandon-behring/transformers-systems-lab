"""The numbers contract — scenario records, benchmark reports, and the consistency validator.

Load-bearing governance (plan Phase −1): "every number in the book traces to a runnable
fixture and a measured report." Three pieces:

- a **scenario** (`scenarios/<id>.yml`) fixes a named configuration + the PREDICTED numbers +
  a `numbers_hash` over them (the drift gate — any change to the frozen config or the
  accounting flips the hash and fails CI);
- a **benchmark report** (JSON) carries the frozen reference-model §7 denominators + the
  MEASURED values + `measured_on` provenance;
- `validate()` checks: all denominators present; the live fixture hash matches the scenario
  AND the report (no silent spec drift); and measured ≈ predicted within tolerance
  (predict-then-measure).

Pure stdlib + PyYAML; no torch (so it runs in CPU-only CI too).
"""
from __future__ import annotations
import hashlib
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fixtures.r1_tiny import r1_tiny_report  # noqa: E402

# frozen reference-model.md §7 denominators every benchmark report must carry
REQUIRED_DENOMINATORS = [
    "hw_sw_manifest", "workload_shape", "warmup_steps", "measurement_window",
    "utilization_metric", "price_timestamp", "allocation_method", "uncertainty", "measured_on",
]


def numbers_hash(fixture_report: dict) -> str:
    """Stable sha256 over a fixture's PREDICTED numbers (params/flops/memory).

    This is the drift gate: any change to the frozen R1-tiny config or the accounting
    formulas flips the hash, so a scenario pinned to an old hash fails validation.
    """
    p, f, m = fixture_report["params"], fixture_report["flops_per_token"], fixture_report["memory"]
    canon = {
        "params": {k: p[k] for k in sorted(p)},
        "flops": {k: f[k] for k in sorted(f)},
        "mem": {k: m[k] for k in sorted(m)},
    }
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def build_r1_tiny_scenario() -> dict:
    """Construct the frozen scenario record for the R1-tiny training step from the fixture."""
    r = r1_tiny_report()
    p, f, m = r["params"], r["flops_per_token"], r["memory"]
    return {
        "id": "r1-tiny-train",
        "description": "One R1-tiny training step (global batch 32×512) on Tier-0.",
        "owner": "transformers-systems",
        "fixture": "fixtures/r1_tiny.py::r1_tiny_report",
        "units": {"flops": "1 MAC = 2 FLOP; backward = 2× forward", "memory": "bytes"},
        "numbers_hash": numbers_hash(r),
        "predicted": {
            "params_total": p["total"],
            "flops_train_per_token": f["total_train"],
            "flops_per_step": f["total_train"] * m["tokens_per_step"],
            "train_state_bytes": m["train_state"],
        },
    }


def validate(scenario: dict, report: dict, rel_tol: float = 0.05) -> list[str]:
    """Return a list of problems (empty = pass). CI fails on any problem."""
    problems: list[str] = []
    live = numbers_hash(r1_tiny_report())

    # 1. drift gate — scenario + report must match the live fixture
    if scenario.get("numbers_hash") != live:
        problems.append(
            f"scenario numbers_hash drift: pinned {str(scenario.get('numbers_hash'))[:12]} != live {live[:12]}"
        )
    if report.get("scenario_numbers_hash") != live:
        problems.append("report scenario_numbers_hash != live fixture hash")

    # 2. denominators present (frozen §7)
    dens = report.get("denominators", {})
    for k in REQUIRED_DENOMINATORS:
        if not dens.get(k):
            problems.append(f"missing/empty denominator: {k}")

    # 3. predict-then-measure — every measured metric within tol of the prediction
    predicted = scenario.get("predicted", {})
    for metric, meas in report.get("measured", {}).items():
        if metric in predicted and predicted[metric]:
            rel = abs(meas - predicted[metric]) / abs(predicted[metric])
            if rel > rel_tol:
                problems.append(
                    f"{metric}: measured {meas:g} vs predicted {predicted[metric]:g} "
                    f"(rel {rel:.1%} > {rel_tol:.0%})"
                )
    return problems


def load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


if __name__ == "__main__":
    # regenerate the frozen scenario file from the fixture
    here = os.path.dirname(__file__)
    out = os.path.join(here, "..", "scenarios", "r1-tiny-train.yml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sc = build_r1_tiny_scenario()
    with open(out, "w") as fh:
        yaml.safe_dump(sc, fh, sort_keys=False)
    print(f"wrote {os.path.relpath(out)}  numbers_hash={sc['numbers_hash'][:16]}…")
