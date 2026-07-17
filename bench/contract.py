"""The numbers contract — scenario records, benchmark reports, and the consistency validator.

Load-bearing governance (plan Phase −1): "every number in the book traces to a runnable
fixture and a measured report." Three pieces:

- a **scenario** (`scenarios/<id>.yml`) fixes a named configuration, the PREDICTED numbers, the
  set of **contracted** metrics a report MUST measure, and a `numbers_hash` over the full
  config + predictions (the drift gate);
- a **benchmark report** (JSON) carries the frozen reference-model §7 denominators + the
  MEASURED values + `measured_on` provenance;
- `validate()` checks: report/scenario identity; the live fixture hash matches; every
  contracted metric is present (no empty/partial reports pass) and within tolerance of its
  prediction; and all denominators are recorded.

Pure stdlib + PyYAML; no torch (runs in CPU-only CI too).
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
from dataclasses import asdict

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fixtures.r1_tiny import r1_tiny_report  # noqa: E402

REQUIRED_DENOMINATORS = [
    "hw_sw_manifest", "workload_shape", "warmup_steps", "measurement_window",
    "utilization_metric", "price_timestamp", "allocation_method", "uncertainty", "measured_on",
]
CONTRACTED = ["params_total", "train_state_bytes"]  # metrics a report MUST measure (exactly)


def numbers_hash(fixture_report: dict) -> str:
    """sha256 over the FULL frozen config + predicted numbers + counting units.

    Includes the config so two semantically different configs that happen to share aggregate
    numbers still hash differently (Codex #6). Any change to config or accounting flips it.
    """
    p, f, m = fixture_report["params"], fixture_report["flops_per_token"], fixture_report["memory"]
    canon = {
        "config": {k: v for k, v in sorted(asdict(fixture_report["config"]).items())},
        "params": {k: p[k] for k in sorted(p)},
        "flops": {k: f[k] for k in sorted(f)},
        "mem": {k: m[k] for k in sorted(m)},
        "units": {"flop": "1 MAC = 2 FLOP; bwd = 2x fwd", "mem": "bytes",
                  "train_state": "fp32 AdamW: weight+grad+m+v = 16 B/param"},
    }
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def build_r1_tiny_scenario() -> dict:
    r = r1_tiny_report()
    p, f, m = r["params"], r["flops_per_token"], r["memory"]
    return {
        "id": "r1-tiny-train",
        "description": "One R1-tiny fp32 AdamW training step on Tier-0.",
        "owner": "transformers-systems",
        "fixture": "fixtures/r1_tiny.py::r1_tiny_report",
        "contracted": list(CONTRACTED),
        "numbers_hash": numbers_hash(r),
        "predicted": {
            "params_total": p["total"],
            "train_state_bytes": m["train_state"],
            # informational (not contracted until a FLOP-measuring lab exists):
            "flops_train_per_token": f["total_train"],
            "flops_per_step_bt32x512": f["total_train"] * m["tokens_per_step"],
        },
    }


def validate(scenario: dict, report: dict, rel_tol: float = 0.02) -> list[str]:
    """Return a list of problems (empty = pass). CI fails on any problem."""
    problems: list[str] = []
    live = numbers_hash(r1_tiny_report())

    # 0. identity
    if report.get("scenario_id") != scenario.get("id"):
        problems.append(f"report scenario_id {report.get('scenario_id')!r} != scenario {scenario.get('id')!r}")

    # 1. drift gate — scenario + report must match the live fixture hash
    if scenario.get("numbers_hash") != live:
        problems.append(f"scenario numbers_hash drift: pinned {str(scenario.get('numbers_hash'))[:12]} != live {live[:12]}")
    if report.get("scenario_numbers_hash") != live:
        problems.append("report scenario_numbers_hash != live fixture hash")

    # 2. denominators present (frozen §7)
    dens = report.get("denominators", {})
    for k in REQUIRED_DENOMINATORS:
        if dens.get(k) in (None, ""):
            problems.append(f"missing/empty denominator: {k}")

    # 3. the contracted metric set must be measured EXACTLY (no empty/partial/extra — Codex #1)
    contracted = set(scenario.get("contracted", []))
    measured = report.get("measured", {})
    missing, extra = contracted - set(measured), set(measured) - contracted
    if missing:
        problems.append(f"contracted metrics not measured: {sorted(missing)}")
    if extra:
        problems.append(f"unexpected measured metrics (not contracted): {sorted(extra)}")

    # 4. predict-then-measure — each contracted metric within tol of its prediction
    predicted = scenario.get("predicted", {})
    for metric in contracted & set(measured):
        if metric not in predicted:
            problems.append(f"contracted metric {metric} has no prediction")
            continue
        pred, meas = predicted[metric], measured[metric]
        rel = abs(meas - pred) / abs(pred) if pred else (0.0 if meas == 0 else float("inf"))
        if rel > rel_tol:
            problems.append(f"{metric}: measured {meas:g} vs predicted {pred:g} (rel {rel:.1%} > {rel_tol:.0%})")
    return problems


def load_yaml(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "scenarios", "r1-tiny-train.yml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sc = build_r1_tiny_scenario()
    with open(out, "w") as fh:
        yaml.safe_dump(sc, fh, sort_keys=False)
    print(f"wrote {os.path.relpath(out)}  numbers_hash={sc['numbers_hash'][:16]}…  contracted={sc['contracted']}")
