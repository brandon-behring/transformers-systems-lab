#!/usr/bin/env bash
# Gate B driver: write a DCP checkpoint at world-size 2, load + verify it at world-size 1.
# `set -e` makes the driver fail if EITHER torchrun stage exits nonzero — so the gate is enforced.
set -euo pipefail
cd "$(dirname "$0")/.."
export GATE_B_CKPT="/home/brandon_behring/.claude/jobs/0de54cce/tmp/gate_b_$$/ckpt"
.venv/bin/torchrun --standalone --nproc-per-node=2 dist/gate_b.py save
.venv/bin/torchrun --standalone --nproc-per-node=1 dist/gate_b.py load
echo "Gate B driver: both stages exit 0 (cross-world-size DCP load enforced)"
