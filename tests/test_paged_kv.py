"""Acceptance wrapper for the M4 paged-KV parity gate.

Runs ``serve/paged_kv.py`` (exit 0 = cached incremental attention reproduces
the full-recompute greedy tokens) as a ``tests/`` entry point, so the gate is
reachable the same way as the other tests instead of living only as a script
self-check. Requires torch (Tier-0); self-skips loudly on the torch-less
hosted CI lane. Run with ``python3 tests/test_paged_kv.py`` (exit 0 = pass).
"""
import os
import subprocess
import sys

try:
    import torch  # noqa: F401
except ImportError:
    print("[skip] torch not installed (hosted CPU lane) — run on Tier-0")
    sys.exit(0)

script = os.path.join(os.path.dirname(__file__), "..", "serve", "paged_kv.py")
proc = subprocess.run([sys.executable, script])
if proc.returncode != 0:
    print(f"FAIL: paged-KV parity gate exited {proc.returncode}")
    sys.exit(1)
print("OK: paged-KV parity gate passed")
