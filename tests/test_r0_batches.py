"""Golden test for the built R0 batch artifact (the training-input contract).

Locks what the data-pipeline chapter promises: shape (256, 512), dtype uint16
(little-endian), every id inside the vocabulary (max id < 8192), and the
artifact's sha256 equal to the value recorded in ``r0/manifest.yml`` — so a
pipeline change that alters the training input trips CI rather than silently
retraining on different data. Pure stdlib; run with
``python3 tests/test_r0_batches.py`` (exit 0 = pass).
"""
import ast
import hashlib
import os
import re
import struct
import sys
from array import array

ROOT = os.path.join(os.path.dirname(__file__), "..")
NPY = os.path.join(ROOT, "r0", "data", "train_batches.npy")
MANIFEST = os.path.join(ROOT, "r0", "manifest.yml")

EXPECT_SHAPE = (256, 512)
EXPECT_DESCR = "<u2"  # little-endian uint16
VOCAB = 8192


def manifest_value(key: str) -> str:
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            m = re.match(rf"\s*{key}:\s*(\S+)\s*$", line)
            if m:
                return m.group(1)
    raise AssertionError(f"{key} not found in {MANIFEST}")


def read_npy_header(f):
    magic = f.read(6)
    assert magic == b"\x93NUMPY", f"bad npy magic {magic!r}"
    major = f.read(1)[0]
    f.read(1)  # minor
    if major == 1:
        (hlen,) = struct.unpack("<H", f.read(2))
    else:
        (hlen,) = struct.unpack("<I", f.read(4))
    return ast.literal_eval(f.read(hlen).decode("latin1").strip())


failures = []
with open(NPY, "rb") as f:
    header = read_npy_header(f)
    payload = f.read()

if header.get("descr") != EXPECT_DESCR:
    failures.append(f"dtype: {header.get('descr')!r} != {EXPECT_DESCR!r}")
if header.get("fortran_order"):
    failures.append("fortran_order is True (expected C order)")
if tuple(header.get("shape", ())) != EXPECT_SHAPE:
    failures.append(f"shape: {header.get('shape')} != {EXPECT_SHAPE}")

expected_bytes = EXPECT_SHAPE[0] * EXPECT_SHAPE[1] * 2
if len(payload) != expected_bytes:
    failures.append(f"payload: {len(payload)} bytes != {expected_bytes}")
else:
    ids = array("H")
    ids.frombytes(payload)
    if sys.byteorder != "little":
        ids.byteswap()
    mx = max(ids)
    if mx >= VOCAB:
        failures.append(f"max token id {mx} >= vocab {VOCAB}")

file_sha = hashlib.sha256(open(NPY, "rb").read()).hexdigest()
want_sha = manifest_value("batches_sha256")
if file_sha != want_sha:
    failures.append(f"sha256 {file_sha} != manifest batches_sha256 {want_sha}")

# The manifest's own view of the batch contract must agree with the golden.
for key, want in (("context_len", "512"), ("n_sequences", "256"), ("dtype", "uint16")):
    got = manifest_value(key)
    if got != want:
        failures.append(f"manifest {key}: {got} != {want}")

if failures:
    print("FAIL r0 batch golden:")
    for msg in failures:
        print(f"  - {msg}")
    sys.exit(1)

print(f"OK r0 batches: shape {EXPECT_SHAPE}, uint16, max id {mx} < {VOCAB}, sha256 matches manifest")
