"""Materialize the R0 data contract: pinned corpus → BPE tokenizer → fixed token batches.

Produces the checksummed fixtures M1a-real trains on (retiring the synthetic tokens), plus
`r0/manifest.yml` recording every sha256 so the corpus/tokenizer/batches are immutable and
reproducible (reference-model.md §2). Byte-level BPE, vocab 8192, with the document-control
vocabulary reserved (so R1 text and R3 doc-VLM share one tokenizer).

Run: ``.venv/bin/python r0/build_r0.py``  (streams a small FineWeb-Edu slice; falls back to a
pinned public-domain text if the dataset is unreachable — the manifest records which).
"""
from __future__ import annotations
import hashlib
import io
import os
import sys
import urllib.request

import numpy as np
import yaml

HERE = os.path.dirname(__file__)
CORPUS = os.path.join(HERE, "corpus", "corpus.txt")
TOKJSON = os.path.join(HERE, "tokenizer", "tokenizer.json")
BATCHES = os.path.join(HERE, "data", "train_batches.npy")
MANIFEST = os.path.join(HERE, "manifest.yml")

N_DOCS = 8000
VOCAB = 8192
CONTEXT = 512
N_SEQS = 256  # fixed batch fixture
SPECIAL = ["<|endoftext|>", "<|pad|>", "<page>", "</page>", "<region>",
           "<table>", "<tr>", "<td>", "<formula>", "<eos_doc>"]
# pinned public-domain fallback (Gutenberg: "The Art of War"), used only if FineWeb-Edu is unreachable
FALLBACK_URL = "https://www.gutenberg.org/cache/epub/132/pg132.txt"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_corpus() -> dict:
    """Assemble the pinned corpus; return provenance dict."""
    os.makedirs(os.path.dirname(CORPUS), exist_ok=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        with open(CORPUS, "w", encoding="utf-8") as out:
            n = 0
            for row in ds:
                t = (row.get("text") or "").strip()
                if t:
                    out.write(t.replace("\n", " ") + "\n")
                    n += 1
                if n >= N_DOCS:
                    break
        return {"source": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT",
                "split": "train", "n_docs": n, "license": "ODC-By-1.0"}
    except Exception as e:  # noqa: BLE001 — deliberate fallback, recorded in the manifest
        print(f"[r0] FineWeb-Edu unavailable ({type(e).__name__}: {e}); using pinned fallback", file=sys.stderr)
        raw = urllib.request.urlopen(FALLBACK_URL, timeout=60).read().decode("utf-8", "replace")
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        with open(CORPUS, "w", encoding="utf-8") as out:
            out.write("\n".join(lines) + "\n")
        return {"source": FALLBACK_URL, "config": "gutenberg-pd-fallback",
                "split": "-", "n_docs": len(lines), "license": "public-domain"}


def train_tokenizer():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    os.makedirs(os.path.dirname(TOKJSON), exist_ok=True)
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB, special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train([CORPUS], trainer)
    tok.save(TOKJSON)
    return tok


def build_batches(tok):
    os.makedirs(os.path.dirname(BATCHES), exist_ok=True)
    eot = tok.token_to_id("<|endoftext|>")
    ids: list[int] = []
    with open(CORPUS, encoding="utf-8") as fh:
        for line in fh:
            ids.extend(tok.encode(line.rstrip("\n")).ids)
            ids.append(eot)
            if len(ids) >= (N_SEQS + 1) * CONTEXT:
                break
    arr = np.array(ids[: N_SEQS * CONTEXT], dtype=np.uint16).reshape(N_SEQS, CONTEXT)
    np.save(BATCHES, arr)
    return arr


def main():
    prov = build_corpus()
    corpus_sha = sha256_file(CORPUS)
    tok = train_tokenizer()
    tok_sha = sha256_file(TOKJSON)
    arr = build_batches(tok)
    batch_sha = sha256_file(BATCHES)

    manifest = {
        "schema": "r0/1",
        "dataset": {**prov, "corpus_sha256": corpus_sha, "corpus_bytes": os.path.getsize(CORPUS)},
        "tokenizer": {"kind": "byte-level-BPE", "vocab_size": VOCAB, "actual_vocab": tok.get_vocab_size(),
                      "special_tokens": SPECIAL, "tokenizer_sha256": tok_sha},
        "batches": {"context_len": CONTEXT, "n_sequences": int(arr.shape[0]),
                    "dtype": "uint16", "batches_sha256": batch_sha},
        "normalization": "strip + newline->space per doc; NFC implicit via UTF-8",
    }
    with open(MANIFEST, "w") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False)
    print(f"[r0] corpus docs={prov['n_docs']} ({prov['source']})  vocab={tok.get_vocab_size()}  "
          f"batches={arr.shape}\n[r0] shas: corpus={corpus_sha[:12]} tok={tok_sha[:12]} batches={batch_sha[:12]}")
    print(f"[r0] wrote {os.path.relpath(MANIFEST)}")


if __name__ == "__main__":
    main()
