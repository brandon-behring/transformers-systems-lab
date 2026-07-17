"""VLM interface-risk spike — prove the document-AI plumbing connects (Codex/plan-scoped).

NOT a feasibility study: it is the *interface* spike. A deterministic rendered-page fixture ->
tiny patch-embed vision encoder -> MLP projector -> R1-tiny decoder -> the real R0 document-control
markup vocabulary, overfitting ONE (image, markup) pair until greedy decoding reproduces the exact
markup. Proves image-tokens -> projector -> decoder -> valid-markup wiring end-to-end on Tier-0,
with no model downloads. Real frozen encoders (SAM/SigLIP2) + constrained decoding (XGrammar) +
a stratified eval set are the next steps (Tier-1 / later).

Run: ``.venv/bin/python vlm/spike.py`` (exit 0 = the pipeline overfits one doc to exact markup).
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.r1 import R1
from fixtures.r1_tiny import R1TinyConfig

HERE = os.path.dirname(__file__)


class TinyVisionEncoder(nn.Module):
    """Patchify a rendered page into visual tokens (stand-in for a frozen SAM/SigLIP2 encoder)."""
    def __init__(self, d_model, patch=8):
        super().__init__()
        self.proj = nn.Conv2d(3, d_model, kernel_size=patch, stride=patch)

    def forward(self, img):                      # img: (1,3,H,W)
        return self.proj(img).flatten(2).transpose(1, 2)   # (1, n_vis, d_model)


class DocVLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dec = R1(cfg)
        self.enc = TinyVisionEncoder(cfg.d_model)
        self.projector = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
                                       nn.Linear(cfg.d_model, cfg.d_model))

    def _run(self, img, text_ids):
        vis = self.projector(self.enc(img))       # (1, n_vis, d)
        txt = self.dec.embed(text_ids)            # (1, T, d)
        h = torch.cat([vis, txt], dim=1)
        cos, sin = self.dec.rope_cos.to(h.device), self.dec.rope_sin.to(h.device)
        for blk in self.dec.blocks:
            h = blk(h, cos, sin)
        return F.linear(self.dec.norm(h), self.dec.embed.weight), vis.shape[1]

    def forward(self, img, text_in, targets):
        logits, n_vis = self._run(img, text_in)
        pred = logits[:, n_vis:n_vis + targets.shape[1]]
        return F.cross_entropy(pred.reshape(-1, pred.size(-1)), targets.reshape(-1))

    @torch.no_grad()
    def generate(self, img, bos, n):
        seq = bos.clone()
        for _ in range(n):
            logits, n_vis = self._run(img, seq)
            seq = torch.cat([seq, logits[:, -1:].argmax(-1)], dim=1)
        return seq[:, 1:]                          # drop BOS


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = R1TinyConfig()

    tok = Tokenizer.from_file(os.path.join(HERE, "..", "r0", "tokenizer", "tokenizer.json"))
    bos_id = tok.token_to_id("<|endoftext|>")
    markup = torch.tensor([tok.encode("<page><formula>E=mc^2</formula></page>").ids], device=dev)
    L = markup.shape[1]
    bos = torch.tensor([[bos_id]], device=dev)
    text_in = torch.cat([bos, markup[:, :-1]], dim=1)          # teacher forcing

    # deterministic "rendered page" fixture (stand-in for pypdfium2 output)
    img = torch.Generator(device="cpu").manual_seed(7)
    page = torch.randn(1, 3, 32, 32, generator=img).to(dev)

    model = DocVLM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    first = None
    for step in range(400):
        loss = model(page, text_in, markup)
        opt.zero_grad(); loss.backward(); opt.step()
        first = loss.item() if first is None else first
    last = loss.item()

    gen = model.generate(page, bos, L)[0].tolist()
    exact = gen == markup[0].tolist()
    print(f"overfit loss {first:.3f} -> {last:.4f} | markup tokens: {L} | n_vis: 16")
    print(f"target markup: {tok.decode(markup[0].tolist())!r}")
    print(f"greedy decode reproduces exact markup: {exact}")
    print("\nVLM interface spike:", "PLUMBING OK (image->encoder->projector->decoder->markup)" if exact else "FAILED")
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
