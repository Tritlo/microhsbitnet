#!/usr/bin/env python
"""M3 oracle: HF reference for the full forward (final hidden + next-token logits).

Loads the model with the int8-reinterpret weight fix (see oracle_m2.py), runs the
forward, and prints the final-norm hidden state and logits in the same format as
the MicroHs `M3` driver.
Run with: uv run --with torch --with transformers --with accelerate python test/oracle_m3.py
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM

PATH = "models/bitnet-b1.58-2B-4T"
IDS = [128000]


def main() -> None:
    model = AutoModelForCausalLM.from_pretrained(PATH, dtype=torch.bfloat16)
    model.eval()
    for mod in model.modules():
        if type(mod).__name__ == "AutoBitLinear":
            mod.weight.data = mod.weight.data.view(torch.int8).to(torch.float32)
    for _, p in model.named_parameters():
        if p.is_floating_point():
            p.data = p.data.float()
    for _, b in model.named_buffers():
        if b.is_floating_point():
            b.data = b.data.float()

    cap = {}
    model.model.norm.register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", o.detach().float().numpy()))

    with torch.no_grad():
        out = model(torch.tensor([IDS]))
    hlast = cap["h"][0][-1]
    logits = out.logits[0, -1].float().numpy()

    print("seq=%d ids=%s" % (len(IDS), IDS))
    print("final hidden (last pos) [0..4]:", [float(x) for x in hlast[:5]])
    print("final hidden L2:", float(np.sqrt((hlast.astype(np.float64) ** 2).sum())))
    print("logits at [0,100,1000,50000,128000]:",
          [float(logits[v]) for v in (0, 100, 1000, 50000, 128000)])
    am = int(logits.argmax())
    print("argmax token:", am, " logit:", float(logits[am]))
    top5 = np.argsort(logits)[-5:][::-1]
    print("top5:", [(int(v), float(logits[v])) for v in top5])


if __name__ == "__main__":
    main()
