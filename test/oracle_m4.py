#!/usr/bin/env python
"""M4 oracle: HF greedy generation on the same prompt ids, to compare against the
MicroHs M4 output. Uses the int8-reinterpret weight fix (see oracle_m2.py).
Run: uv run --with torch --with transformers --with accelerate python test/oracle_m4.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PATH = "models/bitnet-b1.58-2B-4T"
PROMPT = [1502, 25, 22691, 0, 128009, 72803, 25, 220]
NGEN = 12

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

with torch.no_grad():
    out = model.generate(torch.tensor([PROMPT]), max_new_tokens=NGEN,
                         do_sample=False, num_beams=1)
gen = out[0].tolist()[len(PROMPT):]
print("generated ids:", " ".join(str(i) for i in gen))
tok = AutoTokenizer.from_pretrained(PATH)
print("text:", repr(tok.decode(gen, skip_special_tokens=False)))
