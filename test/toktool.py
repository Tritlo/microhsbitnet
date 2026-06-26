#!/usr/bin/env python
"""Tokenizer helper for the M4 generation demo (no BPE in MicroHs — we feed/read
token IDs). Encode a prompt (with the chat template) to IDs, or decode IDs to text.

  uv run --with transformers --with jinja2 python test/toktool.py encode "Hi"
  uv run --with transformers python test/toktool.py decode "128000 9906 ..."
"""
import sys
from transformers import AutoTokenizer

try:
    tok = AutoTokenizer.from_pretrained("models/bitnet-b1.58-2B-4T", fix_mistral_regex=True)
except TypeError:
    tok = AutoTokenizer.from_pretrained("models/bitnet-b1.58-2B-4T")

mode = sys.argv[1]
if mode == "encode":
    msgs = [{"role": "user", "content": sys.argv[2]}]
    out = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  tokenize=True, return_dict=False)
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    print(" ".join(str(int(i)) for i in out))
elif mode == "decode":
    ids = [int(x) for x in sys.argv[2].split()]
    print(tok.decode(ids, skip_special_tokens=False))
