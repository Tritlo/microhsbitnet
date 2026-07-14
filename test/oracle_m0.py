#!/usr/bin/env python3
"""M0 oracle: read the Bonsai GGUF header + spot values with the prism fork's
gguf-py and numpy, printing the same facts as ./bin/m0 (floats to compare
numerically, not textually).

Run with: uv run --with numpy python test/oracle_m0.py
"""

import os
import sys

import numpy as np

PRISM = os.environ.get("PRISM_LLAMA", os.path.expanduser("~/Code/llama.cpp-prism"))
sys.path.insert(0, os.path.join(PRISM, "gguf-py"))
from gguf import GGUFReader  # noqa: E402

MODEL = "models/bonsai-27b/Bonsai-27B-Q1_0.gguf"

r = GGUFReader(MODEL)

raw = np.memmap(MODEL, dtype=np.uint8, mode="r")
print("bytes:", len(raw))
print("kvs:", len(r.fields), " tensors:", len(r.tensors))
print("data_start:", r.data_offset)

KEYS = [
    "general.architecture", "qwen35.block_count", "qwen35.embedding_length",
    "qwen35.feed_forward_length", "qwen35.attention.head_count",
    "qwen35.attention.head_count_kv", "qwen35.attention.key_length",
    "qwen35.full_attention_interval", "qwen35.rope.dimension_count",
    "qwen35.rope.freq_base", "qwen35.attention.layer_norm_rms_epsilon",
    "qwen35.ssm.conv_kernel", "qwen35.ssm.group_count", "qwen35.ssm.inner_size",
    "qwen35.ssm.state_size", "qwen35.ssm.time_step_rank",
    "tokenizer.ggml.model", "tokenizer.ggml.pre",
    "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
]

for k in KEYS:
    f = r.fields.get(k)
    print(f"{k} = {f.contents() if f is not None else 'MISSING'}")

print("rope_sections =", list(r.fields["qwen35.rope.dimension_sections"].contents()))
print("n_tokens =", len(r.fields["tokenizer.ggml.tokens"].data))
print("n_merges =", len(r.fields["tokenizer.ggml.merges"].data))

tens = {t.name: t for t in r.tensors}

for nm in ["token_embd.weight", "output.weight", "output_norm.weight",
           "blk.0.attn_qkv.weight", "blk.0.ssm_conv1d.weight", "blk.3.attn_q.weight"]:
    t = tens[nm]
    # GGUFReader reports shape in GGUF (ne) order already.
    print(nm, list(int(x) for x in t.shape), t.tensor_type.name,
          "off=%d" % (t.data_offset - r.data_offset))

t = tens["output_norm.weight"]
vals = np.frombuffer(raw[t.data_offset:t.data_offset + 20].tobytes(), dtype=np.float32)
print("output_norm[0..4]:", [float(v) for v in vals])


def q1_row(t, row_len, row, c0, n):
    """Dequantize weights [c0, c0+n) of row `row` (18-byte blocks of 128)."""
    out = []
    for c in range(c0, c0 + n):
        j = row * row_len + c
        blk, rem = divmod(j, 128)
        bo = t.data_offset + 18 * blk
        d = np.frombuffer(raw[bo:bo + 2].tobytes(), dtype=np.float16)[0]
        byte = raw[bo + 2 + rem // 8]
        out.append(float(d) if (byte >> (rem % 8)) & 1 else -float(d))
    return out


t = tens["token_embd.weight"]
print("embd[0,0..7]:", q1_row(t, 5120, 0, 0, 8))
print("embd[700,0..7]:", q1_row(t, 5120, 700, 0, 8))
t = tens["blk.3.attn_q.weight"]
print("attn_q[0,0..7]:", q1_row(t, 5120, 0, 0, 8))
