#!/usr/bin/env python3
"""M1 oracle: numpy reference for the fused B1Linear (q8_0 activation quant +
Q1_0 select-dot + group rescale), mirroring the fork's CPU reference float
order exactly. Prints the same lines as ./bin/m1.

Run with: uv run --with numpy python test/oracle_m1.py
"""

import numpy as np

MODEL = "models/bonsai-27b/Bonsai-27B-Q1_0.gguf"

# Offsets from the checked-in ground-truth dump (data_start + tensor offset).
DATA_START = 10992704
QKV_OFF = DATA_START + 362045440   # blk.0.attn_qkv.weight, [5120, 10240] Q1_0
EMB_OFF = DATA_START + 178810880   # token_embd.weight, [5120, 248320] Q1_0

raw = np.memmap(MODEL, dtype=np.uint8, mode="r")

IN_DIM, OUT_DIM = 5120, 10240
x = np.array([((i * 37) % 101 - 50) / 50 for i in range(IN_DIM)], dtype=np.float32)


def quant_q8(x):
    """quantize_row_q8_0_ref: per 32, fp16-round-tripped scale + int8 quants."""
    nb = len(x) // 32
    qs = np.zeros(len(x), dtype=np.int32)
    sc = np.zeros(nb, dtype=np.float32)
    for i in range(nb):
        blk = x[i * 32:(i + 1) * 32]
        amax = np.max(np.abs(blk))
        d = np.float32(amax / 127.0)
        iid = np.float32(1.0 / d) if d else np.float32(0.0)
        sc[i] = np.float32(np.float16(d))
        qs[i * 32:(i + 1) * 32] = np.round(blk * iid).astype(np.int32)
    return qs, sc


def row_bits(off, r, n):
    """Sign bits (+1/-1) and per-128 scales for row r of a Q1_0 tensor."""
    rb = (n // 128) * 18
    row = np.frombuffer(raw[off + r * rb: off + r * rb + rb].tobytes(), dtype=np.uint8)
    blocks = row.reshape(n // 128, 18)
    d = blocks[:, :2].copy().view(np.float16).astype(np.float32).reshape(-1)
    bits = np.unpackbits(blocks[:, 2:], axis=1, bitorder="little").astype(np.int32)
    signs = bits * 2 - 1  # (n//128, 128)
    return signs, d


qs, sc = quant_q8(x)


def dot_row(r):
    signs, d0 = row_bits(QKV_OFF, r, IN_DIM)
    sumf = np.float32(0.0)
    for i in range(IN_DIM // 128):
        sumi = np.float32(0.0)
        for k in range(4):
            a = qs[i * 128 + k * 32: i * 128 + (k + 1) * 32]
            s = signs[i, k * 32:(k + 1) * 32]
            sumi = np.float32(sumi + sc[i * 4 + k] * np.float32(int(np.dot(s, a))))
        sumf = np.float32(sumf + d0[i] * sumi)
    return float(sumf)


out = [dot_row(r) for r in range(8)]
print("qkv out[0..7]:", out)
print("qkv out[2048..2051]:", [dot_row(r) for r in range(2048, 2052)])
print("qkv out[10236..10239]:", [dot_row(r) for r in range(10236, 10240)])

# Full-output sum, in float64 vectorized form (tolerance check, not bit-exact):
signs_all = None
rb = (IN_DIM // 128) * 18
w = np.frombuffer(raw[QKV_OFF: QKV_OFF + OUT_DIM * rb].tobytes(), dtype=np.uint8)
w = w.reshape(OUT_DIM, IN_DIM // 128, 18)
d0 = w[:, :, :2].copy().reshape(OUT_DIM, -1).view(np.float16).astype(np.float64)
bits = np.unpackbits(w[:, :, 2:].reshape(-1, 16), axis=1, bitorder="little").astype(np.int32)
signs = (bits * 2 - 1).astype(np.float64).reshape(OUT_DIM, IN_DIM // 128, 128)
aq = (qs.astype(np.float64) * np.repeat(sc.astype(np.float64), 32)).reshape(IN_DIM // 128, 128)
per_block = np.einsum("rbc,bc->rb", signs, aq)
full = (per_block * d0).sum(axis=1)
print("qkv sum(approx f64):", float(full.sum()))

# Embedding row 42.
signs, d = row_bits(EMB_OFF, 42, IN_DIM)
e = (signs.astype(np.float32) * d[:, None]).reshape(-1)
print("embd42[0..7]:", [float(v) for v in e[:8]])
print("embd42 sum:", float(np.float32(np.sum(e, dtype=np.float32))))
