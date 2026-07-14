#!/usr/bin/env python3
"""M3 oracle: numpy reference for the full-attention block (layer 3), same
synthetic chained inputs as ./bin/m3. Weight matmuls model the kernel's
quantized path (q8_0 activations against dequantized Q1_0 weights) in f64,
so values match to ~1e-4, not bit-exactly.

Run with: uv run --with numpy python test/oracle_m3.py
"""

import numpy as np

MODEL = "models/bonsai-27b/Bonsai-27B-Q1_0.gguf"


# Absolute offsets and (ne-order) shapes from the checked-in tensor dump.
def load_table():
    tab = {}
    with open("test/bonsai-gguf-tensors.txt", encoding="utf-8") as f:
        for line in f:
            if "(" not in line or ")" not in line:
                continue
            name = line.split()[0]
            shape = tuple(int(x) for x in
                          line[line.index("(") + 1:line.index(")")].split(",") if x.strip())
            rest = line[line.index(")") + 1:].split()
            tab[name] = (int(rest[1]), rest[0], shape)
    return tab


TAB = load_table()
raw = np.memmap(MODEL, dtype=np.uint8, mode="r")

HID, NH, NKV, HD, NROT, FFN = 5120, 24, 4, 256, 64, 17408
EPS, THETA = 1e-6, 1e7


def f32(off, n):
    return np.frombuffer(raw[off:off + 4 * n].tobytes(), dtype=np.float32).astype(np.float64)


def deq(off, rows, cols):
    rb = (cols // 128) * 18
    w = np.frombuffer(raw[off:off + rows * rb].tobytes(),
                      dtype=np.uint8).reshape(rows, cols // 128, 18)
    d = w[:, :, :2].copy().reshape(rows, -1).view(np.float16).astype(np.float32)
    bits = np.unpackbits(w[:, :, 2:].reshape(-1, 16), axis=1,
                         bitorder="little").astype(np.int16)
    signs = (bits * 2 - 1).astype(np.float32).reshape(rows, cols // 128, 128)
    return (signs * d[:, :, None]).reshape(rows, cols)


def weight(name):
    off, dtype, shape = TAB["blk.3." + name + ".weight"]
    if dtype == "F32":
        return f32(off, shape[0])
    return deq(off, shape[1], shape[0])  # ne order: (in, out) -> rows=out


W = {k: weight(k) for k in ["attn_norm", "attn_q", "attn_q_norm", "attn_k",
                            "attn_k_norm", "attn_v", "attn_output",
                            "post_attention_norm", "ffn_gate", "ffn_up", "ffn_down"]}


def quant_act(x):
    """Model the kernel's q8_0 activation quant, back to f64 values."""
    n = len(x)
    xb = x.astype(np.float32).reshape(n // 32, 32)
    amax = np.max(np.abs(xb), axis=1)
    d = (amax / np.float32(127.0)).astype(np.float32)
    sc = d.astype(np.float16).astype(np.float32)
    iid = np.where(d > 0, np.float32(1.0) / d, np.float32(0.0))
    qs = np.round(xb * iid[:, None]).astype(np.float64)
    return (qs * sc[:, None].astype(np.float64)).reshape(n)


def b1linear(w, x):
    return w.astype(np.float64) @ quant_act(x)


def rmsnorm(wg, x, n=None):
    s = 1.0 / np.sqrt(np.mean(np.square(x.astype(np.float32)), dtype=np.float32) + np.float32(EPS))
    return wg * (x * float(s))


def rmsnorm_heads(wg, x, nh, hd):
    o = np.zeros_like(x)
    for h in range(nh):
        o[h * hd:(h + 1) * hd] = rmsnorm(wg, x[h * hd:(h + 1) * hd])
    return o


def rope(x, pos, nh, hd, nrot):
    half = nrot // 2
    inv = THETA ** (-2.0 * np.arange(half) / nrot)
    o = x.copy()
    for h in range(nh):
        v = o[h * hd:(h + 1) * hd]
        a = pos * inv
        c, s = np.cos(a), np.sin(a)
        x1, x2 = v[:half].copy(), v[half:nrot].copy()
        v[:half] = x1 * c - x2 * s
        v[half:nrot] = x2 * c + x1 * s
    return o


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


x = np.array([((i * 53) % 199 - 99) / 99 for i in range(HID)], dtype=np.float64)
ck, cv = [], []

for pos in range(3):
    xn = rmsnorm(W["attn_norm"], x)
    qg = b1linear(W["attn_q"], xn)
    q0 = qg.reshape(NH, 2, HD)[:, 0, :].reshape(-1)
    gate = qg.reshape(NH, 2, HD)[:, 1, :].reshape(-1)
    qn = rmsnorm_heads(W["attn_q_norm"], q0, NH, HD)
    kn = rmsnorm_heads(W["attn_k_norm"], b1linear(W["attn_k"], xn), NKV, HD)
    v = b1linear(W["attn_v"], xn)
    qr = rope(qn, pos, NH, HD, NROT)
    kr = rope(kn, pos, NKV, HD, NROT)
    ck.append(kr.reshape(NKV, HD))
    cv.append(v.reshape(NKV, HD))
    att = np.zeros(NH * HD)
    for h in range(NH):
        kv = h // (NH // NKV)
        qh = qr[h * HD:(h + 1) * HD]
        scores = np.array([np.dot(qh, ck[p][kv]) for p in range(pos + 1)]) / np.sqrt(HD)
        w_ = np.exp(scores - scores.max())
        w_ /= w_.sum()
        att[h * HD:(h + 1) * HD] = sum(w_[p] * cv[p][kv] for p in range(pos + 1))
    gated = att * sigmoid(gate)
    o = b1linear(W["attn_output"], gated)
    x1 = x + o
    x1n = rmsnorm(W["post_attention_norm"], x1)
    gf = b1linear(W["ffn_gate"], x1n)
    uf = b1linear(W["ffn_up"], x1n)
    h_ = gf * sigmoid(gf) * uf
    x = x1 + b1linear(W["ffn_down"], h_)
    print(f"pos{pos} out[0..5]:", [float(v_) for v_ in x[:6]])
    print(f"pos{pos} out[5114..5119]:", [float(v_) for v_ in x[-6:]])
    print(f"pos{pos} sum:", float(x.sum()))
