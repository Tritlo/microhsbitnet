#!/usr/bin/env python3
"""M4 oracle: numpy reference for the gated-delta-net block (layer 0), same
chained synthetic inputs as ./bin/m4. Semantics pinned from the fork's CPU
source: ssm_conv (kernel coeff r = window position r, oldest first) + SiLU;
l2_norm = x / max(sqrt(sum x^2), eps); softplus cutoff at 20; the non-KDA
delta rule with q-scale 1/sqrt(d). Matmuls model the kernel's quantized path
in f64 (~1e-3, not bit-exact).

Run with: uv run --with numpy python test/oracle_m4.py
"""

import numpy as np

MODEL = "models/bonsai-27b/Bonsai-27B-Q1_0.gguf"

HID, FFN = 5120, 17408
NK, NV, D, INNER = 16, 48, 128, 6144
KEY = NK * D
CONVDIM = 2 * KEY + INNER
EPS = 1e-6


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
    off, dtype, shape = TAB["blk.0." + name]
    if dtype == "F32":
        return f32(off, int(np.prod(shape)))
    return deq(off, shape[1], shape[0])


W = {k: weight(k) for k in
     ["attn_norm.weight", "attn_qkv.weight", "attn_gate.weight", "ssm_beta.weight",
      "ssm_alpha.weight", "ssm_conv1d.weight", "ssm_dt.bias", "ssm_a",
      "ssm_norm.weight", "ssm_out.weight", "post_attention_norm.weight",
      "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]}
CONV = W["ssm_conv1d.weight"].reshape(CONVDIM, 4)  # ne (4, convDim): 4 contiguous per channel


def quant_act(x):
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


def rmsnorm(wg, x):
    s = 1.0 / np.sqrt(np.mean(np.square(x.astype(np.float32)), dtype=np.float32) + np.float32(EPS))
    return wg * (x * float(s))


def rmsnorm_heads(wg, x, nh, hd):
    o = np.zeros_like(x)
    for h in range(nh):
        o[h * hd:(h + 1) * hd] = rmsnorm(wg, x[h * hd:(h + 1) * hd])
    return o


def l2norm_heads(x, nh, hd):
    o = np.zeros_like(x)
    for h in range(nh):
        v = x[h * hd:(h + 1) * hd]
        o[h * hd:(h + 1) * hd] = v / max(np.sqrt(np.sum(v * v)), EPS)
    return o


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def silu(v):
    return v * sigmoid(v)


def softplus(v):
    return np.where(v > 20.0, v, np.log1p(np.exp(v)))


x = np.array([((i * 53) % 199 - 99) / 99 for i in range(HID)], dtype=np.float64)
ring = np.zeros((3, CONVDIM))
S = np.zeros((NV, D, D))   # S[h, i, j]: i = k-dim, j = v-dim

for pos in range(3):
    xn = rmsnorm(W["attn_norm.weight"], x)
    qkv = b1linear(W["attn_qkv.weight"], xn)
    z = b1linear(W["attn_gate.weight"], xn)
    betaP = b1linear(W["ssm_beta.weight"], xn)
    alphaP = b1linear(W["ssm_alpha.weight"], xn)
    win = np.vstack([ring, qkv[None, :]])          # 4 x convDim, oldest first
    conv = silu(np.einsum("rc,cr->c", win, CONV))
    ring = win[1:]
    q0, k0, v = conv[:KEY], conv[KEY:2 * KEY], conv[2 * KEY:]
    qn = l2norm_heads(q0, NK, D)
    kn = l2norm_heads(k0, NK, D)
    g = W["ssm_a"] * softplus(alphaP + W["ssm_dt.bias"])
    beta = sigmoid(betaP)
    o = np.zeros(NV * D)
    for h in range(NV):
        kh, qh, vh = kn[(h % NK) * D:(h % NK + 1) * D], qn[(h % NK) * D:(h % NK + 1) * D], v[h * D:(h + 1) * D]
        S[h] *= np.exp(g[h])
        delta = (vh - S[h].T @ kh) * beta[h]
        S[h] += np.outer(kh, delta)
        o[h * D:(h + 1) * D] = (S[h].T @ qh) / np.sqrt(D)
    n1 = rmsnorm_heads(W["ssm_norm.weight"], o, NV, D)
    gated = silu(z) * n1
    aout = b1linear(W["ssm_out.weight"], gated)
    x1 = x + aout
    x1n = rmsnorm(W["post_attention_norm.weight"], x1)
    gf = b1linear(W["ffn_gate.weight"], x1n)
    uf = b1linear(W["ffn_up.weight"], x1n)
    x = x1 + b1linear(W["ffn_down.weight"], gf * sigmoid(gf) * uf)
    print(f"pos{pos} out[0..5]:", [float(v_) for v_ in x[:6]])
    print(f"pos{pos} out[5114..5119]:", [float(v_) for v_ in x[-6:]])
    print(f"pos{pos} sum:", float(x.sum()))
