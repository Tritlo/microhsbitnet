#!/usr/bin/env python
"""M1 oracle: independent numpy reference for one list-op BitLinear (layer-0 q_proj).

Reproduces the deterministic input, int8 activation quant, integer matmul, and
rescale, so values can be diffed against the MicroHs `M1` driver.
Run with: uv run --with numpy python test/oracle_m1.py
"""
import json
import struct
import numpy as np

PATH = "models/bitnet-b1.58-2B-4T/model.safetensors"


def main() -> None:
    with open(PATH, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n

        def raw(name):
            meta = hdr[name]
            s, e = meta["data_offsets"]
            f.seek(base + s)
            return f.read(e - s), meta

        def bf16(b):
            u = np.frombuffer(b, dtype=np.uint16).astype(np.uint32)
            return (u << 16).view(np.float32)

        wb, wm = raw("model.layers.0.self_attn.q_proj.weight")
        sb, _ = raw("model.layers.0.self_attn.q_proj.weight_scale")
        packed_rows, in_dim = wm["shape"]
        out_dim = packed_rows * 4
        wscale = float(bf16(sb)[0])

        # Unpack full logical weight [out_dim, in_dim].
        W = np.frombuffer(wb, dtype=np.uint8).reshape(packed_rows, in_dim)
        Wq = np.empty((out_dim, in_dim), dtype=np.int64)
        for i in range(4):
            Wq[i * packed_rows:(i + 1) * packed_rows, :] = ((W >> (2 * i)) & 3).astype(np.int64) - 1

        # Deterministic input + int8 activation quant (per-token absmax).
        c = np.arange(in_dim)
        x = (((c * 73 + 13) % 251) - 125) / 64.0
        amax = max(float(np.abs(x).max()), 1e-5)
        sc = 127.0 / amax
        xq = np.clip(np.round(x * sc), -128, 127).astype(np.int64)

        idots = Wq @ xq
        ys = idots.astype(np.float64) * wscale / sc

        print("inDim=%d outDim=%d" % (in_dim, out_dim))
        print("wscale=%s" % wscale)
        print("ascale=%s" % sc)
        print("xq[0..15]=%s" % [int(v) for v in xq[:16]])
        print("xq sum=%d absSum=%d" % (int(xq.sum()), int(np.abs(xq).sum())))
        print("idots[0..7]=%s" % [int(v) for v in idots[:8]])
        print("idots sum=%d" % int(idots.sum()))
        print("y[0..7]=%s" % [float(v) for v in ys[:8]])


if __name__ == "__main__":
    main()
