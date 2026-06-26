#!/usr/bin/env python
"""M0 oracle: independent numpy reference for the MicroHs safetensors loader.

Reads the same tensors the MicroHs `M0` driver prints and emits the same values,
so the two can be diffed. Run with: uv run --with numpy python test/oracle_m0.py
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

        eb, _ = raw("model.embed_tokens.weight")
        print("embed[0,0..4]:", [float(x) for x in bf16(eb[: 5 * 2])])

        sb, _ = raw("model.layers.0.self_attn.q_proj.weight_scale")
        print("q_proj.weight_scale:", float(bf16(sb)[0]))

        wb, wm = raw("model.layers.0.self_attn.q_proj.weight")
        packed_rows, in_dim = wm["shape"]
        W = np.frombuffer(wb, dtype=np.uint8).reshape(packed_rows, in_dim)
        chunk = packed_rows  # out_dim / 4

        def unpack_row(r):
            i, j = r // chunk, r % chunk
            return ((W[j, :] >> (2 * i)) & 3).astype(np.int16) - 1

        row0 = unpack_row(0)
        print("q_proj trits row0[0..15]:", [int(x) for x in row0[:16]])
        print("q_proj row0 all ternary:", bool(((row0 >= -1) & (row0 <= 1)).all()),
              " len", row0.shape[0])
        print("q_proj trits row700[0..15]:", [int(x) for x in unpack_row(700)[:16]])


if __name__ == "__main__":
    main()
