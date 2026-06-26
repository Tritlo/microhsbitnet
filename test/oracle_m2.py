#!/usr/bin/env python
"""M2 oracle: HF transformers reference for one decoder block (layer 0).

Loads the model in float32 and hooks layer-0 submodules to capture every
intermediate, printing first-3 values + L2 norm at the first/last position in
the same format as the MicroHs `M2` driver.
Run with: uv run --with torch --with transformers --with accelerate python test/oracle_m2.py
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM

PATH = "models/bitnet-b1.58-2B-4T"
IDS = [128000, 791, 6864]


def main() -> None:
    # This transformers version unpacks BitLinear weights to uint8 with a -1->255
    # underflow and its F.linear can't run on CPU. Fix it for a clean float32
    # reference: reinterpret each AutoBitLinear weight as int8 (255 -> -1) and cast
    # to float32 (exact ternary {-1,0,1}), then float-cast remaining tensors.
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

    caps = {}

    def save_out(name):
        def h(mod, inp, out):
            caps[name] = out.detach().float().numpy()
        return h

    def save_io(name):
        def h(mod, inp, out):
            caps[name + "_in"] = inp[0].detach().float().numpy()
            caps[name + "_out"] = out.detach().float().numpy()
        return h

    L = model.model.layers[0]
    L.input_layernorm.register_forward_hook(save_out("hn"))
    L.self_attn.q_proj.register_forward_hook(save_out("q"))
    L.self_attn.k_proj.register_forward_hook(save_out("k"))
    L.self_attn.v_proj.register_forward_hook(save_out("v"))
    L.self_attn.attn_sub_norm.register_forward_hook(save_io("attnsub"))
    L.self_attn.o_proj.register_forward_hook(save_out("o"))
    L.post_attention_layernorm.register_forward_hook(save_out("hn2"))
    L.mlp.gate_proj.register_forward_hook(save_out("gate"))
    L.mlp.up_proj.register_forward_hook(save_out("up"))
    L.mlp.ffn_sub_norm.register_forward_hook(save_io("ffnsub"))
    L.mlp.down_proj.register_forward_hook(save_out("down"))

    with torch.no_grad():
        out = model(torch.tensor([IDS]), output_hidden_states=True)
    hs = out.hidden_states
    embed = hs[0][0].float().numpy()
    layerout = hs[1][0].float().numpy()

    def g(name):
        a = caps[name]
        return a[0] if a.ndim == 3 else a

    o = g("o")
    stages = [
        ("embed", embed), ("hn", g("hn")), ("q", g("q")), ("k", g("k")), ("v", g("v")),
        ("attn", caps["attnsub_in"][0]), ("attnN", caps["attnsub_out"][0]),
        ("o", o), ("xs1", embed + o), ("hn2", g("hn2")),
        ("gate", g("gate")), ("up", g("up")),
        ("gated", caps["ffnsub_in"][0]), ("gatedN", caps["ffnsub_out"][0]),
        ("down", g("down")), ("out", layerout),
    ]
    positions = [0, len(IDS) - 1]
    print("seq=%d ids=%s" % (len(IDS), IDS))
    for lbl, st in stages:
        for pos in positions:
            vec = st[pos]
            head = ", ".join(str(float(x)) for x in vec[:3])
            l2 = float(np.sqrt((vec.astype(np.float64) ** 2).sum()))
            print("%s p%d: [%s] L2=%s" % (lbl, pos, head, l2))


if __name__ == "__main__":
    main()
