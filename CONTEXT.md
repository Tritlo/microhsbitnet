# BitNetComb (1-bit branch)

Inference for the 1-bit Bonsai 27B model implemented in MicroHs, where the binary
weights are interpreted as subset selection rather than numeric multiplications.

## Language

**MicroHs**:
The target Haskell implementation — Lennart Augustsson's extended subset of Haskell
that compiles via combinators (`~/Code/MicroHs`). The implementation language and runtime.
_Avoid_: GHC Haskell, "plain Haskell".

**1-bit Bonsai 27B**:
The PrismML model being run — a binary (Q1_0) re-expression of Qwen3.6-27B: 64 blocks,
hidden 5120, hybrid attention (48 gated-delta-net layers + 16 full-attention layers,
every 4th), untied 1-bit embeddings and LM head. The specific artifact is
`models/bonsai-27b/Bonsai-27B-Q1_0.gguf`.
_Avoid_: Bonsai (ambiguous — a whole model family), Qwen3.6 (the fp16 parent).

**Bit (weight)**:
A single binary weight, one of {-1, +1}, stored as one sign bit (LSB-first) with an
fp16 scale shared per 128-weight group.
_Avoid_: trit, sign (alone).

**Selection interpretation**:
The project's central idea, inherited from the ternary list-op thesis: a weight bit
names *include* or *exclude*, not a multiplier. A binary dot product is
`scale · (2·Σ_selected − Σ_all)` — selection plus one per-row correction; no
per-weight negation, no multiplications in the matmul.

**B1Linear**:
A linear layer whose weight matrix is Q1_0 (sign bits + group scales). The unit to
which the selection interpretation applies. Includes the embedding table and LM head.
_Avoid_: BitLinear (the ternary ancestor), linear/dense/projection (ambiguous).

**GDN (gated delta net)**:
The linear-attention layer kind (48 of 64): causal 4-tap conv, L2-normed q/k,
per-head 128×128 recurrent state updated by the gated delta rule
(`S ← S·e^g + k⊗((v − Sᵀk)·β)`), output gated by `silu(z)`.
_Avoid_: SSM/Mamba (different recurrence), attention (alone).

**W1A8**:
The compute scheme: 1-bit weights (fp16 group-128 scales) and 8-bit integer
activations quantized q8_0-style (per-32 absmax), so the selection fold is integer
arithmetic rescaled once per group pair.
_Avoid_: "1.58-bit", int8 (alone).
