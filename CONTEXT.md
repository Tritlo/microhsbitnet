# BitNetComb

Inference for the BitNet b1.58 2B4T model implemented in MicroHs, where the
ternary weights are interpreted as list operations rather than numeric multiplications.

## Language

**MicroHs**:
The target Haskell implementation — Lennart Augustsson's extended subset of Haskell
that compiles via combinators (`~/Code/MicroHs`). The implementation language and runtime.
_Avoid_: GHC Haskell, "plain Haskell".

**BitNet b1.58 2B4T**:
The Microsoft model being run. A Llama-style decoder (30 layers, hidden 2560, GQA
20/5 heads, RoPE, squared-ReLU FFN) whose linear layers are ternary. The specific
artifact is the *packed* deployment variant.
_Avoid_: BitNet (ambiguous — refers to the framework), b1.5B.

**Trit**:
A single ternary weight value, one of {-1, 0, +1}.
_Avoid_: bit, ternary digit.

**List-op interpretation**:
The project's central idea: a trit names an operation on an activation rather than a
multiplier — -1 = negate, 0 = drop, +1 = identity. A ternary dot product `W · x` is then
a fold that maps each trit over its activation and sums the kept terms, e.g.
`[x1,x2,x3] × [-1,0,+1] = -x1 + x3`. No multiplications in the matmul.

**BitLinear**:
A linear layer whose weight matrix is ternary (trits) with a single per-tensor scale.
The unit to which the list-op interpretation applies.
_Avoid_: linear, dense, projection (when specifically meaning the ternary kind).

**W1.58A8**:
The model's quantization scheme: 1.58-bit ternary **weights** (absmean per-tensor scale)
and 8-bit integer **activations** (absmax per-token scale). Because activations are int8,
the list-op fold is exact integer arithmetic.
_Avoid_: "1-bit", int8 (alone, ambiguous between weights and activations).
