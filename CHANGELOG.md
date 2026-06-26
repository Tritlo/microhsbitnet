# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial project setup.
- Fetched `microsoft/bitnet-b1.58-2B-4T` (packed 1.58-bit weights) into `models/` (gitignored).
- Design phase: `CONTEXT.md` glossary and `PLAN.md` (M0–M4 staged ladder and pinned model/packing
  facts).
- **M0 — safetensors loader** (`src/Safetensors.hs`, `src/M0.hs`): minimal JSON header parser,
  bf16→Float decode, and packed ternary unpack. Loads the full 1.18 GB data section resident in
  ~2 s and is verified bit-identical to a numpy oracle (`test/oracle_m0.py`) on bf16 values,
  scales, and trits (incl. a strided row). Documents MicroHs IO gotchas (`gmhs`, broken
  `BS.empty`/`concat`/`readFile`, no `hSeek`/`hFileSize`).
- **M1 — list-op BitLinear** (`src/BitLinear.hs`, `src/Loader.hs`, `src/M1.hs`): int8 activation
  quant (per-token absmax), the two-bucket trit fold (`+1`→pos, `-1`→neg, `0`→drop, value
  `pos−neg` — no negate/multiply in the matmul), and rescale. A real layer-0 q_proj forward
  matches a numpy oracle (`test/oracle_m1.py`) bit-exactly on the integer dots (incl. full-matrix
  sum) and to Float precision on outputs. Loader factored into `Loader.hs`. Measured ~23 s per
  2560×2560 BitLinear in the combinator runtime (perf for the C→B gate).
- **B — `dotRow` workhorse**: folds directly over a row's contiguous packed bytes (routing by
  the 2-bit field, walking int8 activations in lockstep) with no intermediate `[Trit]` list —
  ~2× faster (23 s → ~10.5 s), bit-exact vs `dotTrits`. `dotTrits` kept as the showcase spec.
- **M2 — full decoder block** (`src/Model.hs`, `src/M2.hs`): RMSNorm, RoPE, GQA causal
  attention, both sub-norms, squared-ReLU gated MLP, residuals. Layer-0 output matches an HF
  transformers oracle (`test/oracle_m2.py`) to ~5-6 sig figs at first/last positions. The oracle
  reinterprets this transformers version's broken uint8 (`-1→255` underflow) BitLinear weights as
  int8 to recover proper ternary, loading the model in float32.
- **`bndot` runtime primitive** (branch `runtime-ternary-dot`): vendored MicroHs at
  `vendor/microhs/` (full history) and added one weighted-dot opcode to its runtime — kind 0
  ternary×int8 (BitLinears), kind 1 bf16×float32 (LM head) — done in C, bound via `_primitive`
  (no FFI). Pure MicroHs caps at ~10 s/BitLinear (combinator graph reduction); the primitive makes
  it negligible. `src/Dot.hs` hosts the binding, `bitLinearPrim`, and the LM-head `lmLogit`.
- **M3 — full forward** (`src/M3.hs`): embed → 30 layers → final norm → tied LM head → logits.
  Runs in **~25 s** (was ~55 min pure) and predicts **argmax token 11**, matching the HF oracle
  (`test/oracle_m3.py`). Per-layer math stays oracle-exact; ~0.6% drift on the final hidden is
  expected float-order accumulation over depth.
