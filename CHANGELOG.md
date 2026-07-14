# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased — 1-bit branch]

### Changed

- **Rewrite target: 1-bit Bonsai 27B** (`prism-ml/Bonsai-27B-gguf`, Q1_0). The ternary/BitNet
  path is dropped on this branch (main keeps it). New `PLAN.md` (M0–M6 ladder, pinned qwen35
  hybrid-architecture facts from the prism llama.cpp fork @ `62061f91`), rewritten `CONTEXT.md`
  (selection interpretation, B1Linear, GDN, W1A8). GGUF ground-truth dumps in
  `test/bonsai-gguf-{kv,tensors}.txt`.
- **M0 — GGUF loader** (`src/Gguf.hs`, `src/Loader.hs`, `src/M0.hs`): GGUF v3 header parser
  (typed KVs, tensor table, alignment-padded data section) over one resident ByteString of the
  whole 3.8 GB file (~18 s load, sequential fixed-count reads as before). Large arrays
  (tokenizer vocab/merges) are kept as raw (type, count, offset) slices, not materialized —
  the offset walk is `seq`-strict (a 744 K-element lazy skip chain overflowed the stack).
  Q1_0/f16/f32 spot decoders. Verified against `test/oracle_m0.py` (fork gguf-py + numpy):
  offsets, shapes, F32 values, Q1_0 signs and group scales all match. The ternary-path
  modules, oracles, and Makefile targets are deleted; `make` now fetches the Bonsai GGUF.
- **M1 — Q1_0 kernel** (`cbits/b1dot.c`, `src/Dot.hs`, `src/M1.hs`): fused B1Linear —
  q8_0-style activation quant (per-32 fp16-round-tripped scales, ported ggml fp16
  conversions) + the bit-select row dot, mirroring the fork's
  `ggml_vec_dot_q1_0_q8_0_generic` float order exactly; `b1embed` row dequant; the f32
  helpers (rmsnorm/attend/add) carry over and rope gains partial-rotary (`nRot < headDim`).
  Verified on the real `blk.0.attn_qkv` (5120→10240) against `test/oracle_m1.py`: spot rows
  float-identical, full-output sum to 3e-6, embedding row exact. (The oracle's vectorized
  check initially "failed" — a numpy `uint8` wraparound in `bits*2-1`, not a kernel bug.)
- **M2 — qwen35 tokenizer** (`cbits/tokparse.c`, `src/Tokenizer.hs`, `src/TestTok.hs`): the
  C tables (sorted vocab / merge-rank / decode) now build straight from the GGUF metadata
  arrays — no JSON anywhere; decode keeps control/user-defined tokens literal. The Haskell
  pre-tokenizer implements the qwen35 regex faithfully (all seven alternation branches,
  real Unicode classes via MicroHs `Data.Char`), and BPE drops the `ignore_merges` shortcut
  (qwen35 leaves it off, unlike llama3). All 27 corpus cases in `test/tok-cases.txt` —
  CJK, emoji + skin tones, Icelandic, contractions, whitespace/newline runs — match the
  fork's `llama-tokenize --ids` exactly (`test/oracle_m2.py`), round-trips included.
- **M3 — full-attention block** (`src/Model.hs`, `src/M3.hs`; new kernels
  `bnrmsnorm_heads`/`bnsplit_qgate`/`bnsigmoid_mul`/`bnsilu_mul`): fused Q+gate projection
  (interleaved 256+256 per head), per-head QK RMSNorm, partial neox RoPE (64 of 256 dims,
  θ=1e7), GQA 24/4 attention, sigmoid output gate, SwiGLU FFN, residual skeleton.
  **R1 resolved**: `test/rope_oracle.c` calls the fork's actual `ggml_rope_multi`
  (IMROPE, sections [11,11,10,0]) with equal text positions — max diff vs plain neox
  2.4e-7, so `bnrope` is exact. Layer-3 block on chained synthetic inputs matches
  `test/oracle_m3.py` (f64 numpy modeling the quantized path) to ~1e-4 at pos0,
  ~1e-3 relative after three chained positions — the expected non-bit-exact-oracle
  envelope; bit-exact external validation lands at M5 (fork layer sums / greedy parity).
- **M4 — gated-delta-net block** (`gdn_conv_silu`/`gdn_l2norm_heads`/`gdn_step` in C,
  `GdnL`/`gdnLayerStep` in `Model.hs`, `src/M4.hs`): qkv/z/β/α projections, causal 4-tap
  depthwise conv ring + SiLU, per-head L2-normed q/k (16 heads tiled onto 48 v-heads via
  `h mod 16`), the non-KDA delta rule (`S ← S·eᵍ + k⊗((v − Sᵀk)·β)`, out `Sᵀq/√128`) with
  β/g (sigmoid / `a·softplus(α+dt)`) computed inside the C step, gated per-head RMSNorm ×
  `silu(z)`, `ssm_out`. **R2/R3 resolved from the fork source**: ssm_conv kernel coeff r =
  window position r (oldest first, per-channel contiguous); l2_norm = `x/max(√Σx², eps)`
  (double accumulator); softplus cutoff at 20. Layer-0 outputs match `test/oracle_m4.py`
  to ~3e-6 at pos0, ~1e-3 after three chained positions.
- **M5 — end-to-end generation** (`src/Forward.hs`, `src/Chat.hs`, `src/Sample.hs` restored;
  `b1argmax`/`b1topk` kernels): the full 64-block hybrid forward with pure state threading,
  Q1_0 embedding lookup and LM head, ChatML wrapper (jinja-faithful, `<think>` prefix),
  greedy + seeded temp/top-k/top-p sampling, C-side byte streaming. **Greedy parity: 32
  raw-prompt tokens byte-identical to the fork's CPU `llama-completion -no-cnv --temp 0`**
  ("…The capital of Iceland is **Reykjavík**. It is located on the southwestern coast…");
  ChatML prompt ids match the fork's conversation mode exactly (+ the `<think>` suffix its
  legacy template omits). ~1.0 s/token at 20 threads, 11.4 GB resident.
  Two MicroHs land mines found and routed around: (1) the runtime narrows ByteString
  substring offsets to a C int (`addForPtr`), so tensor slices past 2 GiB wrapped and
  crashed — weights are now `Dot.W = (file, offset)` handles resolved with 64-bit pointer
  arithmetic at the FFI boundary (upstream-report candidate); (2) `BS.putStr` re-encodes
  bytes >127 as UTF-8 (mojibake for model bytes) — streaming output goes through a C
  `fwrite` (`b1tok_write`).
- **M6 — perf pass**: `b1_row` gains an AVX2 path ported from the fork's own
  `ggml_vec_dot_q1_0_q8_0` (byte-mask bit expansion + xor-negate + `maddubs`, 8-lane FMA
  accumulation per row) with the generic-reference scalar under `#else`; last-ULP shifts vs
  the scalar oracle are expected — the numerics now match what the fork actually runs, and
  the 32-token greedy parity stays **byte-identical**. ~1.0 → **~0.48 s/token**; thread
  sweep 12/16/20 flat (memory-bound), default stays 12. README rewritten for the 1-bit
  branch (selection thesis, parity story, MicroHs land mines, new build/run).

## [Unreleased]

### Added

- Initial project setup.
- Fetched `microsoft/bitnet-b1.58-2B-4T` (packed 1.58-bit weights) into `models/` (gitignored).
- Design phase: `CONTEXT.md` glossary and `PLAN.md` (M0–M4 staged ladder and pinned model/packing
  facts).
- **M0 — safetensors loader** (`src/Safetensors.hs`, `src/M0.hs`): minimal JSON header parser,
  bf16→Float decode, and packed ternary unpack. Loads the full 1.18 GB data section resident in
  ~2 s and is verified bit-identical to a numpy oracle (`test/oracle_m0.py`) on bf16 values,
  scales, and trits (incl. a strided row). Documents MicroHs IO gotchas (`gmhs`; no
  `hSeek`/`hFileSize`; and a zero-length-`ByteString` OOM in `BS.empty`/`concat`/`readFile`
  on the MicroHs of the day, since fixed upstream by the zero-size-`malloc` guard `4b616112`).
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
- **`bndot` C kernel via FFI** (`cbits/bndot.c`, bound in `src/Dot.hs`): the hot dot runs in C,
  called by a plain `foreign import ccall` — no MicroHs runtime patch, no custom primitive (MicroHs
  is vendored at `vendor/microhs/` as clean upstream, pinned). Three dedicated functions, one per
  use: `bndot_ternary` (W1.58 ternary×int8 BitLinears), `bndot_bf16` (bf16×f32 LM head), `bndot_f32`
  (f32×f32 attention q·k); each returns a Double, scales applied caller-side (`bitLinearPrim`,
  `lmLogit`, `floatDot`). Bytestrings pass as raw pointers (`unsafeUseAsCStringLen`, no copy); float
  reads use `memcpy` (alignment-safe). Pure MicroHs caps at ~10 s/BitLinear (combinator graph
  reduction); the kernel makes it negligible. Unit-tested in `src/TestKinds.hs`; the list-op
  `dotTrits` stays the spec for the ternary case.
- **M3 — full forward** (`src/M3.hs`): embed → 30 layers → final norm → tied LM head → logits.
  Runs in **~25 s** (was ~55 min pure) and predicts **argmax token 11**, matching the HF oracle
  (`test/oracle_m3.py`). Per-layer math stays oracle-exact; ~0.6% drift on the final hidden is
  expected float-order accumulation over depth.
- **M4 — generation** (`src/M4.hs`, shared forward in `src/Forward.hs`): greedy autoregressive
  decoding with a **KV cache**. Given the chat prompt "User: Hello! Assistant:", it generates
  **"Hello! How can I assist you today?"** — matching HF greedy generation (`test/oracle_m4.py`)
  token-for-token. The cache (`decoderStep` in `src/Model.hs`, all in Haskell) processes the
  prompt once then runs only the new token each step, reusing stored keys/values — ~45 s/token
  vs ~3 min growing without it.
- **Combinator showcase** (`src/Showcase.hs`): renders a BitLinear dot as the combinators each
  trit generates — +1 = `I`, 0 = `K`, -1 = `C P` (negate via a difference pair `P x 0`). Includes
  a tiny SKI/`C`/`B`/`P` reducer that *evaluates* the generated term and confirms it equals
  `dotTrits`, plus iota-tree encodings (`i = \f. f S K`; `I = i i`, `K = i(i(i i))`). Runs on a
  real q_proj row to show the combinators a BitNet weight row generates.
- **Haskell tokenizer** (`src/Tokenizer.hs`): byte-level BPE encode + decode, no Python. The
  GPT-2 byte↔unicode table is computed; tokenizer.json is parsed for vocab + merges into sorted
  `Mhs.Array` lookups (no `Data.Map`), UTF-8 encode/decode via MicroHs's `packUTF8`/`primBSfromUTF8`.
  Encode = hand-written Llama 3 pre-tokenizer +
  byte-level + BPE (`ignore_merges`, lazy merges); the chat template wraps it. `encodeChat
  "Hello!"` → the exact Python ids, and decode round-trips. M4 now takes a prompt *string* and
  prints generated *text* — the whole pipeline (tokenize → model → generate → detokenize) is pure
  MicroHaskell. `test/toktool.py` remains only as the cross-check oracle.
- **Review & simplify pass** (reconciled with a Codex review): removed dead code from
  `BitLinear.hs` (`bitLinear`/`bitLinearFast`/`showcaseRow`, superseded by the `bndot`
  primitive — `Trit`/`dotTrits`/`rowTrits`/`dotRow` stay: the first three are the spec, and
  `dotRow` (the byte-level list-op workhorse) is now a live path the showcase checks the
  combinator rendering against — the spec, the reduced combinators, and `dotRow` over the raw
  packed bytes all agree on a real q_proj row); deduped
  `mkLin`/`mkNorm`/`look` (M2 now imports them from `Forward`); made the KV cache O(1) per token
  (prepend not `++`, attention is order-independent); fused `rmsNorm` into one strict pass and
  precomputed RoPE inverse frequencies; fixed an M4 off-by-one (could emit `nGen+1` tokens) and a
  `utf8enc` 4-byte gap. All outputs unchanged (M2 oracle, M3 token 11, tokenizer round-trip).
- **FFI fast path replaces the runtime primitive** (driven by a PR review): the in-runtime `bndot`
  opcode had two blockers — a `WANT_FLOAT64 0` compile break on five targets and an un-wired
  interpreter (`Translate.hs`) path — and didn't fit MicroHs's minimal/microcontroller framing. All
  three are artifacts of baking a kernel into core MicroHs; moving it to `cbits/bndot.c` behind a
  plain `foreign import ccall` dissolves them, leaving stock MicroHs untouched. The vendored fork is
  reverted to clean upstream (pinned, reproducible) and a `Makefile` captures the build. The kernel
  is trimmed to the three dots BitNet uses (ternary/bf16/f32); the int4/int8/fp16/fp8 modes existed
  only for a general upstream PR, now abandoned. The tokenizer drops its hand-rolled UTF-8 codecs for
  MicroHs's `packUTF8`/`primBSfromUTF8`. All outputs identical (M2 oracle, M3 token 11, M4 text,
  `TestKinds`).
- **Kernel performance: batch, vectorize, thread, and move the per-token math into C** — generation
  6:10 → **0:37** (~10×), bit-identical text. Three stages:
  1. *Batched dots* — one FFI call does a whole BitLinear matmul (`bnmatmul_ternary`) or the whole
     128 K-vocab LM-head argmax (`bnargmax_bf16`), instead of one call per row/token (~800 K → ~210
     FFI crossings per token).
  2. *SIMD + threads* — the ternary inner loop is AVX2 (the `sign_epi8` trick; bit-exact, it's
     integer), and the matmul/argmax/attention loops are OpenMP-parallel via a new `-tunix_omp`
     build target (`-march=native -fopenmp`). Thread count is `BNDOT_THREADS` (default 12).
  3. *Per-token math in C* — the M4 decoder step runs entirely on float32 byte buffers through
     `bnrmsnorm`, `bnbitlinear` (fused absmax-quantize + ternary matmul + rescale), `bnrope`,
     `bnattend`, `bnsqrelu_mul`, `bnadd`, so RMSNorm/RoPE/attention/MLP/residuals leave the
     combinator runtime (`src/Model.hs` `decoderStepC`, `src/Forward.hs` `stepLayersC`, threading
     `ByteString` buffers). This was the big win (batched 4:49 → C-math 0:37).
  Each kernel is cross-checked against its Haskell twin in `TestKinds` (rmsnorm/bitlinear/add/sqrelu
  bit-identical, rope ~1e-7). M1/M2/M3 keep the Haskell batch path, so the oracle checks are
  untouched; M4 generates the identical text.
- **Tokenizer load in C** (27 s → 0.04 s, ~680×): once the math was fast, the tokenizer parse was
  the whole startup. `cbits/tokparse.c` scans tokenizer.json once and builds the vocab/merge/decode
  tables (sorted for binary search) in C, replacing the Haskell parse that turned the 16 MB file into
  a 16-million-char `String` and `sortBy`'d ~400 K entries in the combinator runtime. `Tokenizer.hs`
  keeps the pre-tokenizer, byte-level mapping and BPE merge loop in Haskell and calls C only for the
  three table ops — vocab lookup, merge-rank lookup, id→bytes decode (keys are the canonical UTF-8 of
  a byte-level string, matching the C tables, so results are identical). `encodeChat "Hello!"` still
  gives the exact Python ids and decode round-trips. Full M4 (encode → 30-layer generation → decode)
  is now **0:15** — down from 6:10 at the start of the performance work (~25×).
- **Facts + polish pass** (M4 ~15 s → **~4 s**, ~16 GB → **~1.2–2 GB**): phase timings in M4 (which
  exposed that "prefill" was really lazy norm-weight decode in the combinator runtime → a
  `bf16_to_f32` C kernel cut weight setup 11.6 s → 0.9 s); a direct f32 argmax (no final-norm
  round-trip); CLI args (`m4 PROMPT NGEN`); stop-token stripping from displayed text. The old
  `-H1G` (~16 GB RSS) was a Haskell-graph relic — M4 now runs on the **default heap** in ~1.2–2 GB.
  C cleanups: `bnrope` cached its constant `inv_freq` (was recomputed every call), `bnbitlinear`/
  `bnattend` use stack scratch (was per-call malloc), and `tokparse.c` got bounds checks (the fixed
  caps could overflow on a malformed file).
- **Interactive chat** (`bin/chat`): a multi-turn REPL whose KV cache persists across turns (each
  turn prefills only the new `User: …<eot>Assistant: ` segment). The chat template is generalized
  to multi-message + roles + an optional system prompt (matching the model's `chat_template`).
  Optional **sampling** — a `bntopk_bf16` top-K kernel in C, with the policy (temperature, top-p
  over the top-K, xorshift RNG) in Haskell (`Sample.hs`); env-controlled (`BN_TEMP`/`BN_TOPP`/
  `BN_TOPK`/`BN_SEED`), greedy by default and deterministic per seed.
- **The artifact**: a `README.md` telling the story (thesis → oracle ladder → the 6:10 → ~4 s
  journey → the meaning-in-Haskell/execution-in-C split), and an expanded `Showcase` that tallies
  every ternary weight of the real model (2.08 B: 42.2% drop, 28.8% each ±1) and estimates the
  ~98 billion-node iota tree the whole model would expand to — a computed witness, not a built tree.
