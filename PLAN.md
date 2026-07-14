# Plan — 1-bit Bonsai 27B inference in MicroHs

Run [1-bit Bonsai 27B](models/bonsai-27b/Bonsai-27B-Q1_0.gguf) (PrismML's binary re-expression of
Qwen3.6-27B) in MicroHs, interpreting the 1-bit weights as **subset selection**: a weight bit picks
*include* or *exclude* in `2·Σ_selected − Σ_all` — the binary specialization of the ternary list-op
thesis, with the expensive negate op gone.
The ternary/BitNet path is dropped on this branch. Staged like the original: correct reference
first, oracle-validated per milestone, then speed.

## Model facts (pinned — from the GGUF KVs and the prism fork source)

Ground truth: `general.architecture = qwen35`; graph = the prism llama.cpp fork
(`~/Code/llama.cpp-prism`, commit `62061f91`) `src/models/qwen35.cpp` + `delta-net-base.cpp`;
fused-op reference `ggml-cpu/ops.cpp:ggml_compute_forward_gated_delta_net_one_chunk`. GGUF dumps:
`test/bonsai-gguf-kv.txt`, `test/bonsai-gguf-tensors.txt`.

- **Shape**: 64 blocks, hidden 5120, FFN 17408 (SwiGLU: `down(silu(gate)·up)`), RMSNorm eps 1e-6,
  vocab 248320, ctx 262144 (we cap small), **untied** Q1_0 embeddings + LM head.
- **Hybrid layout**: `full_attention_interval = 4` → layer `i` is **full attention iff (i+1)%4==0**
  (16 layers: 3,7,…,63); the other 48 are **gated delta net (GDN)** linear-attention layers.
- **Block skeleton** (both kinds): `x += Attn(attn_norm(x))`; `x += FFN(post_attention_norm(x))`.
- **Full attention** (24 Q heads / 4 KV heads, head_dim 256, GQA):
  - `attn_q` [5120→12288] is **interleaved per head**: 256 query dims then 256 **gate** dims
    (stride 512/head). `attn_k`/`attn_v` [5120→1024].
  - Per-head RMSNorm on Q and K (`attn_q_norm`/`attn_k_norm`, 256 dims, eps 1e-6), **before RoPE**.
  - RoPE: `ggml_rope_multi`, sections [11,11,10,0] (pairs), n_rot 64, θ=1e7 — **only the first 64
    of 256 dims rotate**; with equal per-section positions (text-only) this reduces to standard
    neox RoPE over 64 dims (**verify vs oracle — risk R1**).
  - `softmax(QᵀK/√256)V` causal → `out · σ(gate)` → `attn_output` [6144→5120].
- **GDN layer** (dims: head 128, 16 k-heads (`ssm.group_count`), 48 v-heads
  (`ssm.time_step_rank`), key_dim 2048, value_dim = inner_size 6144, conv_dim 10240, d_conv 4):
  - Projections of the normed input: `attn_qkv` [5120→10240] (q 2048 | k 2048 | v 6144),
    `attn_gate` (z) [5120→6144], `ssm_beta` [5120→48], `ssm_alpha` [5120→48] — all Q1_0.
  - `β = σ(beta_proj)`; `g = ssm_a ⊙ softplus(alpha_proj + ssm_dt.bias)` (per v-head scalars;
    `ssm_a` is negative, so `exp(g) ∈ (0,1)` is the decay).
  - **Causal depthwise conv, kernel 4** over the 10240 qkv channels (`ssm_conv1d` [4,10240] f32,
    no bias): ring of the last 3 qkv vectors; `out[c] = Σ_{r=0..3} win[r][c]·K[r][c]`, then SiLU
    (**verify kernel orientation — risk R2**).
  - Split conv output → q,k (per-head **L2-normalize**, 128 dims, eps 1e-6 — **risk R3**:
    x/√(Σx²) vs eps placement), v. k-heads repeat 16→48 (k-head `j%16`… `j/3` — pin: v-head `j`
    uses k-head `j mod 16`? fork uses `ggml_repeat_4d` = **tile**, so v-head `j` ← k-head
    `j mod 16`; verify in oracle).
  - **Delta rule, per v-head, token-by-token** (state S: 128×128 f32, mem layout
    `s[j*128+i]`, i = k-dim, j = v-dim; zero-init):
    `S *= exp(g)`; `d[j] = (v[j] − Σᵢ S[i][j]·k[i])·β`; `S[i][j] += k[i]·d[j]`;
    `o[j] = Σᵢ S[i][j]·q[i] / √128`.
  - Output: per-head gated RMSNorm `rms(o; ssm_norm) ⊙ silu(z)` → `ssm_out` [6144→5120].
- **Q1_0 weight format** (`block_q1_0`, 18 B per 128 weights = fp16 scale `d` + 16 sign bytes,
  LSB-first): `w[j] = bit ? +d : −d`. Rows are contiguous blocks; every matmul weight, the
  embedding table and the LM head are Q1_0; norms/conv/ssm scalars are F32.
- **Activation quant**: q8_0-style — per 32 values one f32 scale `max|x|/127`, int8 quants
  (matches the fork's `vec_dot_q1_0_q8_0` pairing: `sumf += d0 · Σ_k d1_k · Σ_b ±a`).
- **Tokenizer**: byte-level BPE (`gpt2` model), pre-type `qwen35` (case-insensitive-contractions
  variant of the GPT-2 regex — see `llama-vocab.cpp:382`), 247587 merges, `add_bos = false`,
  BOS/pad 248044, EOS 248046. ChatML template with `<think>`; we hardcode the minimal ChatML
  wrapper (no Jinja).
- **Sampling defaults** (GGUF KVs): temp 1.0, top-k 20, top-p 0.95. Oracle runs are greedy.
- Out of scope: vision tower (separate mmproj file), DSpark drafter, MTP/NextN (absent from this
  GGUF — 851 tensors, no `blk.64+`), 262K context, KV-cache quantization.

## What carries over from the ternary codebase

- The **loader shape** (sequential fixed-count reads, one resident data ByteString, O(1) slices)
  — retargeted from safetensors to **GGUF v3** (magic+version, u64 counts, typed KVs, tensor
  infos, `general.alignment`-padded data section; we only need types F32=0 and Q1_0=41).
- The **C-kernel-via-FFI architecture**: batched dots + fused per-token vector
  math on f32 byte buffers, OpenMP/AVX2, scales are the caller's job. `bndot.c` is replaced by
  `b1dot.c` (Q1_0×q8 select-dot; AVX2 later via `vpsignb`), plus a `gdn_step` kernel (48×128×128
  state update is C-only territory) and small ops (silu, l2norm, softplus, sigmoid, conv4).
- The **tokenizer C core** (byte-level BPE in `tokparse.c`) — new regex pre-type, GGUF-sourced
  vocab/merges instead of tokenizer.json.
- The oracle discipline: numpy per-milestone oracles in `test/`, plus the fork's
  **`llama-eval-callback`** (CPU) for per-node activation dumps and fork `llama-cli` (greedy) for
  end-to-end token parity.

## Milestones

- **M0 — GGUF loader** (`src/Gguf.hs`, rewrite `src/Loader.hs`, `src/M0.hs`): parse header/KVs/
  tensor table, load the ~3.79 GB data section resident (94 GB RAM box), slice tensors, dequant
  spot-checks. Oracle: `test/oracle_m0.py` via the fork's `gguf-py` (values + offsets bit-exact).
- **M1 — Q1_0 BitLinear** (`cbits/b1dot.c`, `src/Dot.hs`): q8_0-style activation quant + fused
  quantize→select-dot→rescale matmul (`b1linear`), embedding-row dequant (`b1embed`). Oracle:
  numpy dequant on real tensors, ~1e-5.
- **M2 — tokenizer** (`cbits/tokparse.c`, `src/Tokenizer.hs`, `src/TestTok.hs`): qwen35 regex +
  GGUF vocab/merges/added-tokens. Oracle: fork `llama-tokenize` on a corpus incl. specials/CJK/
  emoji/whitespace runs.
- **M3 — full-attention block** (`src/Model.hs`): fused Q+gate proj, QK-norm, neox RoPE(64,1e7),
  GQA attend, σ-gate, FFN. Oracle: `llama-eval-callback` tensor dumps on a short prompt (resolves
  R1).
- **M4 — GDN block**: conv ring + delta rule (`gdn_step` in C) + gated norm. Oracle: same dumps
  (resolves R2/R3); state parity over ≥8 tokens.
- **M5 — end-to-end generate** (`src/Chat.hs`, `src/Forward.hs`): 64-block forward, Q1_0 LM head
  argmax, ChatML wrapper, greedy + seeded temp/top-k/top-p. Oracle: greedy token-for-token parity
  with fork `llama-cli -ngl 0` over ≥32 tokens.
- **M6 — perf**: AVX2 `vpsignb` path, OpenMP sweep, block-sum trick (`2·Σ_sel − Σ_all`) if it wins,
  measure tok/s + RSS. Expectation: ~13× the 2B ternary cost/token ⇒ **O(10 s)/token** initially.

Each milestone = one commit (or a few), oracle in the same commit. CHANGELOG per milestone.

## Toolchain (unchanged)

`gmhs` from `vendor/microhs` (`make` bootstraps); C via `-optc`/FFI as before; `BNDOT_THREADS`
still the thread knob. `../microhs-rust` (Rust runtime port) is a possible later target — not a
dependency of this plan.

## Risks / open questions

- **R1 rope_multi ≡ neox for text**: sections [11,11,10,0] with equal position streams should
  collapse to plain neox over 64 dims — M3 oracle proves or disproves; fallback: implement true
  mrope sections.
- **R2 conv orientation**: kernel index vs window index in `ggml_ssm_conv` — M4 oracle.
- **R3 l2_norm semantics**: ggml's eps placement — M4 oracle.
- **R4 memory**: 3.79 GB weights + ~150 MB GDN states + KV; state threading copies ~150 MB/token
  (48 pure-threaded 3.1 MB states) — acceptable vs O(10 s) tokens; make `gdn_step` write into a
  caller-provided fresh buffer (stays pure at the Haskell layer).
- **R5 26.9 B ≠ 24.8 B**: whitepaper says ~24.8 B language params; the GGUF reports 26.9 B (it
  counts both untied 1.27 B embed + head). No action, just don't be surprised.
