# Plan — BitNet b1.58 inference in MicroHs

Run [BitNet b1.58 2B4T](models/bitnet-b1.58-2B-4T) inference in MicroHs, interpreting
ternary weights as list operations (negate / drop / id). See [CONTEXT.md](CONTEXT.md)
for vocabulary. Staged (win condition **C**):
build the elegant, correct reference, validate against an oracle, measure, then decide on speed.

## Model facts (pinned)

- Llama-style decoder: 30 layers, hidden 2560, FFN 6912, GQA 20 query / 5 KV heads
  (head_dim 128), RoPE θ=500000, RMSNorm (eps 1e-5), squared-ReLU gated FFN, tied embeddings,
  vocab 128256, max ctx 4096.
- Per layer: `input_layernorm → q/k/v_proj → RoPE → GQA → attn_sub_norm → o_proj`,
  then `post_attention_layernorm → gate/up_proj → relu²(gate)⊙up → ffn_sub_norm → down_proj`.
  The `*_sub_norm` are SubLN RMSNorms applied before each output projection.
- **BitLinear weights**: `U8`, **4 trits/byte**, 2 bits each, packed along the output dim with
  **stride = out/4** (byte `j` holds out-rows `j, j+out/4, j+2·out/4, j+3·out/4`).
  Encode: `(trit+1) ∈ {0,1,2}`. Unpack: `((b >> 2i) & 3) - 1`. One per-tensor `weight_scale` (bf16, absmean).
- **Activation quant**: per-token `scale = 127 / max(|x|)` (clamp 1e-5), `x_q = round(x·scale)` → int8.
  Matmul `Wq·xq` is integer adds/subtracts; output rescaled once by `weight_scale / scale`.
- Embeddings + norms + scales are bf16; bf16→f32 = `(bits << 16)` reinterpreted.

## Toolchain

- Compile with **`gmhs`** from this repo: `~/Code/MicroHs/bin/gmhs -isrc -i$MHS/lib Mod -obin/out`
  (the combinator `mhs` OOMs on default heap; `gmhs`, the GHC-built compiler, works).
- MicroHs has `Double`/`Float`, `Word8`, `Data.Bits` (`shiftR`, `.&.`), `ByteString`,
  `Foreign.Storable` (Word8/Word32/Float/Double), `Data.Array`, `STRef`, FFI.

## Runtime primitive (fast path — branch `runtime-ternary-dot`)

Pure MicroHs caps at ~10 s per 2560×2560 BitLinear: the combinator runtime does ~6.5M
graph reductions for one matmul, and `BS.index` + boxed arithmetic dominate. None of the
pure restructurings (4-output batching, unpacked-list bytes, multiply vs conditional) beat it.

So the hot dot is a **runtime primitive**, not FFI: MicroHs is vendored at `vendor/microhs/`
(full history, for upstream `subtree push`) and its runtime gains one opcode —
`bndot kind weights acts shift len` in `vendor/microhs/src/runtime/eval.c` — that does a
general quantized dot in C and returns a Double (`sum_c decode_w(w,c) * decode_a(a,c)`, accumulated
in double). Like ggml's per-type `vec_dot`, it is pure decode-and-accumulate — quantization
**scales/zero-points are the caller's job** (per tensor/group, outside) — and the kind is picked
once, outside the inner loop. Kinds (weight × activation): **0** W1.58 ternary×int8 (BitNet,
`shift` selects the field), **1** bf16×f32, **2** f32×f32 (attention), **3** int8×int8 (W8A8),
**4** int8×f32, **5** int4×int8 (W4A8), **6** int4×f32 (GPTQ/AWQ), **7** fp16×f32, **8** fp8
E4M3×f32. Bound from Haskell in [`src/Dot.hs`](src/Dot.hs) via the `_primitive` keyword (no
`foreign import`). The model uses 0/1/2; all nine are unit-tested in `src/TestKinds.hs`. The
list-op `dotTrits` stays the spec for kind 0.

Result: q_proj ~10 s → negligible; full decoder block (M2) ~3.5 min → 3.5 s; **full 30-layer
forward + LM head (M3) ~55 min → ~25 s**, argmax matching HF (token 11). Per-layer math stays
oracle-exact (M2 ~1e-5); the ~0.6% drift on M3's final hidden is float-order accumulation over
30 layers re-quantized to int8, within the ~1e-2 target.

**Build with the vendored runtime** (the op is compiled into the executable at link time from
`MHSDIR/src/runtime/eval.c`):

    MHSDIR=$PWD/vendor/microhs ~/Code/MicroHs/bin/gmhs -isrc <Module> -obin/<out>

## MicroHs IO constraints (verified)

- **`BS.empty` / `BS.concat` / `BS.pack []` are broken** (OOM — `empty = pack []` and the
  empty-list pack hangs). This also breaks `BS.readFile` / `hGetContents` (`concat = foldr
  append empty`). Use `openBinaryFile` + fixed-count `BS.hGet`, and combine chunks by seeding
  an accumulator with the first read and `BS.append`-ing only non-empty chunks (never `empty`).
- No `hSeek`, no `hFileSize`, no `getFileSize`. Fine: safetensors is self-describing — read the
  8-byte LE u64 header length, read that many header bytes, and the data-section size is
  `max(data_offsets.end)` from the header. Read the data section in chunks (`hGet` short-reads at
  EOF) into one resident `ByteString`; slice tensors with `BS.take`/`drop`/`index`.
- bf16→f32: `(u16 << 16)` reinterpreted via `Foreign` `poke Word32` / `peek Float`.

## Milestones

- **M0 — Loader.** Parse safetensors (8-byte header len + JSON header), read a named tensor's
  bytes, decode bf16, unpack packed ternary rows. Verify a few weights/scales against a
  reference dump.
- **M1 — One BitLinear.** list-op `dot` over int8 activations + scales. Validate **bit-exact**
  (integer path) against the oracle for a real weight + activation vector.
- **M2 — One transformer block.** q/k/v → RoPE → GQA → attn_sub_norm → o_proj → MLP. Validate
  hidden states against HF for layer 0.
- **M3 — Full stack (done-definition).** 30 layers + embeddings + tied LM head → logits for a
  prompt; match HF logits to tolerance (~1e-2 rel). Greedy argmax of final logits.
- **M4 — Generation (later gate).** KV cache + autoregressive greedy loop → text. Attempt only
  after measuring M3 speed; this is where the C→B optimization decision is made.

## Oracle

HF `transformers` (torch) in a `uv` venv, used only to dump golden per-layer activations and
logits for fixed token-ID inputs. Test-time only — never a runtime dependency.

## Memory model (avoid eval spikes)

Packed weights (~1.2 GB) sit resident in `ByteString` — a `ForeignPtr`, **off the combinator
cell-heap**, loaded once and reused across all M4 tokens (no per-token disk re-read). The
discipline that keeps eval flat: **strict per-row unpack/fold/discard** (transient working set
O(row), never O(matrix)). The other M4 growth term is the **KV cache** (~38K ints/token →
blows the 50M-cell heap near full context); keep it off-heap (`Storable` array) or cap context.

## Representation: workhorse + showcase

- **Workhorse** — strict two-bucket fold, primitive int components: the trit *routes* an
  activation to `pos`, `neg`, or nowhere (`drop = K`), value = `pos − neg` at the end. No
  negation or multiplication primitive anywhere. This is what M1–M3 run.
- **Showcase** — the same row×activation reified as renderable data (the trit sequence as
  combinator terms `I` / `C P` / `K`), evaluable to the identical number. Built only on small
  slices. Keep the op reified (don't fuse it away) so an **iota-tree** rendering of the whole
  net ("here is the LLM") stays derivable later — aspirational, not built now.

## Open / deferred

- Exact int8 clamp bound (-127 vs -128) — confirm against oracle in M1.
- bf16-vs-Double drift budget across norms/RoPE — quantify at M2/M3.
- M4 performance strategy (Storable/array hot loop, FFI) — deferred until measured.
