# BitNetComb — 1-bit Bonsai 27B in MicroHaskell, where binary weights are subset selection

Run [PrismML's 1-bit Bonsai 27B](https://huggingface.co/prism-ml/Bonsai-27B-gguf) — a
binary re-expression of Qwen3.6-27B, 3.8 GB of weights at 1.125 bits each — end-to-end in
**[MicroHs](https://github.com/augustss/MicroHs)**, Lennart Augustsson's combinator-graph
Haskell, with no Python at runtime. GGUF loader, qwen35 tokenizer, the 64-block hybrid
decoder (48 gated-delta-net layers + 16 full-attention layers), generation: all
MicroHaskell, accelerated by a small, focused C core.

The conceit (inherited from the ternary BitNet ancestor on `main`, and sharpened): a 1-bit
weight is not a number to multiply by — it is a **Church boolean**. A weight bit names
*include* or *exclude*, and a dot product of a binary row with an activation vector is
**subset selection**:

```
w · x  =  scale · ( 2·Σ selected xᵢ  −  Σ all xᵢ )
```

The `Σ all` correction is paid once per row; each weight is pure selection — `K` or `K I`,
the two simplest combinators there are. Where the ternary encoding paid 35 combinator
leaves for every `−1` (28.8 % of the weights!), the binary model costs **~1 leaf per stored
bit**: the unfolded combinator term sits within a whisker of the model's Shannon size, and
the tokenizer overtakes the entire neural network as the biggest term. (`main` keeps the full
ternary accounting.)

## What works

```
$ ./bin/chat "What is the capital of Iceland?"
<think>
...
</think>
The capital of Iceland is **Reykjavík**. ...

$ BN_RAW=1 ./bin/chat "The capital of Iceland is" 32
<think>

</think>

The capital of Iceland is **Reykjavík**.

It is located on the southwestern coast of the island and serves as the country
```

That second output is **byte-identical** to the prism llama.cpp fork's own CPU output
(`llama-completion -no-cnv --temp 0`) for all 32 greedy tokens — same kernels' float
semantics, same tokenizer, same delta-rule recurrence, reimplemented from source.

- **`bin/chat PROMPT [NGEN]`** — single-turn ChatML with the model's `<think>` prefix
  (the template the model was trained with); `BN_RAW=1` for raw completion. Optional
  sampling (`BN_TEMP`, `BN_TOPP`, `BN_TOPK`, `BN_SEED`); greedy by default.

A **27-billion-parameter** hybrid-attention model runs at **~0.5 s/token** in ~11.4 GB of
RAM on a desktop CPU. (The fork's hand-tuned C++ does ~55 ms/token on the same machine —
a fair price for a decoder you can read.)

## The ladder (M0 → M5), validated against an oracle at every rung

| stage | what | check |
|---|---|---|
| **M0** | GGUF v3 loader (one resident 3.8 GB ByteString) | fork gguf-py + numpy, bit-exact |
| **M1** | fused Q1_0 B1Linear (q8 quant + bit-select dot) | numpy oracle, spot rows float-identical |
| **M2** | qwen35 byte-level BPE from the GGUF arrays | `llama-tokenize --ids`, 27/27 cases exact |
| **M3** | full-attention block (Q+gate, QK-norm, IMROPE, GQA, σ-gate) | ggml rope oracle (2.4e-7) + numpy |
| **M4** | gated-delta-net block (conv ring, L2 q/k, delta rule) | numpy oracle from pinned fork semantics |
| **M5** | 64-block forward + Q1_0 LM head + generation | **byte-identical greedy parity with the fork** |

The three architecture risks were resolved *from the fork's source*, not guessed:
IMROPE with equal text positions reduces exactly to neox over 64 dims
(`test/rope_oracle.c` drives the fork's own `ggml_rope_multi`); `ssm_conv` pairs kernel
coefficient *r* with window position *r*, oldest first; `l2_norm` is
`x / max(√Σx², eps)` with a double accumulator.

## Meaning in Haskell, execution in C

- **Haskell** (`src/`) holds the meaning: the GGUF header parser, the qwen35 pre-tokenizer
  regex (real Unicode classes) and BPE merge loop, the hybrid decoder skeleton
  (`Model.hs`/`Forward.hs` — pure state threading of KV caches, conv rings, and 48
  delta-rule states), the ChatML template, the sampling policy.
- **C** (`cbits/b1dot.c`, `cbits/tokparse.c`) executes the hot folds: the Q1_0 select-dot
  (an AVX2 port of the fork's own kernel, so numerics match what the fork actually runs),
  the delta-rule step (48 heads × 128×128 state), conv/norms/attention/elementwise ops on
  f32 buffers, and the vocab/merge tables. Plain `foreign import ccall` —
  **stock MicroHs, no runtime fork**.

## MicroHs land mines (found the hard way)

- The runtime narrows ByteString **substring offsets to a C int** (`addForPtr`): slicing a
  3.8 GB ByteString past 2 GiB wraps the pointer and segfaults. Weights are therefore
  `(file, offset)` handles resolved with 64-bit pointer arithmetic at the FFI boundary,
  never substring views. (Upstream-report candidate; `BS.index` is 64-bit clean.)
- **`BS.putStr` re-encodes bytes >127 as UTF-8 characters** — mojibake for bytes that are
  already UTF-8. Token streaming goes through a C `fwrite`.
- A 744 K-element lazy offset chain (skipping the tokenizer arrays) overflows the stack —
  the GGUF walker is `seq`-strict.
- No `hSeek`/`hFileSize` (known from the ternary project): the loader reads the whole file
  sequentially in bounded chunks.

## Build & run

Needs a stock [MicroHs](https://github.com/augustss/MicroHs) (`gmhs` at `~/Code/MicroHs`
or set `GMHS`). `make` bootstraps from a fresh clone: inits the pinned-MicroHs submodule
(documentary), fetches the 3.8 GB GGUF from Hugging Face if missing, builds everything.

```
make                                     # submodule + model + build
./bin/chat "Your question" 128           # ChatML + <think>, 128 tokens
BN_RAW=1 ./bin/chat "Once upon" 64       # raw completion (the parity mode)
BN_TEMP=0.8 BN_SEED=42 ./bin/chat "Hi"   # sample (top-k 20 / top-p 0.95 defaults)
BNDOT_THREADS=20 ./bin/chat ...          # kernel threads (default 12)
make m0 m1 m3 m4 testtok                 # the milestone drivers + tokenizer check
```

The oracles (`test/oracle_m*.py`, `test/rope_oracle.c`) validate against the prism
llama.cpp fork at `~/Code/llama.cpp-prism` (set `PRISM_LLAMA`) — test-time only, never at
runtime.

## Layout

- `src/` — `Gguf`/`Loader` (the file), `Dot` (FFI), `Tokenizer`/`Sample`,
  `Model`/`Forward` (the decoder), `Chat`, `M0`–`M4` milestone drivers, `TestTok`.
- `cbits/` — `b1dot.c` (kernels), `tokparse.c` (tokenizer tables + byte output).
- `docs/combinator-estimate.md` — the ternary-era accounting that the 1-bit thesis sharpens.
- `PLAN.md`, `CONTEXT.md`, `CHANGELOG.md` — the staged plan, the glossary, the history.
- `test/` — the oracles and the checked-in GGUF ground-truth dumps.
