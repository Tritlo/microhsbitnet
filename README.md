# BitNetComb — BitNet b1.58 in MicroHaskell, where ternary weights are list operations

Run [Microsoft's BitNet b1.58 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
language model end-to-end in **[MicroHs](https://github.com/augustss/MicroHs)** — Lennart
Augustsson's combinator-graph Haskell — with no Python at runtime. Tokenizer, 30-layer
transformer, KV-cached generation, detokenizer: all MicroHaskell, accelerated by a small,
focused C core.

The conceit: BitNet's weights are **ternary** — every weight is `-1`, `0`, or `+1`. So a weight
is not a number to multiply by; it is a **list operation** on an activation:

```
+1  =  identity   (keep it)
 0  =  drop       (throw it away)
-1  =  negate     (flip its sign)
```

A dot product of a ternary row with an activation vector is therefore a *fold*, not a sum of
products:

```
[x1, x2, x3]  ·  [-1, 0, +1]   =   -x1 + x3        -- negate, drop, identity
```

No multiplications appear in the matmul — only routing each activation into a `+` or `-` bucket
(`pos - neg` at the end). `src/BitLinear.hs` is that idea as code (`dotTrits`), and
`src/Showcase.hs` renders a real weight row as the **combinators** it generates — `+1 = I`,
`0 = K`, `-1 = C·P` (negation as a difference pair) — and even as **iota trees** (`ι = λf. f S K`).
The list-op formulation is the *meaning* of the program; everything else is making it run.

## What works

```
$ ./bin/m4 "Hello!"
...
Hello! How can I assist you today?

$ ./bin/chat
user: What is 2+2?
assistant: 2 + 2 equals 4. A simple arithmetic operation, but it's a fundamental
           building block for more complex calculations...
user: And times 3?
assistant: 2 + 2 = 4, so 3 times 4 equals 12. This is a basic multiplication
           operation...
```

(The second turn reuses the first turn's KV cache — it remembers "2 + 2 = 4".)

- **`bin/m4`** — one-shot: a prompt string in, generated text out, with phase timings.
- **`bin/chat`** — an interactive multi-turn REPL; the KV cache persists across turns. Optional
  sampling (`BN_TEMP`, `BN_TOPP`, `BN_TOPK`, `BN_SEED`); greedy by default.

Both run a real 2.4-billion-parameter model in **~4 seconds** to a short reply, in **~1.2–2 GB**
of RAM, on a desktop CPU.

## The ladder (M0 → M4), validated against an oracle

The model was built in stages, each checked bit-for-bit (or to float precision) against an HF
`transformers` reference (`test/oracle_*.py`, a *test-time-only* Python dependency):

| stage | what | check |
|---|---|---|
| **M0** | safetensors loader, bf16 + packed-ternary decode | numpy oracle, bit-exact |
| **M1** | one list-op `BitLinear` (the two-bucket fold) | integer dots bit-exact |
| **M2** | one full decoder block (RMSNorm, RoPE, GQA, squared-ReLU MLP) | HF oracle, ~5–6 sig figs |
| **M3** | the full 30-layer forward + tied LM head | argmax token matches HF |
| **M4** | greedy generation with a KV cache | matches HF generation token-for-token |

The Haskell decoder (`src/Model.hs`, `src/Forward.hs`) is the **readable specification** and stays
the oracle-validated reference.

## The performance journey

A short reply went from **6 minutes** to **~4 seconds** without ever changing the output —
each step validated against the same fixed result (`Hello! → "Hello! How can I assist you today?"`):

| step | M4 wall-clock | what changed |
|---|---|---|
| pure combinator runtime | ~hours | every dot is graph reduction (~10 s per BitLinear) |
| `bndot` runtime primitive → **FFI C kernel** | 6:10 | the inner dot runs in C (`foreign import ccall`, no runtime patch) |
| batched dots + AVX2 + OpenMP | 4:49 | one call per *whole* matmul / 128 K-vocab argmax; vectorized + threaded |
| **per-token math in C** | 0:37 | RMSNorm/RoPE/attention/MLP/residuals on float32 buffers |
| **C tokenizer parser** | 0:15 | `tokenizer.json` parsed in C (27 s → 0.04 s) instead of as a 16 M-char Haskell `String` |
| norm-decode in C + heap tuning | **~0:04** | bf16→f32 weight setup in C (11.6 s → 0.9 s); runs on the default heap |

Current breakdown (`bin/m4` prints it): model load ~0.9 s · tokenizer 0.04 s · weight setup ~0.9 s
· prefill ~1.1 s · generation ~108 ms/token.

## Meaning in Haskell, execution in C

The guiding split: **MicroHaskell defines the meaning; C executes the hot
folds and vector steps.**

- **Haskell** keeps the list-op spec (`dotTrits`/`dotRow`), the oracle-validated decoder (M2/M3),
  the BPE pre-tokenizer + merge loop, the sampling *policy*, and the combinator showcase.
- **C** (`cbits/bndot.c`, `cbits/tokparse.c`, ~630 lines total) is the executor: the ternary
  matmul (AVX2 `sign_epi8` trick — bit-exact, it's integer), the bf16/f32 dots, the per-token
  vector math, the top-K head, and the `tokenizer.json` parser. It is called by plain
  `foreign import ccall` — **stock MicroHs, no runtime fork**.

About a **quarter** of the code is C; the other ~74 % — including everything that says what the
model *is* — is MicroHaskell.

## Build & run

Needs [MicroHs](https://github.com/augustss/MicroHs) (`gmhs` on `PATH` or at `~/Code/MicroHs`) and
the model files under `models/bitnet-b1.58-2B-4T/` (`model.safetensors`, `tokenizer.json`).

```
make            # build everything (Makefile selects -tunix_omp = -march=native -fopenmp)
./bin/m4 "Your prompt here" 32          # one-shot, 32 tokens
./bin/chat "You are a terse assistant"  # REPL with an optional system prompt
BN_TEMP=0.8 ./bin/chat                  # sample instead of greedy
BNDOT_THREADS=8 ./bin/m4                # set the thread count (default 12)
./bin/showcase                          # render a weight row as combinators / iota trees
make testkinds testtok                  # unit checks (kernels, tokenizer)
```

## Layout

- `src/` — the MicroHaskell: `BitLinear` (the list-op spec), `Model`/`Forward` (the decoder),
  `Tokenizer`/`Sample` (BPE + sampling policy), `Dot` (the FFI bindings), `Showcase`, the `M0`–`M4`
  drivers, `Chat`.
- `cbits/` — the C executor: `bndot.c` (math kernels), `tokparse.c` (tokenizer parser).
- `PLAN.md`, `CONTEXT.md`, `CHANGELOG.md` — the staged plan, the glossary, and the full history.
- `test/` — Python oracles, used only to validate, never at runtime.
