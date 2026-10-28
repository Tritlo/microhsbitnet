# BitNetComb — BitNet b1.58 in MicroHaskell, where ternary weights are list operations

Run [Microsoft's BitNet b1.58 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
language model end-to-end in **[MicroHs](https://github.com/augustss/MicroHs)** — Lennart
Augustsson's combinator-graph Haskell — with no Python at runtime. Tokenizer, 30-layer
transformer, KV-cached generation, detokenizer: all MicroHaskell, accelerated by a small,
focused C core.

## For visitors

A 2B BitNet runs on the GPU in a browser tab, and the numerical program is Haskell.
MicroHs source is checked, lowered through a small cartesian closed category
(CCC) compiler to WGSL, and executed with WebGPU. The token loop is the same
MicroHs program compiled to C and then WebAssembly. The browser page only
holds WebGPU buffers and shows the text.

```text
Haskell source  →  MicroHs combinators  →  CCC / array graph  →  WGSL kernel
Haskell loop    →  MicroHs  →  C  →  WASM  →  WebGPU
```

Locally, from this branch (GHC, Emscripten, and Node on `PATH`; details are in `spike/ccc/README.md`):

```sh
make -C vendor/microhs bin/gmhs bin/cpphs mhs.conf
cd spike/ccc/web && npm ci && cd ../../..
make -C spike/ccc web-demo
cd spike/ccc/web && npm run demo
```

The showcase page is `spike/ccc/web/demo.html`. It streams five prompts and
fetches weights from `microsoft/bitnet-b1.58-2B-4T` on Hugging Face. Swedish
(`Säg hej på svenska.`) is not in that list. It remains in the reference suite
(`make -C spike/ccc web-check`), which still compares every prompt, including
Swedish, with the native C tokens.

What is source-derived: softmax, RMSNorm, residual add, and the squared-ReLU
gate. Their loops come from `ArrayPrograms.hs`, and the GPU graphs for those
loops are read back out of the combinator term MicroHs emits. `S'` is
`f ∘ ⟨g, h⟩`. The table is [spike/ccc/COMBINATORS.md](spike/ccc/COMBINATORS.md).
BitLinear's quantize, trit contribution, and rescale still go through the
expression compiler; the packed-row schedule is still a template, and the
Scott-encoded `if` in the trit contribution is not a CCC arrow yet.
Attention's score, softmax, and mix use those combinator arrows inside one
kernel template. The LM head multiplies and adds with the same combinators,
interpreted as a sequential binary64 dot in the C reference's accumulator
order. The TypeScript file is the WebGPU binding,
with no tensor schedule in it.

The 2026-10-28 Apple M4 / Chrome 156 run passes all 23 browser fixtures and
matches all 52 C-reference tokens across six prompts, including Swedish.
The checked network takes 74.86 s, versus 65.50 s for the unchanged base in
the same browser. Token execution takes 42.40 s versus 34.78 s. These are
single observed runs; the checked interval includes plan setup and diagnostics.

The maximum normalized-vector relative L2 error is 8.85%; 138/209 vectors
are within 1%. The base had 9.79% maximum error and 6/205 vectors within 1%.
The 1% target still fails. Token agreement does not establish numerical parity.
Binary64 add/multiply checks are exact across 32,768 samples. The exponential
has 107 differences from macOS `expf`, all within 1 ULP.

Context remains limited to 2048 positions because attention scores use
workgroup memory. WebGPU is required. The browser reports a missing GPU and
stops. [The detailed results](spike/ccc/README.md#physical-webgpu-review-2026-10-28)
include the initial head slowdown, its correction, and reproduction commands.

GitHub Pages is wired in `.github/workflows/ccc-pages.yml` and does not run on
push. It deploys only after a manual run and only if the repository variable
`ENABLE_PAGES` is `true`. See `spike/ccc/README.md` for the reproduction guide
and `spike/ccc/record-demo.sh` for the Mac recording steps.

The iota-tree viewer lives in [Tritlo/Combinate](https://github.com/Tritlo/Combinate),
not in this repository. A local Combinate patch adds the `KK` and `KA`
combinator tags; nothing here links that tree, so the patch is not vendored.

## MicroHs → CCC → WASM and WebGPU

This branch adds an experimental path that replaces the handwritten C neural
kernels with GPU kernels generated from MicroHs numerical source. The existing
native implementation below remains separate and unchanged.

There are two compilation paths:

```text
MicroHs numerical functions → checked source → MicroHs combinators → categorical and array graphs → WGSL
MicroHs layer and network programs → stock MicroHs → C → Emscripten → WASM
```

WASM executes the application, including model metadata, tensor planning, cache
state, generation, and diagnostics. MicroHs also owns generic resource metadata,
lifetime checks, counters, and execution-plan replay. A browser bridge exposes WebGPU and
HTTP operations through opaque handles. WebGPU executes the generated kernels. Model weights,
activations, and the KV cache remain on the GPU between operations. Softmax runs
on the GPU too. Its complete Haskell definition uses maps and ordered reductions.
The compiler derives its workgroup schedule and reuses the attention score storage.
RMSNorm, residual add, and the squared-ReLU gate now use the same source-derived
array schedule as softmax. BitLinear and attention still have explicit schedules
around source expressions. The LM head dot is sequential binary64.
There is no C neural kernel or CPU softmax fallback in the browser runtime.
The MicroHs runtime itself still comes from upstream C.

The 2026-10-16 experiment ran the complete 30-layer model across six prompts.
Five generated sequences match the native C reference exactly. Swedish diverges
after five matching tokens. The former 1% vector-error target was **not met** on that run;
numerical comparisons are diagnostics, not a compatibility claim.
The sentences below are that M4 measurement. The visitor section above is the
status of the kernels in this branch.

That implementation passed all 23 browser fixture checks and the six-prompt
rerun on Apple M4. All 52 GPU-selected tokens were unchanged from the earlier
runtime. Weights upload once and remain GPU-resident; no bulk weight bytes pass
through WASM. MicroHs prepares and replays reusable plans with persistent GPU
buffers and bind groups. The browser bridge contains no tensor schedule.
The checked suite takes 65.23 s, versus 65.07 s before source-derived softmax
on the same Chrome build. Token execution takes 34.70 s versus 34.76 s.
These are single observations. The earlier uncached MicroHs run took 482.49 s.
The full checked run still exceeds the older TypeScript-controlled 39.78 s.

The [complete process and reproduction guide](spike/ccc/README.md) explains the
categorical representation, source-to-kernel examples, pinned toolchain, build
commands, GPU execution, checks, measurements, and current limits. Start there
for the WASM/WebGPU work. [The milestone plan](spike/ccc/PLAN.md) records the
acceptance criteria. The rest of this README describes the original native path.

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

Taken to its conclusion, the showcase counts every ternary weight in the real model and reports
what the *whole* model is as combinators (computed, not materialized):

```
2,084,044,800 ternary weights   (42.2% drop, 28.8% identity, 28.8% negate)
  as pure S/K/I combinators:   22,557,610,090   (~22.6 billion)
  as iota (ι) trees:           98,058,519,993   (~98 billion)
```

A 2.4-billion-parameter language model is **~22.6 billion `S`/`K`/`I` combinators** — or, in the
single-combinator iota basis, **~98 billion `ι` nodes**. Run `./bin/showcase` to see it.

And that's just the *ternary* part. The **whole program** — the code, the bf16 **embeddings**, and
the tokenizer — compiled to one self-contained pure combinator term (no primitives; numbers as
Church numerals; `./bin/estimate`):

| component (int8-quantized) | combinators | iotas |
|---|---:|---:|
| code (forward, attention, RoPE, softmax, BPE, sampling) | 52,157 | ~0.5 M |
| ternary weights (list-ops) | 22.56 B | 98.06 B |
| int8 embeddings | 34.84 B | 159.22 B |
| int8 norms / scales | 0.14 B | 0.62 B |
| tokenizer strings | 2.13 B | 9.77 B |
| **TOTAL** | **≈ 59.7 B** | **≈ 267.7 B** |

The *code* — the entire algorithm — is **52 K** combinators, a rounding error; the program is
~99.9999 % data. As shipped (bf16 embeddings) it's ~**53 trillion** combinators; quantized to int8
it drops ~900× to ~60 billion, and the elegant ternary core finally shows through. Full accounting
and assumptions in [docs/combinator-estimate.md](docs/combinator-estimate.md).

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

Both run a real 2.4-billion-parameter model in **~3 seconds** to a short reply, in **~2 GB**
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

A short reply went from **6 minutes** to **~3 seconds** without ever changing the output —
each step validated against the same fixed result (`Hello! → "Hello! How can I assist you today?"`):

| step | M4 wall-clock | what changed |
|---|---|---|
| pure combinator runtime | ~hours | every dot is graph reduction (~10 s per BitLinear) |
| `bndot` runtime primitive → **FFI C kernel** | 6:10 | the inner dot runs in C (`foreign import ccall`, no runtime patch) |
| batched dots + AVX2 + OpenMP | 4:49 | one call per *whole* matmul / 128 K-vocab argmax; vectorized + threaded |
| **per-token math in C** | 0:37 | RMSNorm/RoPE/attention/MLP/residuals on float32 buffers |
| **C tokenizer parser** | 0:15 | `tokenizer.json` parsed in C (27 s → 0.04 s) instead of as a 16 M-char Haskell `String` |
| norm-decode in C + heap tuning | **~0:03** | bf16→f32 weight setup in C (11.6 s → 0.9 s); runs on the default heap |

Current breakdown (`bin/m4` prints it): model load ~0.8 s · tokenizer 0.03 s · layer build ~0.65 s
· prefill ~0.6 s · generation ~74 ms/token — **~3.0 s** total on MicroHs v0.16.5 (the toolchain bump
from v0.16.3 shaved ~12 % off prefill + the per-token glue; the C kernels were unchanged).

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

Needs a stock [MicroHs](https://github.com/augustss/MicroHs) (`gmhs` on `PATH` or at
`~/Code/MicroHs` — set `GMHS` to override). `make` bootstraps the rest from a fresh clone: it
inits the pinned-MicroHs submodule (`vendor/microhs/`, documentary — the build uses your installed
`gmhs`) and fetches the model weights + tokenizer (~1.2 GB, only if missing), then builds.

The current MicroHs pin is 0.16.10.0 (`e11ae213`). To use the pinned compiler, run
`make -C vendor/microhs bin/gmhs bin/cpphs mhs.conf`, then pass
`GMHS="$PWD/vendor/microhs/bin/gmhs"` to `make`.

```
make            # init submodule + fetch model (if missing) + build everything
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
- [`spike/ccc/`](spike/ccc/README.md) — a typed categorical compiler experiment with
  a MicroHs WASM program, generated Metal/WGSL kernels, and WebGPU model checks.
