# Categorical GPU compiler spike

The spike demonstrates how to write numerical kernels in MicroHs and run them
on a GPU. It compiles selected functions from `Programs.hs` and `LayerPrograms.hs`
through the upstream MicroHs parser and type checker. It translates the checked scalar expressions
to typed categorical operations. It generates Metal and WGSL kernels from those graphs.
The model weights remain packed input data.

The target is a direct process: write numerical functions in MicroHs, compile
their supported expressions, and execute the generated kernels. The experiment
does not compile arbitrary Haskell to a GPU. It does not infer a complete array
program from list code. The full-network accuracy comparisons are diagnostic;
the former 1% error target is not met.

- [Compilation process](#compilation-process)
- [Build and run](#build-and-run)
- [WASM and WebGPU](#wasm-and-webgpu)
- [Generated BitLinear](#generated-bitlinear)
- [Complete decoder layer](#complete-decoder-layer)
- [Full-network demonstration and results](#full-network-demonstration)
- [Compiler boundary](#compiler-boundary)
- [Validation](#validation)
- [Measurements](#measurements)
- [Source map and modification workflow](#source-map-and-modification-workflow)
- [Troubleshooting](#troubleshooting)

## Compilation process

### 1. Keep the numerical source and control program separate

`Programs.hs` defines scalar arithmetic, ternary contributions, integer addition,
activation quantization, and rescaling. `LayerPrograms.hs` defines the numerical
operations for normalization, rotary embeddings, attention, and elementwise work.
These are ordinary monomorphic Haskell functions, not strings of shader code.

`LayerProgram.hs` and `NetworkProgram.hs` define the order of tensor operations.
They use opaque tensor handles. The 30-layer loop executes in MicroHs WASM.
The handles refer to WebGPU buffers; they do not copy tensor values through WASM.

```text
Build time: GHC runs the compiler driver

  Programs.hs / LayerPrograms.hs
        │ upstream MicroHs parser and type checker
        ▼
  checked expressions with concrete numeric dictionaries
        │ Compiler.hs
        ▼
  Cat input output + explicit array plans
        │ scalar interpreters + array execution templates
        ├── Metal kernels for the native checks
        ├── WGSL kernels for WebGPU
        └── generated C evaluator for the rotary constant table

Build time: stock MicroHs compiles the control program

  WebMain.hs + WebApp.hs + WebModel.hs + WebTensor.hs
      + LayerProgram.hs + NetworkProgram.hs + WebBrowser.hs + WebFFI.hs
        │ gmhs -c
        ▼
  WebMain.c + upstream MicroHs evaluator
        │ Emscripten
        ▼
  microhs.mjs + microhs.wasm

Run time:

  MicroHs application and tensor planning → JavaScript FFI → generic browser bridge
        → generated WGSL → physical WebGPU device
        → readback → fixture checks and network diagnostics
```

GHC is the build-time host compiler. MicroHs is the compiler and runtime for the
WASM program. This distinction matters: the browser does not run GHC, and the
categorical compiler does not run inside WASM.

### 2. Compile checked expressions to typed categorical operations

`Frontend.hs` uses the upstream parser and type checker before desugaring.
`Compiler.hs` recognizes exact primitive names and their concrete dictionaries.
`CCC.hs` represents composition, products, currying, application, and supported
numeric operations as `Cat input output`.
CCC means cartesian closed category. The structural operations represent typed
function composition, products, and functions as values. Numeric primitives
extend this representation for the supported source fragment.

For example, the source function is:

```haskell
scalar :: Float -> Float
scalar input = (input + 1) * (input - 1)
```

The generated `graphs.txt` contains:

```text
scalar = (mulFloat . <(addFloat . <id, const 1.0>), (subFloat . <id, const 1.0>)>)
```

The input is supplied to both branches through a product. The same graph can
run through the categorical CPU interpreter or generate a target expression.
`Arrays.hs` places that expression inside a bounded GPU map or packed-row
reduction. `LayerArrays.hs` and `HeadArrays.hs` provide the layer and head
execution templates.

This follows the approach in Conal Elliott's
[Compiling to categories](https://conal.net/papers/compiling-to-categories/):
typed functional code can have different categorical interpretations. This
experiment uses its own small frontend pass, not the GHC `concat` plugin.
It does not claim a formal semantics proof for arbitrary Haskell or GPU floats.

### 3. Lower before the combinator representation

The compiler reads typed source before MicroHs converts it to combinators.
It does not recover kernels from the final SKI or combinator graph. Types,
primitive identities, and argument structure are available at this earlier
point. There is no need for a linear-SKI intermediate representation here.
This is a choice of compiler boundary, not a claim that combinators cannot
represent the program.

The array operations are explicit. Only their supported numerical expressions
are derived from source. This is the current limit of the automatic translation;
it is not an Accelerate backend or an automatic optimizer for arbitrary folds.

## Build and run

### Prerequisites

The checked path uses macOS on Apple Silicon with a physical Metal GPU. The
native runner uses Objective-C and Metal frameworks. It is not a Linux runner.
Other WebGPU platforms have not been validated by these results.

| Component | Checked version | Role |
| --- | --- | --- |
| MicroHs | 0.16.10.0, `e11ae213710d54c1954a9cb83ff97530c68d2488` | Pinned compiler, frontend, and runtime |
| GHC | 9.12.2 | Builds the host compiler and `gmhs` |
| Apple Clang | 17.0.0 | Native fixtures, Metal runner, rotary constants |
| Emscripten | 6.0.12 | Compiles the MicroHs C output and evaluator to WASM |
| Node | 25.4.0 | TypeScript build and browser check driver |
| Google Chrome | 155.0.8059.40 | Physical WebGPU execution |

These are observed versions, not a tested range of supported versions. Install
Git, make, curl, Python 3 for SDK setup, the Xcode command-line tools, and GHC
before the build. The GHC
packages required by the compiler driver are listed in this directory's
`Makefile`. There is no Cabal or Nix project for the experiment.

### Set up a fresh clone

```sh
git clone --branch ccc --recurse-submodules https://github.com/Tritlo/microhsbitnet.git
cd microhsbitnet
make -C vendor/microhs bin/gmhs bin/cpphs mhs.conf
make model
```

For an existing clone, use `git submodule update --init --recursive` before
building the pinned compiler. The spike uses `vendor/microhs/bin/gmhs`, not a
separate globally installed MicroHs. Do not omit `bin/cpphs` or `mhs.conf`.

`make model` fetches both `model.safetensors` and `tokenizer.json` from
`microsoft/bitnet-b1.58-2B-4T`. The model file is about 1.18 GB. Model files,
generated fixtures, and build outputs are not versioned. The build produces
large reference buffers, and the browser keeps the model and caches resident.
Allow several GB of memory and disk space.

### Check the native kernels

```sh
make -C spike/ccc check
```

This builds the categorical compiler and native reference programs, generates
the shader sources and fixtures, and checks five Metal cases. It does not need
Node or Emscripten. The frontend negative checks run during shader generation.

The execution environment must permit GPU access. The runner fails if it cannot
open a Metal device. It does not substitute a CPU implementation.

### Set up Emscripten and browser dependencies

Install and activate Emscripten 6.0.12 with the
[official SDK](https://emscripten.org/docs/getting_started/downloads.html).
For example, with the SDK outside this repository:

```sh
git clone https://github.com/emscripten-core/emsdk.git ../emsdk
../emsdk/emsdk install 6.0.12
../emsdk/emsdk activate 6.0.12
source ../emsdk/emsdk_env.sh
emcc --version
```

Activate the SDK in each shell that builds WASM. Set `EMCC` when the compiler
is not on `PATH`. Install the locked Node dependencies once, then run:

```sh
cd spike/ccc/web && npm ci && cd ../../..
make -C spike/ccc web-check
```

The check uses the installed Google Chrome application on macOS. Playwright is
the browser driver; the script does not download a separate Chromium build.
Set `CHROME` to an installed executable if it is not at the default path:

```sh
CHROME="/path/to/chrome" make -C spike/ccc web-check
```

The first run also builds the six native C reference cases. Later runs reuse
the suite when its inputs have not changed. `make -C spike/ccc web` builds all
browser assets without launching Chrome. After a complete build,
`npm --prefix spike/ccc/web run check` repeats the browser check without
regenerating native fixtures. Rebuild with make after changing Haskell source.

The driver starts a loopback HTTP server, runs headless Chrome, writes
`bin/ccc-spike/web-result.json`, and closes the browser and server. It fails on
missing or software GPU adapters. This is a bounded integration demonstration,
not an interactive browser chat application or a deployed service.

## WASM and WebGPU

`WebMain.hs` compiles with stock MicroHs to WASM. It submits each job through the
upstream JavaScript FFI. The host returns immediately and schedules the GPU work.
After readback, the host calls a MicroHs export. MicroHs reads and checks every
output value. It computes the synthetic references from `Programs.hs`. It reads
the real-matrix references from the categorical CPU fixtures. The next job starts
only after MicroHs accepts the preceding result. No C numerical kernel or
Asyncify bridge is linked into this WASM module.

`WebApp.hs` owns model setup, prompt generation, EOS decisions, and diagnostics.
`WebModel.hs` parses metadata and chooses tensor byte ranges. `WebTensor.hs`
owns tensor shapes, named weights, cache positions, resource lifetimes, and
dispatch plans. `WebFixtures.hs` owns the fixture sequence and checks.

`web/bridge.ts` is a generic browser binding, compiled to JavaScript. It stores
opaque browser objects and performs HTTP, WebGPU, typed-array, and asynchronous
completion operations. It has no model names, tensor shapes, prompt loops,
EOS policy, or numerical shader bodies. `WebBrowser.hs` exposes these operations
to MicroHs and retains suspended continuations. Commands for one token are
submitted as a batch rather than as asynchronous calls for every GPU API method.
The JavaScript FFI is synchronous. `WebBrowser.hs` stores one request ticket and
its continuation before it calls JavaScript. The bridge defers completion until
WASM returns. It then calls `browserComplete` with that ticket. MicroHs rejects
stale tickets and resumes the application without Asyncify.

`WebJSON.hs` retains integral command values as `Int`. Converting each handle to
`Double` before JSON formatting added substantial WASM interpreter overhead.
Weight representation checks run on resident GPU buffers. Only their status
words return to MicroHs; the application does not scan weights through scalar
FFI calls.

`Arrays.hs` supplies the array execution templates. The supported
source expressions supply their numerical bodies. The test starts a local HTTP
server and installed Chrome. It rejects missing and software adapters. `CHROME`
can select a different Chrome executable.

`WebFFI.hs` contains all JavaScript foreign imports. The pinned compiler numbers
these imports per module, then removes duplicates by number. Imports in separate
modules can collide. The single bridge module avoids this compiler limitation.

### Reusable execution plans

MicroHs prepares four network plans before the first prompt: prefill and decode,
each with and without layer checkpoints. The MicroHs tensor program runs once
to prepare each plan. Shape, tensor-role, and operation-order checks run during preparation.
`WebTensor.hs` pools GPU buffers by allocation slot, byte size, and usage. It
shares compatible buffers across plans. It interns uniforms by representation
and contents. Activation, scratch, uniform, and readback buffers remain allocated
until the engine closes.

MicroHs chooses the plan for each position. It also selects the embedding chunk
and row, checks consecutive cache positions, and supplies six numeric arguments:
chunk index, row, cache byte offset, rotary byte offset, attention prefix length,
and position. It commits cache positions only after successful execution and
readback. No model operation, shape rule, token decision, or EOS decision is
implemented in the browser bridge.

The bridge exposes generic `plan`, `run`, and host-buffer upload commands.
A plan contains dispatch, copy, uniform-write, and clear operations. Numeric
argument references use `["arg", index]`. Binding alternatives come from
MicroHs; the bridge creates their bind groups once. It checks argument ranges
and resource lifetimes before execution. Adjacent dispatches share a compute
pass. Each run creates a fresh command encoder but reuses buffers and bind
groups. The status buffer is cleared before result checks. GPU command buffers
are submitted once, not reused after submission.

The three standalone layer fixtures reuse one plan with different input buffers
and consecutive cache positions. Full-network comparisons stream pairs of Float
values through `withFloats` in MicroHs. The comparison checks finite binary32
values before conversion and accumulates errors in ascending Double order.
It preserves every comparison and first-mismatch record without constructing
diagnostic value lists. Borrowed CPU buffers are freed on success and failure.

In an earlier fixture run on 2026-10-15, Chrome on Apple M4 reported
`apple metal-3`. All five cases passed
exact checks in JavaScript and WASM. Q and down-projection dispatch plus readback
medians were 0.500 ms and 0.800 ms. Initial buffer allocation and upload took
1.100 ms and 2.000 ms. WGSL fetch, compilation, and pipeline creation took
11.700 ms. These browser measurements use two warmups and nine samples. They
exclude fixture fetch, WASM result verification, and initial upload. Browser
timer precision limits these small intervals. They do not establish complete
model throughput.

See `PLAN.md` for the milestone status and acceptance contract.

## Generated BitLinear

`Programs.hs` also defines absolute maxima, activation quantization, and output
rescaling. The compiler translates these functions to typed graphs. `Arrays.hs`
generates two WGSL stages from them. The first stage finds the activation maximum
and quantizes the input. The second stage performs packed ternary multiplication
and rescaling. Both stages run in one command submission. Intermediate values
stay on the GPU.

The input domain contains finite float32 values. The quantizer clamps the maximum
to at least `1e-5`, scales by `127 / maximum`, and rounds to nearest with ties to
even. WGSL `round` has this tie rule. The C reference sets `FE_TONEAREST` and uses
the existing `bnbitlinear`. Both implementations reject invalid packed fields.
The integer reduction uses the same overflow bound as the raw matrix operation.
Float outputs must satisfy `abs(actual - reference) <= 2e-5 * max(1, abs(reference))`.
Quantized activations must match exactly. This is a tested contract, not a claim
of identical floating-point behavior for all inputs and devices.

Eight cases cover tiny and partial-group matrices, real Q and down weights, zero
input, values below the maximum clamp, positive and negative half-way values,
and a non-dyadic activation maximum. All eight passed in Chrome on Apple M4 on
2026-10-15. The integer binary32 backend matched all 10,129 quantized activations
and 5,184 outputs exactly. MicroHs WASM checked every output.

Before the integer binary32 backend, Q and down dispatch plus readback medians
were 0.700 ms and 0.900 ms. Serial C medians were 0.782 ms and 2.132 ms. These
historical GPU timings include reading quantized activations for validation.
They exclude upload and pipeline creation. They do not measure the current
full-model path.

## Complete decoder layer

`LayerProgram.hs` defines the decoder layer in MicroHs. It uses opaque tensor
handles for projections, normalization, RoPE, cached attention, residual addition,
and the MLP. WASM evaluates this program. The host encodes each requested operation.
Weights, activations, and the KV cache remain on the GPU between positions.

`LayerPrograms.hs` contains the numerical functions. `LayerArrays.hs` generates
their WGSL array kernels. Attention uses one GPU kernel for scores, stable
softmax, and weighted values. The numerical source is ordinary MicroHs:

```haskell
softExp :: Float -> Float -> Float
softExp score maximum = exp (score - maximum)
```

The compiler translates this function and
the source functions `plus`, `divide`, and `maximumScore` through CCC to WGSL.
The exponential uses WGSL `exp`. There is no `Softmax.hs`, WASM softmax export,
softmax callback, or intermediate score readback. The RMS and attention dot sums
retain sequential FP32 addition order.

In an earlier standalone layer run, all 7,680 layer outputs passed the
`2e-5 * max(1, abs(reference))` bound across three consecutive positions.
The inputs were real embedding rows
for tokens `128000`, `9906`, and `0`. The C and GPU paths each retained their own
KV cache. Largest absolute errors were `7.629e-6`, `6.104e-5`, and `1.526e-5`.
One-shot dispatch/readback intervals were 6.9 ms, 6.5 ms, and 6.3 ms. These are not
benchmark medians. The intervals include WASM graph evaluation and host encoding.
They exclude weight upload and pipeline creation. The C reference uses the
existing kernels with no fast math, FP contraction disabled, and no OpenMP.
The experimental attention kernel supports 2,048 cached positions. The model
supports 4,096. Extending the kernel limit remains separate work.

The build writes its outputs to `bin/ccc-spike/`:

| Output | Contents |
| --- | --- |
| `graphs.txt` | Categorical expressions for the scalar and ternary functions |
| `generated.metal`, `generated.wgsl` | Scalar and packed-matrix kernels |
| `bitlinear.wgsl` | Activation quantization and ternary projection kernels |
| `layer.wgsl`, `head.wgsl` | Decoder operations, embedding lookup, head, and argmax |
| `checks.wgsl` | GPU checks for packed fields, finite weights, and head results |
| `rotary.c`, `rotary.bin` | Generated constant evaluator and its coefficient table |
| `cases/`, `layer-cases/`, `network-cases/` | Generated independent reference inputs and outputs |
| `web/microhs.mjs`, `web/microhs.wasm` | MicroHs WASM runtime and JavaScript loader |
| `web/bridge.js` | Compiled generic browser bridge |
| `web-result.json` | Adapter, timings, fixture checks, and network diagnostics |

These files are build outputs, not source files to edit or commit.

## Full-network demonstration

The full-network experiment loads all weights by HTTP byte range and executes
the 30-layer MicroHs program in WASM. The embedding and head
use eight GPU buffers to stay below the default storage binding limit. Weights
do not enter the WASM heap. Each consumed token uses one GPU submission. Weights
and the KV cache remain resident between positions. The GPU selects every
generated token. WASM checks that the token ID is valid. The next step uses that
GPU token, not a reference token. Each case stops at its own GPU EOS token or
after ten generated tokens. The reference length does not control generation.

The independent native C fixture suite contains `hello`, `question`, `arithmetic`,
`code`, `swedish`, and `longer`. Its index is
`bin/ccc-spike/network-cases/suite.json`. Each entry contains `name`, `promptText`,
and `manifest`. Manifest file references are relative to that manifest's folder.
The browser uploads weights once for the suite and resets the KV cache between
cases. It checks every consumed position while the input token prefixes match.

The goal is a simple generated-kernel demonstration, not native-library numerical
equivalence. Full-model token equality and the 1% normalized-vector relative L2
comparison are diagnostics, not acceptance gates. Report matching generated-token
counts and the first discrepancy for each case. Continue generation with GPU
tokens after a discrepancy. Stop vector comparisons once a differing token is
consumed, because the inputs then differ. Hidden-vector, per-element, and logit
comparisons are also diagnostics. Fixture and kernel checks retain their exact
or tolerance requirements. Standalone layer checks retain the
`2e-5 * max(1, abs(reference))` bound.

`LayerPrograms.hs` defines the rotary frequency, product, cosine, and sine.
The compiler emits a native C constant evaluator from these checked graphs.
Clang uses the same paired sine/cosine library operation as the reference.
Separate GHC and WASM trigonometry can give different rounded results. The
generated evaluator runs only during the build. It writes the 1 MiB `rotary.bin`
table. The GPU copies the required 512-byte row for each position. This table
matches all 262,144 optimized C coefficients exactly on this host. No C model
kernel is linked into WASM.

### Validated application boundary

The 2026-10-15 migration passes the stock MicroHs and Emscripten build, strict
TypeScript build, all 16 browser fixture checks, and all five native Metal
checks. Separate checks reject NaN input, an invalid packed field, a NaN norm
weight, an out-of-bounds plan offset, and a released plan dependency.
No neural application logic remains in TypeScript. The generic browser bridge
has 672 source lines, including reusable command execution. The separate Node
check driver has 68 lines.

The bridge keeps downloaded weight bytes in opaque host buffers. Haskell chooses
their GPU allocations and upload commands without copying bulk weights into
WASM. BF16 norm weights use the existing GPU embedding decoder to produce FP32
norm buffers. Haskell-generated representation checks validate packed fields
and finite BF16 values on the GPU. Head-result validation also runs on the GPU.
Only metadata, scalar scales, and diagnostic readbacks need CPU inspection.

The six-prompt run records these residency counters:

| Counter | Value |
| --- | ---: |
| Decoder and final-norm buffers | 331 |
| Shared embedding/head chunks | 8 |
| Raw weight payload bytes | 1,178,562,560 |
| Resident model-buffer bytes after norm conversion | 1,179,443,200 |
| Bulk model-weight bytes copied into WASM | 0 |
| Prepared network plans | 4 |
| Network GPU submissions | 209 |
| New GPU buffers during network execution | 0 |
| New bind groups during network execution | 0 |
| New plans during network execution | 0 |
| Resident GPU bytes after plan setup and after execution | 1,505,591,060 |

Weights upload once for all six cases. The GPU retains these buffers until the
suite finishes. Cache positions reset between cases. Plan setup adds
325,098,760 bytes: the 314,572,800-byte KV cache and 10,525,960 bytes of reusable
scratch, activation, uniform, and readback buffers. Resident bytes do not change
during the 209 positions. The application releases all plan resources when the
engine closes. The total WASM copy counter is not zero: it includes JSON
responses, scalar checks, and diagnostic vectors.

### Result after the application migration

The 2026-10-15 run used Chrome 155.0.8059.40 on Apple M4. All six cases completed
with GPU-selected tokens. Scalar, packed-matrix, BitLinear, and standalone layer
checks passed their existing requirements. All 52 generated tokens are identical
to the previous TypeScript-controlled and uncached MicroHs runs. All 950
diagnostic records are also identical to the uncached MicroHs run.

| Case | Prompt tokens | Generated tokens | Tokens matching C | Execution seconds | Checked driver seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| hello | 8 | 10 | 10 | 2.67 | 5.69 |
| question | 13 | 10 | 10 | 3.36 | 6.54 |
| arithmetic | 20 | 2 | 2 | 3.15 | 6.34 |
| code | 27 | 10 | 10 | 5.66 | 9.57 |
| swedish | 18 | 10 | 5 | 4.15 | 7.57 |
| longer | 77 | 10 | 10 | 15.30 | 21.58 |

Five cases matched the complete C token sequence. `swedish` first differed at
generated token six, position 22: GPU `665`, C `27822`. It continued with its own
GPU tokens. The suite consumed 209 positions and compared 205 final normalized
vectors with matching input histories. Four later Swedish vectors were not
comparable. Six compared vectors met the former 1% bound. The maximum relative
L2 error was 9.7894%. This is not a 1% or exact-token compatibility pass.

The checked network driver took 64.06 s, including 6.75 s of shared plan setup.
The cases spent 34.29 s in token execution and 22.99 s in numerical comparisons,
logging, and remaining application control. Execution includes MicroHs argument
checks, generic bridge calls, GPU work, token checks, and readback. It excludes
the separate reference-vector comparisons. It is not a GPU-only measurement.
The table excludes loading and shared plan setup from each case interval.

Weight payload range fetch, validation, norm conversion, and GPU upload took
another 5.94 s. This interval excludes metadata parsing, scalar scale preparation,
and shader/pipeline setup. The complete application driver took 79.37 s,
including the 2.45 s fixture sequence and setup. There were no CPU softmax
roundtrips.

For the same checked workload, the uncached MicroHs runtime took 482.49 s.
Plan reuse and streaming comparisons reduce that interval by about 7.5 times.
The older TypeScript-controlled driver took 39.78 s, including its diagnostics.
The current checked driver is still about 1.6 times slower than that baseline.
Do not compare the 34.29 s execution-only value directly with a complete checked
driver interval. These are single observations on one host, not benchmark
medians or portable performance guarantees. The older 5.64 s upload measurement
included scalar scale preparation, so the upload intervals also differ.

`bin/ccc-spike/web-result.json` records the results. Its `passed` field covers
execution and fixture checks. The network fields `tokenReferenceMatched` and
`numericalBoundPassed` are both `false` in this run.

`WEBGPU_STRICT=1` selects a separate diagnostic profile. It enables Chrome's
development-only strict-math option. This is not a portable WebGPU requirement
or a production solution. The normal test does not enable this option.

## Compiler boundary

The restrictions in this section apply to numerical source lowered to GPU
kernels. The ordinary MicroHs application can use IO, recursion, local bindings,
and imported helpers. It compiles through stock MicroHs to WASM, not through
the restricted categorical frontend.

`Frontend.hs` calls the upstream parser and type checker before desugaring.
`Compiler.hs` accepts the exact qualified primitive names and resolved instance
dictionaries for the supported operations. It rejects unknown operations.
It does not erase arbitrary dictionaries and assume standard numeric behavior.

`CCC.hs` defines identity, composition, products, a terminal object, currying,
application, and scalar primitives as a typed GADT. Its CPU interpreter supports
these operations. The code interpreter eliminates categorical closures during
code generation. The selected source fragment uses products and scalar operations.
The checks also exercise categorical currying and application directly.

The source fragment accepts monomorphic `Float` arithmetic, absolute values,
minimum and maximum, square root, exponential, trigonometry, and powers. It
accepts bounded rounding, `Int32` conversion, addition, negation, equality,
integer constants, and scalar conditionals. It rejects source-level let bindings,
recursive functions, imported helpers, arbitrary closures, and effects. It resolves generated dictionary
bindings only. The explicit rejection of source closures avoids incorrect capture
when variables are shadowed.

`FloatWGSL.hs` provides compiler helpers for binary32 multiplication, division,
and square root used by the strict kernels. The head uses a separate relaxed
floating-point interpretation. Softmax uses native WGSL `exp`, without a custom
exponential approximation or lookup table. GPU floating-point evaluation is not
a promise to reproduce a CPU library's rounding; see the
[WGSL floating-point rules](https://www.w3.org/TR/WGSL/#floating-point-evaluation).
Small rounding changes can cross activation quantization thresholds. The
full-model comparisons report the resulting differences rather than hiding
them with reference tokens or a CPU fallback.

`Arrays.hs` adds an explicit packed-row map/reduce operation. Its element function
and reducer come from the checked MicroHs source. The driver constructs the array
operation. It does not infer array operations from arbitrary Haskell list folds.
The parallel reduction accepts only the checked `Int32` addition graph with zero
as its identity. The generated kernel uses 128 threads per matrix row. It combines
thread-local sums through a shared-memory reduction.

On this 64-bit host, pinned MicroHs `Int32` arithmetic raises `Overflow` outside its range. The
CPU interpreter checks every integer addition and negation against that range.
It does not use the wrapping arithmetic of GHC `Int32`.
The 32-bit WASM target can differ outside this range. GPU and WASM acceptance
checks use the same proven overflow-free domain.

GPU integer execution requires an overflow-free input domain. The driver evaluates
the source mapper for all 768 combinations of valid fields and signed int8 inputs.
It finds the largest absolute output. It multiplies this bound by the row length.
The driver rejects a matrix if this bound exceeds `maxBound :: Int32`. This proves
that every partial sum remains in range under the parallel reduction order.
The kernel does not implement GPU exceptions. The CPU reference and GPU runner
also reject the unused packed ternary field value 3.

## Validation

The stock MicroHs runtime evaluates nine scalar inputs and fifteen ternary cases.
The categorical CPU interpreter must return the same results. The Metal runner
checks the generated scalar kernel against those reference results bit for bit.

Two synthetic matrices exercise packed fields, signed activations, and a reduction
length that is not divisible by 128. Two matrices come from layer 0 of the real model:
the Q projection and the MLP down projection. All GPU integer results must match
both the categorical CPU interpreter and `bnmatmul_ternary` from `cbits/bndot.c`.

Negative checks reject an unsupported primitive, a source closure, an unproved
reduction graph, a matrix without an overflow proof, and ill-typed source. The
upstream type checker rejects the last case. The stock runtime and CPU interpreter
must both raise `Overflow` for addition beyond `maxBound` and negation of `minBound`.
The runner validates fixture sizes and reports failed GPU commands.

## Measurements

The runner performs two warmups and nine measurements per case. It alternates
the CPU and GPU execution order. It reports medians. It retains GPU buffers for
each matrix. Each measurement submits one dispatch for the complete matrix.

The C baseline uses the existing kernel compiled with Apple Clang `-O3`, without
OpenMP. GPU resident wall time includes command creation, encoding, dispatch,
commit, and completion. The runner reports output copy and validation time
separately and as a combined interval. It also reports initial buffer allocation,
input copies, shader compilation, and GPU timestamps. File reads are outside
these intervals.

These measurements cover individual matrix-vector products. They do not establish
a full-model speedup. Small operations can cost more to dispatch than to run on
the CPU. The spike does not change the existing native inference path. It does
not lower the existing double-precision accumulators or floating-point reductions
to Metal.

Observed results on 2026-10-15, Apple M4:

| Matrix, input × output | Serial C median | GPU resident median | GPU resident + output check median | Initial buffer allocation and copy |
| --- | ---: | ---: | ---: | ---: |
| Q projection, 2560 × 2560 | 0.780 ms | 0.366 ms | 0.367 ms | 0.133 ms |
| Down projection, 6912 × 2560 | 2.102 ms | 0.772 ms | 0.773 ms | 0.444 ms |

The GPU timestamp medians were 0.159 ms and 0.358 ms. This run spent 59.4 ms
compiling the library and creating the pipelines. The resident kernel intervals
exclude this cost. The Metal compiler cache can affect this interval.

All five GPU cases passed bit-exact checks. All four matrix cases also matched
the existing C kernel. The tiny matrix took about 0.15 ms to dispatch. The C
timer rounded its execution time to zero at the printed precision.

## Source map and modification workflow

| Source | Responsibility |
| --- | --- |
| `Programs.hs` | Scalar example and BitLinear numerical functions |
| `LayerPrograms.hs` | Numerical functions for decoder and head operations |
| `Frontend.hs` | Upstream MicroHs parsing, import checking, and type checking |
| `Compiler.hs` | Checked source fragment to typed categorical graphs |
| `CCC.hs` | Typed operations and CPU, Metal, C, and WGSL interpretations |
| `FloatWGSL.hs` | Shared binary32 compiler helpers |
| `Arrays.hs` | Explicit packed-row array plans and GPU reductions |
| `LayerArrays.hs`, `HeadArrays.hs` | Layer, attention, embedding, and head templates |
| `Main.hs` | Compiler driver, source parity checks, and generated outputs |
| `LayerProgram.hs`, `NetworkProgram.hs` | MicroHs tensor program and 30-layer control flow |
| `Rotary.hs` | Rotary table selection |
| `WebMain.hs`, `WebApp.hs` | MicroHs entry point, setup, prompt loop, and result diagnostics |
| `Prepare*.hs`, `bitlinear-reference.c` | Native reference and fixture generation |
| `metal-runner.m` | Physical Metal execution, checks, and timing |
| `WebJSON.hs`, `WebModel.hs` | Metadata parsing, range checks, and model tensor access |
| `WebTensor.hs` | Application tensor roles, resource planning, and cache state |
| `WebFixtures.hs` | Fixture execution, readback validation, and check results |
| `WebBrowser.hs`, `WebFFI.hs` | Generic browser operations and asynchronous continuation bridge |
| `BufferChecks.hs` | Generated GPU representation and result validation kernels |
| `web/bridge.ts` | Generic WebGPU/HTTP object bridge; no neural application logic |
| `web/check.ts` | Local range server and headless Chrome driver |

### Change an existing numerical function

1. Edit its definition in `Programs.hs` or `LayerPrograms.hs`. Keep an explicit
   supported monomorphic signature. Do not edit generated WGSL.
2. Run `make -C spike/ccc fixtures`. This compiles the checked source, compares
   the source and categorical interpretations where covered, and regenerates
   the GPU kernels and reference buffers.
3. Inspect `bin/ccc-spike/graphs.txt` and the relevant generated shader. The graph
   report currently lists the scalar and ternary functions, not every layer
   function.
4. Run `make -C spike/ccc check` for the native scalar and packed-matrix path.
   Run `make -C spike/ccc web-check` for BitLinear, the decoder, and the network.
   A change to a kernel does not automatically update the independent C meaning.
   Fixture failures can therefore expose a changed operation, not just a compiler
   error.

### Add an operation

Use an existing scalar primitive when possible. A new primitive needs explicit
frontend recognition, a typed `Cat` operation, and matching interpreters.
A new array operation also needs an explicit execution template, shape checks,
buffer bindings, and a call from the MicroHs tensor program. Merely defining an
arbitrary Haskell function does not make the compiler select or parallelize it.

Keep JavaScript foreign imports in `WebFFI.hs`. Keep numerical shader bodies in
the Haskell compiler path. Keep application ownership, shapes, generation, and
validation in MicroHs. The browser bridge must expose generic operations only;
it must not become a second implementation of the application.

## Troubleshooting

- **Missing MicroHs files or configuration:** initialize the submodule and run
  `make -C vendor/microhs bin/gmhs bin/cpphs mhs.conf`. The spike requires the
  pinned frontend as well as its compiler executable.
- **`emcc` not found:** activate the SDK in the current shell, or set `EMCC` to
  its executable. Check `emcc --version` before the build.
- **Chrome not found:** set `CHROME` to the installed executable. The default is
  `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`.
- **No GPU, software adapter, or Metal permission error:** run on a host that
  permits physical GPU access. Do not use software rendering to obtain a pass.
- **Missing fixtures or shader files:** run `make -C spike/ccc web`, not only
  `npm run build`. The TypeScript build does not generate numerical kernels,
  native references, or WASM.
- **Model HTTP range error:** use the supplied check driver. Its local server
  returns `206` responses with validated ranges. A basic static server or a
  `file://` page does not provide the same model-loading path.
- **Unsupported source expression:** check the supported fragment above. Source
  lets, recursion, imported helpers, and arbitrary closures are not lowered.
- **Network errors exceed 1%, or tokens differ:** inspect `web-result.json`.
  Execution success does not mean numerical compatibility. The current results
  intentionally retain these differences as diagnostics.

The native C inference path at the repository root remains available. Nothing
in this experiment replaces or changes its runtime behavior.
