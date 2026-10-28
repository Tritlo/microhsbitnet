# MicroHs, WASM, and WebGPU

## Goal

Write numerical code in MicroHs. Compile a supported numerical subset through a
typed categorical and array representation to WGSL. Run the control program as
WASM. Use WebGPU for model computation. Derive complete softmax array operations
from source. Keep reusable lowering rules in the compiler. Other tensor operations
still use explicit array templates and resource plans.

Keep the demonstration simple. Use the existing C implementation as an
independent comparison, not as a requirement for full-model numerical equivalence.
Generated GPU code does not imply that arbitrary Haskell programs can execute on
a GPU.

## Milestones

1. **Complete: native compiler experiment.** Compile source arithmetic and packed
   ternary contributions to Metal. Check stock MicroHs, categorical evaluation,
   C, and GPU results. Prove that integer reductions cannot overflow. Measure
   real Q and down-projection shapes.
2. **Complete: browser execution.** Generate WGSL from the same graphs. Compile
   a MicroHs control program to WASM. Let that program request WebGPU work and
   check returned values. Run scalar, short, partial-group, and real matrix cases
   on a physical GPU. Report compilation, upload, dispatch, and readback costs.
3. **Complete: generated BitLinear.** Add source-defined float quantization and
   rescaling. Compare generated kernels with the fused C operation. State the
   rounding and reduction contracts.
4. **Complete: decoder layer.** Generate normalization, RoPE, attention,
   elementwise operations, and projections. Keep intermediate tensors and the KV
   cache on the GPU. Compare one layer with the existing model implementation.
5. **Complete: complete model.** Load weights without copying them through the
   WASM heap. Handle browser buffer limits. Run prompt prefill and token decode.
   Check logits and generated tokens. Measure complete inference, not only a
   matrix operation.
6. **Complete: GPU softmax.** Compile scalar exponential, addition, division,
   and maximum functions through CCC to WGSL. This first version uses explicit
   softmax array loops. Milestone 11 replaces them with a source-derived plan.
   Use one GPU attention kernel. There is no `Softmax.hs`,
   WASM softmax export, softmax callback, or intermediate score readback.
   Submit once per consumed token. Keep weights and the KV cache resident.
7. **Complete: prompt coverage.** Compare six prompts against
   independent C references. Include arithmetic, questions, code, Unicode, and a longer input.
   Upload weights once for the suite. Reset caches between prompts. Compare
   vectors only while the consumed token prefixes match.
8. **Complete: MicroHs application ownership.** Move model
   metadata, upload planning, tensor shapes and resource lifetimes, cache state,
   generation, EOS, fixture checks, and diagnostics into MicroHs. Replace the
   application-specific TypeScript runtime with a generic browser bridge.
   Keep weight data in opaque host buffers and resident GPU allocations.
   Use generated GPU checks for packed fields and finite BF16 data. Preserve
   the numerical kernels and independent six-prompt workload. Build WASM and
   the strict TypeScript bridge. Rerun all 16 browser fixtures and six prompts.
   Verify zero bulk-weight copies into WASM and one GPU submission per position.
9. **Complete: reusable execution plans.** Prepare prefill and decode plans
   with and without diagnostic checkpoints in MicroHs. Share activation,
   scratch, uniform, and readback buffers across positions and compatible plans.
   Let the generic bridge retain bind groups and replay GPU commands with
   explicit argument slots. Keep token, chunk, plan, and cache decisions in
   MicroHs. Preserve all kernels, numerical checks, and free-running generation.
   Require zero new GPU buffer allocations and zero plan compilation during
   the six-prompt run. Include plan setup in the network driver interval.
   Stream diagnostic comparisons in MicroHs. Preserve finite checks, ascending
   Double sums, tolerances, and first-mismatch records.
10. **Complete: MicroHs replay and resource bookkeeping.** Move generic plan
    compilation, replay, resource metadata, dependency checks, and counters into
    MicroHs. Keep only browser API calls, opaque browser references, and
    asynchronous transport in JavaScript. Retain cached GPU buffers, bind groups,
    and uniform arrays. Use synchronous numeric FFI calls during replay.
    Compile pass boundaries once. Cache lifetime checks until a GPU release.
    Preserve all 16 fixtures, six prompts, 52 generated tokens, and 950 diagnostic
    records. Require zero new GPU buffers, bind groups, or plans during the 209
    positions. Compare the checked workload with the 64.06 s reuse baseline.
11. **Complete: source-derived softmax.** Compile the complete function in
    `ArrayPrograms.hs`, including local bindings, lambdas, and same-module helpers.
    Recognize the typed `SourceArray.map`, `sum`, and `maximum` operations.
    Build a shared array graph with typed scalar captures. Derive an ordered
    workgroup schedule, intermediate storage, map/sum fusion, and final map
    access from that graph. Reuse the attention score buffer when its old value
    has no later use. Keep reductions in ascending index order. Integrate the
    generated plan into the existing attention kernel without new host commands.
    Check stock MicroHs against the graph interpreter. Check six array cases
    and a source-only output change on WebGPU. Rerun the existing 16 fixtures
    and six prompts. Retain resident weights and cached GPU allocations.
    Do not extend this milestone to RMSNorm, BitLinear, or complete network
    scheduling. Do not claim arbitrary list or nested-array compilation.

## Acceptance

Each completed milestone needs an executable check and recorded results. A
browser check must reject a missing GPU. It must not use a CPU or software GPU
as a substitute. Numerical changes require explicit error bounds. Performance
claims must identify the device, workload, and measurement boundary.

The full-model demonstration must use GPU-selected tokens for later steps. WASM
must check that these token IDs are valid. Each case must stop at its own GPU EOS
token or at the independent ten-token limit. Token equality and the 1% relative L2
comparison for final normalized vectors are diagnostics, not acceptance gates.
This replaces the earlier compatibility target with a direct source-to-GPU
demonstration. It does not claim that the earlier 1% target was achieved.
Report matching generated-token counts and the first discrepancy per case.
Continue with GPU tokens after a discrepancy. Stop vector comparisons once a
differing token is consumed. Retain hidden-vector, per-element, and logit
diagnostics. Fixture and kernel checks keep their exact or tolerance requirements.
Standalone layer checks keep the `2e-5 * max(1, abs(reference))` bound.

The suite contains `hello`, `question`, `arithmetic`, `code`, `swedish`, and
`longer`. Its index is `network-cases/suite.json`; each entry contains `name`,
`promptText`, and `manifest`. File references are relative to their manifest's
folder. The suite does not establish an error bound for arbitrary prompts,
long contexts, or other devices.

## Result

The migrated 2026-10-15 Chrome/Apple M4 run completed all six cases: 209 consumed
positions and 52 GPU-selected tokens. Five cases matched C exactly. `swedish`
matched five tokens, then selected `665` instead of C's `27822` at position 22.
In total, 47 of 52 generated tokens matched. The GPU controlled all later inputs
and EOS decisions; C reference lengths did not control generation.

The suite compared 205 normalized vectors before input histories diverged.
Six met the former 1% bound. Maximum relative L2 error was 9.7894%. Execution,
fixture, and standalone layer checks passed. Full-model numerical compatibility
did not pass and remains diagnostic.

All 52 GPU-selected tokens are unchanged from the previous runtime.
The reuse-only checked suite took 64.06 s, versus 482.49 s before reuse.
The interval includes 6.75 s of plan preparation, 34.29 s of token execution,
and 22.99 s of validation and application control. Weight payload fetch,
validation, conversion, and upload took another 5.94 s, excluding metadata,
scalar scale preparation, and pipeline setup. The `hello` case took 5.69 s,
including diagnostics but excluding loading and shared plan preparation.
All 950 diagnostic records are unchanged. The full checked run still exceeds
the TypeScript-controlled baseline of 39.78 s. These are single observed runs,
not benchmark medians. See `README.md` and
`bin/ccc-spike/web-result.json` for details.

RoPE constants come from checked source graphs evaluated through generated
native C at build time. Attention, including softmax, runs in one generated GPU
kernel. No C neural kernel is linked into the WASM runtime.
The typed numerical subset, explicit array templates, bounded suite, and
2048-position limit remain experimental boundaries.

Milestone 8 passes the WASM build, strict TypeScript build, all 16 browser
fixtures, and all five native Metal checks. The model uploads once and retains
1,179,443,200 model-buffer bytes. Bulk model-weight copies into WASM are zero.
The six prompts submit exactly 209 GPU command buffers. Packed-field and NaN
norm corruption checks reject invalid weights. The generic browser bridge is
672 TypeScript lines at that milestone; the separate Node driver is 68 lines.

Milestone 9 passes all 16 browser fixtures and all six prompts. Four plans share
GPU allocations and prebuilt bind groups. The 209 positions create no new GPU
buffers, bind groups, or plans. Resident bytes remain constant during execution.
The 325,098,760-byte plan allocation includes the 314,572,800-byte KV cache.
Negative checks reject NaN input, invalid packed weights, NaN norm weights,
an out-of-bounds plan offset, and a released plan dependency. No neural
application logic moved into TypeScript.

Milestone 10 passes the WASM build, strict TypeScript build, all 16 browser
fixtures, and all six prompts. All 52 tokens and 950 diagnostic records are
unchanged. MicroHs owns generic replay, resource metadata, dependency checks,
and counters. The 290-line browser bridge contains API wrappers, opaque object
references, asynchronous transport, and display code. The network still makes
209 submissions with no new GPU buffers, bind groups, or plans. Weights remain
resident, and bulk weight copies into WASM remain zero.

On Chrome 155.0.8059.82 and Apple M4, the saved JavaScript-replay runtime took
64.86 s and MicroHs replay took 65.07 s for the checked network workload: a 0.3%
increase. Token execution took 34.40 s and 34.76 s. Complete application time
took 80.49 s and 81.79 s. The comparison excludes concurrent GPU checks and uses
the same shaders, weights, references, and prompts. These are single runs, not
a zero-overhead guarantee. Nine focused API probes pass, including invalid
ranges, released resources, missing arguments, invalid bindings, and host bounds.

Milestone 11 passes stock MicroHs/array interpretation checks, all five native
Metal checks, all 23 browser fixtures, and the six-prompt workload on 2026-10-16.
The seven new fixtures cover complete softmax and a source-only outer map that
doubles the result. The largest absolute error is 1.49e-8, below the fixture
bound of `2e-6 * max(1, abs(reference))`.

All 52 GPU tokens and 950 diagnostics match milestone 10. The checked network
interval is 65.23 s versus 65.07 s. Token execution is 34.70 s versus 34.76 s.
These are single observations on the same Chrome version, not benchmark medians.
The 209 positions create no new GPU buffers, bind groups, or plans. The generated
softmax schedule needs no extra workgroup vector or host dispatch. Full-model
numerical limits are unchanged. RMSNorm, BitLinear, and other tensor schedules
remain explicit templates.

## Showcase follow-up

RMSNorm, residual add, and the squared-ReLU gate are now derived from
`ArrayPrograms.hs` the same way softmax is. The attention and BitLinear
schedules are still templates around source expressions. The LM head dot is a
sequential binary64 interpretation of the source multiply and add, matching
`bnargmax_bf16`. Softmax calls `exp32`, checked against libm `expf`.

The showcase page omits Swedish. The reference suite still contains it.
`web-result.json` from a check run reports `demo` and `reference` separately.
The 2026-10-16 Apple figures above are the kernels from before this follow-up.
A physical WebGPU rerun is still required before replacing those figures.
