# MicroHs, WASM, and WebGPU

## Goal

Write numerical code in MicroHs. Compile a supported numerical subset through a
typed categorical and array representation to WGSL. Run the control program as
WASM. Use WebGPU for model computation. Keep handwritten code limited to the
compiler, array execution templates, and resource management.

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
6. **Complete: GPU softmax.** Compile `LayerPrograms.softExp`, defined as
   `exp (score - maximum)`, plus source `plus`, `divide`, and `maximumScore`
   through CCC to WGSL. Use one GPU attention kernel. There is no `Softmax.hs`,
   WASM softmax export, softmax callback, or intermediate score readback.
   Submit once per consumed token. Keep weights and the KV cache resident.
7. **Complete: prompt coverage.** Compare six prompts against
   independent C references. Include arithmetic, questions, code, Unicode, and a longer input.
   Upload weights once for the suite. Reset caches between prompts. Compare
   vectors only while the consumed token prefixes match.

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

The final 2026-10-15 Chrome/Apple M4 run completed all six cases: 209 consumed
positions and 52 GPU-selected tokens. Five cases matched C exactly. `swedish`
matched five tokens, then selected `665` instead of C's `27822` at position 22.
In total, 47 of 52 generated tokens matched. The GPU controlled all later inputs
and EOS decisions; C reference lengths did not control generation.

The suite compared 205 normalized vectors before input histories diverged.
Six met the former 1% bound. Maximum relative L2 error was 9.7894%. Execution,
fixture, and standalone layer checks passed. Full-model numerical compatibility
did not pass and remains diagnostic.

Computation, WASM control, dispatch, readback, and validation took 39.78 s across
the suite. Model fetch, conversion, and GPU upload took another 5.64 s. The
original `hello` case took 3.06 s, excluding loading. These are single observed
runs, not benchmark medians. See `README.md` and
`bin/ccc-spike/web-result.json` for details.

RoPE constants come from checked source graphs evaluated through generated
native C at build time. Attention, including softmax, runs in one generated GPU
kernel. No C neural kernel is linked into the WASM runtime.
The typed numerical subset, explicit array templates, bounded suite, and
2048-position limit remain experimental boundaries.
