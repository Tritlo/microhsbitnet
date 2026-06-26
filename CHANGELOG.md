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
