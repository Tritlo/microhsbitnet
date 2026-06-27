# How big is BitNet b1.58 as pure combinators?

If the whole program — code *and* all the data — were compiled to a single self-contained
combinator term with **no primitives** (numbers as Church numerals, lists/pairs as combinators),
one you could in principle reduce by hand, how many combinators / iotas is it?

`src/Estimate.hs` (`./bin/estimate`) computes this — a small self-contained calculator over
measured tensor/tokenizer statistics (each input's provenance is in the Measurements section). This
note records the encoding, the measurements, and the answer.

## Encoding

- **A ternary weight is a list operation** (the project's thesis): `+1 = I`, `0 = K`, `-1 = C·P`.
  Sizes are exact, from `Showcase` (`size . toSKI` / `size . toIota`): `I` = 1/2, `K` = 1/4,
  `C·P` = 35/155 (S·K·I leaves / iota leaves).
- **A number `n` is a naive unary Church numeral** `⌜n⌝ = λf x. fⁿ x`, built from the `S·B`
  successor: `|⌜n⌝|` ≈ `5n+2` S/K/I leaves = `23n+6` iota leaves. So `c` numbers summing to `v`
  cost `2c+5v` combinators, `6c+23v` iotas.
- **Vocab ids and merge ranks are positional** — the list order *is* the index — so they cost
  nothing. (Church-encoding them explicitly would add the merge ranks ~`Σ 0..280146` ≈ 39 B, which
  would then dominate the tokenizer.)
- Counts are **leaves**, matching `Showcase`'s `size`, as an **unshared tree** ("reduce by hand").
  Allowing sharing (a DAG) collapses repeated subterms — e.g. the weights become "2 B references to
  3 distinct trit-ops" — a wholly different, much smaller number; that is the *compiled* (`.comb`)
  reading, not the *unfolded* one.

## Measurements (from the real model)

- ternary weights: **2,084,044,800** trits (28.8 % `+1`, 42.2 % `0`, 28.8 % `−1`).
- bf16 tensors: **328,775,890** values (the embedding table is 328 M of them); sum of bit-patterns
  **10,580,259,771,600** (avg 32,180).
- int8 (per-row absmax) magnitudes: embeddings **Σ|int8| = 6,837,044,095** (avg 20.8);
  norms/scales **26,876,373**.
- tokenizer: **2,861,544** byte-level codepoints summing to **423,891,442**.
- code: the full `Chat` program (forward, GQA attention, RMSNorm, RoPE, softmax, BPE, generation,
  sampling) compiles to a **52,157**-node `.comb`.

## The answer

### int8-quantized (a hypothetical all-integer BitNet), naive unary Church

| component | combinators | iotas |
|---|---:|---:|
| code (compiled) | 52,157 | ~520,000 |
| ternary weights (list-ops) | 22,557,610,090 | 98,058,519,993 |
| int8 embeddings | 34,841,891,195 | 159,222,026,345 |
| int8 norms / scales | 135,262,925 | 620,799,759 |
| tokenizer strings | 2,125,180,298 | 9,766,672,430 |
| **TOTAL** | **≈ 59.7 billion** | **≈ 267.7 billion** |

### bf16, as shipped (the embeddings are full-precision floats)

The bf16 embedding table, Church-encoded by bit-pattern, dominates everything:

| | combinators | iotas |
|---|---:|---:|
| **TOTAL** | **≈ 52.9 trillion** | **≈ 243.5 trillion** |

### binary numerals instead of unary (each number ≈ `2·bits` nodes)

| | combinators |
|---|---:|
| int8 TOTAL | ≈ 28.5 billion |
| bf16 TOTAL | ≈ 33.8 billion |

## Takeaways

- **The program is ~99.9999 % data.** The entire *algorithm* — what makes it a language model rather
  than a lookup table — is **52 K combinators**, a rounding error against the tens of billions of
  weights and embeddings.
- **The floats are the cost.** As shipped it is ~53 *trillion* combinators, essentially all of it the
  bf16 embedding table — precision the combinator view must encode but that carries little
  information. Quantizing to int8 drops it ~900× to ~60 billion.
- **It hinges on the numeral encoding.** Under unary Church, the int8 embeddings (34.8 B) even
  outweigh the ternary weights (22.6 B) — 328 M numerals at ~21 each beats 2 B trits at ~11 each.
  Under binary numerals the embeddings collapse to ~5 B and the elegant **ternary list-ops dominate
  again** (~28 B total). So the honest range for the int8 model is **~28–60 billion combinators /
  ~100–270 billion iotas**, and for the as-shipped bf16 model, ~53 trillion.
