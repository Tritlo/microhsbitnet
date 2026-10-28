# Combinators are the CCC

MicroHs compiles a typed function to combinators before it runs. On this
branch the GPU kernels for the numeric core are read back out of that
combinator term. `S'`, `B`, and `C` are already the arrows of a cartesian
closed category. The translator in `Combinators.hs` is that reading, and the
WGSL backend is unchanged.

The running example is `scalar` in `Programs.hs`:

```haskell
scalar input = (input + 1) * (input - 1)
```

MicroHs emits, dictionaries and all:

```text
S' ((*) NumFloat) (C ((+) NumFloat) 1.0) (C ((-) NumFloat) 1.0)
```

which is the arrow

```text
mulFloat . <addFloat . <id, const 1.0>, subFloat . <id, const 1.0>>
```

`C (+) 1` is flip: the constant is the second argument, so the addend stays
`const 1` and the input stays `id`.

## Basis

`x` is the arrow's input. Juxtaposition is application. A known primitive such
as `(*)` is the uncurried CCC operation (`mulFloat`), not a runtime closure.

| Combinator | Equation | Arrow |
| --- | --- | --- |
| `I` | `I x = x` | `id` |
| `K v` | `K v x = v` | `const v`, the constant morphism (`v ∘ !`) |
| `A` | `A x y = y` | drop the first argument |
| `B f g` | `B f g x = f (g x)` | `f ∘ g` |
| `C f a` | `C f a x = f x a` | flip; `C (+) 1 = add ∘ ⟨id, const 1⟩` |
| `S' f g h` | `S' f g h x = f (g x) (h x)` | `f ∘ ⟨g, h⟩` |
| `S f g` | `S f g x = f x (g x)` | `f ∘ ⟨id, g⟩` when `f` is a known binary primitive, as in `square = S (*) I` |
| `B' f g h` | `B' f g h x = f g (h x)` | apply `f` to `g`, then compose with `h` |
| `C' f g h` | `C' f g h x = f (g x) h` | `h` is not applied to `x` |
| `C'B x y` | `C'B x y z w = x z (y w)` | `peak = max ∘ ⟨abs ∘ fst, abs ∘ snd⟩` |
| `Z f g` | `Z f g x = f g` | keep a partial application, drop `x` |
| `U` | `U x y = y x` | flipped application |
| `P x y` | `P x y z = z x y` | feed `x` then `y` to `z` |
| `R` | `R x y z = y z x` | permute the last three arguments |
| `O` | `O x y z w = w x y` | the last argument receives `x` and `y` |
| `K2`, `K3`, `K4` | drop 2, 3, or 4 arguments | longer `const` |
| `Y` | `Y f = f (Y f)` | not translated; a kernel has no general recursion |

`S` needs a real exponential, `apply ∘ ⟨f, g⟩`, only when `f` is an unknown
function value rather than a primitive. Squaring is `S (*) I`, and `(*)` is
`mulFloat`, so the arrow is `mul ∘ ⟨id, id⟩` with no `curry` and no `apply`
left in the graph. The translator keeps `curry` and `apply` in its own
evaluator and refuses to emit them into a kernel.

Dictionary arguments (`inst$Data.Num.Num@Primitives.Float` and the other
concrete instances) are the same names the expression compiler already
accepts. `fromInteger` of an integer literal is a constant. Anything else
with that name is rejected.

The translator checks the source signature before type erasure. A scalar
kernel must use the requested scalar types and arity. An array kernel must
take the requested number of `Vector Float` inputs and return `Vector Float`.
Each map or zip binder has a distinct identity. An inner reduction that
depends on an outer element is rejected. Independent reductions can be
captured, including element functions that return only a captured scalar.

## What the combinator term lowers

Checked by `checkCombinators` against the expression compiler, same inputs:

| Source | Combinators that show up | Result |
| --- | --- | --- |
| `scalar`, `square`, `plus`, `product`, `cosine`, `sine` | `S'`, `S`, `C`, `I`, primitives | same values, and `scalar` has the same arrow |
| `normalizer`, `frequency`, `reluMul`, `peak`, `weighted`, `rotateFirst`, `rotateSecond`, `better`, `combine` | `B`, `B'`, `C'`, `C'B`, `S'` | same values |
| `softmax` | `B`, `S'`, `map`, `sum`, `maximum`, `exp`, `/`, `-` | same vectors |
| `rmsNorm` | `S'`, `C`, `zipWith`, `map`, `sum`, `count`, `sqrt` | same vectors |
| `residual`, `gated` | `zipWith`, `S'`, `max` | same vectors |

`SourceArray.map`, `zipWith`, `sum`, `maximum`, and `count` are array
intrinsics. Their element functions are combinator terms, translated with the
table above. The layer shader, the head dot, and the rotary table are built
from these arrows. `softmax`, `rmsNorm`, `residual`, and `gated` in that
shader are the combinator graphs, not the expression-compiler graphs. The two
agree on the checked inputs.

## What does not translate

These are rejected on purpose, and the test fails if one of them starts
translating without a new reading:

- `quantize`, `rescale`, and `contribution`. They mix `round`, `fromIntegral`,
  and a Scott-encoded `if`. A Scott boolean is a function of two continuations,
  which is the exponential case above, not a `select` primitive yet.
- The packed BitLinear reduction. The trit contribution is one of those
  rejected functions, and the packed row is not a `SourceArray` vector.
- `Y`, and any variable that is not a primitive method, a concrete dictionary,
  or one of the five array intrinsics.
- A map or `zipWith` that closes over more than two scalars.
- A nested reduction that depends on an outer map or zip element.

So the claim is literal for the scalar core and for the map/fold programs
above. It is not yet literal for the integer quantizer inside BitLinear.
