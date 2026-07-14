# `System.IO`: positioned I/O unimplemented — `hSeek` / `hFileSize` / `hTell` are undefined values

**Version:** `v0.16.5.0-19-gb1f66ad7` (current master at time of writing)

## Summary

`System.IO` exports the `SeekMode` type and `hIsSeekable`, but the positioned-I/O
operations themselves are missing. `hSeek`, `hFileSize`, and `hTell` are not defined,
so any program that uses them fails at compile time with `undefined value`. As a
result there's no way to do random access into a file — you can only read sequentially
from the current position.

## Reproduction

```haskell
import System.IO

main :: IO ()
main = do
  h  <- openBinaryFile "hello.txt" ReadMode
  sz <- hFileSize h            -- undefined value: hFileSize
  hSeek h AbsoluteSeek 6       -- undefined value: hSeek
  putStrLn (show sz)
```

```
$ gmhs seek.hs -o seek
gmhs: Uncaught exception ...ErrorCall:
"seek.hs": line 6, col 9: undefined value: hFileSize
```

Each of `hFileSize`, `hSeek`, and `hTell` fails the same way in isolation.

## Current state of the surrounding API

- `SeekMode(..)` **is** defined and exported from `System.IO`
  (`AbsoluteSeek | RelativeSeek | SeekFromEnd`) — so the type scaffolding is already there,
  it's just unused.
- `hIsSeekable` is defined but hardcoded to `return False` (`lib/System/IO.hs`), so even
  feature-detecting seekability reports "no".
- `hGetPosn` / `hSetPosn` are also absent.

## Why it matters

Reading a large binary file (in our case a ~1.2 GB safetensors model) without
`hSeek`/`hFileSize` forces a full sequential read of the whole data section into memory,
since you can't size the file up front or jump to a tensor's byte offset. `hGet` works
well and is fast, but positioned reads would let a loader map/slice individual tensors
without holding the entire file resident.

(Unrelated, and noted only so it isn't re-reported: `Data.ByteString`'s `empty`,
`concat`, and `readFile` all work correctly on this version — including a 300 MB
`readFile` in ~1 s — so whatever earlier zero-length-`malloc` "hang"/OOM existed there
is resolved, presumably by `4b616112`.)

## A candidate patch (attached: `scratch-microhs-hseek.patch`)

I put together a deliberately small implementation and verified it; happy to send a PR if
the approach looks right. It's scoped to **real file handles** — what `openFile` and
`openBinaryFile` actually produce — and reports everything else as not-seekable
*honestly* rather than guessing. Three runtime primitives, ~114 lines total:

- **`src/runtime/bfile.c`** (+66) — `seekb(BFILE*, value_t, int)`, `tellb(BFILE*)`, and
  `seekableb(BFILE*)`. Rather than add slots to the `BFILE` vtable (which would mean
  initialising them in all ~17 transducer constructors), it dispatches on the transducer
  type via `p->getb`: `getb_file` → `fseeko`/`ftello`, and the `add_utf8` layer
  (so text handles work) forwards to its wrapped file after dropping its unget. Every
  other transducer — `fd`, in-memory, buffered, crlf, compression — falls through to
  `ERR("…not seekable")`. `seekableb` is a non-disturbing probe (`ftello` to the current
  position, which fails with `ESPIPE` on pipes). Branches are `WANT_*`-guarded.
- **`src/runtime/eval.c`** (+6) — `mhs_seekb` / `mhs_tellb` / `mhs_seekableb` FFI wrappers
  + table entries.
- **`src/MicroHs/FFI.hs`** (+1) — add `"seekb"` / `"tellb"` / `"seekableb"` to
  `runtimeFFI` so the compiler doesn't also generate wrappers (otherwise: duplicate-symbol
  link error).
- **`lib/System/IO.hs`** (+43) — `foreign import`s plus `hSeek :: Handle -> SeekMode ->
  Integer -> IO ()`, `hTell :: Handle -> IO Integer`, `hFileSize :: Handle -> IO Integer`
  (derived as save-pos / seek-end / tell / restore — no third primitive needed).
  `hIsSeekable` now genuinely probes the handle, and `hSeek`/`hTell`/`hFileSize` raise a
  catchable `IllegalOperation` `IOError` (not a runtime `ERR`) on a non-seekable handle.

Offsets use `off_t` (`fseeko`/`ftello`), so they're 64-bit on LP64. Signatures match
GHC's `System.IO`.

### Verified

Built `gmhs` with the patch and exercised, all passing:

- **File handle:** `hFileSize = 12`; `hSeek AbsoluteSeek 6` then `hGetChar` → `'w'`;
  `hTell` → `7`.
- **Text (utf8-layered) handle:** all three `SeekMode`s — `Absolute`/`Relative`/
  `FromEnd` — give `"DGJ"`, `hTell` → `10`.
- **Large file:** `hFileSize`/`hSeek` on a 300 MB binary, offsets up to 314 572 799.
- **Exact `hIsSeekable` + catchable errors:** for the *same* `stdin` handle,
  `echo hi | prog` → `hIsSeekable = False` and `hSeek` throws a caught
  `Handle(stdin): hSeek: illegal operation`; `prog < file` → `hIsSeekable = True` and
  the seek succeeds.
- **No regressions:** `Data.ByteString` `empty`/`concat`/`readFile` patterns still pass,
  and our 30-layer model loader still compiles and behaves identically. Clean compile
  (no new warnings).

### Deliberately out of scope (kept exact, just not yet supported)

These are reported as not-seekable (honest `hIsSeekable = False`, catchable error), so
they can be added later without breaking anyone:

- **fd / buffered / in-memory / crlf handles** — only the `file` (+ `utf8`) path is wired
  up; that's everything `openFile`/`openBinaryFile` produce. The same `p->getb` dispatch
  extends to the others trivially when needed.
- **MSVC (LLP64) large files** — `fseeko`/`ftello` are POSIX; a Windows build would want
  `_fseeki64`/`_ftelli64`.
- **`hGetPosn`/`hSetPosn`** — would want `Eq`/`Show Handle`; easy to layer on
  `hTell`/`hSeek` later.
