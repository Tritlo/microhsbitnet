-- | Estimate the size of the WHOLE BitNet program as a self-contained pure
-- combinator term — code + weights + embeddings + tokenizer — with no primitives
-- (numbers are Church numerals). A pure calculator: every input is a measured
-- constant (provenance below); see docs/combinator-estimate.md for the full
-- accounting, assumptions, and how each number was measured.
--
-- Encoding: a ternary weight is its list-op combinator (+1 = I, 0 = K, -1 = C·P,
-- sizes from Showcase); a number n is a naive unary Church numeral
-- (|n| = 5n+2 S/K/I leaves = 23n+6 iota leaves, from the S·B successor); vocab
-- ids / merge ranks are positional (free); counts are leaves, an unshared tree.
module Estimate(main) where

-- ---- measured constants (provenance) ----

-- ternary trit histogram, from `./bin/showcase` (count_trits over all BitLinears)
nPos, nZero, nNeg :: Integer
nPos = 602187821; nZero = 879693294; nNeg = 602163685

-- per-trit list-op sizes (Showcase: size . toSKI / size . toIota of I / K / C·P)
tIski, tKski, tCski, tIio, tKio, tCio :: Integer
tIski = 1; tKski = 1; tCski = 35;  tIio = 2; tKio = 4; tCio = 155

-- bf16 tensors (embeddings + norms + scales): count and Σ of bit-patterns
nBf16, patSum :: Integer
nBf16 = 328775890; patSum = 10580259771600

-- int8 (per-row absmax) magnitudes: embeddings, then norms+scales (count, Σ|int8|)
embN, embAbs, othN, othAbs :: Integer
embN = 328335360; embAbs = 6837044095
othN = 440530;    othAbs = 26876373

-- tokenizer.json byte-level codepoints (ids/ranks positional): count, Σ
tokN, tokSum :: Integer
tokN = 831311 + 2030233; tokSum = 116600180 + 307291262

-- the program logic, from compiling Chat to a .comb (52,157 graph nodes)
codeComb, codeIo :: Integer
codeComb = 52157; codeIo = 520000        -- ~10x: each combinator -> a few S/K/I -> iota

-- ---- combinator costs ----

-- ternary weights as list ops (exact)
wSKI, wIo :: Integer
wSKI = nPos * tIski + nZero * tKski + nNeg * tCski
wIo  = nPos * tIio  + nZero * tKio  + nNeg * tCio

-- a naive unary Church numeral: c numbers summing to v -> leaves
chSKI, chIO :: Integer -> Integer -> Integer
chSKI c v = 2 * c + 5 * v
chIO  c v = 6 * c + 23 * v

-- binary numerals: ~(2*bits+2) nodes per number, independent of value
-- (int8 ~8 bits, bf16 pattern ~16 bits, codepoint ~10 bits)
binInt8, binBf16 :: Integer
binInt8 = codeComb + wSKI + (embN + othN) * (2 * 8 + 2)  + tokN * 22
binBf16 = codeComb + wSKI +  nBf16        * (2 * 16 + 2) + tokN * 22

comma :: Integer -> String
comma n = let go xs | length xs <= 3 = xs
                    | otherwise = go (take (length xs - 3) xs) ++ "," ++ drop (length xs - 3) xs
          in (if n < 0 then "-" else "") ++ go (show (abs n))

row :: String -> Integer -> Integer -> IO ()
row nm s ic = putStrLn ("  " ++ nm ++ replicate (24 - length nm) ' ' ++ pad (comma s) ++ "   " ++ comma ic)
  where pad x = replicate (16 - length x) ' ' ++ x

main :: IO ()
main = do
  putStrLn ("ternary weights : " ++ comma (nPos + nZero + nNeg)
             ++ "  (+1 " ++ comma nPos ++ ", 0 " ++ comma nZero ++ ", -1 " ++ comma nNeg ++ ")")
  putStrLn ("bf16 values     : " ++ comma nBf16 ++ "   bit-pattern sum: " ++ comma patSum)
  putStrLn ("int8 |embed| sum: " ++ comma embAbs ++ "   int8 |norm/scale| sum: " ++ comma othAbs)
  putStrLn ("tokenizer       : " ++ comma tokN ++ " codepoints summing to " ++ comma tokSum)
  putStrLn ""
  putStrLn "=== the whole program as pure combinators ===              combinators            iotas"
  putStrLn "-- int8-quantized, naive unary Church --"
  row "code (compiled)"   codeComb codeIo
  row "ternary weights"   wSKI wIo
  row "int8 embeddings"   (chSKI embN embAbs) (chIO embN embAbs)
  row "int8 norms/scales" (chSKI othN othAbs) (chIO othN othAbs)
  row "tokenizer strings" (chSKI tokN tokSum) (chIO tokN tokSum)
  row "TOTAL"  (codeComb + wSKI + chSKI embN embAbs + chSKI othN othAbs + chSKI tokN tokSum)
               (codeIo   + wIo  + chIO  embN embAbs + chIO  othN othAbs + chIO  tokN tokSum)
  putStrLn ""
  putStrLn "-- bf16 as-shipped, naive unary Church (the embeddings are floats) --"
  row "bf16 tensors"  (chSKI nBf16 patSum) (chIO nBf16 patSum)
  row "TOTAL"  (codeComb + wSKI + chSKI nBf16 patSum + chSKI tokN tokSum)
               (codeIo   + wIo  + chIO  nBf16 patSum + chIO  tokN tokSum)
  putStrLn ""
  putStrLn "-- binary numerals instead of unary (int8 ~8 bits, bf16 ~16) --"
  row "int8 TOTAL"  binInt8  0
  row "bf16 TOTAL"  binBf16  0
