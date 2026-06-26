-- | The list-op showcase: a BitLinear dot rendered as the combinators it generates.
--
-- A ternary weight is a list operation on its activation: +1 = identity, 0 = drop,
-- -1 = negate. To stay purely combinatorial (no negate primitive), a number x is
-- the difference pair @P x 0@ (value = x - 0); then +1 keeps it (I), -1 swaps it
-- (the pair applied to @C P@, value 0 - x), and 0 drops it (K). The dot sums the
-- pairs and takes pos - neg. So a whole BitLinear row becomes a tree of S/K/I/C/P
-- combinators — and (the aspiration) of the single combinator iota, ι = λf. f S K.
--
-- Run it to see the combinators a real BitNet weight row generates.
module Showcase(main) where

import BitLinear (Trit(..), toTrit, dotTrits, dotRow, rowTrits)
import Safetensors (Tensor(..), lookupTensor)
import Loader (loadModel, slice)

infixl 9 :@

-- | A combinator term: the SKI basis plus C, B, the pair P, the single combinator
-- iota, application, and integer leaves (activations).
data Comb = S | K | I | C | B | P | Iota | Comb :@ Comb | Num Int

-- | Render with left-associated application (parens only on the right).
pretty :: Comb -> String
pretty = go False
  where
    go _ S = "S"; go _ K = "K"; go _ I = "I"; go _ C = "C"; go _ B = "B"
    go _ P = "P"; go _ Iota = "i"; go _ (Num n) = show n
    go paren (a :@ b) = wrap paren (go False a ++ " " ++ go True b)
    wrap True s  = "(" ++ s ++ ")"
    wrap False s = s

-- | Number of leaves (combinator-tree size).
size :: Comb -> Int
size (a :@ b) = size a + size b
size _        = 1

-- | The combinator a trit denotes: +1 = id, 0 = drop, -1 = negate (swap pair).
tritComb :: Trit -> Comb
tritComb Pos  = I
tritComb Zero = K
tritComb Neg  = C :@ P

-- The difference-pair contribution term for one (trit, activation), or Nothing for
-- a dropped (Zero) weight.
elemTerm :: Trit -> Int -> Maybe Comb
elemTerm Pos  x = Just (I :@ pair x)                   -- I (P x 0)      -> (x, 0)
elemTerm Neg  x = Just (pair x :@ (C :@ P))            -- (P x 0) (C P)  -> (0, x)
elemTerm Zero _ = Nothing                              -- dropped (K)

pair :: Int -> Comb
pair x = P :@ Num x :@ Num 0

-- ---------- a tiny combinator reducer (to *evaluate* the generated term) ----------

data Val = VF (Val -> Val) | VN Int

ap :: Val -> Val -> Val
ap (VF f) v = f v
ap (VN _) _ = error "ap: applied a number"

eval :: Comb -> Val
eval I        = VF id
eval K        = VF (\x -> VF (\_ -> x))
eval S        = VF (\f -> VF (\g -> VF (\x -> ap (ap f x) (ap g x))))
eval C        = VF (\f -> VF (\x -> VF (\y -> ap (ap f y) x)))
eval B        = VF (\f -> VF (\g -> VF (\x -> ap f (ap g x))))
eval P        = VF (\x -> VF (\y -> VF (\f -> ap (ap f x) y)))   -- P x y f = f x y
eval Iota     = VF (\f -> ap (ap f (eval S)) (eval K))           -- i f = f S K
eval (Num n)  = VN n
eval (a :@ b) = ap (eval a) (eval b)

-- A difference-pair term -> Int (pos - neg), probing the pair with K and (K I).
evalPair :: Comb -> Int
evalPair t = num (ap v (eval K)) - num (ap v (eval (K :@ I)))   -- P a b K = a, P a b (K I) = b
  where
    v = eval t
    num (VN n) = n
    num _      = error "evalPair: not a number"

-- | The dot of a trit row over int8 activations, computed by *reducing the
-- generated combinators* (should equal 'dotTrits').
dotByComb :: [Trit] -> [Int] -> Int
dotByComb ts xs = sum [ evalPair t | Just t <- zipWith elemTerm ts xs ]

-- ---------- SKI expansion and iota encoding ----------

-- Expand C, B, P to the S/K/I basis (verified against 'eval' in main).
toSKI :: Comb -> Comb
toSKI B = S :@ (K :@ S) :@ K
toSKI C = S :@ (S :@ (K :@ S) :@ (S :@ (K :@ K) :@ S)) :@ (K :@ K)
toSKI P = let c = toSKI C in toSKI B :@ c :@ (c :@ I)   -- P = B C (C I), the vireo
toSKI (a :@ b) = toSKI a :@ toSKI b
toSKI x        = x

-- Encode a pure combinator as a tree over the single combinator iota (Barker):
-- I = ii, K = i(i(ii)), S = i(i(i(ii))).
toIota :: Comb -> Comb
toIota = enc . toSKI
  where
    enc I = Iota :@ Iota
    enc K = Iota :@ (Iota :@ (Iota :@ Iota))
    enc S = Iota :@ (Iota :@ (Iota :@ (Iota :@ Iota)))
    enc (a :@ b) = enc a :@ enc b
    enc x = x

-- ---------- demo ----------

modelPath :: String
modelPath = "models/bitnet-b1.58-2B-4T/model.safetensors"

showOp :: Trit -> String
showOp Pos  = "+1  id      I"
showOp Zero = " 0  drop    K"
showOp Neg  = "-1  negate  C P"

-- @mRow@, when present, is the same dot computed by 'dotRow' walking the raw
-- packed weight bytes (only the real BitNet row has a packed buffer to walk).
demoRow :: String -> [Trit] -> [Int] -> Maybe Int -> IO ()
demoRow title ts xs mRow = do
  putStrLn ("=== " ++ title ++ " ===")
  putStrLn ("trits:        " ++ show (map tritInt ts))
  putStrLn ("activations:  " ++ show xs)
  putStrLn "each weight is a list operation -> a combinator:"
  mapM_ (\t -> putStrLn ("    " ++ showOp t)) ts
  putStrLn ("combinators:  " ++ unwords (map (pretty . tritComb) ts))
  putStrLn ("dot (dotTrits spec):    " ++ show (dotTrits ts xs))
  putStrLn ("dot (reduce combs):     " ++ show (dotByComb ts xs))
  case mRow of
    Just d  -> putStrLn ("dot (dotRow over bytes):" ++ " " ++ show d)
    Nothing -> return ()
  putStrLn ""

tritInt :: Trit -> Int
tritInt Neg = -1
tritInt Zero = 0
tritInt Pos = 1

main :: IO ()
main = do
  -- self-check the SKI expansions used for the iota encoding
  let chk name a b = putStrLn ("  " ++ name ++ ": " ++ (if a == b then "ok" else "FAIL " ++ show (a, b)))
      r t = case eval t of VN n -> n; _ -> error "not int"
  putStrLn "=== SKI expansions verified against the reducer ==="
  chk "B f g x = f(g x)" (r (toSKI B :@ I :@ I :@ Num 7)) (r (B :@ I :@ I :@ Num 7))
  chk "C f x y = f y x"  (r (toSKI C :@ K :@ Num 1 :@ Num 2)) (r (C :@ K :@ Num 1 :@ Num 2))
  chk "P x y K = x"      (r (toSKI P :@ Num 3 :@ Num 4 :@ K)) 3
  chk "P x y (K I) = y"  (r (toSKI P :@ Num 3 :@ Num 4 :@ (K :@ I))) 4
  putStrLn ""

  -- the headline mapping
  putStrLn "=== a ternary weight is a list operation ==="
  putStrLn "  [x1,x2,x3] x [-1,0,+1]  =  [-x1, x3]   (negate, drop, id)"
  putStrLn "  in combinators:  -1 = C P,  0 = K,  +1 = I   (number x = pair P x 0)"
  putStrLn ""

  -- tiny pedagogical example (no packed weight bytes, so no dotRow line)
  demoRow "tiny example  [5,7,3] x [-1,0,+1]"
          [Neg, Zero, Pos] [5, 7, 3] Nothing

  -- the negate operation rendered as an iota tree
  putStrLn "=== the operations as iota trees (i = lambda f. f S K) ==="
  putStrLn ("  +1 = I   = " ++ pretty (toIota I))
  putStrLn ("   0 = K   = " ++ pretty (toIota K))
  let negTree = toIota (C :@ P)
  putStrLn ("  -1 = C P = an iota tree of " ++ show (size negTree) ++ " nodes")
  putStrLn ("           = " ++ take 120 (pretty negTree) ++ " ...")
  putStrLn ""

  -- a real BitNet weight row
  (ts, dat) <- loadModel modelPath
  case lookupTensor "model.layers.0.self_attn.q_proj.weight" ts of
    Nothing -> putStrLn "q_proj missing"
    Just wt -> do
      let packed = slice dat wt
          [packedRows, inDim] = tShape wt
          outDim = packedRows * 4
          chunk  = outDim `div` 4
          trits6 = take 6 (rowTrits packed inDim outDim 0)   -- first 6 weights of row 0
          acts6  = [114, -40, 35, 109, -72, 2]               -- the M1 int8 activations
          -- dotRow walks length(acts6) columns of row 0 straight off the packed
          -- bytes -- the production list-op path, here over the same 6 weights.
          byBytes = dotRow packed inDim chunk 0 acts6
      demoRow "real BitNet q_proj row 0 (first 6 weights)" trits6 acts6 (Just byBytes)
