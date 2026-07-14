-- | The whole 64-block hybrid forward, token-by-token: 48 gated-delta-net
-- blocks + 16 full-attention blocks (every 4th), per-layer state threaded
-- purely, all hot math in C. One 'stepModel' call advances one token and
-- returns the final residual stream (pre output-norm).
module Forward
  ( Layers
  , buildLayers
  , States
  , initStates
  , stepModel
  , look
  ) where

import qualified Data.ByteString as BS
import Gguf (Gguf, GTensor, lookupTensor)
import Model

-- | One decoder block: full attention or gated delta net.
data Layer = LA AttnL | LG GdnL
type Layers = [Layer]

-- | Per-block state: KV cache or (conv ring, delta state).
data St = SA KV | SG GdnState
type States = [St]

look :: Gguf -> String -> GTensor
look g nm = case lookupTensor nm g of
  Just t  -> t
  Nothing -> error ("missing tensor: " ++ nm)

buildLayers :: Gguf -> Cfg -> Layers
buildLayers g cfg =
  [ if isAttnLayer i then LA (loadAttnL g i) else LG (loadGdnL g i)
  | i <- [0 .. cLayers cfg - 1] ]

initStates :: Cfg -> Layers -> States
initStates cfg = map st
  where
    st (LA _) = SA Nothing
    st (LG _) = SG (gdnInit cfg)

-- | Advance every block one token at position @pos@. Strict in each block's
-- output so the state thread doesn't pile up thunks across 64 layers.
stepModel :: Cfg -> Layers -> Int -> States -> BS.ByteString -> (BS.ByteString, States)
stepModel cfg layers pos sts0 x0 = go layers sts0 x0 []
  where
    go [] [] x acc = (x, reverse acc)
    go (LA l : ls) (SA kv : ss) x acc =
      let (x', kv') = attnStep cfg l pos kv x
      in x' `seq` go ls ss x' (SA kv' : acc)
    go (LG l : ls) (SG st : ss) x acc =
      let (x', st') = gdnLayerStep cfg l st x
      in x' `seq` go ls ss x' (SG st' : acc)
    go _ _ _ _ = error "stepModel: layer/state mismatch"
