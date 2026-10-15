module NetworkProgram where

import Control.Monad (foldM)
import LayerProgram (Tensor (..), decoderLayer, normalize, operation, weight)
import Rotary (configureRotary)

foreign export javascript networkStep :: Int -> Int -> Int -> IO Int

-- | Encode all decoder layers for one token and optionally compute the head.
networkStep :: Int -> Int -> Int -> IO Int
networkStep position token withHead = do
    configureRotary position
    embedding <- weight "model.embed_tokens.weight"
    input <- operation "embedding" embedding (Tensor 0) (Tensor 0) token 0
    hidden <- foldM (\current layer -> decoderLayer layer position current) input [0 .. 29]
    normalized <- normalize "model.norm.weight" hidden
    if withHead == 0
        then case hidden of Tensor output -> pure output
        else do
            Tensor output <- operation "head" normalized embedding (Tensor 0) 0 0
            pure output
