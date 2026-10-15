module NetworkProgram where

import Control.Monad (foldM)
import LayerProgram (Tensor (..), decoderLayer, normalize, operation, weight)
import Rotary (configureRotary)
import WebTensor (Engine)

-- | Encode all decoder layers for one token and optionally compute the head.
networkStep :: Engine -> Int -> Int -> Int -> IO Int
networkStep engine position token withHead = do
    configureRotary engine position
    embedding <- weight engine "model.embed_tokens.weight"
    input <- operation engine "embedding" embedding (Tensor 0) (Tensor 0) token 0
    hidden <- foldM (\current layer -> decoderLayer engine layer position current) input [0 .. 29]
    normalized <- normalize engine "model.norm.weight" hidden
    if withHead == 0
        then case hidden of Tensor output -> pure output
        else do
            Tensor output <- operation engine "head" normalized embedding (Tensor 0) 0 0
            pure output
