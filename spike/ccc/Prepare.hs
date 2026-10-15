module Prepare where

import Control.Monad (unless)
import qualified Data.ByteString as BS
import Loader (loadModel, slice)
import Safetensors (Tensor (..), bf16At, lookupTensor)

-- | Extract two packed matrices and their model weight scales.
main :: IO ()
main = do
    (tensors, bytes) <- loadModel "models/bitnet-b1.58-2B-4T/model.safetensors"
    let dump name key =
            case (lookupTensor key tensors, lookupTensor (key ++ "_scale") tensors) of
                (Just tensor, Just scaleTensor) -> do
                    let scaleBytes = slice bytes scaleTensor
                    unless (tDtype scaleTensor == "BF16" && BS.length scaleBytes == 2) $
                        fail ("Expected one BF16 weight scale: " ++ key ++ "_scale")
                    let weightScale = bf16At scaleBytes 0
                    unless (weightScale > 0 && not (isInfinite weightScale)) $
                        fail ("Expected a finite positive weight scale: " ++ key ++ "_scale")
                    BS.writeFile ("bin/ccc-spike/cases/" ++ name ++ ".weights.bin") (slice bytes tensor)
                    let [packedRows, columns] = tShape tensor
                        description = name ++ " " ++ show columns ++ " " ++ show (4 * packedRows)
                    return (description, description ++ " " ++ show weightScale)
                _ -> fail ("Missing weight or weight scale: " ++ key)
    rows <-
        sequence
            [ dump "q-proj" "model.layers.0.self_attn.q_proj.weight"
            , dump "down-proj" "model.layers.0.mlp.down_proj.weight"
            ]
    writeFile "bin/ccc-spike/cases/real-cases.txt" (unlines (map fst rows))
    writeFile "bin/ccc-spike/cases/real-bitlinear-cases.txt" (unlines (map snd rows))
