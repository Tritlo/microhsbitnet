module Rotary (configureRotary) where

import WebTensor (Engine)
import qualified WebTensor

-- | Select 64 compiled cosine and sine pairs for a position below 2048.
configureRotary :: Engine -> Int -> IO ()
configureRotary engine position
    | position < 0 || position >= 2048 = ioError (userError "Rotary position must be between 0 and 2047")
    | otherwise = WebTensor.configureRotary engine position
