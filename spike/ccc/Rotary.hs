module Rotary (configureRotary) where

import WebFFI (selectRotary)

-- | Select 64 compiled cosine and sine pairs for a position below 2048.
configureRotary :: Int -> IO ()
configureRotary position
    | position < 0 || position >= 2048 = ioError (userError "Rotary position must be between 0 and 2047")
    | otherwise = selectRotary position
