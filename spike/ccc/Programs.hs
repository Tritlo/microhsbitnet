module Programs where

import Control.Exception (ArithException (Overflow), evaluate, try)
import Data.Int (Int32)

-- | Reuse a scalar input in two arithmetic expressions.
scalar :: Float -> Float
scalar input = (input + 1) * (input - 1)

-- | Interpret one packed two-bit weight field.
contribution :: Int32 -> Int32 -> Int32
contribution field activation =
    if field == 0
        then negate activation
        else
            if field == 2
                then activation
                else 0

-- | Combine two integer partial sums.
combine :: Int32 -> Int32 -> Int32
combine left right = left + right

-- | Combine absolute activation maxima.
peak :: Float -> Float -> Float
peak left right = max (abs left) (abs right)

-- | Quantize one finite activation with round-to-nearest, ties-to-even.
quantize :: Float -> Float -> Int32
quantize input maximum = round (max (-128) (min 127 (input * (127 / max maximum 0.00001))))

-- | Restore the model weight scale and the activation scale.
rescale :: Int32 -> Float -> Float -> Float
rescale accumulator weightScale maximum = fromIntegral accumulator * weightScale / (127 / max maximum 0.00001)

-- | Run the source functions in the stock MicroHs runtime.
main :: IO ()
main = do
    print (map scalar [-32, -3.5, -1, -0.5, 0, 0.5, 1, 3.5, 32])
    print [contribution field activation | field <- [0 .. 2], activation <- [-128, -1, 0, 1, 127]]
    addition <- try (evaluate (combine 2147483647 1)) :: IO (Either ArithException Int32)
    negation <- try (evaluate (contribution 0 (-2147483647 - 1))) :: IO (Either ArithException Int32)
    print (case addition of Left Overflow -> True; _ -> False)
    print (case negation of Left Overflow -> True; _ -> False)
    print [peak (-8) 4, peak 0 0.0000001]
    print [quantize value 127 | value <- [-127, -126.5, -2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5, 126.5, 127]]
    print [rescale 1024 0.125 127, rescale (-17) 0.5 0]
