module LayerPrograms where

-- | Square one activation for RMS normalization.
square :: Float -> Float
square input = input * input

-- | Add two finite values.
plus :: Float -> Float -> Float
plus left right = left + right

-- | Multiply two finite values.
product :: Float -> Float -> Float
product left right = left * right

-- | Compute the inverse RMS from the sum of squares.
normalizer :: Float -> Float -> Float
normalizer total count = 1 / sqrt (total / count + 0.00001)

-- | Apply an RMS weight and inverse RMS.
weighted :: Float -> Float -> Float -> Float
weighted weight input scale = weight * (input * scale)

-- | Apply squared ReLU and the second MLP branch.
reluMul :: Float -> Float -> Float
reluMul gate up = (max 0 gate) * (max 0 gate) * up

-- | Compute one rotary embedding frequency.
frequency :: Float -> Float -> Float
frequency index headDimension = 500000 ** ((-2) * index / headDimension)

-- | Compute a cosine for the compiled rotary table.
cosine :: Float -> Float
cosine angle = cos angle

-- | Compute a sine for the compiled rotary table.
sine :: Float -> Float
sine angle = sin angle

-- | Rotate the first component with staged cosine and sine factors.
rotateFirst :: Float -> Float -> Float -> Float -> Float
rotateFirst first second cosine sine = first * cosine - second * sine

-- | Rotate the second component with staged cosine and sine factors.
rotateSecond :: Float -> Float -> Float -> Float -> Float
rotateSecond first second cosine sine = second * cosine + first * sine

-- | Compare finite logits for greedy token selection.
better :: Float -> Float -> Bool
better candidate current = candidate > current
