{-# LANGUAGE GADTs #-}

module Arrays (ArrayPlan, packedMatVec, runArray, metalSource, wgslSource, bitLinearSource) where

import CCC
import Control.Monad (unless)
import Data.Bits (shiftR, (.&.))
import qualified Data.ByteString as BS
import Data.Int (Int32, Int8)
import Data.List (foldl')

-- | A fused map and integer sum over each packed matrix row.
data ArrayPlan = PackedMatVec (Cat (Int32, Int32) Int32) (Cat (Int32, Int32) Int32)

-- | Require the source reducer to be Int32 addition with its standard identity.
packedMatVec :: Cat (Int32, Int32) Int32 -> Cat (Int32, Int32) Int32 -> Either String ArrayPlan
packedMatVec mapper reducer = case reducer of
    Compose AddInt (Pair First Second) -> Right (PackedMatVec mapper reducer)
    _ -> Left "Parallel reduction requires the checked Int32 addition reducer"

-- | Interpret packed rows without changing the model representation.
runArray :: ArrayPlan -> Int -> Int -> BS.ByteString -> BS.ByteString -> Either String [Int32]
runArray (PackedMatVec mapper reducer) columns rows weights activations = do
    unless (columns > 0 && columns <= 16777215 && rows > 0 && rows `mod` 4 == 0) $
        Left "Invalid matrix dimensions or unsafe integer reduction bound"
    unless (toInteger columns * toInteger rows <= 128 * 1024 * 1024) $
        Left "The spike accepts at most 128 million matrix entries"
    unless (BS.length weights == columns * (rows `div` 4) && BS.length activations == columns) $
        Left "Matrix or activation buffer length does not match its shape"
    unless (BS.all (\byte -> all (\offset -> ((byte `shiftR` offset) .&. 3) < 3) [0, 2, 4, 6]) weights) $
        Left "The packed matrix contains an invalid ternary field"
    let maximumContribution =
            maximum
                [ abs (toInteger (runCat mapper (field, activation)))
                | field <- [0 .. 2]
                , activation <- [-128 .. 127]
                ]
    unless (toInteger columns * maximumContribution <= toInteger (maxBound :: Int32)) $
        Left "Cannot prove that every reordered Int32 partial sum avoids overflow"
    let chunk = rows `div` 4
        rowSum row =
            let offset = (row `mod` chunk) * columns
                fieldShift = 2 * (row `div` chunk)
                step total column =
                    let field = fromIntegral ((BS.index weights (offset + column) `shiftR` fieldShift) .&. 3)
                        activation = fromIntegral (fromIntegral (BS.index activations column) :: Int8)
                     in runCat reducer (total, runCat mapper (field, activation))
             in foldl' step 0 [0 .. columns - 1]
    Right (map rowSum [0 .. rows - 1])

-- | Lower scalar graphs and the array skeleton to two Metal kernels.
metalSource :: Cat Float Float -> ArrayPlan -> Either String String
metalSource scalar (PackedMatVec mapper reducer) = do
    scalarCode <- emitCat scalar (FloatCode "inputs[index]")
    elementCode <- emitCat mapper (PairCode (IntCode "field") (IntCode "activation"))
    laneCode <- emitCat reducer (PairCode (IntCode "partial") (IntCode "element"))
    treeCode <- emitCat reducer (PairCode (IntCode "scratch[lane]") (IntCode "scratch[lane + stride]"))
    let FloatCode scalarBody = scalarCode
        IntCode elementBody = elementCode
        IntCode laneBody = laneCode
        IntCode treeBody = treeCode
    Right $
        unlines
            [ "#include <metal_stdlib>"
            , "using namespace metal;"
            , "#pragma clang fp contract(off)"
            , "kernel void scalar(device const float* inputs [[buffer(0)]],"
            , "                   device float* outputs [[buffer(1)]],"
            , "                   constant uint& count [[buffer(2)]],"
            , "                   uint index [[thread_position_in_grid]]) {"
            , "  if (index < count) outputs[index] = " ++ scalarBody ++ ";"
            , "}"
            , "kernel void matvec(device const uchar* weights [[buffer(0)]],"
            , "                   device const char* activations [[buffer(1)]],"
            , "                   device int* output [[buffer(2)]],"
            , "                   constant uint& columns [[buffer(3)]],"
            , "                   constant uint& rows [[buffer(4)]],"
            , "                   uint lane [[thread_index_in_threadgroup]],"
            , "                   uint3 group [[threadgroup_position_in_grid]]) {"
            , "  uint row = group.x;"
            , "  if (row >= rows) return;"
            , "  threadgroup int scratch[128];"
            , "  uint chunk = rows / 4;"
            , "  uint offset = (row % chunk) * columns;"
            , "  uint shift = 2 * (row / chunk);"
            , "  int partial = 0;"
            , "  for (uint column = lane; column < columns; column += 128) {"
            , "    int field = int((weights[offset + column] >> shift) & 3);"
            , "    int activation = int(activations[column]);"
            , "    int element = " ++ elementBody ++ ";"
            , "    partial = " ++ laneBody ++ ";"
            , "  }"
            , "  scratch[lane] = partial;"
            , "  threadgroup_barrier(mem_flags::mem_threadgroup);"
            , "  for (uint stride = 64; stride > 0; stride >>= 1) {"
            , "    if (lane < stride) scratch[lane] = " ++ treeBody ++ ";"
            , "    threadgroup_barrier(mem_flags::mem_threadgroup);"
            , "  }"
            , "  if (lane == 0) output[row] = scratch[0];"
            , "}"
            ]

{- | Lower the graphs to WGSL kernels. Pad packed buffers to whole u32 words.
Use the checked CPU domain and overflow bound before dispatch.
-}
wgslSource :: Cat Float Float -> ArrayPlan -> Either String String
wgslSource scalar (PackedMatVec mapper reducer) = do
    scalarCode <- emitWGSL scalar (FloatCode "scalarInputs[index]")
    elementCode <- emitWGSL mapper (PairCode (IntCode "field") (IntCode "activation"))
    laneCode <- emitWGSL reducer (PairCode (IntCode "partial") (IntCode "element"))
    treeCode <- emitWGSL reducer (PairCode (IntCode "scratch[lane]") (IntCode "scratch[lane + stride]"))
    let FloatCode scalarBody = scalarCode
        IntCode elementBody = elementCode
        IntCode laneBody = laneCode
        IntCode treeBody = treeCode
    Right $
        wgslPrelude
            ++ unlines
                [ "@group(0) @binding(0) var<storage, read> scalarInputs: array<f32>;"
                , "@group(0) @binding(1) var<storage, read_write> scalarOutputs: array<f32>;"
                , "@group(0) @binding(2) var<uniform> scalarParams: vec4<u32>;"
                , "@group(0) @binding(0) var<storage, read> weights: array<u32>;"
                , "@group(0) @binding(1) var<storage, read> activations: array<u32>;"
                , "@group(0) @binding(2) var<storage, read_write> output: array<i32>;"
                , "@group(0) @binding(3) var<uniform> matvecParams: vec4<u32>;"
                , "var<workgroup> scratch: array<i32, 128>;"
                , "@compute @workgroup_size(128)"
                , "fn scalar(@builtin(global_invocation_id) globalId: vec3<u32>) {"
                , "  let index = globalId.x;"
                , "  if (index < scalarParams.x) {"
                , "    scalarOutputs[index] = " ++ scalarBody ++ ";"
                , "  }"
                , "}"
                , "@compute @workgroup_size(128)"
                , "fn matvec(@builtin(workgroup_id) groupId: vec3<u32>,"
                , "          @builtin(local_invocation_index) lane: u32) {"
                , "  let row = groupId.x;"
                , "  let columns = matvecParams.x;"
                , "  let rows = matvecParams.y;"
                , "  if (row >= rows) { return; }"
                , "  let chunk = rows / 4u;"
                , "  let offset = (row % chunk) * columns;"
                , "  let shift = 2u * (row / chunk);"
                , "  var partial: i32 = 0;"
                , "  for (var column = lane; column < columns; column += 128u) {"
                , "    let byteIndex = offset + column;"
                , "    let packed = (weights[byteIndex / 4u] >> (8u * (byteIndex % 4u))) & 255u;"
                , "    let field = i32((packed >> shift) & 3u);"
                , "    let activationByte = (activations[column / 4u] >> (8u * (column % 4u))) & 255u;"
                , "    let activation = bitcast<i32>(activationByte << 24u) >> 24u;"
                , "    let element = " ++ elementBody ++ ";"
                , "    partial = " ++ laneBody ++ ";"
                , "  }"
                , "  scratch[lane] = partial;"
                , "  workgroupBarrier();"
                , "  for (var stride = 64u; stride > 0u; stride >>= 1u) {"
                , "    if (lane < stride) { scratch[lane] = " ++ treeBody ++ "; }"
                , "    workgroupBarrier();"
                , "  }"
                , "  if (lane == 0u) { output[row] = scratch[0]; }"
                , "}"
                ]

-- | Generate finite-input BitLinear with source-defined quantization and scales.
bitLinearSource :: Cat (Float, Float) Float -> Cat (Float, Float) Int32 -> Cat (Int32, (Float, Float)) Float -> ArrayPlan -> Either String String
bitLinearSource peak quantize rescale (PackedMatVec mapper reducer) = do
    case quantize of
        Compose RoundFloat (Compose MaxFloat (Pair (FloatConstant lower) (Compose MinFloat (Pair (FloatConstant upper) _))))
            | lower == -128 && upper == 127 -> pure ()
        _ -> Left "BitLinear requires source quantization clamped to the signed int8 range before rounding"
    peakLaneCode <- emitWGSL peak (PairCode (FloatCode "maximum") (FloatCode "inputs[column]"))
    peakTreeCode <- emitWGSL peak (PairCode (FloatCode "maxima[lane]") (FloatCode "maxima[lane + stride]"))
    quantizedCode <- emitWGSL quantize (PairCode (FloatCode "inputs[column]") (FloatCode "maxima[0]"))
    elementCode <- emitWGSL mapper (PairCode (IntCode "field") (IntCode "quantized[column]"))
    laneSumCode <- emitWGSL reducer (PairCode (IntCode "partial") (IntCode "element"))
    treeSumCode <- emitWGSL reducer (PairCode (IntCode "sums[lane]") (IntCode "sums[lane + stride]"))
    scaledCode <- emitWGSL rescale (PairCode (IntCode "sums[0]") (PairCode (FloatCode "scales.x") (FloatCode "peak[0]")))
    let FloatCode peakLane = peakLaneCode
        FloatCode peakTree = peakTreeCode
        IntCode quantized = quantizedCode
        IntCode element = elementCode
        IntCode laneSum = laneSumCode
        IntCode treeSum = treeSumCode
        FloatCode scaled = scaledCode
    pure $
        wgslPrelude
            ++ unlines
                [ "@group(0) @binding(0) var<storage, read> weights: array<u32>;"
                , "@group(0) @binding(1) var<storage, read> inputs: array<f32>;"
                , "@group(0) @binding(2) var<storage, read_write> quantized: array<i32>;"
                , "@group(0) @binding(3) var<storage, read_write> outputs: array<f32>;"
                , "@group(0) @binding(4) var<uniform> dimensions: vec4<u32>;"
                , "@group(0) @binding(5) var<uniform> scales: vec4<f32>;"
                , "@group(0) @binding(6) var<storage, read_write> peak: array<f32>;"
                , "var<workgroup> maxima: array<f32, 128>;"
                , "var<workgroup> sums: array<i32, 128>;"
                , "@compute @workgroup_size(128)"
                , "fn quantize(@builtin(local_invocation_index) lane: u32) {"
                , "  var maximum = 0.0f;"
                , "  for (var column = lane; column < dimensions.x; column += 128u) {"
                , "    maximum = " ++ peakLane ++ ";"
                , "  }"
                , "  maxima[lane] = maximum;"
                , "  workgroupBarrier();"
                , "  for (var stride = 64u; stride > 0u; stride >>= 1u) {"
                , "    if (lane < stride) { maxima[lane] = " ++ peakTree ++ "; }"
                , "    workgroupBarrier();"
                , "  }"
                , "  if (lane == 0u) { peak[0] = maxima[0]; }"
                , "  for (var column = lane; column < dimensions.x; column += 128u) {"
                , "    quantized[column] = " ++ quantized ++ ";"
                , "  }"
                , "}"
                , "@compute @workgroup_size(128)"
                , "fn bitlinear(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {"
                , "  let row = group.x;"
                , "  if (row >= dimensions.y) { return; }"
                , "  let chunk = dimensions.y / 4u;"
                , "  let shift = 2u * (row / chunk);"
                , "  var partial: i32 = 0;"
                , "  for (var column = lane; column < dimensions.x; column += 128u) {"
                , "    let byteIndex = (row % chunk) * dimensions.x + column;"
                , "    let packed = (weights[byteIndex / 4u] >> (8u * (byteIndex % 4u))) & 255u;"
                , "    let field = i32((packed >> shift) & 3u);"
                , "    let element = " ++ element ++ ";"
                , "    partial = " ++ laneSum ++ ";"
                , "  }"
                , "  sums[lane] = partial;"
                , "  workgroupBarrier();"
                , "  for (var stride = 64u; stride > 0u; stride >>= 1u) {"
                , "    if (lane < stride) { sums[lane] = " ++ treeSum ++ "; }"
                , "    workgroupBarrier();"
                , "  }"
                , "  if (lane == 0u) { outputs[row] = " ++ scaled ++ "; }"
                , "}"
                ]
