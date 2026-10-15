module WebModel (
    Model,
    TensorInfo (..),
    openModel,
    modelTensors,
    tensor,
    readTensor,
    tensorScale,
)
where

import Control.Monad (unless)
import Safetensors (bf16ToFloat)
import WebBrowser (Browser, command, dropHost, fetchRange, hostNumber, hostSize, liftBrowser, request)
import WebJSON (JSON (..), asString, int, parseJSON)

-- | Store the model URL, data offset, file size, and validated tensor metadata.
data Model = Model String Int Int [TensorInfo]

-- | Describe one tensor and its byte range within the model data section.
data TensorInfo = TensorInfo
    { tensorName :: String
    , tensorDtype :: String
    , tensorShape :: [Int]
    , tensorBegin :: Int
    , tensorEnd :: Int
    }
    deriving (Eq, Show)

-- | Read the model header through exact HTTP ranges. Keep weights outside WASM.
openModel :: String -> Browser Model
openModel url = do
    (prefix, totalBytes) <- fetchRange url 0 8
    prefixSize <- liftBrowser (hostSize prefix)
    unless (prefixSize == 8) $ liftBrowser $ do
        dropHost prefix
        fail "Truncated safetensors prefix"
    (low, high) <- liftBrowser $ do
        low <- hostNumber prefix 0 "u32"
        high <- hostNumber prefix 4 "u32"
        dropHost prefix
        pure (low, high)
    headerLength <- either (liftBrowser . fail) pure (headerSize totalBytes low high)
    (header, headerTotal) <- fetchRange url 8 headerLength
    fetchedSize <- liftBrowser (hostSize header)
    unless (headerTotal == totalBytes && fetchedSize == headerLength) $ liftBrowser $ do
        dropHost header
        fail "Model length changed or the header range is truncated"
    headerText <- request (command "text" [int header])
    liftBrowser (dropHost header)
    let dataStart = 8 + headerLength
    tensors <- either (liftBrowser . fail) pure (parseJSON (asString headerText) >>= parseTensors (totalBytes - dataStart))
    pure (Model url dataStart totalBytes tensors)

-- | Return the validated tensor table in header order.
modelTensors :: Model -> [TensorInfo]
modelTensors (Model _ _ _ tensors) = tensors

-- | Find a required tensor by name.
tensor :: Model -> String -> TensorInfo
tensor model name = case filter ((== name) . tensorName) (modelTensors model) of
    info : _ -> info
    [] -> error ("Missing model tensor: " ++ name)

-- | Fetch a tensor slice into a host handle. The caller must release the handle.
readTensor :: Model -> String -> Int -> Maybe Int -> Browser Int
readTensor model@(Model url dataStart totalBytes _) name firstByte byteCount = do
    let info = tensor model name
        tensorBytes = tensorEnd info - tensorBegin info
        count = maybe (tensorBytes - firstByte) id byteCount
    unless (firstByte >= 0 && firstByte < tensorBytes && count > 0 && count <= tensorBytes - firstByte) $
        liftBrowser (fail ("Invalid model tensor slice: " ++ name))
    (host, currentTotal) <- fetchRange url (dataStart + tensorBegin info + firstByte) count
    size <- liftBrowser (hostSize host)
    unless (currentTotal == totalBytes && size == count) $ liftBrowser $ do
        dropHost host
        fail ("Model length changed or the tensor range is truncated: " ++ name)
    pure host

-- | Read the named BF16 scalar. Require a finite positive scale.
tensorScale :: Model -> String -> Browser Double
tensorScale model name = do
    let info = tensor model name
    unless (tensorDtype info == "BF16" && product (map toInteger (tensorShape info)) == 1) $
        liftBrowser (fail ("Invalid BF16 model scale: " ++ name))
    host <- readTensor model name 0 (Just 2)
    bits <- liftBrowser $ do
        bits <- hostNumber host 0 "u16"
        dropHost host
        pure bits
    raw <- either (liftBrowser . fail) pure (metadataInt ("BF16 scale bits: " ++ name) (JNumber bits))
    unless (raw <= 65535) $ liftBrowser (fail ("Invalid BF16 scale bits: " ++ name))
    let scale = realToFrac (bf16ToFloat (fromIntegral raw))
    unless (scale > 0 && not (isNaN scale || isInfinite scale)) $
        liftBrowser (fail ("Expected a finite positive model scale: " ++ name))
    pure scale

-- | Validate a header length and file size within signed 32-bit byte offsets.
headerSize :: Int -> Double -> Double -> Either String Int
headerSize totalBytes low high
    | high /= 0 = Left "Safetensors header length exceeds 32 bits"
    | totalBytes < 8 || toInteger totalBytes > 2147483647 = Left "Model size exceeds signed 32-bit byte offsets"
    | otherwise = do
        count <- metadataInt "safetensors header length" (JNumber low)
        if count <= 0 || count > 16 * 1024 * 1024 || count > totalBytes - 8
            then Left "Invalid safetensors header length"
            else Right count

-- | Validate all tensor entries and optional string metadata.
parseTensors :: Int -> JSON -> Either String [TensorInfo]
parseTensors dataBytes (JObject fields) = do
    case lookup "__metadata__" fields of
        Nothing -> Right ()
        Just (JObject metadata)
            | all isString (map snd metadata) -> Right ()
        _ -> Left "Invalid safetensors metadata object"
    mapM (parseTensor dataBytes) (filter ((/= "__metadata__") . fst) fields)
  where
    isString (JString _) = True
    isString _ = False
parseTensors _ _ = Left "Invalid safetensors tensor table"

-- | Check tensor field types, dimensions, byte ranges, and exact storage size.
parseTensor :: Int -> (String, JSON) -> Either String TensorInfo
parseTensor dataBytes (name, JObject fields) =
    case (lookup "dtype" fields, lookup "shape" fields, lookup "data_offsets" fields) of
        (Just (JString dtype), Just (JArray dimensions), Just (JArray [beginValue, endValue])) -> do
            shape <- mapM (metadataInt ("tensor dimension: " ++ name)) dimensions
            begin <- metadataInt ("tensor begin offset: " ++ name) beginValue
            end <- metadataInt ("tensor end offset: " ++ name) endValue
            width <- case dtype of
                "BF16" -> Right 2
                "F32" -> Right 4
                "U8" -> Right 1
                "I8" -> Right 1
                _ -> Left ("Unsupported model tensor dtype: " ++ name)
            if any (<= 0) shape || end <= begin || end > dataBytes
                then Left ("Invalid model tensor shape or range: " ++ name)
                else
                    if product (map toInteger shape) * width /= toInteger end - toInteger begin
                        then Left ("Model tensor byte size differs from its shape: " ++ name)
                        else Right (TensorInfo name dtype shape begin end)
        _ -> Left ("Invalid model tensor metadata: " ++ name)
parseTensor _ (name, _) = Left ("Expected model tensor object: " ++ name)

-- | Read a nonnegative integer that fits a signed 32-bit byte offset.
metadataInt :: String -> JSON -> Either String Int
metadataInt _ (JNumber number)
    | not (isNaN number || isInfinite number)
        && number >= 0
        && number <= 2147483647
        && number == fromInteger (truncate number :: Integer) =
        Right (truncate number)
metadataInt label _ = Left ("Invalid nonnegative 32-bit integer for " ++ label)
