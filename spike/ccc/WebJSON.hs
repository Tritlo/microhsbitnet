module WebJSON (
    JSON (..),
    parseJSON,
    renderJSON,
    field,
    optional,
    asInt,
    asDouble,
    asString,
    asArray,
    asObject,
    asBool,
    int,
)
where

import Data.Char (chr, ord)
import Data.List (intercalate)
import Numeric (showHex)

-- | Represent a JSON value. Object keys must be distinct.
data JSON
    = JNull
    | JBool Bool
    | JInt Int
    | JNumber Double
    | JString String
    | JArray [JSON]
    | JObject [(String, JSON)]
    deriving (Eq, Show)

-- | Parse one complete JSON value. Reject duplicate keys and nonfinite numbers.
parseJSON :: String -> Either String JSON
parseJSON input = do
    (value, remaining) <- parseValue input
    if null (whitespace remaining)
        then Right value
        else Left "Unexpected text after JSON value"

-- | Render a JSON value with escaped strings and finite numbers.
renderJSON :: JSON -> String
renderJSON value = case value of
    JNull -> "null"
    JBool True -> "true"
    JBool False -> "false"
    JInt number -> show number
    JNumber number
        | isNaN number || isInfinite number -> error "Cannot render a nonfinite JSON number"
        | otherwise -> show number
    JString text -> '"' : concatMap escape text ++ "\""
    JArray values -> "[" ++ intercalate "," (map renderJSON values) ++ "]"
    JObject fields -> "{" ++ intercalate "," [renderJSON (JString key) ++ ":" ++ renderJSON item | (key, item) <- fields] ++ "}"

-- | Return a required object field. Fail if the key is absent.
field :: String -> JSON -> JSON
field key value = case optional key value of
    Just item -> item
    Nothing -> error ("Missing JSON field: " ++ key)

-- | Return an optional object field. Fail if the value is not an object.
optional :: String -> JSON -> Maybe JSON
optional key = lookup key . asObject

-- | Read a finite integral number within the bounds of Int.
asInt :: JSON -> Int
asInt (JInt number) = number
asInt value =
    let number = asDouble value
        integer = truncate number :: Integer
     in if number == fromInteger integer && integer >= toInteger (minBound :: Int) && integer <= toInteger (maxBound :: Int)
            then fromInteger integer
            else error "Expected a JSON integer within Int bounds"

-- | Read a finite JSON number.
asDouble :: JSON -> Double
asDouble (JInt number) = fromIntegral number
asDouble (JNumber number)
    | not (isNaN number || isInfinite number) = number
asDouble _ = error "Expected a finite JSON number"

-- | Read a JSON string.
asString :: JSON -> String
asString (JString text) = text
asString _ = error "Expected a JSON string"

-- | Read a JSON array.
asArray :: JSON -> [JSON]
asArray (JArray values) = values
asArray _ = error "Expected a JSON array"

-- | Read a JSON object.
asObject :: JSON -> [(String, JSON)]
asObject (JObject fields) = fields
asObject _ = error "Expected a JSON object"

-- | Read a JSON Boolean.
asBool :: JSON -> Bool
asBool (JBool value) = value
asBool _ = error "Expected a JSON Boolean"

-- | Construct an integral JSON number.
int :: Int -> JSON
int = JInt

-- | Remove only the four whitespace characters that JSON permits.
whitespace :: String -> String
whitespace = dropWhile (`elem` " \t\r\n")

-- | Parse a JSON value and retain the unconsumed input.
parseValue :: String -> Either String (JSON, String)
parseValue input = case whitespace input of
    'n' : 'u' : 'l' : 'l' : remaining -> Right (JNull, remaining)
    't' : 'r' : 'u' : 'e' : remaining -> Right (JBool True, remaining)
    'f' : 'a' : 'l' : 's' : 'e' : remaining -> Right (JBool False, remaining)
    '"' : remaining -> do
        (text, rest) <- parseString remaining
        Right (JString text, rest)
    '[' : remaining -> parseArray (whitespace remaining)
    '{' : remaining -> parseObject (whitespace remaining)
    remaining -> parseNumber remaining

-- | Parse a string after its opening quote. Reject invalid Unicode scalars.
parseString :: String -> Either String (String, String)
parseString = collect []
  where
    collect characters ('"' : remaining) = Right (reverse characters, remaining)
    collect characters ('\\' : remaining) = do
        (character, rest) <- parseEscape remaining
        collect (character : characters) rest
    collect characters (character : remaining)
        | ord character < 0x20 = Left "Unescaped control character in JSON string"
        | ord character >= 0xD800 && ord character <= 0xDFFF = Left "Unpaired surrogate in JSON string"
        | otherwise = collect (character : characters) remaining
    collect _ [] = Left "Unterminated JSON string"

-- | Decode a JSON escape after its backslash.
parseEscape :: String -> Either String (Char, String)
parseEscape input = case input of
    '"' : remaining -> Right ('"', remaining)
    '\\' : remaining -> Right ('\\', remaining)
    '/' : remaining -> Right ('/', remaining)
    'b' : remaining -> Right ('\b', remaining)
    'f' : remaining -> Right ('\f', remaining)
    'n' : remaining -> Right ('\n', remaining)
    'r' : remaining -> Right ('\r', remaining)
    't' : remaining -> Right ('\t', remaining)
    'u' : remaining -> do
        (first, rest) <- hexQuad remaining
        if first >= 0xD800 && first <= 0xDBFF
            then case rest of
                '\\' : 'u' : following -> do
                    (second, trailing) <- hexQuad following
                    if second >= 0xDC00 && second <= 0xDFFF
                        then Right (chr (0x10000 + (first - 0xD800) * 1024 + second - 0xDC00), trailing)
                        else Left "Invalid low surrogate in JSON string"
                _ -> Left "Missing low surrogate in JSON string"
            else
                if first >= 0xDC00 && first <= 0xDFFF
                    then Left "Unpaired low surrogate in JSON string"
                    else Right (chr first, rest)
    _ -> Left "Invalid JSON string escape"

-- | Read the four hexadecimal digits in a Unicode escape.
hexQuad :: String -> Either String (Int, String)
hexQuad = collect 4 0
  where
    collect 0 value remaining = Right (value, remaining)
    collect count value (character : remaining)
        | character >= '0' && character <= '9' = collect (count - 1) (value * 16 + ord character - ord '0') remaining
        | character >= 'a' && character <= 'f' = collect (count - 1) (value * 16 + ord character - ord 'a' + 10) remaining
        | character >= 'A' && character <= 'F' = collect (count - 1) (value * 16 + ord character - ord 'A' + 10) remaining
    collect _ _ _ = Left "Invalid Unicode escape in JSON string"

-- | Parse an array after its opening bracket. Reject trailing commas.
parseArray :: String -> Either String (JSON, String)
parseArray (']' : remaining) = Right (JArray [], remaining)
parseArray input = collect [] input
  where
    collect values remaining = do
        (value, rest) <- parseValue remaining
        case whitespace rest of
            ',' : following -> collect (value : values) following
            ']' : following -> Right (JArray (reverse (value : values)), following)
            _ -> Left "Expected a comma or closing bracket in JSON array"

-- | Parse an object after its opening brace. Reject duplicate keys.
parseObject :: String -> Either String (JSON, String)
parseObject ('}' : remaining) = Right (JObject [], remaining)
parseObject input = collect [] input
  where
    collect fields remaining = case whitespace remaining of
        '"' : following -> do
            (key, rest) <- parseString following
            if key `elem` map fst fields
                then Left ("Duplicate JSON object key: " ++ key)
                else case whitespace rest of
                    ':' : afterColon -> do
                        (value, afterValue) <- parseValue afterColon
                        case whitespace afterValue of
                            ',' : next -> collect ((key, value) : fields) next
                            '}' : next -> Right (JObject (reverse ((key, value) : fields)), next)
                            _ -> Left "Expected a comma or closing brace in JSON object"
                    _ -> Left "Expected a colon after JSON object key"
        _ -> Left "Expected a JSON object key"

-- | Recognize an ASCII decimal digit.
decimalDigit :: Char -> Bool
decimalDigit character = character >= '0' && character <= '9'

-- | Parse a number with the JSON integer, fraction, and exponent grammar.
parseNumber :: String -> Either String (JSON, String)
parseNumber input = do
    let (sign, unsigned) = case input of
            '-' : remaining -> ("-", remaining)
            _ -> ("", input)
    (whole, afterWhole) <- case unsigned of
        '0' : remaining -> Right ("0", remaining)
        character : _
            | character >= '1' && character <= '9' -> Right (span decimalDigit unsigned)
        _ -> Left "Expected a JSON value"
    (fraction, afterFraction) <- case afterWhole of
        '.' : remaining -> do
            (digits, rest) <- requiredDigits remaining
            Right ('.' : digits, rest)
        _ -> Right ("", afterWhole)
    (exponent, trailing) <- case afterFraction of
        character : remaining
            | character == 'e' || character == 'E' -> do
                let (exponentSign, unsignedExponent) = case remaining of
                        '+' : rest -> ("+", rest)
                        '-' : rest -> ("-", rest)
                        _ -> ("", remaining)
                (digits, rest) <- requiredDigits unsignedExponent
                Right ('e' : exponentSign ++ digits, rest)
        _ -> Right ("", afterFraction)
    case reads (sign ++ whole ++ fraction ++ exponent) of
        [(number, "")]
            | not (isNaN number || isInfinite number) -> Right (JNumber number, trailing)
        _ -> Left "Invalid or nonfinite JSON number"

-- | Read a nonempty sequence of decimal digits.
requiredDigits :: String -> Either String (String, String)
requiredDigits input = case span decimalDigit input of
    ([], _) -> Left "Expected a decimal digit in JSON number"
    result -> Right result

-- | Escape one Unicode scalar for a JSON string.
escape :: Char -> String
escape character = case character of
    '"' -> "\\\""
    '\\' -> "\\\\"
    '\b' -> "\\b"
    '\f' -> "\\f"
    '\n' -> "\\n"
    '\r' -> "\\r"
    '\t' -> "\\t"
    _
        | ord character < 0x20 -> "\\u" ++ replicate (4 - length digits) '0' ++ digits
        | ord character >= 0xD800 && ord character <= 0xDFFF -> error "Cannot render an unpaired Unicode surrogate"
        | otherwise -> [character]
  where
    digits = showHex (ord character) ""
