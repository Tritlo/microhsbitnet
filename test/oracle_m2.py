#!/usr/bin/env python3
"""M2 oracle: run the fork's llama-tokenize over test/tok-cases.txt (same
unescaping as TestTok.hs, --no-parse-special to match encodeText) and print
one python-style id list per case — diffable against ./bin/testtok output.

Run with: python3 test/oracle_m2.py
"""

import os
import subprocess
import tempfile

PRISM = os.environ.get("PRISM_LLAMA", os.path.expanduser("~/Code/llama.cpp-prism"))
TOKENIZE = os.path.join(PRISM, "build-cuda", "bin", "llama-tokenize")
MODEL = "models/bonsai-27b/Bonsai-27B-Q1_0.gguf"


def unescape(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            c = s[i + 1]
            out.append({"n": "\n", "r": "\r", "t": "\t", "\\": "\\"}.get(c, "\\" + c))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


with open("test/tok-cases.txt", encoding="utf-8") as f:
    cases = [line for line in f.read().split("\n") if line]

for raw in cases:
    text = unescape(raw)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tf:
        tf.write(text)
        path = tf.name
    try:
        r = subprocess.run(
            [TOKENIZE, "-m", MODEL, "-f", path, "--ids", "--no-bos",
             "--no-parse-special", "--no-escape", "--log-disable"],
            capture_output=True, text=True, check=True)
        print(r.stdout.strip().replace(", ", ","))
    finally:
        os.unlink(path)
