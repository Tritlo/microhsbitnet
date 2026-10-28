#!/bin/sh
# Record the showcase on a machine with a physical WebGPU adapter.
# The cloud VM used for this branch has no physical GPU, so it cannot
# produce the token video. Chrome there only exposes a software adapter,
# which the page rejects.
#
# From a checkout of this branch, on macOS with Google Chrome:
#   ./spike/ccc/record-demo.sh
# Then record the window (QuickTime or Chrome's tab capture) while tokens
# stream into the transcript. Ctrl-C stops the local server.
#
# The page fetches weights from Hugging Face. A local model is not required
# for the recording. The reference check (Swedish included) is separate:
#   make -C spike/ccc web-check
# That check still needs models/bitnet-b1.58-2B-4T/ and the generated fixtures.

set -e
cd "$(dirname "$0")/../.."
if [ ! -d spike/ccc/web/node_modules ]; then
  (cd spike/ccc/web && npm ci)
fi
make -C spike/ccc web-demo
cd spike/ccc/web
echo "Opening the showcase. Record this window. The process exits when generation finishes."
DEMO=1 HEADED=1 npm run demo
