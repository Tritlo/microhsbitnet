CCC PROGRAM GALLERY

RUN LOCALLY
Extract ccc-demo.zip into an empty directory. Open a terminal in that directory:

  python3 -m http.server 8879 --bind 127.0.0.1

Open http://localhost:8879/ in a browser. No npm install or build is needed.
The gallery is static and remains readable without WebGPU. The small scalar
example needs a WebGPU browser and HTTPS or localhost. It loads no model.
Do not use a file:// URL for the GPU example. To serve a public copy, use HTTPS.
No hosting or publication is configured by this package.

OPTIONAL FULL-MODEL EXPERIMENT
Open http://localhost:8879/model/web/demo.html . The runtime is included.
Starting the model page downloads 1.18 GB of weights (1,178,623,988 bytes), plus
the tokenizer, from Hugging Face. Neither file is in this archive. A physical
WebGPU adapter, network access, and several GB of free memory are required.
The page runs five fixed prompts. It is a research demonstration.
The six-prompt reference check matched 52/52 tokens, but its 1% numerical error
target still fails. Token agreement does not establish numerical equivalence.

WHAT THE PAGE SHOWS
Compiler outputs are prepared at build time. The browser runs exported WGSL;
it does not compile edited Haskell. The GPU does not reduce the iota pictures.
The compact view keeps function and dictionary names. The expanded view includes
all reachable Haskell definitions, including unused methods and error paths.
Runtime operations, foreign calls, and typed literals remain named leaves.
Counts measure syntax occurrences, not executed GPU instructions.
Full vector SVG files can exceed browser DOM capacity. Preview the PNG files.
Comparison poster SVGs embed the expanded raster. Separate full SVG sources
and indexed tree grammars describe every expanded occurrence.

REBUILD FROM THE SOURCE SNAPSHOT
source.zip contains sibling Combinate and microhsbitnet-ccc-review directories.
It omits the model, installed dependencies, build outputs, and MicroHs sources.
See SOURCE-REVISIONS.json for exact revisions and dirty-state records.
Prerequisites: Git, make, GHC and the packages in spike/ccc/Makefile, Node/npm,
Clang, Python 3, zip, and an installed Chrome for image generation. The validated
host is macOS on Apple Silicon. See the source README files for tool versions.
From the directory where source.zip was extracted:

  git clone https://github.com/augustss/MicroHs.git microhsbitnet-ccc-review/vendor/microhs
  git -C microhsbitnet-ccc-review/vendor/microhs checkout --detach e11ae213710d54c1954a9cb83ff97530c68d2488
  make -C microhsbitnet-ccc-review/vendor/microhs bin/gmhs bin/cpphs mhs.conf
  make -C microhsbitnet-ccc-review model HF_BASE=https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/resolve/04c3b9ad9361b824064a1f25ea60a8be9599b127
  make -C microhsbitnet-ccc-review/spike/ccc graphics-data
  cd Combinate
  npm ci
  node --import tsx scripts/ccc-graphics.ts ../microhsbitnet-ccc-review/bin/ccc-spike/graphics-data.json ../../rebuilt-graphics
  node --import tsx scripts/ccc-package.ts ../../rebuilt-graphics ../microhsbitnet-ccc-review ../../rebuilt-demo

The model download is required to regenerate the checkpoint statistics, but is
not required to serve this package. The source snapshot contains current files,
not just the Git revisions; uncommitted changes are included in selected paths.
The optional model runtime needs a separate Emscripten build. See
microhsbitnet-ccc-review/spike/ccc/README.md and its web-demo Makefile target.
Read SOURCE-NOTICE.txt and LICENSES before reusing source or generated material.
