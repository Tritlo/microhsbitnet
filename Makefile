# Build the BitNet-b1.58-in-MicroHaskell demos.
#
# Needs a MicroHs install (stock — no patch, no fork): point GMHS at its gmhs, or
# have ~/Code/MicroHs/bin/gmhs. gmhs auto-detects its own lib + runtime. The
# fast-path dot is the C kernel cbits/bndot.c, linked via FFI (plain
# `foreign import ccall`); its OpenMP + AVX2 flags go straight to cc with
# -optc/-optl. Modules that import Dot need the kernel; the rest don't.
#
# Run the demo:  ./bin/m4 ["prompt" [nGen]]   -- runs in ~2GB with the default heap
# (BNDOT_THREADS=N to set the thread count). The Haskell batch path (M2/M3) still
# wants a big nursery, e.g.:  ./bin/m3 +RTS -H800M -RTS

GMHS   ?= $(HOME)/Code/MicroHs/bin/gmhs
MHS     = $(GMHS) -isrc -optc -fopenmp -optc -march=native -optl -fopenmp
KERNEL  = -Icbits cbits/bndot.c

# One command from a fresh clone: fetch the pinned MicroHs (submodule) + the
# model weights, then build everything. (Re-runs are cheap: the submodule and the
# model files are only fetched if missing.)
MODEL_DIR  := models/bitnet-b1.58-2B-4T
HF_BASE    := https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/resolve/main

.PHONY: all fast plain clean setup submodule model
all: submodule model fast plain

setup: all   # alias

# The pinned MicroHs source (documentary — the build uses your installed gmhs).
submodule:
	@git submodule update --init --recursive 2>/dev/null || true

# Fetch the model weights + tokenizer into models/ (gitignored). ~1.2 GB once.
model: $(MODEL_DIR)/model.safetensors $(MODEL_DIR)/tokenizer.json
$(MODEL_DIR)/model.safetensors $(MODEL_DIR)/tokenizer.json:
	@mkdir -p $(MODEL_DIR)
	curl -L --fail -C - -o $@ $(HF_BASE)/$(@F)

# Fast path (import Dot -> link the bndot kernel).
fast: m1 m2 m3 m4 chat testkinds
# No fast path.
plain: m0 testtok showcase

# The tokenizer parser (cbits/tokparse.c) is linked by the tokenizer users (M4, TestTok).
m1:        ; $(MHS) $(KERNEL) M1        -obin/m1
m2:        ; $(MHS) $(KERNEL) M2        -obin/m2
m3:        ; $(MHS) $(KERNEL) M3        -obin/m3
m4:        ; $(MHS) $(KERNEL) cbits/tokparse.c M4 -obin/m4
chat:      ; $(MHS) $(KERNEL) cbits/tokparse.c Chat -obin/chat
testkinds: ; $(MHS) $(KERNEL) TestKinds -obin/testkinds
m0:        ; $(MHS) M0 -obin/m0
testtok:   ; $(MHS) -Icbits cbits/tokparse.c TestTok -obin/testtok
showcase:  ; $(MHS) $(KERNEL) Showcase -obin/showcase
estimate:  ; $(MHS) Estimate -obin/estimate

clean: ; rm -f bin/m0 bin/m1 bin/m2 bin/m3 bin/m4 bin/testkinds bin/testtok bin/showcase
