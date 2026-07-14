# Build the 1-bit-Bonsai-in-MicroHaskell demos.
#
# Needs a recent stock MicroHs (≥ v0.16.5 — the one pinned in vendor/microhs).
# Point GMHS at its gmhs, or have ~/Code/MicroHs/bin/gmhs — and make sure that
# binary was rebuilt from the current source. gmhs auto-detects its own lib +
# runtime. The fast-path kernels live in cbits/ and are linked via FFI (plain
# `foreign import ccall`); OpenMP + AVX2 flags go straight to cc with
# -optc/-optl. Modules that import Dot need the kernel; the rest don't.

GMHS   ?= $(HOME)/Code/MicroHs/bin/gmhs
MHS     = $(GMHS) -isrc -optc -fopenmp -optc -march=native -optl -fopenmp
KERNEL  = -Icbits cbits/b1dot.c

# One command from a fresh clone: fetch the pinned MicroHs (submodule) + the
# model weights, then build everything. (Re-runs are cheap: the submodule and
# the model file are only fetched if missing.)
MODEL_DIR  := models/bonsai-27b
MODEL      := $(MODEL_DIR)/Bonsai-27B-Q1_0.gguf
HF_BASE    := https://huggingface.co/prism-ml/Bonsai-27B-gguf/resolve/main

.PHONY: all fast plain clean setup submodule model
all: submodule model fast plain

setup: all   # alias

# The pinned MicroHs source (documentary — the build uses your installed gmhs).
submodule:
	@git submodule update --init --recursive 2>/dev/null || true

# Fetch the model weights (tokenizer included — GGUF embeds it). ~3.8 GB once.
model: $(MODEL)
$(MODEL):
	@mkdir -p $(MODEL_DIR)
	curl -L --fail -C - -o $@ $(HF_BASE)/$(@F)

# Fast path (import Dot -> link the b1dot kernel). Grows as milestones land.
fast:
# No fast path.
plain: m0

m0: ; $(MHS) M0 -obin/m0

clean: ; rm -f bin/m0
