ROOT := ../..
GHC ?= ghc
BUILD := $(ROOT)/bin/ccc-spike
EMCC ?= emcc
GHCFLAGS := -O2 -XScopedTypeVariables -XTypeSynonymInstances -XMultiParamTypeClasses -XFlexibleInstances
GHCFLAGS += -i$(ROOT)/vendor/microhs/ghc -i$(ROOT)/vendor/microhs/src -i.
GHCFLAGS += -I$(ROOT)/vendor/microhs/src/runtime -I$(ROOT)/vendor/microhs/src/runtime/unix
GHCFLAGS += -outputdir $(BUILD)/ghc
GHCFLAGS += -package mtl -package pretty -package haskeline -package process -package time
GHCFLAGS += -package ghc-prim -package containers -package deepseq -package directory
GHCFLAGS += -package text -package bytestring -package filepath -package array
GHCFLAGS += -Wno-x-partial

.PHONY: all check web web-check fixtures network-fixtures
all: $(BUILD)/ccc-compile $(BUILD)/source $(BUILD)/prepare $(BUILD)/metal-runner

fixtures: $(BUILD)/ccc-compile $(BUILD)/source $(BUILD)/prepare $(BUILD)/bitlinear-reference $(BUILD)/prepare-layer
	@mkdir -p $(BUILD)/cases
	cd $(ROOT) && bin/ccc-spike/prepare
	cd $(ROOT) && bin/ccc-spike/ccc-compile
	cd $(ROOT) && bin/ccc-spike/bitlinear-reference bin/ccc-spike/cases
	cd $(ROOT) && bin/ccc-spike/prepare-layer

check: all fixtures
	cd $(ROOT) && bin/ccc-spike/metal-runner bin/ccc-spike/generated.metal bin/ccc-spike/cases

web: fixtures network-fixtures $(BUILD)/web/microhs.mjs
	cd web && npm run build

web-check: web
	cd web && npm run check

network-fixtures: $(BUILD)/network-cases/suite.json

$(BUILD)/network-cases/suite.json: $(BUILD)/prepare-network $(ROOT)/models/bitnet-b1.58-2B-4T/model.safetensors $(ROOT)/models/bitnet-b1.58-2B-4T/tokenizer.json
	cd $(ROOT) && bin/ccc-spike/prepare-network

$(BUILD)/prepare-network: PrepareNetwork.hs $(wildcard $(ROOT)/src/*.hs) $(ROOT)/cbits/bndot.c $(ROOT)/cbits/tokparse.c
	@mkdir -p $(BUILD)
	cd $(ROOT) && vendor/microhs/bin/gmhs -ispike/ccc -isrc -optc -fno-fast-math -optc -ffp-contract=off PrepareNetwork -Icbits cbits/bndot.c cbits/tokparse.c -obin/ccc-spike/prepare-network

$(BUILD)/web/WebMain.c: WebMain.hs WebFFI.hs WebBrowser.hs WebJSON.hs WebModel.hs WebApp.hs WebFixtures.hs WebTensor.hs Programs.hs LayerPrograms.hs Rotary.hs LayerProgram.hs NetworkProgram.hs $(ROOT)/src/Safetensors.hs
	@mkdir -p $(BUILD)/web
	cd $(ROOT) && vendor/microhs/bin/gmhs -c -ispike/ccc -isrc WebMain -obin/ccc-spike/web/WebMain.c

$(BUILD)/web/microhs.mjs: $(BUILD)/web/WebMain.c
	$(EMCC) -O2 -I$(ROOT)/vendor/microhs/src/runtime -I$(ROOT)/vendor/microhs/src/runtime/unix -DUSE_SYSTEM_RAW -Wno-address-of-packed-member $< $(ROOT)/vendor/microhs/src/runtime/eval.c --no-entry -lm -sEXPORTED_FUNCTIONS=_mhs_init,_malloc,_free -sEXPORTED_RUNTIME_METHODS=HEAPU8 -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=MicroHsGPU -sENVIRONMENT=web,node -sALLOW_MEMORY_GROWTH=1 -sSTACK_SIZE=5242880 -sEXIT_RUNTIME=0 -o $@

$(BUILD)/ccc-compile: Main.hs Frontend.hs Compiler.hs CCC.hs FloatWGSL.hs Arrays.hs LayerArrays.hs LayerPrograms.hs HeadArrays.hs BufferChecks.hs
	@mkdir -p $(BUILD)/ghc
	$(GHC) $(GHCFLAGS) Main.hs -o $@

$(BUILD)/source: Programs.hs
	@mkdir -p $(BUILD)
	cd $(ROOT) && vendor/microhs/bin/gmhs -ispike/ccc Programs -obin/ccc-spike/source

$(BUILD)/prepare: Prepare.hs
	@mkdir -p $(BUILD)
	cd $(ROOT) && vendor/microhs/bin/gmhs -ispike/ccc -isrc Prepare -obin/ccc-spike/prepare

$(BUILD)/prepare-layer: PrepareLayer.hs $(wildcard $(ROOT)/src/*.hs) $(ROOT)/cbits/bndot.c
	@mkdir -p $(BUILD)
	cd $(ROOT) && vendor/microhs/bin/gmhs -ispike/ccc -isrc -optc -fno-fast-math -optc -ffp-contract=off PrepareLayer -Icbits cbits/bndot.c -obin/ccc-spike/prepare-layer

$(BUILD)/metal-runner: metal-runner.m $(ROOT)/cbits/bndot.c $(ROOT)/cbits/bndot.h
	@mkdir -p $(BUILD)
	clang -O3 -fobjc-arc -framework Foundation -framework Metal -I$(ROOT)/cbits metal-runner.m $(ROOT)/cbits/bndot.c -o $@

$(BUILD)/bitlinear-reference: bitlinear-reference.c $(ROOT)/cbits/bndot.c $(ROOT)/cbits/bndot.h
	@mkdir -p $(BUILD)
	clang -std=c11 -O3 -fno-fast-math -ffp-contract=off -Wno-unknown-pragmas -I$(ROOT)/cbits bitlinear-reference.c $(ROOT)/cbits/bndot.c -lm -o $@
