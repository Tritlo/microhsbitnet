# Build the BitNet-b1.58-in-MicroHaskell demos.
#
# MicroHs is vendored (clean upstream, pinned) at vendor/microhs and supplies the
# lib + runtime via MHSDIR. The fast-path dot is the C kernel cbits/bndot.c, linked
# in via FFI (plain `foreign import ccall`, no runtime patch). Modules that import
# Dot need the kernel; the rest don't.
#
# Run a built demo with a big nursery, e.g.:  ./bin/m3 +RTS -H800M -RTS

GMHS   ?= $(HOME)/Code/MicroHs/bin/gmhs
MHSDIR := $(CURDIR)/vendor/microhs
MHS     = MHSDIR=$(MHSDIR) $(GMHS) -isrc
KERNEL  = -Icbits cbits/bndot.c

.PHONY: all fast plain clean
all: fast plain

# Fast path (import Dot -> link the bndot kernel).
fast: m1 m2 m3 m4 testkinds
# No fast path.
plain: m0 testtok showcase

m1:        ; $(MHS) $(KERNEL) M1        -obin/m1
m2:        ; $(MHS) $(KERNEL) M2        -obin/m2
m3:        ; $(MHS) $(KERNEL) M3        -obin/m3
m4:        ; $(MHS) $(KERNEL) M4        -obin/m4
testkinds: ; $(MHS) $(KERNEL) TestKinds -obin/testkinds
m0:        ; $(MHS) M0       -obin/m0
testtok:   ; $(MHS) TestTok  -obin/testtok
showcase:  ; $(MHS) Showcase -obin/showcase

clean: ; rm -f bin/m0 bin/m1 bin/m2 bin/m3 bin/m4 bin/testkinds bin/testtok bin/showcase
