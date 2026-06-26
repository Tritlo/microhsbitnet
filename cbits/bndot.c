/* The three inner weighted dots of BitNet b1.58 2B4T inference, in C so the loops
 * run natively instead of as MicroHs combinator graph reduction. Called from
 * Haskell via plain `foreign import ccall` (see src/Dot.hs) — no runtime patch, no
 * custom primitive. Quantization scales are the caller's job (ggml-style vec_dot):
 * each returns the raw dot as a double (accumulator headroom) and Haskell
 * multiplies in the weight/activation scales.
 *
 * Float reads go through memcpy so an arbitrary ByteString offset (substrings are
 * not aligned) never causes a misaligned load; bf16 is assembled little-endian. */

#include "bndot.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static double load_f32(const unsigned char *p) {
  float f;
  memcpy(&f, p, sizeof f);
  return (double)f;
}

/* W1.58 ternary (2-bit field, value-1 ∈ {-1,0,1}) × int8. */
double bndot_ternary(const void *wv, const void *av, int n, int shift) {
  const unsigned char *w = (const unsigned char *)wv;
  const unsigned char *a = (const unsigned char *)av;
  double acc = 0.0;
  int c;
  if (shift < 0 || shift > 6) {
    fprintf(stderr, "bndot_ternary: bad shift %d\n", shift);
    abort();
  }
  for (c = 0; c < n; c++)
    acc += (double)(((w[c] >> shift) & 3) - 1) * (double)((int8_t)a[c]);
  return acc;
}

/* bf16 (high 16 bits of a float32, little-endian) × float32. */
double bndot_bf16(const void *wv, const void *av, int n) {
  const unsigned char *w = (const unsigned char *)wv;
  const unsigned char *a = (const unsigned char *)av;
  double acc = 0.0;
  int c;
  for (c = 0; c < n; c++) {
    uint32_t bits = ((uint32_t)((uint32_t)w[2 * c] | ((uint32_t)w[2 * c + 1] << 8))) << 16;
    float wv32;
    memcpy(&wv32, &bits, sizeof wv32);
    acc += (double)wv32 * load_f32(a + 4 * c);
  }
  return acc;
}

/* float32 × float32. */
double bndot_f32(const void *wv, const void *av, int n) {
  const unsigned char *w = (const unsigned char *)wv;
  const unsigned char *a = (const unsigned char *)av;
  double acc = 0.0;
  int c;
  for (c = 0; c < n; c++)
    acc += load_f32(w + 4 * c) * load_f32(a + 4 * c);
  return acc;
}
