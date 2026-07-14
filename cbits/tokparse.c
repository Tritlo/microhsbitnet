/* The C tables of the qwen35 byte-level BPE tokenizer, built straight from the
 * GGUF metadata arrays (length-prefixed strings — no JSON anywhere): the vocab
 * and merge tables sorted for binary-search lookup, and the id->bytes decode
 * table. The interesting logic (pre-tokenizer regex, BPE merge loop) stays in
 * Haskell and calls b1tok_vlookup / b1tok_mlookup / b1tok_decode.
 *
 * Keys are the token strings exactly as stored in the GGUF (the byte-level
 * representation, canonical UTF-8) — what the Haskell side passes as packUTF8
 * of a byte-level string. Decode maps an id to model bytes: normal tokens via
 * the byte-level alphabet's inverse (level_to_byte per codepoint); control /
 * user-defined tokens keep their literal bytes. */

#include "tokparse.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* ---- byte-level alphabet: codepoint -> model byte (inverse of the GPT-2 map) ---- */
static int g_nonprint[68];
static void init_nonprint(void) {
  int n = 0, b;
  for (b = 0; b <= 32; b++)    g_nonprint[n++] = b;
  for (b = 127; b <= 160; b++) g_nonprint[n++] = b;
  g_nonprint[n++] = 173;
}
static int level_to_byte(int cp) { return cp < 256 ? cp : g_nonprint[cp - 256]; }

/* Decode the UTF-8 codepoint at p (< end); advance *adv. */
static int utf8_next(const unsigned char *p, const unsigned char *end, int *adv) {
  unsigned char c = p[0];
  if (c < 0x80) { *adv = 1; return c; }
  if (c < 0xE0 && p + 1 < end) { *adv = 2; return ((c & 0x1F) << 6) | (p[1] & 0x3F); }
  if (c < 0xF0 && p + 2 < end) { *adv = 3; return ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F); }
  if (p + 3 < end) { *adv = 4; return ((c & 0x07) << 18) | ((p[1] & 0x3F) << 12) | ((p[2] & 0x3F) << 6) | (p[3] & 0x3F); }
  *adv = 1; return c;
}

/* ---- the parsed tables (static; one tokenizer per process) ---- */
typedef struct { uint32_t off, len; int32_t id; } VEntry;
typedef struct { uint32_t aoff, alen, boff, blen; int32_t rank; } MEntry;

static unsigned char *g_keybuf;
static VEntry *g_vocab; static int g_nvocab;
static unsigned char *g_mkeybuf;
static MEntry *g_merges; static int g_nmerges;
static unsigned char *g_decbuf;
static uint32_t *g_decoff, *g_declenv;
static const unsigned char *g_ttypes; static int g_nttypes;
static int32_t *g_ttype_copy;

static int kcmp(const unsigned char *a, int la, const unsigned char *b, int lb) {
  int m = la < lb ? la : lb;
  int c = memcmp(a, b, m);
  if (c) return c;
  return la - lb;
}
static int vcmp(const void *x, const void *y) {
  const VEntry *a = x, *b = y;
  return kcmp(g_keybuf + a->off, a->len, g_keybuf + b->off, b->len);
}
static int mcmp(const void *x, const void *y) {
  const MEntry *a = x, *b = y;
  int c = kcmp(g_mkeybuf + a->aoff, a->alen, g_mkeybuf + b->aoff, b->alen);
  if (c) return c;
  return kcmp(g_mkeybuf + a->boff, a->blen, g_mkeybuf + b->boff, b->blen);
}

/* Build the tables from the GGUF arrays: `toks`/`merges` point at the first
 * length-prefixed (u64 LE) string of the tokenizer.ggml.tokens / .merges
 * arrays; `ttypes` at the first i32 of .token_type. Returns 0 on success. */
int b1tok_load(const void *toksv, int ntoks, const void *ttypesv,
               const void *mergesv, int nmerges) {
  init_nonprint();
  const unsigned char *p;
  int i;
  uint64_t total;

  g_ttype_copy = (int32_t *)malloc(sizeof(int32_t) * (size_t)ntoks);
  memcpy(g_ttype_copy, ttypesv, sizeof(int32_t) * (size_t)ntoks);
  g_ttypes = (const unsigned char *)g_ttype_copy;
  g_nttypes = ntoks;

  /* vocab: key bytes are the raw strings; decode bytes never longer. */
  total = 0;
  p = (const unsigned char *)toksv;
  for (i = 0; i < ntoks; i++) {
    uint64_t l; memcpy(&l, p, 8);
    total += l; p += 8 + l;
  }
  g_keybuf  = (unsigned char *)malloc((size_t)total + 16);
  g_decbuf  = (unsigned char *)malloc((size_t)total + 16);
  g_vocab   = (VEntry *)malloc(sizeof(VEntry) * (size_t)ntoks);
  g_decoff  = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)ntoks);
  g_declenv = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)ntoks);
  g_nvocab = ntoks;

  uint32_t koff = 0, doff = 0;
  p = (const unsigned char *)toksv;
  for (i = 0; i < ntoks; i++) {
    uint64_t l; memcpy(&l, p, 8);
    const unsigned char *sb = p + 8, *se = sb + l;
    memcpy(g_keybuf + koff, sb, l);
    g_vocab[i].off = koff; g_vocab[i].len = (uint32_t)l; g_vocab[i].id = i;
    koff += (uint32_t)l;
    g_decoff[i] = doff;
    if (g_ttype_copy[i] == 1) {           /* NORMAL: byte-level inverse */
      const unsigned char *q = sb;
      while (q < se) {
        int adv, cp = utf8_next(q, se, &adv);
        g_decbuf[doff++] = (unsigned char)level_to_byte(cp);
        q += adv;
      }
    } else {                              /* control / user-defined: literal */
      memcpy(g_decbuf + doff, sb, l);
      doff += (uint32_t)l;
    }
    g_declenv[i] = doff - g_decoff[i];
    p = se;
  }

  /* merges: "left right" split at the literal space. */
  total = 0;
  p = (const unsigned char *)mergesv;
  for (i = 0; i < nmerges; i++) {
    uint64_t l; memcpy(&l, p, 8);
    total += l; p += 8 + l;
  }
  g_mkeybuf = (unsigned char *)malloc((size_t)total + 16);
  g_merges  = (MEntry *)malloc(sizeof(MEntry) * (size_t)nmerges);
  g_nmerges = nmerges;

  uint32_t moff = 0;
  p = (const unsigned char *)mergesv;
  for (i = 0; i < nmerges; i++) {
    uint64_t l; memcpy(&l, p, 8);
    const unsigned char *sb = p + 8;
    uint64_t sp = 0;
    while (sp < l && sb[sp] != ' ') sp++;
    g_merges[i].aoff = moff;
    memcpy(g_mkeybuf + moff, sb, sp); moff += (uint32_t)sp;
    g_merges[i].alen = (uint32_t)sp;
    g_merges[i].boff = moff;
    if (sp < l) { memcpy(g_mkeybuf + moff, sb + sp + 1, l - sp - 1); moff += (uint32_t)(l - sp - 1); }
    g_merges[i].blen = (uint32_t)(sp < l ? l - sp - 1 : 0);
    g_merges[i].rank = i;
    p = sb + l;
  }

  qsort(g_vocab, g_nvocab, sizeof(VEntry), vcmp);
  qsort(g_merges, g_nmerges, sizeof(MEntry), mcmp);
  return 0;
}

/* byte-level token (canonical UTF-8) -> id, or -1. */
int b1tok_vlookup(const void *sv, int len) {
  const unsigned char *s = (const unsigned char *)sv;
  int lo = 0, hi = g_nvocab;
  while (lo < hi) {
    int m = (lo + hi) >> 1;
    int c = kcmp(g_keybuf + g_vocab[m].off, g_vocab[m].len, s, len);
    if (c == 0) return g_vocab[m].id;
    if (c < 0) lo = m + 1; else hi = m;
  }
  return -1;
}

/* merge pair (two byte-level UTF-8 strings) -> rank, or -1. */
int b1tok_mlookup(const void *av, int alen, const void *bv, int blen) {
  const unsigned char *a = (const unsigned char *)av, *b = (const unsigned char *)bv;
  int lo = 0, hi = g_nmerges;
  while (lo < hi) {
    int m = (lo + hi) >> 1;
    int c = kcmp(g_mkeybuf + g_merges[m].aoff, g_merges[m].alen, a, alen);
    if (c == 0) c = kcmp(g_mkeybuf + g_merges[m].boff, g_merges[m].blen, b, blen);
    if (c == 0) return g_merges[m].rank;
    if (c < 0) lo = m + 1; else hi = m;
  }
  return -1;
}

/* decode: id -> model bytes. Writes min(len, cap) into out; returns the length. */
int b1tok_decode(int id, void *out, int cap) {
  if (id < 0 || id >= g_nvocab) return 0;
  int n = (int)g_declenv[id];
  int w = n < cap ? n : cap;
  memcpy(out, g_decbuf + g_decoff[id], w);
  return n;
}

/* the GGUF token_type of an id (1 = normal, 3 = control, ...). */
int b1tok_ttype(int id) {
  if (id < 0 || id >= g_nttypes) return 0;
  return (int)g_ttype_copy[id];
}
