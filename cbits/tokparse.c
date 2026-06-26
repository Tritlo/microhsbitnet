/* A C parser for tokenizer.json (Llama-3 byte-level BPE). It scans the raw bytes
 * once, extracts the vocab / merges / added-token tables, and sorts them for
 * binary-search lookup — replacing the Haskell parse that turned the 16 MB file
 * into a 16-million-char String and sorted ~400 K entries in the combinator
 * runtime (~27 s -> well under a second). The interesting logic (pre-tokenizer,
 * BPE) stays in Haskell and calls bntok_vlookup / bntok_mlookup / bntok_decode.
 *
 * Keys are the canonical UTF-8 of the token's codepoints — exactly what the
 * Haskell side passes as packUTF8 of a byte-level string — so lookups match. The
 * decode table maps an id to the model bytes (vocab: the byte-level alphabet's
 * inverse `level_to_byte`; added tokens: the codepoints as raw bytes, overriding
 * the vocab entry, mirroring the old accumArray). */

#include "tokparse.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* ---- byte-level alphabet: codepoint -> model byte (inverse of the GPT-2 map) ---- */
static int g_nonprint[68];
static void init_nonprint(void) {
  int n = 0, b;
  for (b = 0; b <= 32; b++)   g_nonprint[n++] = b;
  for (b = 127; b <= 160; b++) g_nonprint[n++] = b;
  g_nonprint[n++] = 173;
}
static int level_to_byte(int cp) { return cp < 256 ? cp : g_nonprint[cp - 256]; }

/* canonical UTF-8 of a codepoint (cp <= 0xFFFF here); returns bytes written. */
static int utf8_emit(int cp, unsigned char *o) {
  if (cp < 0x80)  { o[0] = cp; return 1; }
  if (cp < 0x800) { o[0] = 0xC0 | (cp >> 6); o[1] = 0x80 | (cp & 0x3F); return 2; }
  o[0] = 0xE0 | (cp >> 12); o[1] = 0x80 | ((cp >> 6) & 0x3F); o[2] = 0x80 | (cp & 0x3F); return 3;
}

static int hexd(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return 0;
}

/* Parse a JSON string body (p just past the opening quote) into codepoints,
 * mirroring the Haskell parseJStr (which ran on already-UTF8-decoded text, so a
 * raw multi-byte char becomes one codepoint). Returns the position past the
 * closing quote; fills cps[0..*ncp). */
static const char *parse_jstr(const char *p, const char *end, int *cps, int *ncp) {
  int n = 0;
  while (p < end && n < 1020) {
    unsigned char c = (unsigned char)*p;
    if (c == '"') { p++; break; }
    if (c == '\\' && p + 1 < end) {
      unsigned char e = (unsigned char)p[1];
      if (e == 'u' && p + 5 < end) {
        cps[n++] = (hexd(p[2]) << 12) | (hexd(p[3]) << 8) | (hexd(p[4]) << 4) | hexd(p[5]);
        p += 6;
      } else if (e == 'n') { cps[n++] = 10; p += 2; }
      else if (e == 'r') { cps[n++] = 13; p += 2; }
      else if (e == 't') { cps[n++] = 9;  p += 2; }
      else { cps[n++] = e; p += 2; }
    } else if (c < 0x80) { cps[n++] = c; p++; }
    else if (c < 0xE0 && p + 1 < end) {
      cps[n++] = ((c & 0x1F) << 6) | ((unsigned char)p[1] & 0x3F); p += 2;
    } else if (c < 0xF0 && p + 2 < end) {
      cps[n++] = ((c & 0x0F) << 12) | (((unsigned char)p[1] & 0x3F) << 6) | ((unsigned char)p[2] & 0x3F);
      p += 3;
    } else if (p + 3 < end) {
      cps[n++] = ((c & 0x07) << 18) | (((unsigned char)p[1] & 0x3F) << 12)
                 | (((unsigned char)p[2] & 0x3F) << 6) | ((unsigned char)p[3] & 0x3F);
      p += 4;
    } else p++;
  }
  *ncp = n;
  return p;
}

/* Find the byte just after the first occurrence of `pat` in [p,end). */
static const char *after_pat(const char *p, const char *end, const char *pat) {
  int plen = (int)strlen(pat);
  for (; p + plen <= end; p++)
    if (memcmp(p, pat, plen) == 0) return p + plen;
  return end;
}

/* ---- the parsed tables (static; one tokenizer per process) ---- */
typedef struct { uint32_t off, len; int32_t id; } VEntry;
typedef struct { uint32_t aoff, alen, boff, blen; int32_t rank; } MEntry;

static unsigned char *g_keybuf; static uint32_t g_keylen;
static VEntry *g_vocab; static int g_nvocab;
static unsigned char *g_mkeybuf; static uint32_t g_mkeylen;
static MEntry *g_merges; static int g_nmerges;

#define MAXID 200000
static unsigned char *g_decbuf; static uint32_t g_declen_total;
static uint32_t g_decoff[MAXID]; static uint32_t g_declenv[MAXID];

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

/* Parse tokenizer.json. Returns 0 on success. */
int bntok_load(const void *bytes, int len) {
  const char *s = (const char *)bytes, *end = s + len;
  int cps[1024];
  init_nonprint();
  g_keybuf  = (unsigned char *)malloc(len);
  g_mkeybuf = (unsigned char *)malloc(len);
  g_decbuf  = (unsigned char *)malloc(len);
  g_vocab   = (VEntry *)malloc(sizeof(VEntry) * 200000);
  g_merges  = (MEntry *)malloc(sizeof(MEntry) * 500000);
  g_keylen = g_mkeylen = g_declen_total = 0;
  g_nvocab = g_nmerges = 0;
  memset(g_decoff, 0, sizeof g_decoff);
  memset(g_declenv, 0, sizeof g_declenv);

  /* vocab: "vocab": { "tok": id, ... } */
  const char *p = after_pat(s, end, "\"vocab\":");
  while (p < end && *p != '{') p++;
  if (p < end) p++;
  for (;;) {
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r' || *p == ',')) p++;
    if (p >= end || *p != '"') break;
    int ncp; p = parse_jstr(p + 1, end, cps, &ncp);
    while (p < end && *p != ':') p++;
    if (p < end) p++;
    while (p < end && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) p++;
    int id = 0; while (p < end && *p >= '0' && *p <= '9') { id = id * 10 + (*p - '0'); p++; }
    /* key = canonical UTF-8 of the codepoints */
    uint32_t off = g_keylen;
    for (int i = 0; i < ncp; i++) g_keylen += utf8_emit(cps[i], g_keybuf + g_keylen);
    g_vocab[g_nvocab].off = off; g_vocab[g_nvocab].len = g_keylen - off; g_vocab[g_nvocab].id = id;
    g_nvocab++;
    /* decode bytes = level_to_byte(codepoints) */
    if (id >= 0 && id < MAXID) {
      uint32_t doff = g_declen_total;
      for (int i = 0; i < ncp; i++) g_decbuf[g_declen_total++] = (unsigned char)level_to_byte(cps[i]);
      g_decoff[id] = doff; g_declenv[id] = g_declen_total - doff;
    }
  }

  /* added_tokens: [ { "id": N, ..., "content": "str", ... }, ... ] -> decode override */
  const char *ap = after_pat(s, end, "\"added_tokens\":");
  const char *aend = ap; while (aend < end && *aend != ']') aend++;
  for (;;) {
    const char *q = after_pat(ap, aend, "\"id\":");
    if (q >= aend) break;
    while (q < aend && (*q == ' ' || *q == '\n' || *q == '\t' || *q == '\r')) q++;
    int id = 0; while (q < aend && *q >= '0' && *q <= '9') { id = id * 10 + (*q - '0'); q++; }
    const char *cq = after_pat(q, aend, "\"content\":");
    while (cq < aend && *cq != '"') cq++;
    if (cq >= aend) break;
    int ncp; const char *r = parse_jstr(cq + 1, aend, cps, &ncp);
    if (id >= 0 && id < MAXID) {
      uint32_t doff = g_declen_total;
      for (int i = 0; i < ncp; i++) g_decbuf[g_declen_total++] = (unsigned char)(cps[i] & 0xFF);
      g_decoff[id] = doff; g_declenv[id] = g_declen_total - doff;   /* override vocab */
    }
    ap = r;
  }

  /* merges: [ "left right", ... ] (space-separated byte-level strings) */
  const char *mp = after_pat(s, end, "\"merges\":");
  const char *mend = mp; while (mend < end && *mend != ']') mend++;
  for (;;) {
    while (mp < mend && (*mp == ' ' || *mp == '\n' || *mp == '\t' || *mp == '\r' || *mp == ',')) mp++;
    if (mp >= mend || *mp != '"') break;
    int ncp; mp = parse_jstr(mp + 1, mend, cps, &ncp);
    int sp = 0; while (sp < ncp && cps[sp] != 32) sp++;          /* split at the literal space */
    uint32_t aoff = g_mkeylen;
    for (int i = 0; i < sp; i++) g_mkeylen += utf8_emit(cps[i], g_mkeybuf + g_mkeylen);
    uint32_t alen = g_mkeylen - aoff;
    uint32_t boff = g_mkeylen;
    for (int i = sp + 1; i < ncp; i++) g_mkeylen += utf8_emit(cps[i], g_mkeybuf + g_mkeylen);
    uint32_t blen = g_mkeylen - boff;
    g_merges[g_nmerges].aoff = aoff; g_merges[g_nmerges].alen = alen;
    g_merges[g_nmerges].boff = boff; g_merges[g_nmerges].blen = blen;
    g_merges[g_nmerges].rank = g_nmerges;
    g_nmerges++;
  }

  qsort(g_vocab, g_nvocab, sizeof(VEntry), vcmp);
  qsort(g_merges, g_nmerges, sizeof(MEntry), mcmp);
  return 0;
}

/* byte-level token (canonical UTF-8) -> id, or -1. */
int bntok_vlookup(const void *sv, int len) {
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

/* merge pair (two canonical UTF-8 strings) -> rank, or -1. */
int bntok_mlookup(const void *av, int alen, const void *bv, int blen) {
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
int bntok_decode(int id, void *out, int cap) {
  if (id < 0 || id >= MAXID) return 0;
  int n = (int)g_declenv[id];
  int w = n < cap ? n : cap;
  memcpy(out, g_decbuf + g_decoff[id], w);
  return n;
}
