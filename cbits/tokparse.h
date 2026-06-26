#ifndef TOKPARSE_H
#define TOKPARSE_H

/* A C parser for tokenizer.json (Llama-3 byte-level BPE). See tokparse.c. */

/* Parse the file bytes and build the vocab/merge/decode tables. Returns 0 on ok. */
int bntok_load(const void *bytes, int len);

/* byte-level token (canonical UTF-8) -> id, or -1 if absent. */
int bntok_vlookup(const void *s, int len);

/* merge pair (two canonical UTF-8 strings) -> rank, or -1 if absent. */
int bntok_mlookup(const void *a, int alen, const void *b, int blen);

/* decode id -> model bytes; writes min(len, cap) into out, returns the length. */
int bntok_decode(int id, void *out, int cap);

#endif
