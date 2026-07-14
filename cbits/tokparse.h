#ifndef TOKPARSE_H
#define TOKPARSE_H

/* The C tables of the qwen35 byte-level BPE tokenizer. See tokparse.c. */

/* Build the tables from the GGUF metadata arrays (length-prefixed strings for
 * tokens/merges, i32s for token_type). Returns 0 on ok. */
int b1tok_load(const void *toks, int ntoks, const void *ttypes,
               const void *merges, int nmerges);

/* byte-level token (canonical UTF-8) -> id, or -1 if absent. */
int b1tok_vlookup(const void *s, int len);

/* merge pair (two byte-level UTF-8 strings) -> rank, or -1 if absent. */
int b1tok_mlookup(const void *a, int alen, const void *b, int blen);

/* decode id -> model bytes; writes min(len, cap) into out, returns the length. */
int b1tok_decode(int id, void *out, int cap);

/* the GGUF token_type of an id (1 = normal, 3 = control, ...). */
int b1tok_ttype(int id);

#endif
