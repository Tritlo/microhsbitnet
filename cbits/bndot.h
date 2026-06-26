#ifndef BNDOT_H
#define BNDOT_H

/* The three weighted dots BitNet b1.58 2B4T inference needs. See bndot.c. */

/* W1.58 ternary weights {-1,0,+1}, 4 per byte as 2-bit fields (`shift` 0/2/4/6
 * selects the field), dotted with int8 activations. */
double bndot_ternary(const void *w, const void *a, int n, int shift);

/* bf16 weights dotted with float32 activations. */
double bndot_bf16(const void *w, const void *a, int n);

/* float32 weights dotted with float32 activations. */
double bndot_f32(const void *w, const void *a, int n);

#endif
