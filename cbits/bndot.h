#ifndef BNDOT_H
#define BNDOT_H

/* The hot kernels of BitNet b1.58 2B4T inference. See bndot.c. */

/* ---- dot kernels (batch path) ---- */

/* A full W1.58 BitLinear: packed ternary weight × int8 acts → outDim raw dots. */
void bnmatmul_ternary(const void *w, const void *a, int inDim, int outDim, double *out);

/* The tied LM head: argmax over `vocab` of bf16 embedding · f32 hidden. */
int bnargmax_bf16(const void *embed, const void *hF32, int hidden, int vocab, double *bestVal);

/* One float32 × float32 dot (attention q·k). */
double bndot_f32(const void *w, const void *a, int n);

/* ---- per-token vector math (generation path) ---- */

/* out = w * x / sqrt(mean(x^2) + eps). */
void bnrmsnorm(const void *w, const void *x, int n, float eps, float *out);

/* Fused BitLinear: absmax-quantize x to int8, ternary matmul, rescale → out. */
void bnbitlinear(const void *packed, float wscale, const void *x, int inDim, int outDim, float *out);

/* In-place rotary embedding on each of nHeads heads of length headDim. */
void bnrope(void *vec, int pos, int nHeads, int headDim, float theta);

/* GQA causal attention of one query against nPos cached keys/values. */
void bnattend(const void *q, const void *cK, const void *cV, int nPos,
              int nHeads, int nKV, int headDim, float scale, float *out);

/* Squared-ReLU gated MLP: out = max(0,g)^2 * u. */
void bnsqrelu_mul(const void *g, const void *u, int n, float *out);

/* Residual add: out = a + b. */
void bnadd(const void *a, const void *b, int n, float *out);

#endif
