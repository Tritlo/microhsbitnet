#ifndef B1DOT_H
#define B1DOT_H

/* 1-bit (Q1_0) kernels */
void b1matmul(const void *w, const void *x, int inDim, int outDim, float *out);
void b1embed(const void *w, int row, int rowLen, float *out);

/* f32 vector math */
void bnrmsnorm(const void *w, const void *x, int n, float eps, float *out);
void bnrope(void *vec, int pos, int nHeads, int headDim, int nRot, float theta);
void bnattend(const void *q, const void *cK, const void *cV, int nPos,
              int nHeads, int nKV, int headDim, float scale, float *out);
void bnadd(const void *a, const void *b, int n, float *out);

#endif
