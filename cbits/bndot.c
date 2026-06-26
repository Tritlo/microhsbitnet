/* The hot kernels of BitNet b1.58 2B4T inference, in C so the loops run natively
 * instead of as MicroHs combinator graph reduction. Called from Haskell via plain
 * `foreign import ccall` (see src/Dot.hs) — no runtime patch, no custom primitive.
 *
 * Two groups:
 *   - the dot kernels (bnmatmul_ternary / bnargmax_bf16 / bndot_f32), batched so
 *     one FFI call does a whole matmul / vocab scan;
 *   - the per-token vector math of the generation path (bnbitlinear, bnrmsnorm,
 *     bnrope, bnattend, bnsqrelu_mul, bnadd), so the whole decoder step runs in C
 *     on float32 buffers rather than as combinator graph reduction.
 *
 * Quantization scales are the caller's job (ggml-style vec_dot). Thread count is
 * BNDOT_THREADS (default 12). Float reads in the dot kernels use memcpy
 * (alignment-safe); the vector-math kernels take naturally-aligned float buffers. */

#include "bndot.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* Thread count: BNDOT_THREADS env (default 12), read once. */
static int bn_threads(void) {
  static int t = 0;
  if (t == 0) {
    const char *e = getenv("BNDOT_THREADS");
    t = e ? atoi(e) : 12;
    if (t < 1) t = 1;
  }
  return t;
}

static double load_f32(const unsigned char *p) {
  float f;
  memcpy(&f, p, sizeof f);
  return (double)f;
}

#if defined(__AVX2__)
static long hsum_epi32(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  __m128i s  = _mm_add_epi32(lo, hi);
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
  return (long)_mm_cvtsi128_si32(s);
}
#endif

/* One ternary row dot: sum_c (((w[c] >> shift) & 3) - 1) * (int8)a[c]. Integer,
 * so the vectorized and scalar paths give exactly the same value. */
static long ternary_row(const unsigned char *w, const signed char *a, int n, int shift) {
  long acc = 0;
  int c = 0;
#if defined(__AVX2__)
  const __m128i vsh = _mm_cvtsi32_si128(shift);
  const __m256i m3  = _mm256_set1_epi8(3);
  const __m256i one = _mm256_set1_epi8(1);
  __m256i vacc = _mm256_setzero_si256();
  for (; c + 32 <= n; c += 32) {
    __m256i wb   = _mm256_loadu_si256((const __m256i *)(w + c));
    __m256i fld  = _mm256_and_si256(_mm256_srl_epi16(wb, vsh), m3); /* 2-bit field, per byte */
    __m256i trit = _mm256_sub_epi8(fld, one);                      /* field-1 ∈ {-1,0,1} */
    __m256i va   = _mm256_loadu_si256((const __m256i *)(a + c));    /* int8 activations */
    __m256i prod = _mm256_sign_epi8(va, trit);                     /* trit*act (sign trick) */
    __m256i p16  = _mm256_maddubs_epi16(one, prod);
    __m256i p32  = _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
    vacc = _mm256_add_epi32(vacc, p32);
  }
  acc += hsum_epi32(vacc);
#endif
  for (; c < n; c++)
    acc += (long)(((w[c] >> shift) & 3) - 1) * (long)a[c];
  return acc;
}

/* ---------- dot kernels (used by the batch path: M1/M2/M3) ---------- */

/* A full W1.58 BitLinear: packed weight × int8 activations, outDim raw dots. */
void bnmatmul_ternary(const void *wv, const void *av, int inDim, int outDim, double *out) {
  const unsigned char *w = (const unsigned char *)wv;
  const signed char   *a = (const signed char *)av;
  int chunk = outDim / 4;
  int r;
  #pragma omp parallel for num_threads(bn_threads()) schedule(static)
  for (r = 0; r < outDim; r++) {
    int shift = 2 * (r / chunk);
    const unsigned char *wr = w + (long)(r % chunk) * inDim;
    out[r] = (double)ternary_row(wr, a, inDim, shift);
  }
}

/* The tied LM head: argmax over `vocab` of (bf16 embedding row v) · (f32 hidden).
 * Each dot is a sequential double — identical to one-at-a-time — so only the vocab
 * scan is parallel. Earliest index wins ties. Returns argmax; *bestVal its logit. */
int bnargmax_bf16(const void *embed, const void *hF32, int hidden, int vocab, double *bestVal) {
  const unsigned char *e = (const unsigned char *)embed;
  const unsigned char *h = (const unsigned char *)hF32;
  int bestI = 0;
  float bestV = -INFINITY;
  #pragma omp parallel num_threads(bn_threads())
  {
    int   li = 0;
    float lv = -INFINITY;
    int v;
    #pragma omp for schedule(static)
    for (v = 0; v < vocab; v++) {
      const unsigned char *ev = e + (long)v * hidden * 2;
      double acc = 0.0;
      int c;
      for (c = 0; c < hidden; c++) {
        uint32_t bits = ((uint32_t)((uint32_t)ev[2 * c] | ((uint32_t)ev[2 * c + 1] << 8))) << 16;
        float wv;
        memcpy(&wv, &bits, sizeof wv);
        acc += (double)wv * load_f32(h + 4 * c);
      }
      float l = (float)acc;
      if (l > lv) { lv = l; li = v; }
    }
    #pragma omp critical
    {
      if (lv > bestV || (lv == bestV && li < bestI)) { bestV = lv; bestI = li; }
    }
  }
  *bestVal = (double)bestV;
  return bestI;
}

/* Decode n bf16 values (little-endian) to float32. For one-time weight setup
 * (e.g. the RMSNorm weights) — bf16->f32 is the high 16 bits, so (bits << 16). */
void bf16_to_f32(const void *srcv, int n, float *out) {
  const unsigned char *s = (const unsigned char *)srcv;
  int i;
  for (i = 0; i < n; i++) {
    uint32_t bits = ((uint32_t)((uint32_t)s[2 * i] | ((uint32_t)s[2 * i + 1] << 8))) << 16;
    memcpy(&out[i], &bits, sizeof out[i]);
  }
}

/* One float32 × float32 dot (attention q·k in the batch path). */
double bndot_f32(const void *wv, const void *av, int n) {
  const unsigned char *w = (const unsigned char *)wv;
  const unsigned char *a = (const unsigned char *)av;
  double acc = 0.0;
  int c;
  for (c = 0; c < n; c++)
    acc += load_f32(w + 4 * c) * load_f32(a + 4 * c);
  return acc;
}

/* ---------- per-token vector math (the generation path: M4) ----------
 * All take/produce naturally-aligned float32 buffers. Float ops are sequenced to
 * match the Haskell reference (left-fold sums, single-precision libm). */

/* RMSNorm: out[i] = w[i] * x[i] / sqrt(mean(x^2) + eps). */
void bnrmsnorm(const void *wv, const void *xv, int n, float eps, float *out) {
  const float *w = (const float *)wv;
  const float *x = (const float *)xv;
  float ss = 0.0f;
  int i;
  for (i = 0; i < n; i++) ss += x[i] * x[i];
  float s = 1.0f / sqrtf(ss / (float)n + eps);
  for (i = 0; i < n; i++) out[i] = w[i] * (x[i] * s);
}

/* A full BitLinear, fused: absmax-quantize the f32 activations to int8, ternary
 * matmul, rescale. out[r] = dot_r * wscale / actscale. */
void bnbitlinear(const void *packed, float wscale, const void *xv, int inDim, int outDim, float *out) {
  const float *x = (const float *)xv;
  float amax = 0.0f;
  int i;
  for (i = 0; i < inDim; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
  if (amax < 1.0e-5f) amax = 1.0e-5f;
  float sc = 127.0f / amax;
  signed char xq[inDim];   /* stack scratch (inDim <= 6912) */
  for (i = 0; i < inDim; i++) {
    long q = lrintf(x[i] * sc);
    if (q < -128) q = -128; else if (q > 127) q = 127;
    xq[i] = (signed char)q;
  }
  const unsigned char *w = (const unsigned char *)packed;
  int chunk = outDim / 4;
  int r;
  #pragma omp parallel for num_threads(bn_threads()) schedule(static)
  for (r = 0; r < outDim; r++) {
    int shift = 2 * (r / chunk);
    const unsigned char *wr = w + (long)(r % chunk) * inDim;
    out[r] = (float)ternary_row(wr, xq, inDim, shift) * wscale / sc;
  }
}

/* Rotary position embedding, in place, on each of nHeads heads of length headDim.
 * inv_freq[k] = theta^(-2k/headDim); rotate the (k, half+k) pair by pos*inv_freq[k]. */
void bnrope(void *vecv, int pos, int nHeads, int headDim, float theta) {
  float *vec = (float *)vecv;
  int half = headDim / 2;
  int k, h;
  /* inv_freq is constant across the run; compute it once (keyed by headDim/theta).
   * Called from the single Haskell thread, so the static cache needs no lock. */
  static float inv[512];
  static int   inv_hd = 0;
  static float inv_th = 0.0f;
  if (inv_hd != headDim || inv_th != theta) {
    for (k = 0; k < half; k++) inv[k] = powf(theta, -2.0f * (float)k / (float)headDim);
    inv_hd = headDim; inv_th = theta;
  }
  for (h = 0; h < nHeads; h++) {
    float *hv = vec + h * headDim;
    for (k = 0; k < half; k++) {
      float a  = (float)pos * inv[k];
      float c  = cosf(a), s = sinf(a);
      float x1 = hv[k], x2 = hv[half + k];
      hv[k]        = x1 * c - x2 * s;
      hv[half + k] = x2 * c + x1 * s;
    }
  }
}

/* GQA causal attention for one new query against the cached keys/values. q is
 * nHeads*headDim; cK/cV are nPos*(nKV*headDim). Per query head: scaled dot with
 * each cached key, softmax, weighted sum of cached values. */
void bnattend(const void *qv, const void *cKv, const void *cVv, int nPos,
              int nHeads, int nKV, int headDim, float scale, float *out) {
  const float *q  = (const float *)qv;
  const float *cK = (const float *)cKv;
  const float *cV = (const float *)cVv;
  int groups = nHeads / nKV;
  int kvDim  = nKV * headDim;
  int h;
  #pragma omp parallel for num_threads(bn_threads()) schedule(static)
  for (h = 0; h < nHeads; h++) {
    int kv = h / groups;
    const float *qh = q + h * headDim;
    float scores[nPos];   /* per-thread stack scratch */
    float m = -INFINITY;
    int p, c;
    for (p = 0; p < nPos; p++) {
      const float *kh = cK + (long)p * kvDim + kv * headDim;
      float d = 0.0f;
      for (c = 0; c < headDim; c++) d += qh[c] * kh[c];
      d *= scale;
      scores[p] = d;
      if (d > m) m = d;
    }
    float sum = 0.0f;
    for (p = 0; p < nPos; p++) { scores[p] = expf(scores[p] - m); sum += scores[p]; }
    float *oh = out + h * headDim;
    for (c = 0; c < headDim; c++) oh[c] = 0.0f;
    for (p = 0; p < nPos; p++) {
      float wgt = scores[p] / sum;
      const float *vh = cV + (long)p * kvDim + kv * headDim;
      for (c = 0; c < headDim; c++) oh[c] += wgt * vh[c];
    }
  }
}

/* Squared-ReLU gated MLP elementwise: out[i] = max(0, g[i])^2 * u[i]. */
void bnsqrelu_mul(const void *gv, const void *uv, int n, float *out) {
  const float *g = (const float *)gv;
  const float *u = (const float *)uv;
  int i;
  for (i = 0; i < n; i++) { float r = g[i] > 0.0f ? g[i] : 0.0f; out[i] = r * r * u[i]; }
}

/* Residual add: out[i] = a[i] + b[i]. */
void bnadd(const void *av, const void *bv, int n, float *out) {
  const float *a = (const float *)av;
  const float *b = (const float *)bv;
  int i;
  for (i = 0; i < n; i++) out[i] = a[i] + b[i];
}
