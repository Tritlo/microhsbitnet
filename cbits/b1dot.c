/* The hot kernels of 1-bit Bonsai 27B inference, in C so the loops run natively
 * instead of as MicroHs combinator graph reduction. Called from Haskell via
 * plain `foreign import ccall` (see src/Dot.hs) — no runtime patch, no custom
 * primitive.
 *
 * The 1-bit pieces mirror the prism llama.cpp fork's CPU reference exactly
 * (ggml_vec_dot_q1_0_q8_0_generic + quantize_row_q8_0_ref, including the fp16
 * round-trip of the activation scales), so a row dot here equals the fork's
 * bit for bit. The f32 vector helpers (rmsnorm/rope/attend/add) carry over
 * from the ternary kernel; rope gains partial-rotary support (n_rot < headDim).
 *
 * Quantization scales live inside the Q1_0 blocks; activation quantization is
 * fused into the matmul. Thread count is BNDOT_THREADS (default 12). */

#include "b1dot.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

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

/* ---------- fp16, ported from ggml-impl.h so rounding matches the fork ---- */

static inline float fp32_from_bits(uint32_t w) {
  float f; memcpy(&f, &w, sizeof f); return f;
}
static inline uint32_t fp32_to_bits(float f) {
  uint32_t w; memcpy(&w, &f, sizeof w); return w;
}

static float f16_to_f32(uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;
  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
  const float exp_scale = 0x1.0p-112f;
  const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result = sign |
    (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

static uint16_t f32_to_f16(float f) {
  const float scale_to_inf  = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;
  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);
  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (uint16_t)((sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

static inline float f16_at(const unsigned char *p) {
  uint16_t h; memcpy(&h, p, sizeof h); return f16_to_f32(h);
}

/* ---------- Q1_0 (18-byte blocks: fp16 scale + 128 sign bits, LSB-first) --- */

/* Quantize n f32 activations q8_0-style: per 32 values one scale (fp16
 * round-tripped, exactly like quantize_row_q8_0_ref) and 32 int8 quants. */
static void quant_q8(const float *x, int n, signed char *qs, float *sc) {
  int nb = n / 32;
  for (int i = 0; i < nb; i++) {
    float amax = 0.0f;
    for (int j = 0; j < 32; j++) {
      float v = fabsf(x[i * 32 + j]);
      if (v > amax) amax = v;
    }
    float d = amax / 127.0f;
    float id = d ? 1.0f / d : 0.0f;
    sc[i] = f16_to_f32(f32_to_f16(d));
    for (int j = 0; j < 32; j++) qs[i * 32 + j] = (signed char)roundf(x[i * 32 + j] * id);
  }
}

/* One Q1_0 row dot against pre-quantized activations: bit set -> +a, clear ->
 * -a. Accumulation order matches ggml_vec_dot_q1_0_q8_0_generic. */
static float b1_row(const unsigned char *w, const signed char *a, const float *sc, int n) {
  int nb = n / 128;
  float sumf = 0.0f;
  for (int i = 0; i < nb; i++) {
    const unsigned char *blk = w + 18 * i;
    float d0 = f16_at(blk);
    float sumi = 0.0f;
    for (int k = 0; k < 4; k++) {
      const unsigned char *bits = blk + 2 + 4 * k;
      const signed char *qy = a + i * 128 + k * 32;
      int sumi_block = 0;
      for (int b = 0; b < 4; ++b, qy += 8) {
        unsigned mask = bits[b];
        sumi_block += ((mask & 0x01) ? qy[0] : -qy[0])
                   +  ((mask & 0x02) ? qy[1] : -qy[1])
                   +  ((mask & 0x04) ? qy[2] : -qy[2])
                   +  ((mask & 0x08) ? qy[3] : -qy[3])
                   +  ((mask & 0x10) ? qy[4] : -qy[4])
                   +  ((mask & 0x20) ? qy[5] : -qy[5])
                   +  ((mask & 0x40) ? qy[6] : -qy[6])
                   +  ((mask & 0x80) ? qy[7] : -qy[7]);
      }
      sumi += sc[i * 4 + k] * (float)sumi_block;
    }
    sumf += d0 * sumi;
  }
  return sumf;
}

/* A full B1Linear, fused: quantize the f32 activations once, then outDim row
 * dots of the Q1_0 weight (row-major, inDim/128 blocks per row). */
void b1matmul(const void *wv, const void *xv, int inDim, int outDim, float *out) {
  const unsigned char *w = (const unsigned char *)wv;
  const float *x = (const float *)xv;
  int nb = inDim / 32;
  signed char *qs = (signed char *)malloc((size_t)inDim + (size_t)nb * sizeof(float));
  float *sc = (float *)(qs + inDim);
  quant_q8(x, inDim, qs, sc);
  size_t rowBytes = (size_t)(inDim / 128) * 18;
  int r;
  #pragma omp parallel for num_threads(bn_threads()) schedule(static)
  for (r = 0; r < outDim; r++)
    out[r] = b1_row(w + (size_t)r * rowBytes, qs, sc, inDim);
  free(qs);
}

/* Dequantize one Q1_0 row of rowLen weights (the embedding lookup). */
void b1embed(const void *wv, int row, int rowLen, float *out) {
  const unsigned char *w = (const unsigned char *)wv + (size_t)row * (rowLen / 128) * 18;
  for (int i = 0; i < rowLen / 128; i++) {
    const unsigned char *blk = w + 18 * i;
    float d = f16_at(blk);
    for (int j = 0; j < 128; j++) {
      unsigned bit = (blk[2 + j / 8] >> (j % 8)) & 1;
      out[i * 128 + j] = bit ? d : -d;
    }
  }
}

/* ---------- f32 vector math (carried from the ternary kernel) ------------- */

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

/* Rotary position embedding, in place, neox pairing with partial rotary: in
 * each head only the first nRot dims rotate, pairs (k, k+nRot/2),
 * inv_freq[k] = theta^(-2k/nRot); dims >= nRot pass through. */
void bnrope(void *vecv, int pos, int nHeads, int headDim, int nRot, float theta) {
  float *vec = (float *)vecv;
  int half = nRot / 2;
  int k, h;
  static float inv[512];
  static int   inv_nr = 0;
  static float inv_th = 0.0f;
  if (inv_nr != nRot || inv_th != theta) {
    for (k = 0; k < half; k++) inv[k] = powf(theta, -2.0f * (float)k / (float)nRot);
    inv_nr = nRot; inv_th = theta;
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

/* Residual add: out[i] = a[i] + b[i]. */
void bnadd(const void *av, const void *bv, int n, float *out) {
  const float *a = (const float *)av;
  const float *b = (const float *)bv;
  int i;
  for (i = 0; i < n; i++) out[i] = a[i] + b[i];
}

/* Per-head RMSNorm: each of nHeads spans of headDim normalized independently
 * against the shared headDim-long weight (the q/k norms). */
void bnrmsnorm_heads(const void *wv, const void *xv, int nHeads, int headDim,
                     float eps, float *out) {
  const float *w = (const float *)wv;
  const float *x = (const float *)xv;
  for (int h = 0; h < nHeads; h++) {
    const float *xh = x + h * headDim;
    float *oh = out + h * headDim;
    float ss = 0.0f;
    for (int i = 0; i < headDim; i++) ss += xh[i] * xh[i];
    float s = 1.0f / sqrtf(ss / (float)headDim + eps);
    for (int i = 0; i < headDim; i++) oh[i] = w[i] * (xh[i] * s);
  }
}

/* Deinterleave the fused Q projection: per head, headDim query dims then
 * headDim gate dims (stride 2*headDim). */
void bnsplit_qgate(const void *qgv, int nHeads, int headDim, float *q, float *gate) {
  const float *qg = (const float *)qgv;
  for (int h = 0; h < nHeads; h++) {
    memcpy(q    + h * headDim, qg + h * 2 * headDim,           sizeof(float) * (size_t)headDim);
    memcpy(gate + h * headDim, qg + h * 2 * headDim + headDim, sizeof(float) * (size_t)headDim);
  }
}

/* Output gate: out[i] = a[i] * sigmoid(g[i]). */
void bnsigmoid_mul(const void *gv, const void *av, int n, float *out) {
  const float *g = (const float *)gv;
  const float *a = (const float *)av;
  for (int i = 0; i < n; i++) out[i] = a[i] / (1.0f + expf(-g[i]));
}

/* SwiGLU elementwise: out[i] = silu(g[i]) * u[i]. */
void bnsilu_mul(const void *gv, const void *uv, int n, float *out) {
  const float *g = (const float *)gv;
  const float *u = (const float *)uv;
  for (int i = 0; i < n; i++) out[i] = (g[i] / (1.0f + expf(-g[i]))) * u[i];
}

/* ---------- gated delta net (the 48 linear-attention layers) -------------- */

/* Causal 4-tap depthwise conv + SiLU, one token: ring holds the previous
 * 3 channel vectors (oldest first), x the current one; kernel is per-channel
 * contiguous (ggml ssm_conv orientation: coefficient r pairs with window
 * position r, 0 = oldest). */
void gdn_conv_silu(const void *ringv, const void *xv, const void *kernelv,
                   int channels, float *out) {
  const float *ring = (const float *)ringv;
  const float *x    = (const float *)xv;
  const float *ker  = (const float *)kernelv;
  for (int c = 0; c < channels; c++) {
    float s = ring[0 * channels + c] * ker[c * 4 + 0]
            + ring[1 * channels + c] * ker[c * 4 + 1]
            + ring[2 * channels + c] * ker[c * 4 + 2]
            + x[c]                   * ker[c * 4 + 3];
    out[c] = s / (1.0f + expf(-s));
  }
}

/* Per-head L2 norm: y = x / max(sqrt(sum x^2), eps), sum in double
 * (ggml_compute_forward_l2_norm_f32 semantics). */
void gdn_l2norm_heads(const void *xv, int nHeads, int headDim, float eps, float *out) {
  const float *x = (const float *)xv;
  for (int h = 0; h < nHeads; h++) {
    const float *xh = x + h * headDim;
    double ss = 0.0;
    for (int i = 0; i < headDim; i++) ss += (double)(xh[i] * xh[i]);
    float scale = 1.0f / fmaxf(sqrtf((float)ss), eps);
    for (int i = 0; i < headDim; i++) out[h * headDim + i] = xh[i] * scale;
  }
}

/* The gated delta rule, one token, all heads — mirrors the fused CPU op
 * (ggml_compute_forward_gated_delta_net_one_chunk, non-KDA):
 *   beta = sigmoid(betaP);  g = a * softplus(alphaP + dt)
 *   S *= exp(g);  d[j] = (v[j] - sum_i S[i][j] k[i]) * beta
 *   S[i][j] += k[i] d[j];  out[j] = sum_i S[i][j] q[i] / sqrt(d)
 * State layout transposed: s[j*d + i]. q/k carry nK heads; v-head h uses
 * q/k head (h % nK). state_out gets the updated copy (caller owns purity). */
void gdn_step(const void *state_in, const void *qv, const void *kv, const void *vv,
              const void *alphav, const void *dtv, const void *av, const void *betav,
              int nV, int nK, int d, float *out, float *state_out) {
  const float *q  = (const float *)qv;
  const float *k  = (const float *)kv;
  const float *v  = (const float *)vv;
  const float *al = (const float *)alphav;
  const float *dt = (const float *)dtv;
  const float *a  = (const float *)av;
  const float *bp = (const float *)betav;
  const float scale = 1.0f / sqrtf((float)d);
  memcpy(state_out, state_in, sizeof(float) * (size_t)nV * d * d);
  int h;
  #pragma omp parallel for num_threads(bn_threads()) schedule(static)
  for (h = 0; h < nV; h++) {
    float *S = state_out + (size_t)h * d * d;
    const float *kh = k + (size_t)(h % nK) * d;
    const float *qh = q + (size_t)(h % nK) * d;
    const float *vh = v + (size_t)h * d;
    float sp = al[h] + dt[h];
    sp = (sp > 20.0f) ? sp : logf(1.0f + expf(sp));
    float g = a[h] * sp;
    float beta = 1.0f / (1.0f + expf(-bp[h]));
    float eg = expf(g);
    float delta[512]; /* d <= 512 */
    for (int j = 0; j < d * d; j++) S[j] *= eg;
    for (int j = 0; j < d; j++) {
      float sum = 0.0f;
      const float *Sj = S + (size_t)j * d;
      for (int i = 0; i < d; i++) sum += Sj[i] * kh[i];
      delta[j] = (vh[j] - sum) * beta;
    }
    for (int j = 0; j < d; j++) {
      float *Sj = S + (size_t)j * d;
      float dj = delta[j];
      for (int i = 0; i < d; i++) Sj[i] += kh[i] * dj;
    }
    float *oh = out + (size_t)h * d;
    for (int j = 0; j < d; j++) {
      float sum = 0.0f;
      const float *Sj = S + (size_t)j * d;
      for (int i = 0; i < d; i++) sum += Sj[i] * qh[i];
      oh[j] = sum * scale;
    }
  }
}
