/* R1 oracle: does qwen35's ggml_rope_multi (IMROPE, sections [11,11,10,0],
 * n_rot 64, theta 1e7) with equal per-stream text positions reduce to plain
 * neox rope over the first 64 dims? Computes both on the same input and
 * prints the max abs diff. */
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define HD 256
#define NH 24
#define NT 3
#define NROT 64

static void my_rope(float *vec, int pos, int nHeads, int headDim, int nRot, float theta) {
  int half = nRot / 2;
  for (int h = 0; h < nHeads; h++) {
    float *hv = vec + h * headDim;
    for (int k = 0; k < half; k++) {
      float a = (float)pos * powf(theta, -2.0f * (float)k / (float)nRot);
      float c = cosf(a), s = sinf(a);
      float x1 = hv[k], x2 = hv[half + k];
      hv[k]        = x1 * c - x2 * s;
      hv[half + k] = x2 * c + x1 * s;
    }
  }
}

int main(void) {
  struct ggml_init_params ip = { 1024*1024*64, NULL, 0 };
  struct ggml_context *ctx = ggml_init(ip);

  struct ggml_tensor *q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HD, NH, NT);
  struct ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NT * 4);

  float *qd = (float *)q->data;
  for (int i = 0; i < HD*NH*NT; i++) qd[i] = sinf(0.37f * (float)i) * 2.0f;

  int32_t *pd = (int32_t *)pos->data;
  /* llama.cpp text layout: 4 streams of n_tokens each (stream-major), all equal */
  int positions[NT] = {0, 1, 7};
  for (int s = 0; s < 4; s++)
    for (int t = 0; t < NT; t++)
      pd[s*NT + t] = positions[t];

  int sections[4] = {11, 11, 10, 0};
  struct ggml_tensor *out = ggml_rope_multi(ctx, q, pos, NULL, NROT, sections,
      GGML_ROPE_TYPE_IMROPE, 262144, 10000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

  struct ggml_cgraph *gf = ggml_new_graph(ctx);
  ggml_build_forward_expand(gf, out);
  ggml_graph_compute_with_ctx(ctx, gf, 4);

  /* my neox on a copy of the raw input */
  float *mine = (float *)malloc(sizeof(float)*HD*NH*NT);
  for (int i = 0; i < HD*NH*NT; i++) mine[i] = sinf(0.37f * (float)i) * 2.0f;
  for (int t = 0; t < NT; t++)
    my_rope(mine + t*HD*NH, positions[t], NH, HD, NROT, 10000000.0f);

  float *od = (float *)out->data;
  float maxd = 0; int maxi = -1;
  for (int i = 0; i < HD*NH*NT; i++) {
    float d = fabsf(od[i] - mine[i]);
    if (d > maxd) { maxd = d; maxi = i; }
  }
  printf("max abs diff = %g at %d (ggml=%g mine=%g)\n", maxd, maxi,
         maxi >= 0 ? od[maxi] : 0.0f, maxi >= 0 ? mine[maxi] : 0.0f);
  printf("sample ggml[0..5] = %g %g %g %g %g %g\n", od[0], od[1], od[2], od[3], od[4], od[5]);
  printf("pos7 head0 dims 0,32 ggml: %g %g  mine: %g %g\n",
         od[2*HD*NH+0], od[2*HD*NH+32], mine[2*HD*NH+0], mine[2*HD*NH+32]);
  return 0;
}
