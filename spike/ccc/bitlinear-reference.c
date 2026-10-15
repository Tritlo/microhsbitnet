#define _POSIX_C_SOURCE 200809L

#include "bndot.h"
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#error "Build the serial BitLinear reference without OpenMP"
#endif

_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24,
               "Float fixtures require IEEE binary32");

typedef struct {
  char name[64];
  int columns;
  int rows;
  float weight_scale;
} Case;

enum Pattern { MIXED, ZERO, NEARZERO, ROUNDING, NONDYADIC };

static void fail(const char *message) {
  fprintf(stderr, "bitlinear-reference: %s\n", message);
  exit(EXIT_FAILURE);
}

static FILE *open_file(const char *directory, const char *name,
                       const char *suffix, const char *mode) {
  char path[4096];
  int length = snprintf(path, sizeof path, "%s/%s%s", directory, name, suffix);
  if (length < 0 || (size_t)length >= sizeof path)
    fail("Fixture path is too long");
  FILE *file = fopen(path, mode);
  if (!file) {
    perror(path);
    exit(EXIT_FAILURE);
  }
  return file;
}

static void *allocate(size_t bytes) {
  void *memory = malloc(bytes);
  if (!memory)
    fail("Cannot allocate fixture memory");
  return memory;
}

static void write_file(const char *directory, const char *name,
                       const char *suffix, const void *bytes, size_t count) {
  FILE *file = open_file(directory, name, suffix, "wb");
  if (fwrite(bytes, 1, count, file) != count || fclose(file) != 0)
    fail("Cannot write fixture");
}

static float read_scale(const char *directory, const Case *fixture) {
  FILE *file = open_file(directory, "real-bitlinear-cases.txt", "", "r");
  Case entry;
  while (fscanf(file, "%63s %d %d %f", entry.name, &entry.columns, &entry.rows,
                &entry.weight_scale) == 4) {
    if (strcmp(entry.name, fixture->name) == 0) {
      if (entry.columns != fixture->columns || entry.rows != fixture->rows ||
          !isfinite(entry.weight_scale) || entry.weight_scale <= 0)
        fail("Invalid model weight scale or matrix shape");
      fclose(file);
      return entry.weight_scale;
    }
  }
  fail("Missing model weight scale");
  return 0;
}

static void make_inputs(float *inputs, int columns, enum Pattern pattern) {
  static const float mixed[] = {0,          0x1p-20f,    -0x1p-20f,    0x1p-18f,
                                -0x1p-18f,  127.0f / 64, -127.0f / 64, 0x1p-7f,
                                3.0f / 128, -0x1p-7f,    -3.0f / 128};
  static const float nearzero[] = {0,         0x1p-20f, -0x1p-20f, 0x1p-18f,
                                   -0x1p-18f, 0x1p-17f, -0x1p-17f};
  static const float rounding[] = {127,  -127,  0.5f, -0.5f, 1.5f,   -1.5f,
                                   2.5f, -2.5f, 3.5f, -3.5f, 126.5f, -126.5f};
  for (int column = 0; column < columns; column++) {
    switch (pattern) {
    case ZERO:
      inputs[column] = 0;
      break;
    case NEARZERO:
      inputs[column] = nearzero[column % (sizeof nearzero / sizeof *nearzero)];
      break;
    case ROUNDING:
      inputs[column] = rounding[column % (sizeof rounding / sizeof *rounding)];
      break;
    case NONDYADIC:
      inputs[column] = column == 0 ? 1.234567f : column == 1 ? -1.234567f :
          ((float)((column % 253) - 126) + 0.5f) / (127.0f / 1.234567f);
      break;
    case MIXED:
      inputs[column] = (float)((column * 73 + 13) % 251 - 125) / 64;
      if (column % 32 < (int)(sizeof mixed / sizeof *mixed))
        inputs[column] = mixed[column % 32];
      break;
    }
  }
}

static float quantize(const float *inputs, int columns, int32_t *quantized,
                      enum Pattern pattern) {
  float maximum = 1.0e-5f;
  for (int column = 0; column < columns; column++) {
    if (!isfinite(inputs[column]))
      fail("Activation must be finite");
    maximum = fmaxf(maximum, fabsf(inputs[column]));
  }
  float scale = 127.0f / maximum;
  static const int32_t rounding[] = {127, -127, 0, 0,  2,   -2,
                                     2,   -2,   4, -4, 126, -126};
  static const int32_t nearzero[] = {0, 12, -12, 48, -48, 97, -97};
  for (int column = 0; column < columns; column++) {
    float rounded = nearbyintf(inputs[column] * scale);
    quantized[column] = (int32_t)fmaxf(-128, fminf(127, rounded));
    if (pattern == ROUNDING &&
        quantized[column] !=
            rounding[column % (sizeof rounding / sizeof *rounding)])
      fail("Nearest-even rounding check failed");
    if (pattern == NEARZERO &&
        quantized[column] !=
            nearzero[column % (sizeof nearzero / sizeof *nearzero)])
      fail("Activation scale floor check failed");
  }
  return scale;
}

static double seconds(void) {
  struct timespec timestamp;
  if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0)
    fail("Cannot read monotonic clock");
  return (double)timestamp.tv_sec + (double)timestamp.tv_nsec * 1.0e-9;
}

static int compare_double(const void *left, const void *right) {
  double first = *(const double *)left;
  double second = *(const double *)right;
  return (first > second) - (first < second);
}

static void run_case(const char *directory, FILE *manifest, const Case *fixture,
                     enum Pattern pattern) {
  int columns = fixture->columns;
  int rows = fixture->rows;
  if (columns <= 0 || columns > 6912 || rows <= 0 || rows % 4 != 0 ||
      (int64_t)columns * rows > 128 * 1024 * 1024)
    fail("Invalid matrix shape");
  size_t weight_bytes = (size_t)columns * (size_t)(rows / 4);
  uint8_t *weights = allocate(weight_bytes);
  float *inputs = allocate((size_t)columns * sizeof *inputs);
  int32_t *quantized = allocate((size_t)columns * sizeof *quantized);
  float *output = allocate((size_t)rows * sizeof *output);
  FILE *file = open_file(directory, pattern == MIXED ? fixture->name : "tail",
                         ".weights.bin", "rb");
  if (fread(weights, 1, weight_bytes, file) != weight_bytes ||
      fgetc(file) != EOF)
    fail("Packed weight size does not match the matrix shape");
  fclose(file);
  for (size_t offset = 0; offset < weight_bytes; offset++)
    for (int shift = 0; shift < 8; shift += 2)
      if (((weights[offset] >> shift) & 3) == 3)
        fail("Invalid ternary weight field");
  make_inputs(inputs, columns, pattern);
  float activation_scale = quantize(inputs, columns, quantized, pattern);
  for (int warmup = 0; warmup < 2; warmup++)
    bnbitlinear(weights, fixture->weight_scale, inputs, columns, rows, output);
  double samples[9];
  for (int repetition = 0; repetition < 9; repetition++) {
    double start = seconds();
    bnbitlinear(weights, fixture->weight_scale, inputs, columns, rows, output);
    samples[repetition] = seconds() - start;
  }
  for (int row = 0; row < rows; row++) {
    int shift = 2 * (row / (rows / 4));
    size_t offset = (size_t)(row % (rows / 4)) * (size_t)columns;
    int32_t total = 0;
    for (int column = 0; column < columns; column++)
      total += (((weights[offset + (size_t)column] >> shift) & 3) - 1) *
               quantized[column];
    float expected = (float)total * fixture->weight_scale / activation_scale;
    if (!isfinite(output[row]) || output[row] != expected)
      fail("bnbitlinear differs from the nearest-even integer reference");
  }
  if (pattern != MIXED)
    write_file(directory, fixture->name, ".weights.bin", weights, weight_bytes);
  write_file(directory, fixture->name, ".float.bin", inputs,
             (size_t)columns * sizeof *inputs);
  write_file(directory, fixture->name, ".quantized.bin", quantized,
             (size_t)columns * sizeof *quantized);
  write_file(directory, fixture->name, ".bitlinear.bin", output,
             (size_t)rows * sizeof *output);
  if (fprintf(manifest, "%s %d %d %.9g\n", fixture->name, columns, rows,
              (double)fixture->weight_scale) < 0)
    fail("Cannot write BitLinear manifest");
  qsort(samples, 9, sizeof *samples, compare_double);
  printf("%s: quantized i32 and bnbitlinear reference PASS; "
         "serial_bnbitlinear_median_ms=%.6f activation_scale=%.9g\n",
         fixture->name, samples[4] * 1000, (double)activation_scale);
  free(output);
  free(quantized);
  free(inputs);
  free(weights);
}

int main(int argc, char **argv) {
  if (argc != 2)
    fail("Usage: bitlinear-reference CASE_DIRECTORY");
  uint32_t endian = 1;
  if (*(const uint8_t *)&endian != 1 || fesetround(FE_TONEAREST) != 0)
    fail("Requires little-endian storage and nearest-even rounding");
  const char *names[] = {"tiny", "tail", "q-proj", "down-proj"};
  FILE *cases = open_file(argv[1], "cases.txt", "", "r");
  FILE *manifest = open_file(argv[1], "bitlinear-cases.txt", "", "w");
  Case fixture;
  Case tail;
  for (int index = 0; index < 4; index++) {
    if (fscanf(cases, "%63s %d %d", fixture.name, &fixture.columns,
               &fixture.rows) != 3 ||
        strcmp(fixture.name, names[index]) != 0)
      fail("Expected tiny, tail, q-proj, down-proj in cases.txt");
    fixture.weight_scale =
        index < 2 ? (index == 0 ? 0.5f : 0.25f) : read_scale(argv[1], &fixture);
    run_case(argv[1], manifest, &fixture, MIXED);
    if (index == 1)
      tail = fixture;
  }
  fclose(cases);
  if (tail.columns != 131 || tail.rows != 12)
    fail("Edge fixtures require the 131 by 12 tail matrix");
  const char *edges[] = {"zero", "nearzero", "rounding", "nondyadic"};
  for (int index = 0; index < 4; index++) {
    snprintf(tail.name, sizeof tail.name, "%s", edges[index]);
    run_case(argv[1], manifest, &tail, (enum Pattern)(index + 1));
  }
  if (fclose(manifest) != 0)
    fail("Cannot close BitLinear manifest");
  puts("Two warmups and nine timed bnbitlinear calls per case; no OpenMP. "
       "Fixture generation, validation, and file IO are outside timing.");
  return EXIT_SUCCESS;
}
