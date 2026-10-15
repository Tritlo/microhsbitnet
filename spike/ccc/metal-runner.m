#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "bndot.h"
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_OPENMP)
#error "Build without OpenMP for the serial C baseline."
#endif
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "The fixture format requires a little-endian host."
#endif

enum { Repetitions = 9, MaximumCases = 64 };
static const NSUInteger MaximumFixtureBytes = 256 * 1024 * 1024;
static const uint64_t MaximumMatvecTerms = 128 * 1024 * 1024;

@interface RunnerCase : NSObject
@property(nonatomic, copy) NSString *name;
@property(nonatomic) uint32_t inputDimension;
@property(nonatomic) uint32_t outputDimension;
@property(nonatomic, strong) NSData *firstData;
@property(nonatomic, strong) NSData *secondData;
@property(nonatomic, strong) NSData *expected;
@property(nonatomic, strong) NSMutableData *cpuOutput;
@property(nonatomic, strong) NSMutableData *readback;
@property(nonatomic, strong) id<MTLBuffer> firstBuffer;
@property(nonatomic, strong) id<MTLBuffer> secondBuffer;
@property(nonatomic, strong) id<MTLBuffer> outputBuffer;
@end
@implementation RunnerCase
@end

typedef struct {
    double wall;
    double device;
} GPUTiming;

static void Fail(NSString *format, ...) NS_FORMAT_FUNCTION(1, 2) __attribute__((noreturn));
static void Fail(NSString *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    NSString *message = [[NSString alloc] initWithFormat:format arguments:arguments];
    va_end(arguments);
    fprintf(stderr, "metal-runner: %s\n", message.UTF8String);
    exit(EXIT_FAILURE);
}

static double Now(void) {
    struct timespec timestamp;
    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0) Fail(@"clock_gettime failed");
    return timestamp.tv_sec + timestamp.tv_nsec * 1e-9;
}

static NSData *ReadFile(NSString *path, NSUInteger expectedBytes, NSUInteger *budget) {
    NSError *error = nil;
    NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfItemAtPath:path error:&error];
    if (!attributes) Fail(@"cannot stat %@: %@", path, error.localizedDescription);
    unsigned long long length = [attributes fileSize];
    if (![attributes[NSFileType] isEqualToString:NSFileTypeRegular] || length == 0 || length > *budget)
        Fail(@"%@ must be a nonempty regular file within the remaining %lu-byte limit", path, (unsigned long)*budget);
    if (expectedBytes != 0 && length != expectedBytes)
        Fail(@"%@ has %llu bytes; expected %lu", path, length, (unsigned long)expectedBytes);
    NSData *data = [NSData dataWithContentsOfFile:path options:0 error:&error];
    if (!data || data.length != length) Fail(@"cannot read stable file %@: %@", path, error.localizedDescription);
    *budget -= data.length;
    return data;
}

static uint32_t Dimension(NSString *text) {
    const char *digits = text.UTF8String;
    if (!digits || digits[0] < '0' || digits[0] > '9') Fail(@"invalid dimension: %@", text);
    char *end = NULL;
    errno = 0;
    unsigned long long dimension = strtoull(digits, &end, 10);
    if (errno || *end || dimension == 0 || dimension > INT_MAX) Fail(@"invalid dimension: %@", text);
    return (uint32_t)dimension;
}

static uint32_t Word(NSData *data, NSUInteger index) {
    uint32_t value;
    memcpy(&value, (const uint8_t *)data.bytes + index * sizeof(value), sizeof(value));
    return value;
}

static NSArray<RunnerCase *> *LoadCases(NSString *directory) {
    NSUInteger budget = MaximumFixtureBytes;
    NSMutableArray<RunnerCase *> *cases = [NSMutableArray array];
    RunnerCase *scalar = [RunnerCase new];
    scalar.name = @"scalar";
    scalar.firstData = ReadFile([directory stringByAppendingPathComponent:@"scalar-input.bin"], 0, &budget);
    if (scalar.firstData.length % sizeof(float) || scalar.firstData.length / sizeof(float) > 1024 * 1024)
        Fail(@"scalar input must contain 1 to 1048576 float32 values");
    scalar.outputDimension = (uint32_t)(scalar.firstData.length / sizeof(float));
    scalar.expected = ReadFile([directory stringByAppendingPathComponent:@"scalar-expected.bin"], scalar.firstData.length, &budget);
    for (NSUInteger index = 0; index < scalar.outputDimension; index++) {
        if ((Word(scalar.firstData, index) & 0x7f800000u) == 0x7f800000u ||
            (Word(scalar.expected, index) & 0x7f800000u) == 0x7f800000u)
            Fail(@"scalar fixture contains a nonfinite value at index %lu", (unsigned long)index);
    }
    [cases addObject:scalar];

    NSUInteger manifestBudget = 64 * 1024;
    NSData *manifest = ReadFile([directory stringByAppendingPathComponent:@"cases.txt"], 0, &manifestBudget);
    NSString *text = [[NSString alloc] initWithData:manifest encoding:NSUTF8StringEncoding];
    if (!text || [text rangeOfString:@"\0"].location != NSNotFound) Fail(@"cases.txt must be UTF-8 text without NUL bytes");
    NSCharacterSet *whitespace = [NSCharacterSet whitespaceCharacterSet];
    NSCharacterSet *invalidName = [[NSCharacterSet characterSetWithCharactersInString:
        @"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-."] invertedSet];
    NSMutableSet<NSString *> *names = [NSMutableSet setWithObject:@"scalar"];
    NSUInteger lineNumber = 0;
    for (NSString *line in [text componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]]) {
        lineNumber++;
        NSMutableArray<NSString *> *fields = [NSMutableArray array];
        for (NSString *field in [line componentsSeparatedByCharactersInSet:whitespace])
            if (field.length) [fields addObject:field];
        if (fields.count == 0) continue;
        if (fields.count != 3) Fail(@"cases.txt line %lu requires NAME IN_DIM OUT_DIM", (unsigned long)lineNumber);
        NSString *name = fields[0];
        if (name.length > 128 || [name rangeOfCharacterFromSet:invalidName].location != NSNotFound ||
            [name isEqualToString:@"."] || [name isEqualToString:@".."] || [names containsObject:name])
            Fail(@"invalid or duplicate case name: %@", name);
        if (cases.count > MaximumCases) Fail(@"at most %d matvec cases are allowed", MaximumCases);
        [names addObject:name];
        RunnerCase *testCase = [RunnerCase new];
        testCase.name = name;
        testCase.inputDimension = Dimension(fields[1]);
        testCase.outputDimension = Dimension(fields[2]);
        uint64_t terms = (uint64_t)testCase.inputDimension * testCase.outputDimension;
        if (testCase.outputDimension % 4 || testCase.inputDimension > INT32_MAX / 128 || terms > MaximumMatvecTerms)
            Fail(@"%@ requires OUT_DIM divisible by 4, IN_DIM <= %d, and at most %llu terms", name,
                 INT32_MAX / 128, (unsigned long long)MaximumMatvecTerms);
        testCase.firstData = ReadFile([directory stringByAppendingPathComponent:[name stringByAppendingString:@".weights.bin"]],
                                      (NSUInteger)(terms / 4), &budget);
        testCase.secondData = ReadFile([directory stringByAppendingPathComponent:[name stringByAppendingString:@".acts.bin"]],
                                       testCase.inputDimension, &budget);
        testCase.expected = ReadFile([directory stringByAppendingPathComponent:[name stringByAppendingString:@".expected.bin"]],
                                     (NSUInteger)testCase.outputDimension * sizeof(int32_t), &budget);
        const uint8_t *weights = testCase.firstData.bytes;
        for (NSUInteger index = 0; index < testCase.firstData.length; index++)
            if ((weights[index] & (weights[index] >> 1) & 0x55u) != 0)
                Fail(@"%@ contains unused ternary code 3 at byte %lu", name, (unsigned long)index);
        testCase.cpuOutput = [NSMutableData dataWithLength:(NSUInteger)testCase.outputDimension * sizeof(double)];
        [cases addObject:testCase];
    }
    if (cases.count == 1) Fail(@"cases.txt must contain at least one matvec case");
    return cases;
}

static id<MTLComputePipelineState> Pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString *name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) Fail(@"shader entry '%@' is missing", name);
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) Fail(@"cannot create %@ pipeline: %@", name, error.localizedDescription);
    if (pipeline.staticThreadgroupMemoryLength > device.maxThreadgroupMemoryLength)
        Fail(@"%@ exceeds device threadgroup memory", name);
    return pipeline;
}

static void AllocateBuffers(id<MTLDevice> device, RunnerCase *testCase) {
    if (testCase.firstData.length > device.maxBufferLength || testCase.secondData.length > device.maxBufferLength ||
        testCase.expected.length > device.maxBufferLength) Fail(@"%@ exceeds the device buffer limit", testCase.name);
    testCase.firstBuffer = [device newBufferWithBytes:testCase.firstData.bytes length:testCase.firstData.length options:MTLResourceStorageModeShared];
    if (testCase.secondData)
        testCase.secondBuffer = [device newBufferWithBytes:testCase.secondData.bytes length:testCase.secondData.length options:MTLResourceStorageModeShared];
    testCase.outputBuffer = [device newBufferWithLength:testCase.expected.length options:MTLResourceStorageModeShared];
    if (!testCase.firstBuffer || (testCase.secondData && !testCase.secondBuffer) || !testCase.outputBuffer)
        Fail(@"%@ Metal buffer allocation failed", testCase.name);
}

static void PoisonOutput(RunnerCase *testCase) {
    uint32_t *output = testCase.outputBuffer.contents;
    for (NSUInteger index = 0; index < testCase.outputDimension; index++) output[index] = ~Word(testCase.expected, index);
}

static GPUTiming RunGPU(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline, RunnerCase *testCase) {
    double start = Now();
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (!command || !encoder) Fail(@"%@ cannot create Metal command buffer or encoder", testCase.name);
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:testCase.firstBuffer offset:0 atIndex:0];
    uint32_t inputDimension = testCase.inputDimension;
    uint32_t outputDimension = testCase.outputDimension;
    if (testCase.secondData) {
        [encoder setBuffer:testCase.secondBuffer offset:0 atIndex:1];
        [encoder setBuffer:testCase.outputBuffer offset:0 atIndex:2];
        [encoder setBytes:&inputDimension length:sizeof(inputDimension) atIndex:3];
        [encoder setBytes:&outputDimension length:sizeof(outputDimension) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake(outputDimension, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    } else {
        [encoder setBuffer:testCase.outputBuffer offset:0 atIndex:1];
        [encoder setBytes:&outputDimension length:sizeof(outputDimension) atIndex:2];
        NSUInteger threads = MIN((NSUInteger)128, pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(outputDimension, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    double wall = Now() - start;
    if (command.status != MTLCommandBufferStatusCompleted)
        Fail(@"%@ GPU command failed (status %lu): %@", testCase.name, (unsigned long)command.status, command.error.localizedDescription);
    double device = NAN;
    if (command.GPUStartTime > 0 && command.GPUEndTime > command.GPUStartTime)
        device = command.GPUEndTime - command.GPUStartTime;
    return (GPUTiming){ .wall = wall, .device = device };
}

static double ValidateGPU(RunnerCase *testCase) {
    double start = Now();
    memcpy(testCase.readback.mutableBytes, testCase.outputBuffer.contents, testCase.expected.length);
    if (memcmp(testCase.readback.bytes, testCase.expected.bytes, testCase.expected.length) != 0) {
        for (NSUInteger index = 0; index < testCase.outputDimension; index++) {
            uint32_t actual = Word(testCase.readback, index);
            uint32_t expected = Word(testCase.expected, index);
            if (actual != expected)
                Fail(@"%@ GPU mismatch at index %lu: bits 0x%08x, expected 0x%08x", testCase.name, (unsigned long)index, actual, expected);
        }
    }
    return Now() - start;
}

static double RunCPU(RunnerCase *testCase) {
    const void *weights = testCase.firstData.bytes;
    const void *activations = testCase.secondData.bytes;
    double *output = testCase.cpuOutput.mutableBytes;
    double start = Now();
    bnmatmul_ternary(weights, activations, (int)testCase.inputDimension, (int)testCase.outputDimension, output);
    return Now() - start;
}

static void ValidateCPU(RunnerCase *testCase) {
    const double *output = testCase.cpuOutput.bytes;
    for (NSUInteger index = 0; index < testCase.outputDimension; index++) {
        uint32_t bits = Word(testCase.expected, index);
        int32_t expected;
        memcpy(&expected, &bits, sizeof(expected));
        if (output[index] != (double)expected)
            Fail(@"%@ C mismatch at row %lu: %.17g, expected %d", testCase.name, (unsigned long)index, output[index], expected);
    }
}

static int CompareDouble(const void *left, const void *right) {
    double leftValue = *(const double *)left;
    double rightValue = *(const double *)right;
    return (leftValue > rightValue) - (leftValue < rightValue);
}

static double Median(double *samples, NSUInteger count) {
    qsort(samples, count, sizeof(*samples), CompareDouble);
    return count % 2 ? samples[count / 2] : (samples[count / 2 - 1] + samples[count / 2]) / 2;
}

static void Benchmark(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline, RunnerCase *testCase) {
    for (NSUInteger warmup = 0; warmup < 2; warmup++) {
        @autoreleasepool {
            if (testCase.secondData) { RunCPU(testCase); ValidateCPU(testCase); }
            PoisonOutput(testCase);
            RunGPU(queue, pipeline, testCase);
            ValidateGPU(testCase);
        }
    }
    double cpu[Repetitions], wall[Repetitions], device[Repetitions], readback[Repetitions], validated[Repetitions];
    NSUInteger deviceSamples = 0;
    for (NSUInteger repetition = 0; repetition < Repetitions; repetition++) {
        @autoreleasepool {
            for (NSUInteger phase = 0; phase < 2; phase++) {
                if ((repetition + phase) % 2 == 0) {
                    if (testCase.secondData) { cpu[repetition] = RunCPU(testCase); ValidateCPU(testCase); }
                } else {
                    PoisonOutput(testCase);
                    GPUTiming timing = RunGPU(queue, pipeline, testCase);
                    wall[repetition] = timing.wall;
                    readback[repetition] = ValidateGPU(testCase);
                    validated[repetition] = wall[repetition] + readback[repetition];
                    if (isfinite(timing.device)) device[deviceSamples++] = timing.device;
                }
            }
        }
    }
    printf("%s PASS count=%u", testCase.name.UTF8String, testCase.outputDimension);
    if (testCase.secondData)
        printf(" in_dim=%u C_serial_median_ms=%.6f", testCase.inputDimension, Median(cpu, Repetitions) * 1e3);
    printf(" GPU_resident_wall_median_ms=%.6f", Median(wall, Repetitions) * 1e3);
    if (deviceSamples)
        printf(" GPU_timestamp_median_ms=%.6f (%lu/%d samples)", Median(device, deviceSamples) * 1e3, (unsigned long)deviceSamples, Repetitions);
    else printf(" GPU_timestamp_median_ms=unavailable");
    printf(" readback_validate_median_ms=%.6f resident_plus_readback_validate_median_ms=%.6f\n",
           Median(readback, Repetitions) * 1e3, Median(validated, Repetitions) * 1e3);
}

int main(int argumentCount, const char *arguments[]) {
    @autoreleasepool {
        if (argumentCount != 3) Fail(@"usage: metal-runner GENERATED_METAL CASE_DIRECTORY");
        NSString *sourcePath = [NSString stringWithUTF8String:arguments[1]];
        NSString *directory = [NSString stringWithUTF8String:arguments[2]];
        if (!sourcePath || !directory) Fail(@"arguments must be UTF-8 paths");
        NSArray<RunnerCase *> *cases = LoadCases(directory);
        NSUInteger sourceBudget = 4 * 1024 * 1024;
        NSData *sourceData = ReadFile(sourcePath, 0, &sourceBudget);
        NSString *source = [[NSString alloc] initWithData:sourceData encoding:NSUTF8StringEncoding];
        if (!source) Fail(@"shader source must be UTF-8 text");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) Fail(@"no Metal GPU is available; GPU access may be blocked by the execution sandbox. No CPU fallback was used.");
        NSError *error = nil;
        MTLCompileOptions *options = [MTLCompileOptions new];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        if (@available(macOS 15.0, *)) options.mathMode = MTLMathModeSafe;
        else options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        double compileStart = Now();
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) Fail(@"Metal shader compilation failed: %@", error.localizedDescription);
        id<MTLComputePipelineState> scalarPipeline = Pipeline(device, library, @"scalar");
        id<MTLComputePipelineState> matvecPipeline = Pipeline(device, library, @"matvec");
        double compileTime = Now() - compileStart;
        if (matvecPipeline.maxTotalThreadsPerThreadgroup < 128 || device.maxThreadsPerThreadgroup.width < 128)
            Fail(@"matvec requires 128 threads per threadgroup");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) Fail(@"cannot create Metal command queue");
        printf("device: %s\nshader_compile_and_pipeline_ms=%.6f\n", device.name.UTF8String, compileTime * 1e3);
        printf("Limits: 64 matvec cases, 256 MiB fixture data, 1048576 scalar values, 134217728 matvec terms per case.\n");
        printf("Two warmups, %d repetitions; CPU/GPU order alternates for matvec. C baseline: serial, without OpenMP.\n", Repetitions);
        printf("Resident GPU wall includes command creation, encoding, dispatch, commit, wait; excludes upload and readback/validation.\n");
        printf("Readback/validation copies shared output to host memory and compares every bit; resident-plus-readback includes both measured intervals.\n");
        printf("Buffers remain resident per case. Each sample submits one dispatch for the full scalar batch or one matvec.\n");
        printf("These bounded batch timings do not measure full model inference or establish application speedup.\n");
        double allocationTime = 0;
        for (RunnerCase *testCase in cases) {
            testCase.readback = [NSMutableData dataWithLength:testCase.expected.length];
            double allocationStart = Now();
            AllocateBuffers(device, testCase);
            double elapsed = Now() - allocationStart;
            allocationTime += elapsed;
            printf("%s initial_buffer_allocation_upload_ms=%.6f\n", testCase.name.UTF8String, elapsed * 1e3);
        }
        printf("initial_buffer_allocation_upload_total_ms=%.6f (shared memory; input copy, output allocation; excludes file I/O)\n", allocationTime * 1e3);
        for (RunnerCase *testCase in cases)
            Benchmark(queue, testCase.secondData ? matvecPipeline : scalarPipeline, testCase);
        printf("ALL CASES PASS (%lu): GPU bit-exact; matvec C raw dots exact.\n", (unsigned long)cases.count);
    }
    return EXIT_SUCCESS;
}
