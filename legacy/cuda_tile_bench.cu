/*
 * CUDA Tile C++ GEMM Benchmark
 * Target: Jetson AGX Thor (sm_110, Blackwell)
 *
 * Compile:
 *   nvcc -std=c++20 -arch=sm_110 -O3 -enable-tile -o cuda_tile_bench cuda_tile_bench.cu
 *
 * Usage:
 *   ./cuda_tile_bench                          # default sweep
 *   ./cuda_tile_bench 512 1024 2048            # custom sizes
 *   ./cuda_tile_bench --precision bf16 4096    # only BF16
 */

#include <cuda_runtime.h>
#include <cuda_tile.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

using namespace cuda::tiles;

/* ============================================================
 * GEMM kernel using CUDA Tile API.
 *
 * Each block computes one TILE_M x TILE_N output tile.
 * Tile primitives (load, mma, store) handle intra-block
 * parallelism automatically — we never touch threadIdx.
 *
 * Input element type T is templated:
 *   - float          -> FP32 path, mma uses TF32 tensor cores
 *   - __nv_bfloat16  -> BF16 path, mma uses BF16 tensor cores (FP32 accum)
 *   - __half         -> FP16 path, mma uses FP16 tensor cores (FP32 accum)
 * ============================================================ */
template <typename T, int TILE_M, int TILE_N, int TILE_K>
__tile_global__ void gemm_tile_kernel(float* __restrict__ C,
                                      const T* __restrict__ A,
                                      const T* __restrict__ B,
                                      int M, int N, int K)
{
    using A_tile_shape = shape<TILE_M, TILE_K>;
    using B_tile_shape = shape<TILE_K, TILE_N>;
    using C_tile_shape = shape<TILE_M, TILE_N>;
    using C_fp32_tile  = tile<float, C_tile_shape>;

    int bx = detail::bid<0>();
    int by = detail::bid<1>();

    int bm = bx * TILE_M;
    int bn = by * TILE_N;

    auto a_ext = extents<int, dynamic_extent, dynamic_extent>{M, K};
    auto b_ext = extents<int, dynamic_extent, dynamic_extent>{K, N};
    auto c_ext = extents<int, dynamic_extent, dynamic_extent>{M, N};

    auto a_span = tensor_span(A, a_ext, layout_left{});
    auto b_span = tensor_span(B, b_ext, layout_left{});
    auto c_span = tensor_span(C, c_ext, layout_left{});

    auto a_view = partition_view(a_span, A_tile_shape{});
    auto b_view = partition_view(b_span, B_tile_shape{});
    auto c_view = partition_view(c_span, C_tile_shape{});

    C_fp32_tile c_tile = zeros<C_fp32_tile>();

    for (int k = 0; k < K; k += TILE_K) {
        auto a_tile = a_view.load(bm, k);
        auto b_tile = b_view.load(k, bn);
        c_tile = mma(a_tile, b_tile, c_tile);
    }

    c_view.store(c_tile, bm, bn);
}

/* ============================================================
 * Benchmark framework
 * ============================================================ */

struct BenchResult {
    char   config_name[40];
    int    M, N, K;
    int    tile_m, tile_n, tile_k;
    double gflops;
    double latency_ms;
    bool   valid;
};

/* fill_random: produce uniform values in [-1, 1] of type T. */
template <typename T>
static void fill_random(T* ptr, size_t n);
template <>
void fill_random<float>(float* ptr, size_t n) {
    for (size_t i = 0; i < n; i++)
        ptr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}
template <>
void fill_random<__nv_bfloat16>(__nv_bfloat16* ptr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        ptr[i] = __float2bfloat16(v);
    }
}
template <>
void fill_random<__half>(__half* ptr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        ptr[i] = __float2half(v);
    }
}

/* Run a single benchmark configuration. */
template <typename T, int TILE_M, int TILE_N, int TILE_K>
static BenchResult run_bench(int M, int N, int K, const char* name,
                             int warmup_iters, int bench_iters)
{
    BenchResult res = {};
    strncpy(res.config_name, name, sizeof(res.config_name) - 1);
    res.M = M; res.N = N; res.K = K;
    res.tile_m = TILE_M; res.tile_n = TILE_N; res.tile_k = TILE_K;
    res.valid = false;
    res.gflops = 0.0;
    res.latency_ms = 0.0;

    size_t sz_a = (size_t)M * K * sizeof(T);
    size_t sz_b = (size_t)K * N * sizeof(T);
    size_t sz_c = (size_t)M * N * sizeof(float);

    T     *h_A     = (T*)malloc(sz_a);
    T     *h_B     = (T*)malloc(sz_b);
    float *h_C_dev = (float*)malloc(sz_c);
    T     *d_A = nullptr, *d_B = nullptr;
    float *d_C = nullptr;
    float  elapsed_ms = 0.0f;
    double flops      = (double)M * N * K * 2.0;
    bool   has_nan    = false;
    bool   ok         = true;

    cudaMalloc(&d_A, sz_a);
    cudaMalloc(&d_B, sz_b);
    cudaMalloc(&d_C, sz_c);

    fill_random<T>(h_A, M * K);
    fill_random<T>(h_B, K * N);

    cudaMemcpy(d_A, h_A, sz_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz_b, cudaMemcpyHostToDevice);

    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);
    dim3 block(1, 1, 1);

    for (int i = 0; i < warmup_iters; i++) {
        cudaMemset(d_C, 0, sz_c);
        gemm_tile_kernel<T, TILE_M, TILE_N, TILE_K><<<grid, block>>>(d_C, d_A, d_B, M, N, K);
    }
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("    [ERROR] warmup launch: %s\n", cudaGetErrorString(e));
        ok = false;
    }

    if (ok) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaMemset(d_C, 0, sz_c);
        cudaEventRecord(start);
        for (int i = 0; i < bench_iters; i++) {
            gemm_tile_kernel<T, TILE_M, TILE_N, TILE_K><<<grid, block>>>(d_C, d_A, d_B, M, N, K);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsed_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("    [ERROR] timed launch: %s\n", cudaGetErrorString(e));
            ok = false;
        }
    }

    if (ok) {
        res.gflops     = flops * bench_iters / (elapsed_ms * 1e-3) / 1e9;
        res.latency_ms = elapsed_ms / bench_iters;

        cudaMemcpy(h_C_dev, d_C, sz_c, cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < (size_t)M * N; i++) {
            if (isnan(h_C_dev[i]) || isinf(h_C_dev[i])) {
                has_nan = true;
                break;
            }
        }
        res.valid = !has_nan;
    }

    free(h_A); free(h_B); free(h_C_dev);
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);
    return res;
}

/* ============================================================
 * Tile configuration sweep
 * ============================================================ */

struct TileConfig {
    int tile_m, tile_n, tile_k;
    const char* name;
};

static const TileConfig kConfigs[] = {
    {16, 16,  8, "16x16x8"},
    {16, 16, 16, "16x16x16"},
    {16, 16, 32, "16x16x32"},
    {16, 32, 16, "16x32x16"},
    {16, 32, 32, "16x32x32"},
    {32, 32, 16, "32x32x16"},
    {32, 32, 32, "32x32x32"},
    {32, 64, 32, "32x64x32"},
    {64, 64, 16, "64x64x16"},
    {64, 64, 32, "64x64x32"},
    {128,128, 16, "128x128x16"},
    {128,128, 32, "128x128x32"},
};
static const int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

/* Run one configuration by index — explicit template instantiations
 * avoid the runtime-shape dispatch issue in the tile compiler. */
template <typename T>
static BenchResult dispatch(int c, int M, int N, int K, int warmup, int bench) {
    switch (c) {
        case  0: return run_bench<T, 16, 16,  8>(M, N, K, kConfigs[0].name, warmup, bench);
        case  1: return run_bench<T, 16, 16, 16>(M, N, K, kConfigs[1].name, warmup, bench);
        case  2: return run_bench<T, 16, 16, 32>(M, N, K, kConfigs[2].name, warmup, bench);
        case  3: return run_bench<T, 16, 32, 16>(M, N, K, kConfigs[3].name, warmup, bench);
        case  4: return run_bench<T, 16, 32, 32>(M, N, K, kConfigs[4].name, warmup, bench);
        case  5: return run_bench<T, 32, 32, 16>(M, N, K, kConfigs[5].name, warmup, bench);
        case  6: return run_bench<T, 32, 32, 32>(M, N, K, kConfigs[6].name, warmup, bench);
        case  7: return run_bench<T, 32, 64, 32>(M, N, K, kConfigs[7].name, warmup, bench);
        case  8: return run_bench<T, 64, 64, 16>(M, N, K, kConfigs[8].name, warmup, bench);
        case  9: return run_bench<T, 64, 64, 32>(M, N, K, kConfigs[9].name, warmup, bench);
        case 10: return run_bench<T,128,128, 16>(M, N, K, kConfigs[10].name, warmup, bench);
        case 11: return run_bench<T,128,128, 32>(M, N, K, kConfigs[11].name, warmup, bench);
        default: return BenchResult{};
    }
}

template <typename T>
static void run_precision_sweep(const char* prec_name, const cudaDeviceProp& prop,
                                 int num_problems, const int* prob_sizes,
                                 const char* const* prob_names,
                                 size_t input_elem_size,
                                 int warmup_iters, int bench_iters,
                                 double& best_gflops, char* best_name,
                                 int& best_problem, int& pass_count, int& total_count)
{
    printf("\n============================================================\n");
    printf("  %s GEMM Sweep (M=N=K)\n", prec_name);
    printf("============================================================\n");
    printf("%-14s  %-8s  %10s  %10s  %-6s\n",
           "Tile", "Problem", "GFLOPS", "Lat(ms)", "Valid");
    printf("------------------------------------------------------------\n");

    for (int p = 0; p < num_problems; p++) {
        int prob = prob_sizes[p];
        int M = prob, N = prob, K = prob;

        // Memory check: A + B + C in input size + float for C
        size_t mem_needed = (size_t)M * K * input_elem_size * 2
                          + (size_t)M * N * sizeof(float);
        if (mem_needed > prop.totalGlobalMem * 0.5) {
            printf("\n  %-8s: skipped (would need %.1f GB)\n",
                   prob_names[p], mem_needed / 1e9);
            continue;
        }

        printf("\n  Problem: %s (M=N=K=%d)\n", prob_names[p], M);

        for (int c = 0; c < kNumConfigs; c++) {
            total_count++;
            BenchResult res = dispatch<T>(c, M, N, K, warmup_iters, bench_iters);

            const char* vstr = res.valid ? "PASS" : "FAIL";
            printf("    %-12s  %-8s  %10.1f  %9.3f  %s\n",
                   res.config_name, prob_names[p],
                   res.gflops, res.latency_ms, vstr);

            if (res.valid) {
                pass_count++;
                if (res.gflops > best_gflops) {
                    best_gflops = res.gflops;
                    strncpy(best_name, res.config_name, sizeof(*best_name) * 40 - 1);
                    best_problem = prob;
                }
            }
        }
    }
}

/* ============================================================
 * Main
 * ============================================================ */

int main(int argc, char* argv[])
{
    printf("============================================================\n");
    printf("  CUDA Tile C++ GEMM Benchmark (Jetson AGX Thor, sm_110)\n");
    printf("============================================================\n\n");

    int device_id = 0;
    cudaGetDevice(&device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    printf("Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  SM Count: %d\n", prop.multiProcessorCount);
    printf("  Global Memory: %.1f GB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // Theoretical peak estimates for sm_110 Blackwell (Jetson AGX Thor)
    // 20 SMs, 4 tensor cores per SM, boost 1.575 GHz
    double gpu_clock_ghz = 1.575;
    int    sm_count      = prop.multiProcessorCount;

    // FP32 (CUDA core FMA path): 128 FP32 cores per SM * 2 ops/cycle
    double fp32_cuda_peak = gpu_clock_ghz * sm_count * 256.0;
    // FP32 via TF32 tensor cores: 1024 ops/cycle/TC, 4 TC/SM
    double tf32_tensor_peak = gpu_clock_ghz * sm_count * 4.0 * 1024.0;
    // BF16/FP16 tensor cores: 2x TF32 dense
    double fp16_tensor_peak = 2.0 * tf32_tensor_peak;

    printf("\n--- Theoretical Peak Estimates ---\n");
    printf("  GPU Clock (max): %.3f GHz\n", gpu_clock_ghz);
    printf("  SMs: %d, TC/SM: 4\n", sm_count);
    printf("  FP32 CUDA core:  %.1f GFLOPS\n", fp32_cuda_peak);
    printf("  TF32 tensor:     %.1f GFLOPS\n", tf32_tensor_peak);
    printf("  BF16/FP16 tensor:%.1f GFLOPS\n\n", fp16_tensor_peak);

    // Parse args: optional --precision <fp32|bf16|fp16|all>
    int  run_fp32 = 1, run_bf16 = 1, run_fp16 = 1;
    int  arg_idx  = 1;
    if (argc > 1 && strcmp(argv[1], "--precision") == 0 && argc > 2) {
        arg_idx = 3;
        run_fp32 = run_bf16 = run_fp16 = 0;
        if (strcmp(argv[2], "fp32") == 0)      run_fp32 = 1;
        else if (strcmp(argv[2], "bf16") == 0) run_bf16 = 1;
        else if (strcmp(argv[2], "fp16") == 0) run_fp16 = 1;
        else if (strcmp(argv[2], "all") == 0)  run_fp32 = run_bf16 = run_fp16 = 1;
        else { fprintf(stderr, "Unknown precision: %s\n", argv[2]); return 1; }
    }

    // Problem sizes
    int    prob_sizes[16];
    const char* prob_names[16];
    char   prob_name_bufs[16][16];
    int    num_problems = 4;
    int    defaults[]   = {512, 1024, 2048, 4096};
    for (int i = 0; i < num_problems; i++) {
        prob_sizes[i] = defaults[i];
        snprintf(prob_name_bufs[i], 16, "%d", defaults[i]);
        prob_names[i] = prob_name_bufs[i];
    }
    if (arg_idx < argc) {
        num_problems = (argc - arg_idx < 16) ? argc - arg_idx : 16;
        for (int i = 0; i < num_problems; i++) {
            prob_sizes[i] = atoi(argv[arg_idx + i]);
            snprintf(prob_name_bufs[i], 16, "%d", prob_sizes[i]);
            prob_names[i] = prob_name_bufs[i];
        }
    }

    int warmup_iters = 5;
    int bench_iters  = 20;

    int    pass_count = 0, total_count = 0;

    struct BestEntry {
        double gflops;
        char   name[40];
        int    problem;
        const char* prec;
    };
    BestEntry best_fp32 = {0.0, "", 0, "FP32"};
    BestEntry best_bf16 = {0.0, "", 0, "BF16"};
    BestEntry best_fp16 = {0.0, "", 0, "FP16"};
    char buf_fp32[40] = "", buf_bf16[40] = "", buf_fp16[40] = "";
    int  prob_fp32 = 0, prob_bf16 = 0, prob_fp16 = 0;

    if (run_fp32) {
        run_precision_sweep<float>("FP32", prop, num_problems, prob_sizes, prob_names,
            sizeof(float), warmup_iters, bench_iters,
            best_fp32.gflops, buf_fp32, prob_fp32,
            pass_count, total_count);
        strncpy(best_fp32.name, buf_fp32, sizeof(best_fp32.name) - 1);
        best_fp32.problem = prob_fp32;
    }
    if (run_bf16) {
        run_precision_sweep<__nv_bfloat16>("BF16", prop, num_problems, prob_sizes, prob_names,
            sizeof(__nv_bfloat16), warmup_iters, bench_iters,
            best_bf16.gflops, buf_bf16, prob_bf16,
            pass_count, total_count);
        strncpy(best_bf16.name, buf_bf16, sizeof(best_bf16.name) - 1);
        best_bf16.problem = prob_bf16;
    }
    if (run_fp16) {
        run_precision_sweep<__half>("FP16", prop, num_problems, prob_sizes, prob_names,
            sizeof(__half), warmup_iters, bench_iters,
            best_fp16.gflops, buf_fp16, prob_fp16,
            pass_count, total_count);
        strncpy(best_fp16.name, buf_fp16, sizeof(best_fp16.name) - 1);
        best_fp16.problem = prob_fp16;
    }

    printf("\n============================================================\n");
    printf("  Summary\n");
    printf("============================================================\n");
    printf("Total configs run: %d  (passed: %d)\n", total_count, pass_count);

    if (run_fp32 && best_fp32.gflops > 0.0) {
        printf("\nFP32 best: %.1f GFLOPS  (tile=%s, problem=%d)\n",
               best_fp32.gflops, best_fp32.name, best_fp32.problem);
        printf("  vs FP32 CUDA core peak (%.0f GFLOPS): %.1f%%\n",
               fp32_cuda_peak, best_fp32.gflops / fp32_cuda_peak * 100.0);
        printf("  vs TF32 tensor peak    (%.0f GFLOPS): %.1f%%\n",
               tf32_tensor_peak, best_fp32.gflops / tf32_tensor_peak * 100.0);
    }
    if (run_bf16 && best_bf16.gflops > 0.0) {
        printf("\nBF16 best: %.1f GFLOPS  (tile=%s, problem=%d)\n",
               best_bf16.gflops, best_bf16.name, best_bf16.problem);
        printf("  vs BF16 tensor peak    (%.0f GFLOPS): %.1f%%\n",
               fp16_tensor_peak, best_bf16.gflops / fp16_tensor_peak * 100.0);
    }
    if (run_fp16 && best_fp16.gflops > 0.0) {
        printf("\nFP16 best: %.1f GFLOPS  (tile=%s, problem=%d)\n",
               best_fp16.gflops, best_fp16.name, best_fp16.problem);
        printf("  vs FP16 tensor peak    (%.0f GFLOPS): %.1f%%\n",
               fp16_tensor_peak, best_fp16.gflops / fp16_tensor_peak * 100.0);
    }
    printf("\n");
    return 0;
}
