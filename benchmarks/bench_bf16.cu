/* Quick BF16 GEMM timing — does it use tensor cores? */
#include <cuda_runtime.h>
#include <cuda_tile.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <algorithm>

using namespace cuda::tiles;

template <int TM, int TN, int TK>
__tile_global__ void gemm_bf16(float* C, const __nv_bfloat16* A, const __nv_bfloat16* B, int M, int N, int K)
{
    using Ash = shape<TM, TK>;
    using Bsh = shape<TK, TN>;
    using Csh = shape<TM, TN>;
    using Ctil = tile<float, Csh>;

    int bx = detail::bid<0>();
    int by = detail::bid<1>();
    int bm = bx * TM;
    int bn = by * TN;

    auto a_ext = extents<int, dynamic_extent, dynamic_extent>{M, K};
    auto b_ext = extents<int, dynamic_extent, dynamic_extent>{K, N};
    auto c_ext = extents<int, dynamic_extent, dynamic_extent>{M, N};
    auto a_sp = tensor_span(A, a_ext, layout_left{});
    auto b_sp = tensor_span(B, b_ext, layout_left{});
    auto c_sp = tensor_span(C, c_ext, layout_left{});
    auto a_vw = partition_view(a_sp, Ash{});
    auto b_vw = partition_view(b_sp, Bsh{});
    auto c_vw = partition_view(c_sp, Csh{});

    Ctil c = zeros<Ctil>();
    for (int k = 0; k < K; k += TK) {
        c = mma(a_vw.load(bm, k), b_vw.load(k, bn), c);
    }
    c_vw.store(c, bm, bn);
}

static void fill_bf16(__nv_bfloat16* p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = ((float)rand()/RAND_MAX)*2.0f - 1.0f;
        p[i] = __float2bfloat16(v);
    }
}

/* Read GPU clock from sysfs (cur_freq in kHz) */
static double read_sysfs_clock_khz() {
    FILE *f = fopen("/sys/devices/platform/bus@0/d0b0000000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/gpu-gpc-0/devfreq/gpu-gpc-0/cur_freq", "r");
    if (!f) {
        // Try alternative paths
        static const char* paths[] = {
            "/sys/devices/platform/13940000.gpu/devfreq/13940000.gpu/cur_freq",
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
            nullptr
        };
        for (int i = 0; paths[i]; i++) {
            f = fopen(paths[i], "r");
            if (f) break;
        }
    }
    if (!f) return 1575000.0;  // fallback: 1575 MHz
    double freq_hz = 0;
    if (fscanf(f, "%lf", &freq_hz) == 1) {
        fclose(f);
        return freq_hz / 1000.0;  // Hz -> kHz
    }
    fclose(f);
    return 1575000.0;
}

/* Per-iteration statistics */
struct IterStats {
    double times[512];  // per-iteration GFLOPS
    int count;
    double min_val;
    double max_val;
    double mean;
    double stddev;

    IterStats() : count(0), min_val(1e30), max_val(0), mean(0), stddev(0) {}

    void add(double val) {
        if (count < 512) times[count] = val;
        count++;
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    void compute() {
        if (count == 0) return;
        double sum = 0;
        for (int i = 0; i < count; i++) sum += times[i];
        mean = sum / count;
        double sq_sum = 0;
        for (int i = 0; i < count; i++) {
            double d = times[i] - mean;
            sq_sum += d * d;
        }
        stddev = count > 1 ? sqrt(sq_sum / (count - 1)) : 0;
    }
};

template <int TM, int TN, int TK>
IterStats bench_one(int M, int N, int K, int warmup, int iters) {
    size_t sa = (size_t)M*K*sizeof(__nv_bfloat16);
    size_t sb = (size_t)K*N*sizeof(__nv_bfloat16);
    size_t sc = (size_t)M*N*sizeof(float);

    __nv_bfloat16 *dA, *dB; float *dC;
    __nv_bfloat16 *hA = (__nv_bfloat16*)malloc(sa);
    __nv_bfloat16 *hB = (__nv_bfloat16*)malloc(sb);
    float *hC = (float*)malloc(sc);
    cudaMalloc(&dA, sa); cudaMalloc(&dB, sb); cudaMalloc(&dC, sc);

    fill_bf16(hA, (size_t)M*K);
    fill_bf16(hB, (size_t)K*N);
    cudaMemcpy(dA, hA, sa, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sb, cudaMemcpyHostToDevice);

    dim3 grid((M+TM-1)/TM, (N+TN-1)/TN);
    dim3 block(1,1,1);

    for (int i = 0; i < warmup; i++) {
        cudaMemset(dC, 0, sc);
        gemm_bf16<TM,TN,TK><<<grid, block>>>(dC, dA, dB, M, N, K);
    }
    cudaDeviceSynchronize();

    IterStats stats;
    bool has_nan = false;
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    for (int i = 0; i < iters; i++) {
        // Zero C ONCE per iteration, outside kernel timing
        cudaMemset(dC, 0, sc);
        cudaEventRecord(s);
        gemm_bf16<TM,TN,TK><<<grid, block>>>(dC, dA, dB, M, N, K);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms = 0;
        cudaEventElapsedTime(&ms, s, e);

        double flops = (double)M*N*K*2.0;
        double gflops = flops / (ms * 1e-3) / 1e9;
        stats.add(gflops);
    }

    // Validate once after all iterations (no NaN/Inf)
    cudaMemset(dC, 0, sc);
    gemm_bf16<TM,TN,TK><<<grid, block>>>(dC, dA, dB, M, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(hC, dC, sc, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < (size_t)M*N; i++) {
        if (isnan(hC[i]) || isinf(hC[i])) { has_nan = true; break; }
    }

    stats.compute();
    if (has_nan) { stats.min_val = -1; stats.mean = -1; }

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);

    return stats;
}

int main(int argc, char* argv[]) {
    int seed = 42;
    int iterations = 50;
    int warmup = 5;

    // Parse CLI arguments
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--seed=", 7) == 0) seed = atoi(argv[i] + 7);
        else if (strncmp(argv[i], "--iterations=", 13) == 0) iterations = atoi(argv[i] + 13);
        else if (strncmp(argv[i], "--warmup=", 9) == 0) warmup = atoi(argv[i] + 9);
    }

    srand(seed);

    /* Environment fingerprint */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int sm_count = prop.multiProcessorCount;
    double clock_khz = read_sysfs_clock_khz();
    double clock_ghz = clock_khz / 1e6;
    // BF16 peak: NVIDIA stated 1032 TF for Thor SM110a
    double peak_tflops = 1032.0;

    printf("BF16 GEMM timing — does the tile compiler use tensor cores?\n");
    printf("If we see > 5000 GFLOPS, tensor cores are being used.\n\n");
    printf("=== Environment ===\n");
    printf("Device:    %s (sm_%d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SM count:  %d\n", sm_count);
    printf("Clock:     %.1f MHz (%.3f GHz)\n", clock_khz/1000.0, clock_ghz);
    printf("Peak: %.0f TFLOPS\n", peak_tflops);
    printf("Seed:      %d\n", seed);
    printf("Warmup:    %d  Iterations: %d\n\n", warmup, iterations);

    int sizes[] = {512, 1024, 2048, 4096};
    int configs[][3] = {
        {16, 16, 16},
        {16, 32, 16},
        {32, 32, 16},
        {32, 32, 32},
        {64, 64, 16},
        {64, 64, 32},
        {128,128, 16},
    };
    int ncfg = sizeof(configs)/(sizeof(int)*3);

    printf("%-12s", "Tile");
    for (int s = 0; s < 4; s++) printf("  M=N=K=%-4d", sizes[s]);
    printf("\n");
    printf("%-12s", "--------");
    for (int s = 0; s < 4; s++) printf("  -------------");
    printf("\n");

    for (int c = 0; c < ncfg; c++) {
        int TM = configs[c][0], TN = configs[c][1], TK = configs[c][2];
        printf("%dx%dx%-4d", TM, TN, TK);
        for (int s = 0; s < 4; s++) {
            int M = sizes[s];
            IterStats stats;
            // dispatch on tile config
            if      (TM==16 && TN==16 && TK==16) stats = bench_one<16,16,16>(M,M,M,warmup,iterations);
            else if (TM==16 && TN==32 && TK==16) stats = bench_one<16,32,16>(M,M,M,warmup,iterations);
            else if (TM==32 && TN==32 && TK==16) stats = bench_one<32,32,16>(M,M,M,warmup,iterations);
            else if (TM==32 && TN==32 && TK==32) stats = bench_one<32,32,32>(M,M,M,warmup,iterations);
            else if (TM==64 && TN==64 && TK==16) stats = bench_one<64,64,16>(M,M,M,warmup,iterations);
            else if (TM==64 && TN==64 && TK==32) stats = bench_one<64,64,32>(M,M,M,warmup,iterations);
            else if (TM==128 && TN==128 && TK==16) stats = bench_one<128,128,16>(M,M,M,warmup,iterations);

            if (stats.mean < 0) {
                printf("  [FAIL]");
            } else {
                printf("  %.0f±%.0f GF", stats.mean, stats.stddev);
            }
        }
        printf("\n");
    }

    printf("\nSummary (per-tile, per-size): mean GFLOPS ± stddev\n");
    printf("Peak: %.0f TFLOPS\n", peak_tflops);

    return 0;
}
