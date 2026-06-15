/* FP8 GEMM timing — E4M3 and E5M2 across sizes.
 * Single tile config (64x64x16) to avoid multi-instantiation timeout.
 */
#include <cuda_runtime.h>
#include <cuda_tile.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <type_traits>

using namespace cuda::tiles;

template <typename T, int TM, int TN, int TK>
__tile_global__ void gemm_fp8(float* C, const T* A, const T* B, int M, int N, int K)
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

static void fill_fp8_e4m3(__nv_fp8_e4m3* p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = ((float)rand()/RAND_MAX)*2.0f - 1.0f;
        p[i] = __nv_fp8_e4m3(v);
    }
}
static void fill_fp8_e5m2(__nv_fp8_e5m2* p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = ((float)rand()/RAND_MAX)*2.0f - 1.0f;
        p[i] = __nv_fp8_e5m2(v);
    }
}

template <typename T>
double bench(int M, int N, int K, int warmup, int iters) {
    constexpr int TM = 64, TN = 64, TK = 16;
    size_t sa = (size_t)M*K*sizeof(T);
    size_t sb = (size_t)K*N*sizeof(T);
    size_t sc = (size_t)M*N*sizeof(float);
    T *dA, *dB; float *dC;
    T *hA = (T*)malloc(sa), *hB = (T*)malloc(sb);
    float *hC = (float*)malloc(sc);
    cudaMalloc(&dA, sa); cudaMalloc(&dB, sb); cudaMalloc(&dC, sc);

    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        fill_fp8_e4m3(hA, (size_t)M*K);
        fill_fp8_e4m3(hB, (size_t)K*N);
    } else {
        fill_fp8_e5m2(hA, (size_t)M*K);
        fill_fp8_e5m2(hB, (size_t)K*N);
    }
    cudaMemcpy(dA, hA, sa, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sb, cudaMemcpyHostToDevice);

    dim3 grid((M+TM-1)/TM, (N+TN-1)/TN);
    dim3 block(1,1,1);

    for (int i = 0; i < warmup; i++) {
        cudaMemset(dC, 0, sc);
        gemm_fp8<T,TM,TN,TK><<<grid, block>>>(dC, dA, dB, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEvent_t s, ev;
    cudaEventCreate(&s); cudaEventCreate(&ev);
    cudaMemset(dC, 0, sc);
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) {
        gemm_fp8<T,TM,TN,TK><<<grid, block>>>(dC, dA, dB, M, N, K);
    }
    cudaEventRecord(ev);
    cudaEventSynchronize(ev);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, ev);

    cudaMemcpy(hC, dC, sc, cudaMemcpyDeviceToHost);
    bool has_nan = false;
    for (size_t i = 0; i < (size_t)M*N; i++) {
        if (isnan(hC[i]) || isinf(hC[i])) { has_nan = true; break; }
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    if (has_nan) return -1;
    double flops = (double)M*N*K*2.0;
    return flops * iters / (ms * 1e-3) / 1e9;
}

int main() {
    printf("FP8 GEMM (E4M3 and E5M2) — 64x64x16 tile, M=N=K\n");
    printf("If tensor cores used, expect >>305 GFLOPS (BF16 baseline) at 4096\n\n");

    int sizes[] = {512, 1024, 2048, 4096};
    printf("%-8s  %12s  %12s\n", "Size", "E4M3 GFLOPS", "E5M2 GFLOPS");
    printf("--------------------------------------------------\n");

    for (int s = 0; s < 4; s++) {
        int M = sizes[s];
        double g_e4 = bench<__nv_fp8_e4m3>(M, M, M, 3, 10);
        double g_e5 = bench<__nv_fp8_e5m2>(M, M, M, 3, 10);
        printf("%-8d  %12.1f  %12.1f\n", M, g_e4, g_e5);
    }
    return 0;
}
