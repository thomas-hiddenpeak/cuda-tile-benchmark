/* Minimal single-config BF16 GEMM timing across sizes.
 * Just one kernel instantiation, multiple problem sizes.
 * Answers: does the tile compiler use BF16 tensor cores? */
#include <cuda_runtime.h>
#include <cuda_tile.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace cuda::tiles;

template <int TM, int TN, int TK>
__tile_global__ void gemm_bf16(float* C, const __nv_bfloat16* A,
                              const __nv_bfloat16* B, int M, int N, int K)
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

int main() {
    printf("BF16 GEMM single-config (64x64x16) across sizes\n");
    printf("If tensor cores are used, expect >>2000 GFLOPS at 4096\n\n");

    int sizes[] = {512, 1024, 2048, 4096};
    constexpr int TM = 64, TN = 64, TK = 16;

    printf("%-10s  %10s  %10s  %6s\n", "Size", "GFLOPS", "Lat(ms)", "Valid");
    printf("--------------------------------------------------\n");

    for (int s = 0; s < 4; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
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
        dim3 block(1, 1, 1);  // tile compiler figures out actual threads

        // Warmup
        for (int i = 0; i < 3; i++) {
            cudaMemset(dC, 0, sc);
            gemm_bf16<TM,TN,TK><<<grid, block>>>(dC, dA, dB, M, N, K);
        }
        cudaDeviceSynchronize();

        cudaEvent_t sev, eev;
        cudaEventCreate(&sev); cudaEventCreate(&eev);
        int iters = 10;
        cudaMemset(dC, 0, sc);
        cudaEventRecord(sev);
        for (int i = 0; i < iters; i++) {
            gemm_bf16<TM,TN,TK><<<grid, block>>>(dC, dA, dB, M, N, K);
        }
        cudaEventRecord(eev);
        cudaEventSynchronize(eev);
        float ms = 0;
        cudaEventElapsedTime(&ms, sev, eev);

        cudaMemcpy(hC, dC, sc, cudaMemcpyDeviceToHost);
        bool has_nan = false;
        for (size_t i = 0; i < (size_t)M*N; i++) {
            if (isnan(hC[i]) || isinf(hC[i])) { has_nan = true; break; }
        }

        double flops = (double)M*N*K*2.0;
        double gflops = flops * iters / (ms * 1e-3) / 1e9;
        printf("%-10d  %10.1f  %10.3f  %s\n",
               M, gflops, ms/iters, has_nan ? "FAIL" : "PASS");

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        free(hA); free(hB); free(hC);
    }
    return 0;
}
