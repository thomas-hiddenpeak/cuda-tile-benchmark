/*
 * NVFP4 GEMM Kernel using CUTLASS block-scaled FP4 MMA
 * sm_110a: 20 SMs x 32,768 FLOPs/cycle x 1.575 GHz = ~1,032 TFLOPS (dense)
 */

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "cute/arch/mma_sm120.hpp"

using namespace cute;

__global__ void nvfp4_gemm_blockscaled_kernel(
    const __nv_fp4_e2m1* __restrict__ A,
    const __nv_fp4_e2m1* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for FP4 tiles
    __shared__ uint32_t sA[128][64];
    __shared__ uint32_t sB[64][128];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tx + (ty + tz * 2) * 32;

    int warp_id = tid / 32;
    int lane = tid % 32;

    int bm = blockIdx.x * 128;
    int bn = blockIdx.y * 128;

    int warp_m = warp_id % 4;
    int warp_n = warp_id / 4;

    int start_m = bm + warp_m * 32;
    int start_n = bn + warp_n * 64;

    float accum[128] = {0};

    using MMA = SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<float_e2m1_t, float_e2m1_t, float, float_ue8m0_t, 32>;

    for (int k = 0; k < K; k += 32) {
        for (int i = tx; i < 128; i += 32) {
            for (int j = ty; j < 32; j += 32) {
                int idx = (bm + i) * K + (k + j);
                uint32_t val = 0;
                if (idx < (size_t)M * K)
                    val = (uint32_t)A[idx];
                sA[i][j] = val;
            }
        }

        __syncthreads();

        for (int i = tx; i < 32; i += 32) {
            for (int j = ty; j < 8; j += 32) {
                int idx = (k + i) * N + (bn + j);
                uint32_t val = 0;
                if (idx < (size_t)K * N)
                    val = (uint32_t)B[idx];
                sB[i][j] = val;
            }
        }

        __syncthreads();

        int a_row = lane / 4 * 4;
        int a_col = lane % 4 * 2;

        uint32_t regA[64];
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 8; c++) {
                int row = a_row + r;
                int col = a_col + c;
                regA[r * 8 + c] = (row < 16 && col < 32) ? sA[row][col] : 0;
            }
        }

        uint32_t regB[16];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 4; c++) {
                int col = a_col + c;
                regB[r * 2 + c] = (col < 8) ? sB[r][col] : 0;
            }
        }

        uint8_t sfa = 0x80;
        uint8_t sfb = 0x80;

        for (int i = 0; i < 32; i++) {
            int out_row = i / 4 * 4 + a_row % 4;
            int out_col = i % 4 + a_col / 2 * 2;

            float c0 = accum[out_row * 8 + out_col];
            float c1 = accum[out_row * 8 + out_col + 1];
            float c2 = accum[out_row * 8 + out_col + 2];
            float c3 = accum[out_row * 8 + out_col + 3];

            uint32_t a0 = regA[i * 2];
            uint32_t a1 = regA[i * 2 + 1];
            uint32_t a2 = regA[i * 2 + 2];
            uint32_t a3 = regA[i * 2 + 3];

            uint32_t b0 = regB[i];
            uint32_t b1 = regB[i + 1];

            float d0, d1, d2, d3;
            MMA::fma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sfa, sfb);

            accum[out_row * 8 + out_col] = d0;
            accum[out_row * 8 + out_col + 1] = d1;
            accum[out_row * 8 + out_col + 2] = d2;
            accum[out_row * 8 + out_col + 3] = d3;
        }

        __syncthreads();
    }

    for (int r = 0; r < 16; r++) {
        for (int c = 0; c < 8; c++) {
            int m = start_m + r;
            int n = start_n + c;
            if (m < M && n < N) {
                C[m * N + n] = accum[r * 8 + c];
            }
        }
    }
}

static void fill_fp4(__nv_fp4_e2m1* p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = ((float)rand() / 32768.0f) * 2.0f - 1.0f;
        p[i] = __nv_fp4_e2m1(v);
    }
}

double bench(int M, int N, int K, int warmup, int iters) {
    size_t sa = (size_t)M * K * sizeof(__nv_fp4_e2m1);
    size_t sb = (size_t)K * N * sizeof(__nv_fp4_e2m1);
    size_t sc = (size_t)M * N * sizeof(float);

    size_t free_mem, total;
    cudaMemGetInfo(&free_mem, &total);
    if (sa + sb + sc > free_mem * 0.8) return -1;

    __nv_fp4_e2m1 *dA, *dB;
    float *dC;

    __nv_fp4_e2m1 *hA = (__nv_fp4_e2m1*)malloc(sa);
    __nv_fp4_e2m1 *hB = (__nv_fp4_e2m1*)malloc(sb);

    if (!hA || !hB) return -1;

    cudaMalloc(&dA, sa);
    cudaMalloc(&dB, sb);
    cudaMalloc(&dC, sc);

    if (cudaGetLastError() != cudaSuccess) {
        ::free(hA); ::free(hB);
        return -1;
    }

    fill_fp4(hA, (size_t)M * K);
    fill_fp4(hB, (size_t)K * N);

    cudaMemcpy(dA, hA, sa, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sb, cudaMemcpyHostToDevice);

    dim3 grid((M + 128 - 1) / 128, (N + 128 - 1) / 128);
    dim3 block(32, 2, 4);

    for (int i = 0; i < warmup; i++) {
        cudaMemset(dC, 0, sc);
        nvfp4_gemm_blockscaled_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaMemset(dC, 0, sc);
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) {
        nvfp4_gemm_blockscaled_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms;
    cudaEventElapsedTime(&ms, s, e);

    cudaEventDestroy(s);
    cudaEventDestroy(e);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    ::free(hA);
    ::free(hB);

    double flops = (double)M * N * K * 2.0;
    return flops * iters / (ms * 1e-3) / 1e9;
}

int main() {
    printf("=== NVFP4 Block-Scaled MMA Kernel (CUTLASS PTX) ===\n");
    printf("sm_110a: 20 SMs x 32,768 FLOPs/cycle x 1.575 GHz\n");
    printf("Theoretical Peak (dense): ~1,032 TFLOPS\n");
    printf("Theoretical Peak (sparse): ~2,064 TFLOPS\n\n");

    int sizes[] = {2048, 4096};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("%-10s  %10s  %10s  %8s\n", "M=N=K", "GFLOPS", "TFLOPS", "Eff%%");
    printf("--------------------------------------------------------------------\n");

    for (int i = 0; i < n_sizes; i++) {
        int M = sizes[i];
        double mem_gb = ((double)M * M * 2 + (double)M * M * 2 + (double)M * M * 4) / 1024.0 / 1024.0 / 1024.0;

        printf("  [ %d ] M=N=K=%d (GPU mem = %.1f GB)\n", i, M, mem_gb);
        fflush(stdout);

        double gflops = bench(M, M, M, 3, 10);
        if (gflops < 0) {
            printf("  -> OOM\n");
            continue;
        }

        double tflops = gflops / 1000.0;
        double eff = gflops / 1032000.0 * 100.0;
        printf("    -> %10.0f  %8.1f  %6.1f%%\n\n", gflops, tflops, eff);
        fflush(stdout);

        if (eff > 50) {
            printf("  >>> REACHED %.1f%% EFFICIENCY at %d^3\n", eff, M);
            break;
        }
    }

    return 0;
}
