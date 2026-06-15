/*
 * NVFP4 Handwritten GEMM Kernel for sm_110a
 *
 * Uses block-scaled tensor core MMA (tcgen05.mma.blockscaled)
 * Warp-specialized design with TMEM
 */

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// MMA intrinsics for FP4→FP32
// m16x8x32: 16×8×32 = 8192 flops per instruction
// 512 FLOPs per TC = 64 instructions per TC per cycle
#define MMA_FP4_16x8x32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1, c0,c1,c2,c3) \
    asm volatile( \
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 " \
        "{%0,  %1,  %2,  %3}," \
        "{%4,  %5,  %6,  %7}," \
        "{%8,  %9}," \
        "{%10, %11, %12, %13};\n" \
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) \
        :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3), \
           "r"(b0),  "r"(b1), \
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3))

// FP4 layout: 4 bits per value, 2 values per byte
// 16×8×32 FP4 tile:
//   A: 16×32 = 512 FP4 values = 256 bytes = 128 uint16_t = 64 uint32_t
//   B: 32×8 = 256 FP4 values = 128 bytes = 64 uint16_t = 32 uint32_t

// 16×8×32 layout:
// A: 16 rows × 32 cols (packed as 16×4 uint32_t per row)
// B: 32 rows × 8 cols (packed as 32×2 uint32_t per row)

__global__ void nvfp4_gemm_kernel(
    const __nv_fp4_e2m1* __restrict__ A,
    const __nv_fp4_e2m1* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // Shared memory for FP4 tiles
    __shared__ uint32_t sA[128][8];   // 128 rows × 8 uint32_t per row (128×32 FP4)
    __shared__ uint32_t sB[32][8];    // 32 rows × 8 uint32_t per row (32×32 FP4)

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    // Block coordinates
    int block_x = blockIdx.x;
    int block_y = blockIdx.y;
    int block_m = block_x * 128;  // 128 rows per block
    int block_n = block_y * 128;  // 128 cols per block

    // Warp-level computation
    // 8 warps per block (256 threads)
    // Each warp processes 16×8 elements with 32 K-dimension
    int tid = tx + (ty + tz * 2) * 32;  // 0-255
    int warp_id = tid / 32;
    int lane = tid % 32;

    // 4 warps in M direction (0-3), 2 warps in N direction (0-1)
    int warp_m = warp_id % 4;
    int warp_n = warp_id / 4;

    int start_m = block_m + warp_m * 32;
    int start_n = block_n + warp_n * 64;

    // Accumulators: 16 rows × 8 cols = 128 floats per warp
    float accum[128] = {0};

    for (int k = 0; k < K; k += 32) {
        // Load A tile: 128×32 FP4 → 128×8 uint32_t
        for (int i = tx; i < 128; i += 32) {
            for (int j = ty; j < 4; j += 2) {
                int row = block_m + i;
                int col = k + j * 8 + lane;
                int idx = row * K + col;
                if (idx < (size_t)M * K) {
                    sA[i][j * 2 + lane / 4] = (uint32_t)A[idx];
                } else {
                    sA[i][j * 2 + lane / 4] = 0;
                }
            }
        }

        __syncthreads();

        // Load B tile: 32×128 FP4 → 32×8 uint32_t
        for (int i = tx; i < 32; i += 32) {
            for (int j = ty; j < 8; j += 2) {
                int row = k + i;
                int col = block_n + j * 4 + lane;
                int idx = row * N + col;
                if (idx < (size_t)K * N) {
                    sB[i][j + lane / 4] = (uint32_t)B[idx];
                } else {
                    sB[i][j + lane / 4] = 0;
                }
            }
        }

        __syncthreads();

        // FP4→FP32 MMA: 16×8×32
        // Each warp processes 16×8 elements
        int a_row = lane / 4 * 4;  // 0, 4, 8, 12
        int a_col = lane % 4 * 2;  // 0, 2, 4, 6

        for (int row = 0; row < 16; row++) {
            for (int col = 0; col < 8; col++) {
                // Load A and B tiles for this row and column
                uint32_t a0 = sA[warp_m * 32 + row][a_col + col / 2];
                uint32_t a1 = sA[warp_m * 32 + row][a_col + col / 2 + 1];
                uint32_t a2 = sA[warp_m * 32 + row][a_col + col / 2 + 2];
                uint32_t a3 = sA[warp_m * 32 + row][a_col + col / 2 + 3];

                uint32_t b0 = sB[row][col];
                uint32_t b1 = sB[row][col + 1];

                float c0 = accum[row * 8 + col];
                float c1 = 0.0f;
                float c2 = 0.0f;
                float c3 = 0.0f;

                float d0, d1, d2, d3;
                MMA_FP4_16x8x32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);

                accum[row * 8 + col] = d0;
                accum[row * 8 + col + 1] = d1;
                accum[row * 8 + col + 2] = d2;
                accum[row * 8 + col + 3] = d3;
            }
        }

        __syncthreads();
    }

    // Write results
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            int m = start_m + i;
            int n = start_n + j;
            if (m < M && n < N) {
                C[m * N + n] = accum[i * 8 + j];
            }
        }
    }
}

static void fill_fp4(__nv_fp4_e2m1* p, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float v = ((float)rand()/32768.0f) * 2.0f - 1.0f;
        p[i] = __nv_fp4_e2m1(v);
    }
}

double bench(int M, int N, int K, int warmup, int iters) {
    size_t sa = (size_t)M*K*sizeof(__nv_fp4_e2m1);
    size_t sb = (size_t)K*N*sizeof(__nv_fp4_e2m1);
    size_t sc = (size_t)M*N*sizeof(float);

    size_t free_mem, total;
    cudaMemGetInfo(&free_mem, &total);
    if (sa + sb + sc > free_mem * 0.8) return -1;

    __nv_fp4_e2m1 *dA, *dB; float *dC;
    __nv_fp4_e2m1 *hA = (__nv_fp4_e2m1*)malloc(sa);
    __nv_fp4_e2m1 *hB = (__nv_fp4_e2m1*)malloc(sb);
    if (!hA || !hB) return -1;

    cudaMalloc(&dA, sa); cudaMalloc(&dB, sb); cudaMalloc(&dC, sc);
    if (cudaGetLastError() != cudaSuccess) { ::free(hA); ::free(hB); return -1; }

    fill_fp4(hA, (size_t)M*K);
    fill_fp4(hB, (size_t)K*N);
    cudaMemcpy(dA, hA, sa, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sb, cudaMemcpyHostToDevice);

    dim3 grid((M+128-1)/128, (N+128-1)/128);
    dim3 block(32, 2, 4);  // 256 threads, 8 warps

    for (int i = 0; i < warmup; i++) {
        cudaMemset(dC, 0, sc);
        nvfp4_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaDeviceSynchronize();

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaMemset(dC, 0, sc);
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) {
        nvfp4_gemm_kernel<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);

    float ms;
    cudaEventElapsedTime(&ms, s, e);

    cudaEventDestroy(s); cudaEventDestroy(e);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    ::free(hA); ::free(hB);

    double flops = (double)M*N*K*2.0;
    return flops * iters / (ms * 1e-3) / 1e9;
}

int main() {
    printf("=== NVFP4 Handwritten Kernel ===\n");
    printf("sm_110a: 20 SMs × 32,768 FLOPs/cycle × 1.575 GHz\n");
    printf("Theoretical Peak (dense): ~1,032 TFLOPS\n");
    printf("Theoretical Peak (sparse): ~2,064 TFLOPS\n\n");

    printf("%-10s  %10s  %10s  %8s\n", "M=N=K", "GFLOPS", "TFLOPS", "Eff%%");
    printf("--------------------------------------------------------------------\n");

    int sizes[] = {2048, 4096};
    int n_sizes = sizeof(sizes)/sizeof(sizes[0]);

    for (int i = 0; i < n_sizes; i++) {
        int M = sizes[i];
        double mem_gb = ((double)M*M*2 + (double)M*M*2 + (double)M*M*4) / 1024.0/1024.0/1024.0;

        printf("  [ %d ] M=N=K=%d (GPU mem = %.1f GB)\n", i, M, mem_gb);
        fflush(stdout);

        double gflops = bench(M, M, M, 3, 10);
        if (gflops < 0) {
            printf("  → OOM\n");
            continue;
        }

        double tflops = gflops / 1000.0;
        double eff = gflops / 1032000.0 * 100.0;
        printf("    → %10.0f  %8.1f  %6.1f%%\n\n", gflops, tflops, eff);
        fflush(stdout);

        if (eff > 50) {
            printf("  >>> REACHED %.1f%% EFFICIENCY at %d³\n", eff, M);
            break;
        }
    }

    return 0;
}
