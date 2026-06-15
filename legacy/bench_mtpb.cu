/* Multi-tile-per-block FP32 GEMM.
 * Each block computes BT_M x BT_N output tiles of shape (TILE_M, TILE_N).
 * Loads A_block (BLOCK_M, TILE_K) and B_block (TILE_K, BLOCK_N) once per K
 * iteration, reuses them across BT_M * BT_N mmas.
 *
 * Uses extract() to get sub-tiles from the larger blocks.
 */
#include <cuda_runtime.h>
#include <cuda_tile.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace cuda::tiles;

template <int TILE_M, int TILE_N, int TILE_K, int BT_M, int BT_N>
__tile_global__ void gemm_mtpb(float* C, const float* A, const float* B, int M, int N, int K)
{
    constexpr int BLOCK_M = BT_M * TILE_M;
    constexpr int BLOCK_N = BT_N * TILE_N;

    using A_block_shape = shape<BLOCK_M, TILE_K>;
    using B_block_shape = shape<TILE_K, BLOCK_N>;
    using A_sub_shape = shape<TILE_M, TILE_K>;
    using B_sub_shape = shape<TILE_K, TILE_N>;
    using C_sub_shape = shape<TILE_M, TILE_N>;
    using C_sub_tile = tile<float, C_sub_shape>;

    int bx = detail::bid<0>();
    int by = detail::bid<1>();
    int bm = bx * BLOCK_M;
    int bn = by * BLOCK_N;

    auto a_ext = extents<int, dynamic_extent, dynamic_extent>{M, K};
    auto b_ext = extents<int, dynamic_extent, dynamic_extent>{K, N};
    auto c_ext = extents<int, dynamic_extent, dynamic_extent>{M, N};
    auto a_sp = tensor_span(A, a_ext, layout_left{});
    auto b_sp = tensor_span(B, b_ext, layout_left{});
    auto c_sp = tensor_span(C, c_ext, layout_left{});

    auto a_view = partition_view(a_sp, A_block_shape{});
    auto b_view = partition_view(b_sp, B_block_shape{});
    auto c_view = partition_view(c_sp, C_sub_shape{});

    // BT_M x BT_N array of accumulators
    C_sub_tile c_tiles[BT_M][BT_N];
    for (int m = 0; m < BT_M; m++)
        for (int n = 0; n < BT_N; n++)
            c_tiles[m][n] = zeros<C_sub_tile>();

    // K loop with data reuse
    for (int k = 0; k < K; k += TILE_K) {
        // Load larger A and B blocks once
        auto a_block = a_view.load(bm, k);
        auto b_block = b_view.load(k, bn);

        // BT_M * BT_N mmas reusing the same a_block/b_block
        for (int m = 0; m < BT_M; m++) {
            for (int n = 0; n < BT_N; n++) {
                auto a_sub = extract(a_block, A_sub_shape{}, m * TILE_M, 0);
                auto b_sub = extract(b_block, B_sub_shape{}, 0, n * TILE_N);
                c_tiles[m][n] = mma(a_sub, b_sub, c_tiles[m][n]);
            }
        }
    }

    // Store each sub-tile
    for (int m = 0; m < BT_M; m++)
        for (int n = 0; n < BT_N; n++)
            c_view.store(c_tiles[m][n], bm + m * TILE_M, bn + n * TILE_N);
}

static void fill_random(float* p, size_t n) {
    for (size_t i = 0; i < n; i++)
        p[i] = ((float)rand()/RAND_MAX)*2.0f - 1.0f;
}

template <int TM, int TN, int TK, int BTM, int BTN>
double bench(int M, int N, int K, int warmup, int iters) {
    constexpr int BLOCK_M = BTM * TM;
    constexpr int BLOCK_N = BTN * TN;
    size_t sa = (size_t)M*K*sizeof(float);
    size_t sb = (size_t)K*N*sizeof(float);
    size_t sc = (size_t)M*N*sizeof(float);
    float *dA, *dB, *dC;
    float *hA = (float*)malloc(sa), *hB = (float*)malloc(sb), *hC = (float*)malloc(sc);
    cudaMalloc(&dA, sa); cudaMalloc(&dB, sb); cudaMalloc(&dC, sc);
    fill_random(hA, (size_t)M*K); fill_random(hB, (size_t)K*N);
    cudaMemcpy(dA, hA, sa, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sb, cudaMemcpyHostToDevice);

    dim3 grid((M+BLOCK_M-1)/BLOCK_M, (N+BLOCK_N-1)/BLOCK_N);
    dim3 block(1, 1, 1);

    for (int i = 0; i < warmup; i++) {
        cudaMemset(dC, 0, sc);
        gemm_mtpb<TM,TN,TK,BTM,BTN><<<grid, block>>>(dC, dA, dB, M, N, K);
    }
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { printf("  launch err: %s\n", cudaGetErrorString(e)); return -1; }

    cudaEvent_t s, ev;
    cudaEventCreate(&s); cudaEventCreate(&ev);
    cudaMemset(dC, 0, sc);
    cudaEventRecord(s);
    for (int i = 0; i < iters; i++) {
        gemm_mtpb<TM,TN,TK,BTM,BTN><<<grid, block>>>(dC, dA, dB, M, N, K);
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
    printf("Multi-tile-per-block FP32 GEMM (M=N=K=4096)\n");
    printf("Each block computes BT_M*BT_N output tiles, reuses A/B loads\n\n");
    printf("%-24s  %10s\n", "Config", "GFLOPS");
    printf("--------------------------------------\n");

    int sizes[] = {2048, 4096};
    struct Cfg { int tm, tn, tk, btm, btn; const char* name; };
    Cfg configs[] = {
        {16, 16, 16, 1, 1, "16x16x16 (1x1)"},
        {32, 32, 16, 1, 1, "32x32x16 (1x1)"},
        {64, 64, 16, 1, 1, "64x64x16 (1x1)"},
        {16, 16, 16, 2, 2, "16x16x16 (2x2) B=32x32"},
        {16, 16, 16, 4, 4, "16x16x16 (4x4) B=64x64"},
        {16, 16, 16, 8, 8, "16x16x16 (8x8) B=128x128"},
        {32, 32, 16, 2, 2, "32x32x16 (2x2) B=64x64"},
        {32, 32, 16, 4, 4, "32x32x16 (4x4) B=128x128"},
        {16, 16, 32, 2, 2, "16x16x32 (2x2) B=32x32"},
        {16, 16, 32, 4, 4, "16x16x32 (4x4) B=64x64"},
        {16, 16, 32, 8, 8, "16x16x32 (8x8) B=128x128"},
        {32, 32, 32, 2, 2, "32x32x32 (2x2) B=64x64"},
    };
    int ncfg = sizeof(configs)/sizeof(configs[0]);

    for (int s = 0; s < 2; s++) {
        int M = sizes[s];
        printf("\n--- M=N=K=%d ---\n", M);
        for (int c = 0; c < ncfg; c++) {
            int TM=configs[c].tm, TN=configs[c].tn, TK=configs[c].tk;
            int BTM=configs[c].btm, BTN=configs[c].btn;
            double g = -1;
            if      (TM==16 && TN==16 && TK==16 && BTM==1 && BTN==1) g = bench<16,16,16,1,1>(M,M,M,3,5);
            else if (TM==32 && TN==32 && TK==16 && BTM==1 && BTN==1) g = bench<32,32,16,1,1>(M,M,M,3,5);
            else if (TM==64 && TN==64 && TK==16 && BTM==1 && BTN==1) g = bench<64,64,16,1,1>(M,M,M,3,5);
            else if (TM==16 && TN==16 && TK==16 && BTM==2 && BTN==2) g = bench<16,16,16,2,2>(M,M,M,3,5);
            else if (TM==16 && TN==16 && TK==16 && BTM==4 && BTN==4) g = bench<16,16,16,4,4>(M,M,M,3,5);
            else if (TM==16 && TN==16 && TK==16 && BTM==8 && BTN==8) g = bench<16,16,16,8,8>(M,M,M,3,5);
            else if (TM==32 && TN==32 && TK==16 && BTM==2 && BTN==2) g = bench<32,32,16,2,2>(M,M,M,3,5);
            else if (TM==32 && TN==32 && TK==16 && BTM==4 && BTN==4) g = bench<32,32,16,4,4>(M,M,M,3,5);
            else if (TM==16 && TN==16 && TK==32 && BTM==2 && BTN==2) g = bench<16,16,32,2,2>(M,M,M,3,5);
            else if (TM==16 && TN==16 && TK==32 && BTM==4 && BTN==4) g = bench<16,16,32,4,4>(M,M,M,3,5);
            else if (TM==16 && TN==16 && TK==32 && BTM==8 && BTN==8) g = bench<16,16,32,8,8>(M,M,M,3,5);
            else if (TM==32 && TN==32 && TK==32 && BTM==2 && BTN==2) g = bench<32,32,32,2,2>(M,M,M,3,5);
            printf("  %-22s  %10.1f\n", configs[c].name, g);
        }
    }
    return 0;
}
