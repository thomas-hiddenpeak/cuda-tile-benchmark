/* Minimal BF16 GEMM test to verify the tile compiler accepts BF16 */
#include <cuda_runtime.h>
#include <cuda_tile.h>
#include <cuda_bf16.h>
#include <stdio.h>

using namespace cuda::tiles;

template <int TILE_M, int TILE_N, int TILE_K>
__tile_global__ void gemm_bf16_kernel(float* C, const __nv_bfloat16* A, const __nv_bfloat16* B, int M, int N, int K)
{
    using A_shape = shape<TILE_M, TILE_K>;
    using B_shape = shape<TILE_K, TILE_N>;
    using C_shape = shape<TILE_M, TILE_N>;
    using C_tile  = tile<float, C_shape>;

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

    auto a_view = partition_view(a_span, A_shape{});
    auto b_view = partition_view(b_span, B_shape{});
    auto c_view = partition_view(c_span, C_shape{});

    C_tile c_tile = zeros<C_tile>();
    for (int k = 0; k < K; k += TILE_K) {
        auto at = a_view.load(bm, k);
        auto bt = b_view.load(k, bn);
        c_tile = mma(at, bt, c_tile);
    }
    c_view.store(c_tile, bm, bn);
}

int main()
{
    const int M = 1024, N = 1024, K = 1024;
    const int TM = 64, TN = 64, TK = 16;
    size_t sa = (size_t)M*K*sizeof(__nv_bfloat16);
    size_t sb = (size_t)K*N*sizeof(__nv_bfloat16);
    size_t sc = (size_t)M*N*sizeof(float);

    __nv_bfloat16 *dA, *dB; float *dC;
    cudaMalloc(&dA, sa); cudaMalloc(&dB, sb); cudaMalloc(&dC, sc);
    cudaMemset(dA, 0, sa); cudaMemset(dB, 0, sb); cudaMemset(dC, 0, sc);

    dim3 grid((M+TM-1)/TM, (N+TN-1)/TN);
    dim3 block(1,1,1);
    gemm_bf16_kernel<TM,TN,TK><<<grid, block>>>(dC, dA, dB, M, N, K);
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("BF16 launch error: %s\n", cudaGetErrorString(e));
        return 1;
    }
    printf("BF16 kernel launched OK (M=N=K=%d, tile=%dx%dx%d)\n", M, TM, TN, TK);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
