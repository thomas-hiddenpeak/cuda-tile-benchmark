/* Minimal FP8 GEMM — verify the tile API accepts FP8 (E4M3 and E5M2) */
#include <cuda_runtime.h>
#include <cuda_tile.h>
#include <cuda_fp8.h>
#include <stdio.h>

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

int main()
{
    const int M = 1024, N = 1024, K = 1024;
    const int TM = 64, TN = 64, TK = 16;
    size_t sa = (size_t)M*K*sizeof(__nv_fp8_e4m3);
    size_t sb = (size_t)K*N*sizeof(__nv_fp8_e4m3);
    size_t sc = (size_t)M*N*sizeof(float);

    __nv_fp8_e4m3 *dA, *dB; float *dC;
    cudaMalloc(&dA, sa); cudaMalloc(&dB, sb); cudaMalloc(&dC, sc);
    cudaMemset(dA, 0, sa); cudaMemset(dB, 0, sb); cudaMemset(dC, 0, sc);

    dim3 grid((M+TM-1)/TM, (N+TN-1)/TN);
    dim3 block(1,1,1);

    // E4M3
    gemm_fp8<__nv_fp8_e4m3, TM, TN, TK><<<grid, block>>>(dC, dA, dB, M, N, K);
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("FP8 E4M3 launch error: %s\n", cudaGetErrorString(e));
        return 1;
    }
    printf("FP8 E4M3 kernel launched OK (M=N=K=%d, tile=%dx%dx%d)\n", M, TM, TN, TK);

    // E5M2
    __nv_fp8_e5m2 *dA2, *dB2;
    cudaMalloc(&dA2, sa); cudaMalloc(&dB2, sb);
    cudaMemset(dA2, 0, sa); cudaMemset(dB2, 0, sb);
    gemm_fp8<__nv_fp8_e5m2, TM, TN, TK><<<grid, block>>>(dC, dA2, dB2, M, N, K);
    cudaDeviceSynchronize();
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        printf("FP8 E5M2 launch error: %s\n", cudaGetErrorString(e));
        return 1;
    }
    printf("FP8 E5M2 kernel launched OK (M=N=K=%d, tile=%dx%dx%d)\n", M, TM, TN, TK);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(dA2); cudaFree(dB2);
    return 0;
}
