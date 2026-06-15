/* NVFP4 (E2M1) GEMM via cuBLASLt — Blackwell micro-scaling format.
 * A, B: FP4 E2M1 with VEC16 E4M3 scale factors per 16 K-elements.
 * C: FP16 output.
 *
 * NVFP4 layout (column-major, cuBLASLt convention):
 *   A: (M, K) FP4     scale_A: (M, K/16) E4M3
 *   B: (K, N) FP4     scale_B: (K/16, N) E4M3
 *   C = A * B : (M, N) FP16
 */
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} } while(0)
#define CHECK_LT(x)   do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("cuBLASLt err %s:%d: %d\n",__FILE__,__LINE__,(int)s); return 1;} } while(0)

int main() {
    printf("NVFP4 (E2M1) GEMM via cuBLASLt on Thor (sm_110)\n");
    printf("Block-scaled FP4: VEC16 E4M3 scales per 16 K-elements\n\n");

    cublasLtHandle_t handle;
    CHECK_LT(cublasLtCreate(&handle));

    int sizes[] = {512, 1024, 2048, 4096};
    printf("%-8s  %12s  %10s  %-6s\n", "Size", "GFLOPS", "Lat(ms)", "Valid");
    printf("--------------------------------------------------\n");

    for (int s = 0; s < 4; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        // K must be divisible by 16 for VEC16 scales
        if (K % 16 != 0) { printf("%d: K not /16, skip\n", M); continue; }

        // FP4 storage: 2 elements per byte
        size_t sa = (size_t)M * K / 2;
        size_t sb = (size_t)K * N / 2;
        size_t sc = (size_t)M * N * sizeof(__half);
        // E4M3 scale factors: one per 16 K-elements
        size_t sA_sc = (size_t)M * (K / 16) * sizeof(__nv_fp8_e4m3);
        size_t sB_sc = (size_t)(K / 16) * N * sizeof(__nv_fp8_e4m3);

        __nv_fp4_storage_t *dA, *dB;
        __half *dC;
        __nv_fp8_e4m3 *dA_scale, *dB_scale;
        CHECK_CUDA(cudaMalloc(&dA, sa));
        CHECK_CUDA(cudaMalloc(&dB, sb));
        CHECK_CUDA(cudaMalloc(&dC, sc));
        CHECK_CUDA(cudaMalloc(&dA_scale, sA_sc));
        CHECK_CUDA(cudaMalloc(&dB_scale, sB_sc));

        // Fill with random FP4 data
        // For simplicity, fill with random bytes (not perfectly uniform FP4 but valid)
        unsigned char *hA = (unsigned char*)malloc(sa);
        unsigned char *hB = (unsigned char*)malloc(sb);
        __nv_fp8_e4m3 *hA_sc = (__nv_fp8_e4m3*)malloc(sA_sc);
        __nv_fp8_e4m3 *hB_sc = (__nv_fp8_e4m3*)malloc(sB_sc);
        for (size_t i = 0; i < sa; i++) hA[i] = rand() & 0xFF;
        for (size_t i = 0; i < sb; i++) hB[i] = rand() & 0xFF;
        // Set all scales to 1.0 (E4M3 representation of 1.0)
        __nv_fp8_e4m3 one = __nv_fp8_e4m3(1.0f);
        for (size_t i = 0; i < sA_sc / sizeof(__nv_fp8_e4m3); i++) hA_sc[i] = one;
        for (size_t i = 0; i < sB_sc / sizeof(__nv_fp8_e4m3); i++) hB_sc[i] = one;
        CHECK_CUDA(cudaMemcpy(dA, hA, sa, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB, hB, sb, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dA_scale, hA_sc, sA_sc, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(dB_scale, hB_sc, sB_sc, cudaMemcpyHostToDevice));

        // Matrix layouts (column-major)
        cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
        CHECK_LT(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, M, K, M));
        CHECK_LT(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, K, N, K));
        CHECK_LT(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, M, N, M));

        // Matmul descriptor
        cublasLtMatmulDesc_t opDesc;
        CHECK_LT(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        // Scale mode: VEC16 UE4M3 for A and B
        cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        CHECK_LT(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
            &scaleMode, sizeof(scaleMode)));
        CHECK_LT(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
            &scaleMode, sizeof(scaleMode)));
        // Scale pointers
        CHECK_LT(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &dA_scale, sizeof(dA_scale)));
        CHECK_LT(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &dB_scale, sizeof(dB_scale)));

        // Preference and heuristic
        cublasLtMatmulPreference_t pref;
        CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
        size_t ws = 64 * 1024 * 1024;
        CHECK_LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &ws, sizeof(ws)));
        void *dWS = nullptr;
        CHECK_CUDA(cudaMalloc(&dWS, ws));

        cublasLtMatmulHeuristicResult_t heur = {};
        int returned = 0;
        cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(handle, opDesc,
            Adesc, Bdesc, Cdesc, Cdesc, pref, 1, &heur, &returned);
        if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
            printf("%-8d  FP4 not supported by cuBLASLt heuristic (status=%d, returned=%d)\n",
                   M, (int)st, returned);
            // Cleanup and continue
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatmulDescDestroy(opDesc);
            cublasLtMatrixLayoutDestroy(Adesc);
            cublasLtMatrixLayoutDestroy(Bdesc);
            cublasLtMatrixLayoutDestroy(Cdesc);
            cudaFree(dA); cudaFree(dB); cudaFree(dC);
            cudaFree(dA_scale); cudaFree(dB_scale); cudaFree(dWS);
            free(hA); free(hB); free(hA_sc); free(hB_sc);
            continue;
        }

        // Warmup
        float alpha = 1.0f, beta = 0.0f;
        for (int i = 0; i < 3; i++) {
            CHECK_CUDA(cudaMemset(dC, 0, sc));
            CHECK_LT(cublasLtMatmul(handle, opDesc, &alpha, dA, Adesc, dB, Bdesc,
                &beta, dC, Cdesc, dC, Cdesc, &heur.algo, dWS, ws, 0));
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        // Timed
        cudaEvent_t sev, eev;
        cudaEventCreate(&sev); cudaEventCreate(&eev);
        int iters = 10;
        CHECK_CUDA(cudaMemset(dC, 0, sc));
        cudaEventRecord(sev);
        for (int i = 0; i < iters; i++) {
            cublasLtMatmul(handle, opDesc, &alpha, dA, Adesc, dB, Bdesc,
                &beta, dC, Cdesc, dC, Cdesc, &heur.algo, dWS, ws, 0);
        }
        cudaEventRecord(eev);
        cudaEventSynchronize(eev);
        float ms = 0;
        cudaEventElapsedTime(&ms, sev, eev);

        // Validate (no NaN/Inf in C)
        __half *hC = (__half*)malloc(sc);
        CHECK_CUDA(cudaMemcpy(hC, dC, sc, cudaMemcpyDeviceToHost));
        bool has_nan = false;
        for (size_t i = 0; i < (size_t)M*N; i++) {
            float v = __half2float(hC[i]);
            if (isnan(v) || isinf(v)) { has_nan = true; break; }
        }
        double flops = (double)M*N*K*2.0;
        double gflops = flops * iters / (ms * 1e-3) / 1e9;
        printf("%-8d  %12.1f  %10.3f  %s\n", M, gflops, ms/iters, has_nan ? "FAIL" : "PASS");

        // Cleanup
        free(hC);
        free(hA); free(hB); free(hA_sc); free(hB_sc);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        cudaFree(dA_scale); cudaFree(dB_scale); cudaFree(dWS);
        cudaEventDestroy(sev); cudaEventDestroy(eev);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatmulDescDestroy(opDesc);
        cublasLtMatrixLayoutDestroy(Adesc);
        cublasLtMatrixLayoutDestroy(Bdesc);
        cublasLtMatrixLayoutDestroy(Cdesc);
    }

    cublasLtDestroy(handle);
    return 0;
}
