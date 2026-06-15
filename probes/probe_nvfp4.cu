/* NVFP4 GEMM — try multiple compute/output/scale combinations to see which (if any) cuBLASLt accepts on Thor. */
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA err %d: %s\n",__LINE__,cudaGetErrorString(e)); return 1;} } while(0)
#define CHECK_LT(x)   do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){printf("LT err %d: %d\n",__LINE__,(int)s); return 1;} } while(0)

struct TryCfg {
    const char* name;
    cublasComputeType_t compute;
    cudaDataType_t scale_type;
    cudaDataType_t out_type;
    cublasLtMatmulMatrixScale_t scale_mode;
};

int try_one(cublasLtHandle_t handle, const TryCfg& c, int M, int N, int K) {
    size_t sa = (size_t)M*K/2, sb = (size_t)K*N/2;
    size_t sC;
    if (c.out_type == CUDA_R_16F) sC = (size_t)M*N*sizeof(__half);
    else if (c.out_type == CUDA_R_16BF) sC = (size_t)M*N*sizeof(__nv_bfloat16);
    else sC = (size_t)M*N*sizeof(float);
    size_t sSc = (size_t)M*(K/16)*sizeof(__nv_fp8_e4m3);

    __nv_fp4_storage_t *dA, *dB;
    void *dC;
    __nv_fp8_e4m3 *dAs, *dBs;
    CHECK_CUDA(cudaMalloc(&dA, sa));
    CHECK_CUDA(cudaMalloc(&dB, sb));
    CHECK_CUDA(cudaMalloc(&dC, sC));
    CHECK_CUDA(cudaMalloc(&dAs, sSc));
    CHECK_CUDA(cudaMalloc(&dBs, sSc));

    unsigned char *hA=(unsigned char*)malloc(sa), *hB=(unsigned char*)malloc(sb);
    for (size_t i=0;i<sa;i++) hA[i]=0x88;  // valid FP4
    for (size_t i=0;i<sb;i++) hB[i]=0x88;
    CHECK_CUDA(cudaMemcpy(dA,hA,sa,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB,hB,sb,cudaMemcpyHostToDevice));
    __nv_fp8_e4m3 one = __nv_fp8_e4m3(1.0f);
    __nv_fp8_e4m3 *hAs=(__nv_fp8_e4m3*)malloc(sSc), *hBs=(__nv_fp8_e4m3*)malloc(sSc);
    for (size_t i=0;i<sSc/sizeof(__nv_fp8_e4m3);i++){hAs[i]=one;hBs[i]=one;}
    CHECK_CUDA(cudaMemcpy(dAs,hAs,sSc,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBs,hBs,sSc,cudaMemcpyHostToDevice));

    cublasLtMatrixLayout_t Ad, Bd, Cd;
    CHECK_LT(cublasLtMatrixLayoutCreate(&Ad, CUDA_R_4F_E2M1, M, K, M));
    CHECK_LT(cublasLtMatrixLayoutCreate(&Bd, CUDA_R_4F_E2M1, K, N, K));
    CHECK_LT(cublasLtMatrixLayoutCreate(&Cd, c.out_type, M, N, M));

    cublasLtMatmulDesc_t op;
    CHECK_LT(cublasLtMatmulDescCreate(&op, c.compute, c.scale_type));
    CHECK_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &c.scale_mode, sizeof(c.scale_mode)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &c.scale_mode, sizeof(c.scale_mode)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dAs, sizeof(dAs)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dBs, sizeof(dBs)));

    cublasLtMatmulPreference_t pref;
    CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = 64*1024*1024;
    CHECK_LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
    void *dWS=nullptr; CHECK_CUDA(cudaMalloc(&dWS, ws));

    cublasLtMatmulHeuristicResult_t heur{};
    int ret=0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(handle, op, Ad, Bd, Cd, Cd, pref, 1, &heur, &ret);
    const char* res;
    if (st == CUBLAS_STATUS_SUCCESS && ret > 0) res = "HEUR_OK";
    else if (st == CUBLAS_STATUS_NOT_SUPPORTED) res = "NOT_SUP";
    else { static char buf[32]; snprintf(buf,32,"ERR_%d",(int)st); res = buf; }

    // Also try direct call with NULL algo
    void *dWS2=nullptr; CHECK_CUDA(cudaMalloc(&dWS2, ws));
    float alpha=1.0f, beta=0.0f;
    cublasStatus_t st2 = cublasLtMatmul(handle, op, &alpha, dA, Ad, dB, Bd, &beta, dC, Cd, dC, Cd, NULL, dWS2, ws, 0);
    const char* res2 = (st2 == CUBLAS_STATUS_SUCCESS) ? "NULL_OK" :
                       (st2 == CUBLAS_STATUS_NOT_SUPPORTED) ? "NULL_NS" : "NULL_ERR";
    printf("  %-30s  heur=%-8s  null=%-8s\n", c.name, res, res2);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(op);
    cublasLtMatrixLayoutDestroy(Ad);
    cublasLtMatrixLayoutDestroy(Bd);
    cublasLtMatrixLayoutDestroy(Cd);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(dAs); cudaFree(dBs); cudaFree(dWS); cudaFree(dWS2);
    free(hA); free(hB); free(hAs); free(hBs);
    return 0;
}

int main() {
    cublasLtHandle_t handle;
    CHECK_LT(cublasLtCreate(&handle));
    int M=1024, N=1024, K=1024;
    printf("NVFP4 cuBLASLt config probe on Thor (sm_110), M=N=K=%d\n", M);
    printf("A,B = CUDA_R_4F_E2M1; scale mode varies; compute + output vary.\n\n");

    TryCfg cfgs[] = {
        // compute, scale_type, out_type, scale_mode
        {"C32F outF VEC16",     CUBLAS_COMPUTE_32F,        CUDA_R_32F, CUDA_R_32F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3},
        {"C32F outF VEC32",     CUBLAS_COMPUTE_32F,        CUDA_R_32F, CUDA_R_32F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0},
        {"C16F outH VEC16",     CUBLAS_COMPUTE_16F,        CUDA_R_16F, CUDA_R_16F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3},
        {"C16F outH VEC32",     CUBLAS_COMPUTE_16F,        CUDA_R_16F, CUDA_R_16F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0},
        {"C32F outH VEC16",     CUBLAS_COMPUTE_32F,        CUDA_R_32F, CUDA_R_16F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3},
        {"C32F outBF VEC16",    CUBLAS_COMPUTE_32F,        CUDA_R_32F, CUDA_R_16BF, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3},
        {"C16F outBF VEC16",    CUBLAS_COMPUTE_16F,        CUDA_R_16F, CUDA_R_16BF, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3},
        {"C32F_F16 outF VEC16", CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F, CUDA_R_32F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3},
        {"C32F_FBF outF VEC16", CUBLAS_COMPUTE_32F_FAST_16BF, CUDA_R_32F, CUDA_R_32F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3},
        {"C16F outF VEC16",     CUBLAS_COMPUTE_16F,        CUDA_R_16F, CUDA_R_32F, CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3},
    };
    for (auto& c : cfgs) try_one(handle, c, M, N, K);

    cublasLtDestroy(handle);
    return 0;
}
