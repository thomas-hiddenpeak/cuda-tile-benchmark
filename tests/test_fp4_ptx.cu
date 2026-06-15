#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <stdint.h>
#include <stdio.h>

__global__ void test_blockscaled_fp4_mma(const __nv_fp4_e2m1* A, const __nv_fp4_e2m1* B, float* C) {
    uint32_t a0 = (uint32_t)A[0];
    uint32_t a1 = (uint32_t)A[1];
    uint32_t a2 = (uint32_t)A[2];
    uint32_t a3 = (uint32_t)A[3];
    
    uint32_t b0 = (uint32_t)B[0];
    uint32_t b1 = (uint32_t)B[1];
    
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
    float d0, d1, d2, d3;
    
    // Use the exact CUTLASS SM120 block-scaled format
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
           "r"(b0),  "r"(b1),
           "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3),
           "r"(0x80u),
           "h"((uint16_t)0), "h"((uint16_t)0),
           "r"(0x80u),
           "h"((uint16_t)0), "h"((uint16_t)0));
    
    C[0] = d0; C[1] = d1; C[2] = d2; C[3] = d3;
}

int main() {
    printf("Compiled successfully!\n");
    return 0;
}
