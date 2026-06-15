#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int clockKhz;
    cudaDeviceGetAttribute(&clockKhz, cudaDevAttrClockRate, 0);
    double clockGhz = clockKhz / 1e6;
    int sm_count = prop.multiProcessorCount;
    
    printf("=== SM110 FP4 Peak Analysis ===\n");
    printf("SM count: %d\n", sm_count);
    printf("Clock: %.1f MHz (%.3f GHz)\n", clockKhz/1000.0, clockGhz);
    printf("\n");
    
    // SM110 FP4 block-scaled MMA specs
    // Each SM has 2 UMMA (TCGEN05) modules
    // Each UMMA does 128x128xK FP4 elements per cycle
    // Each FP4 multiply-add = 2 FLOPs
    // Total: 2 UMMA x 128 x 128 x 2 FLOPs = 65,536 FLOPs/SM/cycle
    
    // Current formula in bench_nvfp4_ptx.cu:
    // 128 x 128 x 2 = 32,768 FLOPs (only 1 UMMA)
    double peak_current = sm_count * 128.0 * 128.0 * 2.0 * clockGhz;
    printf("Current formula (1 UMMA):  %.1f TFLOPS\n", peak_current / 1000.0);
    
    // Correct formula: 2 UMMA units per SM
    // 2 x 128 x 128 x 2 = 65,536 FLOPs/SM/cycle
    double peak_correct = sm_count * 2.0 * 128.0 * 128.0 * 2.0 * clockGhz;
    printf("Correct formula (2 UMMA):  %.1f TFLOPS\n", peak_correct / 1000.0);
    
    printf("\n");
    printf("Your reference: 1035 TF (default power)\n");
    printf("Ratio (2 UMMA / 1 UMMA):   %.1f x\n", peak_correct / peak_current);
    printf("Ratio (1035 / current):    %.1f x\n", 1035.0 / (peak_current / 1000.0));
    printf("Ratio (1035 / correct):    %.1f x\n", 1035.0 / (peak_correct / 1000.0));
    
    double clock_for_1035_2umma = 1035.0 * 1000.0 / (sm_count * 2.0 * 128.0 * 128.0 * 2.0);
    printf("\nClock for 1035 TF (2 UMMA):  %.0f MHz\n", clock_for_1035_2umma * 1000.0);
    
    double clock_for_1035_1umma = 1035.0 * 1000.0 / (sm_count * 128.0 * 128.0 * 2.0);
    printf("Clock for 1035 TF (1 UMMA):  %.0f MHz\n", clock_for_1035_1umma * 1000.0);
    
    return 0;
}
