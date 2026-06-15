#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int clockRate = 0;
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);
    double clockGhz = clockRate / 1e6;
    
    printf("Device: %s (sm_%d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("SMEM per SM: %d bytes (%.1f KB)\n", prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor / 1024.0);
    printf("Base clock: %.1f MHz (%.3f GHz)\n", clockRate/1000.0, clockGhz);
    
    // FP4 Peak
    double peak_tflops = prop.multiProcessorCount * 128.0 * 128.0 * 2.0 * clockGhz / 1000.0;
    printf("FP4 Peak (base clock): %.1f TFLOPS\n", peak_tflops);
    
    // SMEM per SM = 96KB for SM110
    int smem_bytes = prop.sharedMemPerMultiprocessor;
    int stage_bytes = 256*128*2 + 256*128*2 + 256*128/16 + 256*128/16 + 4096;
    printf("  Each stage: ~%d bytes\n", stage_bytes);
    int max_stages = smem_bytes / stage_bytes;
    printf("  Max stages at full SMEM: %d\n", max_stages);
    printf("  Auto stage count (K128): ~6 stages\n");
    
    // Occupancy
    printf("\n=== Occupancy Analysis ===\n");
    printf("Cluster shape: 2×4×1 (8 thread blocks per cluster)\n");
    printf("SMEM per SM: %d bytes\n", prop.sharedMemPerMultiprocessor);
    printf("Stage 256x256x128: A(%d B) + B(%d B) + SFA(%d B) + SFB(%d B) + Pipeline(%d B)\n",
        256*128*2, 256*128*2, 256*128/16, 256*128/16, 4096);
    printf("Stage bytes: %d\n", stage_bytes);
    printf("Max blocks per SM (with cluster 2x4x1): 4 blocks per SM\n");
    
    // 20 SMs × 4 blocks = 80 blocks total
    // For 4096x4096: 16x16 = 256 blocks → ~64 blocks per SM!
    printf("\nFor 4096x4096:\n");
    printf("  Grid: 16×16 = 256 blocks total\n");
    printf("  Blocks per SM: 256/20 = 12.8\n");
    printf("  This means 4096 has plenty of blocks for full occupancy\n");
    
    // 2048x2048: 8x8 = 64 blocks → 3.2 blocks per SM
    printf("\nFor 2048x2048:\n");
    printf("  Grid: 8×8 = 64 blocks total\n");
    printf("  Blocks per SM: 64/20 = 3.2\n");
    
    // 1024x1024: 4x4 = 16 blocks → 0.8 blocks per SM
    printf("\nFor 1024x1024:\n");
    printf("  Grid: 4×4 = 16 blocks total\n");
    printf("  Blocks per SM: 16/20 = 0.8\n");
    printf("  → Low occupancy! This is why 1024 has lower efficiency\n");
    
    return 0;
}
