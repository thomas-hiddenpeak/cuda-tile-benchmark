/***************************************************************************************************
 * Multi-source FP4 peak FLOPS calculator for NVIDIA Thor (sm_110a)
 *
 * Verifies GPU clock via sysfs devfreq (accurate) vs CUDA API (may be stale on Thor).
 * Computes theoretical FP4 dense/sparse peak based on verified clock.
 **************************************************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Read sysfs devfreq current frequency
static int read_sysfs_clock_mhz() {
    FILE *f = fopen("/sys/devices/platform/bus@0/d0b0000000.pcie/pci0000:00/0000:00:00.0/0000:01:00.0/gpu-gpc-0/devfreq/gpu-gpc-0/cur_freq", "r");
    if (!f) return -1;
    char buf[64] = {0};
    if (fgets(buf, sizeof(buf), f) == NULL) {
        fclose(f);
        return -1;
    }
    fclose(f);
    // Strip whitespace/newline
    char *end = buf;
    while (*end && (*end != '\n' && *end != '\r' && *end != ' ')) end++;
    *end = 0;
    long long hz = atoll(buf);
    return (int)(hz / 1000000);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int sm_count = prop.multiProcessorCount;

    // Source A: CUDA API (may be stale on Thor)
    int cudaClockKhz;
    cudaDeviceGetAttribute(&cudaClockKhz, cudaDevAttrClockRate, 0);
    double cudaClockMhz = cudaClockKhz / 1000.0;

    // Source B: sysfs devfreq (accurate)
    int sysfsClockMhz = read_sysfs_clock_mhz();

    // Source C: Memory clock
    int memClockKhz;
    cudaDeviceGetAttribute(&memClockKhz, cudaDevAttrMemoryClockRate, 0);
    double memClockMhz = memClockKhz / 1000.0;

    printf("==========================================================\n");
    printf("  FP4 Peak FLOPS Calculator — NVIDIA Thor (sm_110a)\n");
    printf("==========================================================\n\n");

    printf("GPU: %s\n", prop.name);
    printf("SMs: %d\n\n", sm_count);

    printf("Clock Sources:\n");
    printf("  CUDA devAttrClockRate:  %6.0f MHz  (may be stale)\n", cudaClockMhz);
    printf("  sysfs devfreq cur_freq: %6d MHz  (accurate)\n", sysfsClockMhz > 0 ? sysfsClockMhz : -1);
    printf("  Memory clock:           %6.0f MHz\n\n", memClockMhz);

    // Use sysfs if available, else fall back to CUDA
    double gpuClockMhz = (sysfsClockMhz > 0) ? sysfsClockMhz : cudaClockMhz;
    printf("Using GPU clock: %.0f MHz\n\n", gpuClockMhz);

    // SM110 FP4 throughput: 32,768 FLOPS/cycle/SM (dense)
    const double FLOPS_PER_CYCLE_FP4_DENSE  = 32768.0;
    const double FLOPS_PER_CYCLE_FP4_SPARSE = 65536.0;

    double peak_dense  = sm_count * FLOPS_PER_CYCLE_FP4_DENSE  * gpuClockMhz / 1e6;
    double peak_sparse = sm_count * FLOPS_PER_CYCLE_FP4_SPARSE * gpuClockMhz / 1e6;

    printf("Theoretical Peak:\n");
    printf("  FP4 Dense:  %6.0f TFLOPS\n", peak_dense);
    printf("  FP4 Sparse: %6.0f TFLOPS\n\n", peak_sparse);

    // NVIDIA stated peak at MAXN 1575 MHz
    double peak_maxn_dense  = sm_count * FLOPS_PER_CYCLE_FP4_DENSE  * 1575.0 / 1e6;
    double peak_maxn_sparse = sm_count * FLOPS_PER_CYCLE_FP4_SPARSE * 1575.0 / 1e6;

    printf("NVIDIA Stated (1,575 MHz MAXN):\n");
    printf("  FP4 Dense:  ~1,035 TFLOPS\n");
    printf("  FP4 Sparse: ~2,070 TFLOPS\n");
    printf("  Calculated: %6.0f / %6.0f TFLOPS\n\n", peak_maxn_dense, peak_maxn_sparse);

    // Current best benchmark result
    double current_best = 579.0;
    printf("Current Best Benchmark: %.0f TF (M256xN128xK256, C2x2x1)\n", current_best);
    printf("  Efficiency vs actual peak: %.0f%%\n", current_best / peak_dense * 100);
    printf("  Efficiency vs NVIDIA peak: %.0f%%\n", current_best / peak_maxn_dense * 100);
    printf("  Gap to 80%% peak: %.1fx\n", (peak_dense * 0.80) / current_best);
    printf("  Gap to 90%% peak: %.1fx\n", (peak_dense * 0.90) / current_best);

    return 0;
}
