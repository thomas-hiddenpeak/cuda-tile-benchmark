#include <stdio.h>

int main() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("GPU: %s\n", prop.name);
    printf("sm_%d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    
    // Check for sm_110 features
    printf("\nChecking FP4 support:\n");
    printf("  (Will try to compile and run a simple test)\n");
    return 0;
}
