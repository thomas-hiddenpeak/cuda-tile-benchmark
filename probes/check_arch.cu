#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Shared mem per block: %d KB\n", prop.sharedMemPerBlock / 1024);
    return 0;
}
