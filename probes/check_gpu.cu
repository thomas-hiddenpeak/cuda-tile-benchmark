#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    cudaSetDevice(0);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    printf("GPU: %s\n", props.name);
    printf("SMs: %d\n", props.multiProcessorCount);
    printf("Capability: %d.%d\n", props.major, props.minor);
    printf("Shared Memory per SM: %d KB\n", props.sharedMemPerMultiprocessor / 1024);
    return 0;
}
