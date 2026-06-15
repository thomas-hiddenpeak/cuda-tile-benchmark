/*
 * NVFP4 Block-Scaled GEMM Benchmark for SM110 (NVIDIA Thor)
 * Adapted from CUTLASS example 72a_blackwell_nvfp4_bf16_gemm.cu
 */

#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

#include "cutlass/detail/sm100_blockscaled_layout.hpp"

using namespace cute;

// FP4 block-scaled GEMM configuration for SM100/SM110
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC = float;
using ElementD = cutlass::float_e2m1_t;
using ElementAccumulator = float;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Tile shape for FP4 GEMM (matching CUTLASS example)
using MmaTileShape = Shape<_256,_256,_128>;
using ClusterShape = Shape<_2,_2,_1>;

// Epilogue collective builder for FP4 GEMM
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, 8,
    ElementD, LayoutDTag, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

// Mainloop collective builder for FP4 GEMM
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, 32,
    ElementB, LayoutBTag, 32,
    ElementAccumulator,
    MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Stride and Layout types
using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

// Sm1xxBlkScaledConfig for SFA/SFB layout generation
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// HostTensors
cutlass::HostTensor<ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
cutlass::HostTensor<ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
cutlass::HostTensor<ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
cutlass::HostTensor<ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=== NVFP4 TCGEN05 UMMA Block-Scaled MMA Kernel ===\n");
    printf("Device: %s (sm_%d.%d)\n", prop.name, prop.major, prop.minor);
    printf("SM count: %d\n", prop.multiProcessorCount);

    int clockRate = 0;
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);
    // clockRate is in kHz, convert to GHz: clockRate / 1e6
    double clockGhz = clockRate / 1e6;
    
    // SM100/SM110 FP4 block-scaled MMA: 128x128x2 = 32,768 FLOPs per SM per cycle
    // Peak = SM_count * 32768 FLOPs/cycle * clock_GHz
    // Result in GFLOPS: SM_count * 32768 * clockGhz
    // Result in TFLOPS: SM_count * 32768 * clockGhz / 1000
    double peak_gflops = prop.multiProcessorCount * 128.0 * 128.0 * 2.0 * clockGhz;
    double peak_tflops = peak_gflops / 1000.0;
    printf("Clock: %.1f MHz (%.3f GHz)\n", clockRate/1000.0, clockGhz);
    printf("FP4 Peak: %.1f TFLOPS (%.0f GFLOPS)\n\n", peak_tflops, peak_tflops * 1000.0);

    printf("%-10s  %12s  %10s  %8s\n", "M=N=K", "GFLOPS", "TFLOPS", "Eff%%");
    printf("--------------------------------------------------------------------\n");

    int sizes[] = {1024, 2048, 4096};
    int n_sizes = 3;

    for (int si = 0; si < n_sizes; si++) {
        int M = sizes[si], N = M, K = M;
        
        printf("  M=N=K=%d\n", M);
        fflush(stdout);

        // Set up strides
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

        // Set up layouts
        LayoutA layout_A = make_layout(make_shape(M, K, 1), stride_A);
        LayoutB layout_B = make_layout(make_shape(N, K, 1), stride_B);
        LayoutC layout_C = make_layout(make_shape(M, N, 1), stride_C);
        LayoutD layout_D = make_layout(make_shape(M, N, 1), stride_D);

        // Generate SFA/SFB layouts using Sm1xxBlkScaledConfig
        LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
        LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

        // Allocate host tensors
        block_A.reset(cutlass::make_Coord(size(layout_A)));
        block_B.reset(cutlass::make_Coord(size(layout_B)));
        block_C.reset(cutlass::make_Coord(size(layout_C)));
        block_D.reset(cutlass::make_Coord(size(layout_D)));
        block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
        block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

        // Initialize with random data (matching CUTLASS example)
        cutlass::reference::host::TensorFillRandomUniform(block_A.host_view(), 2021, 2.0, -2.0, 0);
        cutlass::reference::host::TensorFillRandomUniform(block_B.host_view(), 2022, 2.0, -2.0, 0);
        cutlass::reference::host::TensorFillRandomUniform(block_C.host_view(), 2023, 2.0, -2.0, 0);
        cutlass::reference::host::TensorFillRandomUniform(block_SFA.host_view(), 2024, 2.0, -2.0, 0);
        cutlass::reference::host::TensorFillRandomUniform(block_SFB.host_view(), 2025, 2.0, -2.0, 0);

        // Sync to device
        block_A.sync_device();
        block_B.sync_device();
        block_C.sync_device();
        block_SFA.sync_device();
        block_SFB.sync_device();

        // Create CUTLASS GEMM instance
        Gemm gemm;
        
        // Set up arguments (matching CUTLASS example exactly)
        auto arguments = typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            { // Mainloop arguments
              block_A.device_data(), stride_A,
              block_B.device_data(), stride_B,
              block_SFA.device_data(), layout_SFA,
              block_SFB.device_data(), layout_SFB
            },
            { // Epilogue arguments
              {1.0f, 0.0f},
              block_C.device_data(), stride_C,
              block_D.device_data(), stride_D
            }
        };

        size_t workspace_size = Gemm::get_workspace_size(arguments);
        void* workspace;
        cudaMalloc(&workspace, workspace_size);

        // First run (warmup/validation)
        gemm.initialize(arguments, workspace);
        gemm.run();
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  Kernel error: %s\n\n", cudaGetErrorString(err));
            cudaFree(workspace);
            continue;
        }

        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < 20; i++) {
            gemm.initialize(arguments, workspace);
            gemm.run();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        
        // GFLOPS = total_FLOPs / time_seconds / 1e9
        // total_FLOPs = M * N * K * 2 * num_runs (2 FLOPs per FP4 FMA)
        // time_seconds = ms / 1000
        // GFLOPS = total_FLOPs / (ms / 1000) / 1e9 = total_FLOPs / ms * 1000 / 1e9
        double total_flops = (double)M * N * K * 2.0 * 20;
        double gflops = total_flops / ms / 1e6;  // GFLOPS
        double tflops = gflops / 1000.0;          // TFLOPS
        double eff = tflops / peak_tflops * 100.0; // efficiency %

        cudaEventDestroy(start); cudaEventDestroy(stop);
        printf("    -> %10.0f  %8.1f  %6.1f%%\n\n", gflops, tflops, eff);
        fflush(stdout);

        cudaFree(workspace);
    }

    // Rectangular tests (LLM inference patterns)
    printf("\n=== Rectangular Shapes (LLM Patterns) ===\n");
    printf("%-18s  %12s  %10s  %8s\n", "MxNxK", "GFLOPS", "TFLOPS", "Eff%%");
    printf("--------------------------------------------------------------------\n");

    struct RectShape { const char* name; int m, n, k; };
    RectShape rect_shapes[] = {
        {"M4096xN2048xK4096", 4096, 2048, 4096},
        {"M2048xN4096xK4096", 2048, 4096, 4096},
        {"M4096xN4096xK2048", 4096, 4096, 2048},
    };
    int n_rect = 3;

    for (int ri = 0; ri < n_rect; ri++) {
        int M = rect_shapes[ri].m, N = rect_shapes[ri].n, K = rect_shapes[ri].k;
        
        printf("  %s\n", rect_shapes[ri].name);
        fflush(stdout);

        // Set up strides
        StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

        // Set up layouts
        LayoutA layout_A = make_layout(make_shape(M, K, 1), stride_A);
        LayoutB layout_B = make_layout(make_shape(N, K, 1), stride_B);
        LayoutC layout_C = make_layout(make_shape(M, N, 1), stride_C);
        LayoutD layout_D = make_layout(make_shape(M, N, 1), stride_D);

        // Generate SFA/SFB layouts using Sm1xxBlkScaledConfig
        LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
        LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

        // Allocate host tensors
        block_A.reset(cutlass::make_Coord(size(layout_A)));
        block_B.reset(cutlass::make_Coord(size(layout_B)));
        block_C.reset(cutlass::make_Coord(size(layout_C)));
        block_D.reset(cutlass::make_Coord(size(layout_D)));
        block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
        block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

        // Initialize with random data (matching CUTLASS example)
        cutlass::reference::host::TensorFillRandomUniform(block_A.host_view(), 2021, 2.0, -2.0, 0);
        cutlass::reference::host::TensorFillRandomUniform(block_B.host_view(), 2022, 2.0, -2.0, 0);
        cutlass::reference::host::TensorFillRandomUniform(block_C.host_view(), 2023, 2.0, -2.0, 0);
        cutlass::reference::host::TensorFillRandomUniform(block_SFA.host_view(), 2024, 2.0, -2.0, 0);
        cutlass::reference::host::TensorFillRandomUniform(block_SFB.host_view(), 2025, 2.0, -2.0, 0);

        // Sync to device
        block_A.sync_device();
        block_B.sync_device();
        block_C.sync_device();
        block_SFA.sync_device();
        block_SFB.sync_device();

        // Create CUTLASS GEMM instance
        Gemm gemm;
        
        // Set up arguments (matching CUTLASS example exactly)
        auto arguments = typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            { // Mainloop arguments
              block_A.device_data(), stride_A,
              block_B.device_data(), stride_B,
              block_SFA.device_data(), layout_SFA,
              block_SFB.device_data(), layout_SFB
            },
            { // Epilogue arguments
              {1.0f, 0.0f},
              block_C.device_data(), stride_C,
              block_D.device_data(), stride_D
            }
        };

        size_t workspace_size = Gemm::get_workspace_size(arguments);
        void* workspace;
        cudaMalloc(&workspace, workspace_size);

        // First run (warmup/validation)
        gemm.initialize(arguments, workspace);
        gemm.run();
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  Kernel error: %s\n\n", cudaGetErrorString(err));
            cudaFree(workspace);
            continue;
        }

        // Benchmark
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < 20; i++) {
            gemm.initialize(arguments, workspace);
            gemm.run();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        
        double total_flops = (double)M * N * K * 2.0 * 20;
        double gflops = total_flops / ms / 1e6;  // GFLOPS
        double tflops = gflops / 1000.0;          // TFLOPS
        double eff = tflops / peak_tflops * 100.0; // efficiency %

        cudaEventDestroy(start); cudaEventDestroy(stop);
        printf("    -> %10.0f  %8.1f  %6.1f%%\n\n", gflops, tflops, eff);
        fflush(stdout);

        cudaFree(workspace);
    }

    return 0;
}
