#!/bin/bash
# benchmark_fp4_matrix.sh - Systematically test FP4→FP4 tile configurations
# Usage: ./benchmark_fp4_matrix.sh

set -e
cd "$(dirname "$0")"

NVCC=/usr/local/cuda-13.3/bin/nvcc
CUTLASS_DIR=${CUTLASS_DIR:-/home/rm01/opencodeWorkspace/cutlass}
SRC=bench_nvfp4_fp4.cu
OUTPUT=bench_nvfp4_fp4

echo "=== FP4→FP4 Configuration Matrix Test ==="
echo ""

# Configuration matrix - focusing on FP4→FP4 optimal space
# Based on: M128x128xK256 + C2x2x1 = 492 TF baseline
declare -a MMA_M=(128 128 128 128 128 256 256 256 256 64 64 256)
declare -a MMA_N=(128 128 128 256 256 128 128 256 256 128 256 512)
declare -a MMA_K=(128 256 512 128 256 256 512 256 512 256 256 128)
declare -a CLUSTER_M=(2 2 2 2 2 4 4 4 4 2 2 4)
declare -a CLUSTER_N=(2 2 2 4 4 2 2 4 4 4 4 4)
declare -a SFVEC=(16 16 16 16 16 16 16 16 16 16 16 16)

declare -a LABELS=(
    "M128x128-K128-C2x2"
    "M128x128-K256-C2x2"
    "M128x128-K512-C2x2"
    "M128x256-K128-C2x4"
    "M128x256-K256-C2x4"
    "M256x128-K256-C4x2"
    "M256x128-K512-C4x2"
    "M256x256-K256-C4x4"
    "M256x256-K512-C4x4"
    "M64x128-K256-C2x4"
    "M64x256-K256-C2x4"
    "M256x512-K128-C4x4"
)

for i in $(seq 0 $((${#MMA_M[@]} - 1))); do
    M=${MMA_M[$i]}
    N=${MMA_N[$i]}
    K=${MMA_K[$i]}
    CM=${CLUSTER_M[$i]}
    CN=${CLUSTER_N[$i]}
    SF=${SFVEC[$i]}
    label=${LABELS[$i]}
    
    echo "----------------------------------------"
    echo "Config ${i}: ${label}"
    echo "  Tile: ${M}x${N}x${K}, Cluster: ${CM}x${CN}x1, SFVec: ${SF}"
    
    # Temporarily modify source
    cp $SRC bench_nvfp4_fp4.cu.bak
    
    # Replace tile shape
    sed -i "s/using MmaTileShape        = Shape<_128,_128,_256>;/using MmaTileShape        = Shape<_${M},_${N},_${K}>;/" $SRC
    sed -i "s/using ClusterShape        = Shape<_2,_2,_1>;/using ClusterShape        = Shape<_${CM},_${CN},_1>;/" $SRC
    sed -i "s/constexpr int InputSFVectorSize  = 16;/constexpr int InputSFVectorSize  = ${SF};/" $SRC
    sed -i "s/constexpr int OutputSFVectorSize = 16;/constexpr int OutputSFVectorSize = ${SF};/" $SRC
    
    # Build
    $NVCC -std=c++17 -O3 -arch=sm_110a \
      -I $CUTLASS_DIR/include \
      -I $CUTLASS_DIR/tools/util/include \
      -I $CUTLASS_DIR/examples/common \
      --expt-relaxed-constexpr --expt-extended-lambda \
      $SRC -o $OUTPUT -lcudart 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "  Build: OK"
        echo "  Results:"
        
        for size in 2048 4096; do
            result=$($OUTPUT --m=$size --n=$size --k=$size --iterations=10 2>&1)
            gflops=$(echo "$result" | grep "GFLOPS" | awk '{print $NF}')
            tflops=$(echo "$gflops" | awk '{printf "%.1f", $1/1000.0}')
            echo "    M=N=K=${size}:  ${gflops} GFLOPS (${tflops} TF)"
        done
        
        echo ""
    else
        echo "  Build: FAILED"
        cp bench_nvfp4_fp4.cu.bak $SRC
    fi
    
    # Restore original
    cp bench_nvfp4_fp4.cu.bak $SRC
done

rm -f bench_nvfp4_fp4.cu.bak

echo "========================================"
echo "Matrix test complete"
