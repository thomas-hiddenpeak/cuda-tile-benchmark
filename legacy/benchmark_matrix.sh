#!/bin/bash
# benchmark_matrix.sh - Systematically test FP4 GEMM configurations
# Usage: ./benchmark_matrix.sh

set -e
cd "$(dirname "$0")"

NVCC=/usr/local/cuda-13.3/bin/nvcc
CUTLASS_DIR=${CUTLASS_DIR:-/home/rm01/opencodeWorkspace/cutlass}
SRC=bench_nvfp4_ptx.cu
OUTPUT=bench_nvfp4

echo "=== FP4 GEMM Configuration Matrix Test ==="
echo ""

# Configuration matrix
declare -a MMA_M=(256 256 256 512 512 512)
declare -a MMA_N=(256 256 256 256 512 512)
declare -a MMA_K=(128 256 512 128 256 512)
declare -a CLUSTER_M=(2 2 2 4 4 4)
declare -a CLUSTER_N=(4 4 4 2 4 2)

declare -a LABELS=("K128-C2x4" "K256-C2x4" "K512-C2x4" "K128-512-C4x2" "K256-512-C4x4" "K512-512-C4x2")

for i in $(seq 0 $((${#MMA_M[@]} - 1))); do
    M=${MMA_M[$i]}
    N=${MMA_N[$i]}
    K=${MMA_K[$i]}
    CM=${CLUSTER_M[$i]}
    CN=${CLUSTER_N[$i]}
    label=${LABELS[$i]}
    
    echo "----------------------------------------"
    echo "Config ${i}: ${label}"
    echo "  Tile: ${M}x${N}x${K}, Cluster: ${CM}x${CN}x1"
    
    # Temporarily modify source
    cp $SRC bench_nvfp4_ptx.cu.bak
    
    # Replace tile shape
    sed -i "s/using MmaTileShape = Shape<_256,_256,_256>;/using MmaTileShape = Shape<_${M},_${N},_${K}>;/" $SRC
    sed -i "s/using ClusterShape = Shape<_2,_4,_1>;/using ClusterShape = Shape<_${CM},_${CN},_1>;/" $SRC
    
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
        $OUTPUT 2>&1 | grep -E "FP4 Peak:|M=N=K|    ->" | sed 's/^/    /'
        echo ""
    else
        echo "  Build: FAILED"
        cp bench_nvfp4_ptx.cu.bak $SRC
    fi
done

# Restore original
cp bench_nvfp4_ptx.cu.bak $SRC
rm -f bench_nvfp4_ptx.cu.bak

echo "========================================"
echo "Matrix test complete"
