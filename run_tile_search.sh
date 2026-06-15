#!/bin/bash
# Systematic tile shape search for FP4 GEMM optimization
# Tests all tile shapes × cluster configs
# Safe: no sed -i, trap cleanup, JSON persistence, no set -e

cd "$(dirname "$0")"
source env.sh
SRC=bench_nvfp4_fp4.cu

COMPILE_FLAGS="-std=c++17 -O3 -arch=sm_110a ${CUTLASS_INCLUDES} ${CUTLASS_COMPUTE_FLAGS}"

# Tile shapes: M N K
TILE_SHAPES=(
  "128 128 128"
  "128 128 256"
  "128 128 512"
  "256 256 128"
  "256 256 256"
  "256 256 512"
  "512 512 128"
  "512 512 256"
)

# Cluster shapes: X Y Z
CLUSTERS=(
  "1 1 1"
  "2 2 1"
  "2 4 1"
  "4 2 1"
  "4 4 1"
)

TOTAL=$(( ${#TILE_SHAPES[@]} * ${#CLUSTERS[@]} ))

# ── Setup results ──
mkdir -p results
RESULTS_FILE="results/results_$(date +%Y%m%d_%H%M%S).jsonl"

cleanup() { :; }
trap 'cleanup' EXIT INT TERM

get_gpu_name() {
  nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown"
}
GPU_NAME=$(get_gpu_name)

write_result() {
  local config="$1" cluster="$2" gflops="$3" tflops="$4" peak_pct="$5" status="$6"
  local ts
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf '{"config":"%s","cluster":"%s","gflops":%s,"tflops":%s,"peak_pct":%s,"status":"%s","timestamp":"%s","gpu":"%s"}\n' \
    "$config" "$cluster" "$gflops" "$tflops" "$peak_pct" "$status" "$ts" "$GPU_NAME" >> "$RESULTS_FILE"
}

echo "=== FP4 Tile Shape Search (4096³) ==="
echo "Tile shapes: ${#TILE_SHAPES[@]}, Clusters: ${#CLUSTERS[@]}, Total: $TOTAL configs"
printf "%-40s %12s %8s %8s\n" "Config" "GFLOPS" "TF" "Peak%"
printf "%-40s %12s %8s %8s\n" "------" "------" "--" "-----"

count=0
for TILE in "${TILE_SHAPES[@]}"; do
  read -r TM TN TK <<< "$TILE"

  for CLUSTER in "${CLUSTERS[@]}"; do
    read -r CX CY CZ <<< "$CLUSTER"
    count=$((count + 1))

    cfg_name="M${TM}xN${TN}xK${TK}"
    cl_name="C${CX}x${CY}x${CZ}"
    full_label="$cfg_name $cl_name"
    echo -n "[$count/$TOTAL] $full_label ... "

    BIN="bench_nvfp4_fp4.m${TM}n${TN}k${TK}.c${CX}${CY}${CZ}"

    build_log=$(mktemp)
    if $NVCC $COMPILE_FLAGS \
      -DTILE_M=$TM -DTILE_N=$TN -DTILE_K=$TK \
      -DTILE_CLUSTERM=$CX -DTILE_CLUSTERN=$CY -DTILE_CLUSTERZ=$CZ \
      $SRC -o "$BIN" 2>"$build_log"; then

      run_output=$($BIN --m=4096 --n=4096 --k=4096 --iterations=30 2>&1)
      run_exit=$?
      rm -f "$BIN"

      if [ $run_exit -ne 0 ]; then
        echo "RUN_FAIL"
        write_result "$cfg_name" "$cl_name" "0" "0" "0" "RUN_FAIL"
      else
        gflops=$(echo "$run_output" | grep "GFLOPS" | awk '{print $NF}')
        if [ -z "$gflops" ]; then
          echo "NO_GFLOPS"
          write_result "$cfg_name" "$cl_name" "0" "0" "0" "RUN_FAIL"
        else
          tflops=$(echo "$gflops" | awk '{printf "%.0f", $1/1000.0}')
          peak_pct=$(echo "$gflops" "$PEAK_TFLOPS" | awk '{printf "%.1f", ($1/1000.0/$2)*100}')
          printf "%-40s %12.0f %7.0fT %7s%%\n" "$full_label" "$gflops" "$tflops" "$peak_pct"
          write_result "$cfg_name" "$cl_name" "$gflops" "$tflops" "$peak_pct" "PASS"
        fi
      fi
    else
      echo "BUILD_FAIL"
      head -30 "$build_log" >&2
      write_result "$cfg_name" "$cl_name" "0" "0" "0" "BUILD_FAIL"
    fi
    rm -f "$build_log"
  done
done

echo ""
echo "Results saved to $RESULTS_FILE"
echo "=== Tile Search Complete ==="
