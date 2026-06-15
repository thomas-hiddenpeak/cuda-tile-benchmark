#!/bin/bash
# Test a single config: ./run_tile_test.sh M N K CM CN
# Safe: no sed -i, trap cleanup, JSON persistence, no set -e

cd "$(dirname "$0")"
source env.sh
SRC="${PROJECT_ROOT}/benchmarks/bench_nvfp4_fp4.cu"

COMPILE_FLAGS="-std=c++17 -O3 -arch=sm_110a ${PROJECT_INCLUDES} ${CUTLASS_INCLUDES} ${CUTLASS_COMPUTE_FLAGS}"

M=${1:-256}
N=${2:-128}
K=${3:-256}
CM=${4:-2}
CN=${5:-1}
CZ=1

BIN="${PROJECT_ROOT}/bench_nvfp4_fp4.m${M}n${N}k${K}.c${CM}${CN}${CZ}"

cfg_name="M${M}xN${N}xK${K}"
cl_name="C${CM}x${CN}x${CZ}"

# ── Setup results ──
RESULTS_DIR="${PROJECT_ROOT}/results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="${RESULTS_DIR}/results_$(date +%Y%m%d_%H%M%S).jsonl"

cleanup() {
  rm -f "$BIN"
}
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

echo "[1/1] $cfg_name $cl_name ..."

build_log=$(mktemp)
if $NVCC $COMPILE_FLAGS \
  -DTILE_M=$M -DTILE_N=$N -DTILE_K=$K \
  -DTILE_CLUSTERM=$CM -DTILE_CLUSTERN=$CN -DTILE_CLUSTERZ=$CZ \
  $SRC -o "$BIN" 2>"$build_log"; then

  run_output=$($BIN --m=4096 --n=4096 --k=4096 --iterations=30 2>&1)
  run_exit=$?

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
      printf "M%dxN%dxK%d C%dx%dx%d -> %s TF (%.0f GFLOPS) [%s%% peak]\n" \
        $M $N $K $CM $CN $CZ "$tflops" "$gflops" "$peak_pct"
      write_result "$cfg_name" "$cl_name" "$gflops" "$tflops" "$peak_pct" "PASS"
    fi
  fi
else
  echo "BUILD_FAIL"
  head -30 "$build_log" >&2
  write_result "$cfg_name" "$cl_name" "0" "0" "0" "BUILD_FAIL"
fi

rm -f "$build_log"
echo "Results saved to ${RESULTS_DIR}/"
