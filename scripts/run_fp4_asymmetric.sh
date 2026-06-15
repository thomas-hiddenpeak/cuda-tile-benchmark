#!/bin/bash
# FP4→FP4 asymmetric tile search - promising configs only
# Safe: no sed -i, trap cleanup, JSON persistence, no set -e

cd "$(dirname "$0")"
source env.sh
SRC="${PROJECT_ROOT}/benchmarks/bench_nvfp4_fp4.cu"

COMPILE_FLAGS="-std=c++17 -O3 -arch=sm_110a ${PROJECT_INCLUDES} ${CUTLASS_INCLUDES} ${CUTLASS_COMPUTE_FLAGS}"

# ── Build config list: M128 series + M256 series ──
CONFIGS=()

# M128 tiles - N in {128, 192, 256}, clusters {1x1, 2x1, 1x2, 2x2}
for N in 128 192 256; do
  for C in "1 1" "2 1" "1 2" "2 2"; do
    read -r CM CN <<< "$C"
    CONFIGS+=("128 $N 256 $CM $CN 1 M128xN${N}xK256 C${CM}x${CN}x1")
  done
done

# M256 tiles - N in {128, 192, 256}, clusters {2x1, 1x2, 2x2, 4x1}
for N in 128 192 256; do
  for C in "2 1" "1 2" "2 2" "4 1"; do
    read -r CM CN <<< "$C"
    CONFIGS+=("256 $N 256 $CM $CN 1 M256xN${N}xK256 C${CM}x${CN}x1")
  done
done
TOTAL=${#CONFIGS[@]}

# ── Setup results ──
RESULTS_DIR="${PROJECT_ROOT}/results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="${RESULTS_DIR}/results_$(date +%Y%m%d_%H%M%S).jsonl"

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

echo "=== FP4→FP4 Asymmetric Tile Search ==="
echo "Total configs: $TOTAL"
printf "%-40s %12s %8s %8s\n" "Config" "GFLOPS" "TF" "Peak%"
printf "%-40s %12s %8s %8s\n" "------" "------" "--" "-----"

for i in $(seq 0 $((TOTAL - 1))); do
  read -r TM TN TK CM CN CZ CFG_NAME CL_NAME <<< "${CONFIGS[$i]}"
  echo -n "[$((i+1))/$TOTAL] $CFG_NAME ... "

  BIN="${PROJECT_ROOT}/bench_nvfp4_fp4.m${TM}n${TN}.c${CM}${CN}${CZ}"

  build_log=$(mktemp)
  if $NVCC $COMPILE_FLAGS \
    -DTILE_M=$TM -DTILE_N=$TN -DTILE_K=$TK \
    -DTILE_CLUSTERM=$CM -DTILE_CLUSTERN=$CN -DTILE_CLUSTERZ=$CZ \
    $SRC -o "$BIN" 2>"$build_log"; then

    run_output=$($BIN --m=4096 --n=4096 --k=4096 --iterations=30 2>&1)
    run_exit=$?
    rm -f "$BIN"

    if [ $run_exit -ne 0 ]; then
      echo "RUN_FAIL"
      write_result "$CFG_NAME" "$CL_NAME" "0" "0" "0" "RUN_FAIL"
    else
      gflops=$(echo "$run_output" | grep "GFLOPS" | awk '{print $NF}')
      if [ -z "$gflops" ]; then
        echo "NO_GFLOPS"
        write_result "$CFG_NAME" "$CL_NAME" "0" "0" "0" "RUN_FAIL"
      else
        tflops=$(echo "$gflops" | awk '{printf "%.0f", $1/1000.0}')
        peak_pct=$(echo "$gflops" "$PEAK_TFLOPS" | awk '{printf "%.1f", ($1/1000.0/$2)*100}')
        printf "%-40s %12.0f %7.0fT %7s%%\n" "$CFG_NAME" "$gflops" "$tflops" "$peak_pct"
        write_result "$CFG_NAME" "$CL_NAME" "$gflops" "$tflops" "$peak_pct" "PASS"
      fi
    fi
  else
    echo "BUILD_FAIL"
    head -30 "$build_log" >&2
    write_result "$CFG_NAME" "$CL_NAME" "0" "0" "0" "BUILD_FAIL"
  fi
  rm -f "$build_log"
done

echo ""
echo "Results saved to $RESULTS_FILE"
echo "Search complete"
