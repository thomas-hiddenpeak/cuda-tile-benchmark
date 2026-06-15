#!/bin/bash
# FP4→FP4 M256 Series + Cluster Matrix
# Safe: no sed -i, trap cleanup, JSON persistence, no set -e

cd "$(dirname "$0")"
source env.sh
SRC="${PROJECT_ROOT}/benchmarks/bench_nvfp4_fp4.cu"
OUTPUT="${PROJECT_ROOT}/bench_nvfp4_fp4"

COMPILE_FLAGS="-std=c++17 -O3 -arch=sm_110a ${PROJECT_INCLUDES} ${CUTLASS_INCLUDES} ${CUTLASS_COMPUTE_FLAGS}"

BINARY_DIR=""

# ── Configs: TM TN TK CM CN CZ name ──
CONFIGS=(
  "256 256 256 2 2 1 M256xN256 C2x2x1"
  "256 256 256 2 1 1 M256xN256 C2x1x1"
  "256 128 256 2 2 1 M256xN128 C2x2x1"
  "256 128 256 2 1 1 M256xN128 C2x1x1"
  "256  64 256 2 1 1 M256xN64  C2x1x1"
  "256  64 256 2 2 1 M256xN64  C2x2x1"
)
TOTAL=${#CONFIGS[@]}

# ── Setup results ──
RESULTS_DIR="${PROJECT_ROOT}/results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="${RESULTS_DIR}/results_$(date +%Y%m%d_%H%M%S).jsonl"

cleanup() {
  if [ -n "$BINARY_DIR" ] && [ -d "$BINARY_DIR" ]; then
    rm -rf "$BINARY_DIR"
  fi
}
trap 'cleanup' EXIT INT TERM

# ── Helpers ──
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

# ── Print header ──
echo "=== M256 Series + Cluster Matrix ==="
printf "%-35s %10s %8s %8s\n" "Config" "GFLOPS" "TF" "Peak%"
printf "%-35s %10s %8s %8s\n" "------" "------" "--" "-----"

for i in $(seq 0 $((TOTAL - 1))); do
  read -r TM TN TK CM CN CZ CFG_NAME CL_NAME <<< "${CONFIGS[$i]}"
  label="$CFG_NAME $CL_NAME"
  echo -n "[$((i+1))/$TOTAL] $label ... "

  # Unique binary name per config
  BIN="$OUTPUT.m${TM}n${TN}k${TK}.c${CM}${CN}${CZ}"

  # Build
  build_log=$(mktemp)
  if $NVCC $COMPILE_FLAGS \
    -DTILE_M=$TM -DTILE_N=$TN -DTILE_K=$TK \
    -DTILE_CLUSTERM=$CM -DTILE_CLUSTERN=$CN -DTILE_CLUSTERZ=$CZ \
    $SRC -o "$BIN" 2>"$build_log"; then

    # Run benchmark
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
        printf "%-35s %10.0f %7.0fT %7s%%\n" "$label" "$gflops" "$tflops" "$peak_pct"
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
echo "M256 search complete"
