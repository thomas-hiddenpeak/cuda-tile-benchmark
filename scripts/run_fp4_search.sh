#!/bin/bash
# Systematic FP4→FP4 tile search - M×N expansion with SF vector sizes
# Safe: no sed -i, trap cleanup, JSON persistence, no set -e

cd "$(dirname "$0")"
source env.sh
SRC="${PROJECT_ROOT}/benchmarks/bench_nvfp4_fp4.cu"

COMPILE_FLAGS="-std=c++17 -O3 -arch=sm_110a ${PROJECT_INCLUDES} ${CUTLASS_INCLUDES} ${CUTLASS_COMPUTE_FLAGS}"

# M×N combinations (K=256, Cluster=2x2x1 fixed)
declare -a MNS=("64 64" "64 128" "64 256" "128 64" "128 128" "128 256" "256 64" "256 128" "256 256")
declare -a SFVS=("4" "8" "16" "32")

TOTAL=$(( ${#MNS[@]} * ${#SFVS[@]} ))

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

echo "=== FP4→FP4 M×N Tile Matrix ==="
echo "K=256 fixed, Cluster=2x2x1 fixed, Total=$TOTAL configs"
printf "%-35s %12s %8s %8s\n" "Config" "GFLOPS" "TF" "Peak%"
printf "%-35s %12s %8s %8s\n" "------" "------" "--" "-----"

count=0
for MN in "${MNS[@]}"; do
  read -r TM TN <<< "$MN"
  for SF in "${SFVS[@]}"; do
    count=$((count + 1))
    cfg_name="M${TM}xN${TN}xK256 SFVec=${SF}"
    echo -n "[$count/$TOTAL] $cfg_name ... "

    BIN="${PROJECT_ROOT}/bench_nvfp4_fp4.m${TM}n${TN}.sf${SF}"

    build_log=$(mktemp)
    if $NVCC $COMPILE_FLAGS \
      -DTILE_M=$TM -DTILE_N=$TN -DTILE_K=256 \
      -DTILE_CLUSTERM=2 -DTILE_CLUSTERN=2 -DTILE_CLUSTERZ=1 \
      -DTILE_INPUTSF=$SF -DTILE_OUTPUTSF=$SF \
      $SRC -o "$BIN" 2>"$build_log"; then

      run_output=$($BIN --m=4096 --n=4096 --k=4096 --iterations=30 2>&1)
      run_exit=$?
      rm -f "$BIN"

      if [ $run_exit -ne 0 ]; then
        echo "RUN_FAIL"
        write_result "$cfg_name" "C2x2x1" "0" "0" "0" "RUN_FAIL"
      else
        gflops=$(echo "$run_output" | grep "GFLOPS" | awk '{print $NF}')
        if [ -z "$gflops" ]; then
          echo "NO_GFLOPS"
          write_result "$cfg_name" "C2x2x1" "0" "0" "0" "RUN_FAIL"
        else
          tflops=$(echo "$gflops" | awk '{printf "%.0f", $1/1000.0}')
          peak_pct=$(echo "$gflops" "$PEAK_TFLOPS" | awk '{printf "%.1f", ($1/1000.0/$2)*100}')
          printf "%-35s %12.0f %7.0fT %7s%%\n" "$cfg_name" "$gflops" "$tflops" "$peak_pct"
          write_result "$cfg_name" "C2x2x1" "$gflops" "$tflops" "$peak_pct" "PASS"
        fi
      fi
    else
      echo "BUILD_FAIL"
      head -30 "$build_log" >&2
      write_result "$cfg_name" "C2x2x1" "0" "0" "0" "BUILD_FAIL"
    fi
    rm -f "$build_log"
  done
done

echo ""
echo "Results saved to $RESULTS_FILE"
echo "Search complete"
