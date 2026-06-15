#!/bin/bash
# SF Vector Size Search - test SF configs with best tile shape
# Safe: no sed -i, trap cleanup, JSON persistence, no set -e
# Usage: ./run_sf_search.sh [M,N,K] [CM,CN,CZ]
#        Default: 128,128,256 tile + 2,2,1 cluster

cd "$(dirname "$0")"
source env.sh
SRC="${PROJECT_ROOT}/benchmarks/bench_nvfp4_fp4.cu"

COMPILE_FLAGS="-std=c++17 -O3 -arch=sm_110a ${PROJECT_INCLUDES} ${CUTLASS_INCLUDES} ${CUTLASS_COMPUTE_FLAGS}"

# Read best tile and cluster from args
BEST_TILE="${1:-128,128,256}"
BEST_CLUSTER="${2:-2,2,1}"

IFS=',' read -r DEF_TM DEF_TN DEF_TK <<< "$BEST_TILE"
IFS=',' read -r DEF_CM DEF_CN DEF_CZ <<< "$BEST_CLUSTER"

# SF configs: InputSFVectorSize OutputSFVectorSize
SF_CONFIGS=(
  "4 4"
  "8 8"
  "16 16"
  "32 32"
  "64 64"
  "8 16"
  "16 32"
  "8 32"
)
TOTAL=${#SF_CONFIGS[@]}

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

TILE_LABEL="M${DEF_TM}xN${DEF_TN}xK${DEF_TK}"
CL_LABEL="C${DEF_CM}x${DEF_CN}x${DEF_CZ}"

echo "=== SF Vector Size Search (4096³) ==="
echo "Base: $TILE_LABEL, Cluster: $CL_LABEL"
printf "%-35s %12s %8s %8s\n" "Config" "GFLOPS" "TF" "Peak%"
printf "%-35s %12s %8s %8s\n" "------" "------" "--" "-----"

for i in $(seq 0 $((TOTAL - 1))); do
  read -r INSF OUTSF <<< "${SF_CONFIGS[$i]}"
  label="SFVec[${INSF},${OUTSF}]"
  full_label="$TILE_LABEL $label"
  echo -n "[$((i+1))/$TOTAL] $label ... "

  BIN="${PROJECT_ROOT}/bench_nvfp4_fp4.sf${INSF}o${OUTSF}"

  build_log=$(mktemp)
  if $NVCC $COMPILE_FLAGS \
    -DTILE_M=$DEF_TM -DTILE_N=$DEF_TN -DTILE_K=$DEF_TK \
    -DTILE_CLUSTERM=$DEF_CM -DTILE_CLUSTERN=$DEF_CN -DTILE_CLUSTERZ=$DEF_CZ \
    -DTILE_INPUTSF=$INSF -DTILE_OUTPUTSF=$OUTSF \
    $SRC -o "$BIN" 2>"$build_log"; then

    run_output=$($BIN --m=4096 --n=4096 --k=4096 --iterations=30 2>&1)
    run_exit=$?
    rm -f "$BIN"

    if [ $run_exit -ne 0 ]; then
      echo "RUN_FAIL"
      write_result "$full_label" "$CL_LABEL" "0" "0" "0" "RUN_FAIL"
    else
      gflops=$(echo "$run_output" | grep "GFLOPS" | awk '{print $NF}')
      if [ -z "$gflops" ]; then
        echo "NO_GFLOPS"
        write_result "$full_label" "$CL_LABEL" "0" "0" "0" "RUN_FAIL"
      else
        tflops=$(echo "$gflops" | awk '{printf "%.0f", $1/1000.0}')
        peak_pct=$(echo "$gflops" "$PEAK_TFLOPS" | awk '{printf "%.1f", ($1/1000.0/$2)*100}')
        printf "%-35s %12.0f %7.0fT %7s%%\n" "$full_label" "$gflops" "$tflops" "$peak_pct"
        write_result "$full_label" "$CL_LABEL" "$gflops" "$tflops" "$peak_pct" "PASS"
      fi
    fi
  else
    echo "BUILD_FAIL"
    head -30 "$build_log" >&2
    write_result "$full_label" "$CL_LABEL" "0" "0" "0" "BUILD_FAIL"
  fi
  rm -f "$build_log"
done

echo ""
echo "Results saved to $RESULTS_FILE"
echo "SF Search Complete"
