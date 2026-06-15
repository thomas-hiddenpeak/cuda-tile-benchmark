#!/usr/bin/env bash
# =============================================================================
# run_isolated.sh — GPU clock/power isolation for reproducible benchmarking
# =============================================================================
# Locks GPU clock to max frequency, sets MAXN power mode, and runs a benchmark
# binary under controlled conditions.
#
# Usage:
#   ./run_isolated.sh ./bench_nvfp4_fp4 --m=4096 --n=4096 --k=4096 --iterations=10
#
# The script will:
#   1. Lock GPU clock to max frequency (devfreq sysfs)
#   2. Set power profile to MAXN (if nvpmodel available)
#   3. Run the benchmark binary with all passed args
#   4. Restore previous GPU governor state on exit (trap)
#
# Author: Sisyphus
# =============================================================================

set -euo pipefail

GPU_SYSFS="/sys/devices/13a00000.gpu/devfreq/13a00000.gpu"
MAX_FREQ=""
ORIG_GOV=""

# --- Detect and set MAXN power mode ---
set_maxn() {
  if command -v nvpmodel &>/dev/null; then
    echo "[isolated] Setting MAXN power profile..."
    nvpmodel -q 2>/dev/null || true
    # MAXN is typically mode 0 on Jetson
    sudo nvpmodel -m 0 2>/dev/null || echo "[isolated] WARN: nvpmodel -m 0 failed"
  else
    echo "[isolated] nvpmodel not found; skipping"
  fi
}

# --- Lock GPU clock to max ---
lock_gpu_clock() {
  if [[ -d "$GPU_SYSFS" ]]; then
    ORIG_GOV="$(cat "$GPU_SYSFS/governor" 2>/dev/null || echo '')"
    MAX_FREQ="$(cat "$GPU_SYSFS/max_freq" 2>/dev/null || echo '')"

    if [[ -n "$MAX_FREQ" && -w "$GPU_SYSFS/governor" ]]; then
      echo "[isolated] Locking GPU clock to max: $(( MAX_FREQ / 1000 )) MHz"
      echo "userspace" | sudo tee "$GPU_SYSFS/governor" >/dev/null 2>&1 || true
      echo "$MAX_FREQ" | sudo tee "$GPU_SYSFS/cur_freq" >/dev/null 2>&1 || true
    else
      echo "[isolated] WARN: Cannot lock GPU clock (sysfs not writable)"
    fi
  else
    echo "[isolated] WARN: GPU sysfs not found at $GPU_SYSFS"
  fi
}

# --- Restore governor ---
restore_gpu_clock() {
  if [[ -n "$ORIG_GOV" && -d "$GPU_SYSFS" && -w "$GPU_SYSFS/governor" ]]; then
    echo "[isolated] Restoring GPU governor to '$ORIG_GOV'"
    echo "$ORIG_GOV" | sudo tee "$GPU_SYSFS/governor" >/dev/null 2>&1 || true
  fi
}

# --- Trap EXIT to always restore ---
trap restore_gpu_clock EXIT

# --- Main ---
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <benchmark_binary> [args...]"
  exit 1
fi

BENCH="$1"
shift

if [[ ! -x "$BENCH" ]]; then
  echo "[isolated] ERROR: '$BENCH' is not an executable"
  exit 1
fi

echo "[isolated] Benchmark: $BENCH $*"

set_maxn
lock_gpu_clock

echo "[isolated] Starting benchmark..."
echo "============================================================"
"$BENCH" "$@"
EXIT_CODE=$?
echo "============================================================"
echo "[isolated] Benchmark exited with code $EXIT_CODE"

exit $EXIT_CODE
