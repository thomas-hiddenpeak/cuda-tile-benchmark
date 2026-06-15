# =============================================================================
# Environment configuration — source this from shell scripts
# Modify paths below for your setup; commit the template, not your local copy.
# =============================================================================
# Usage in a script (scripts/ subdirectory):
#   source "$(dirname "$0")/env.sh"

export NVCC="${NVCC:-/usr/local/cuda-13.3/bin/nvcc}"
export CUTLASS_DIR="${CUTLASS_DIR:-/home/rm01/opencodeWorkspace/cutlass}"

# Auto-detect CUTLASS commit
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-$(cd "$CUTLASS_DIR" && git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

# Project root (parent of scripts/)
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PROJECT_INCLUDES="-I ${PROJECT_ROOT}/include"

# Derived paths
export CUTLASS_INCLUDES="-I ${CUTLASS_DIR}/include -I ${CUTLASS_DIR}/tools/util/include -I ${CUTLASS_DIR}/examples/common"
export CUTLASS_COMMIT_FLAG="-DCUTLASS_COMMIT=\"${CUTLASS_COMMIT}\""
export CUTLASS_COMPUTE_FLAGS="${CUTLASS_COMPUTE_FLAGS:---expt-relaxed-constexpr --expt-extended-lambda} ${CUTLASS_COMMIT_FLAG}"
export PEAK_TFLOPS=1032
