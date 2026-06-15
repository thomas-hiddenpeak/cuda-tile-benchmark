# =============================================================================
# Environment configuration — included by Makefile
# Override via environment variables or by creating env.local.mk:
#   NVCC = /custom/path/nvcc
#   CUTLASS_DIR = /custom/path/cutlass
# =============================================================================

NVCC       ?= /usr/local/cuda-13.3/bin/nvcc
CUTLASS_DIR ?= /home/rm01/opencodeWorkspace/cutlass

-include env.local.mk

# Auto-detect CUTLASS commit (overridable via env or env.local.mk)
CUTLASS_COMMIT ?= $(shell cd "$(CUTLASS_DIR)" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")

CUTLASS_INCLUDES  := -I $(CUTLASS_DIR)/include -I $(CUTLASS_DIR)/tools/util/include -I $(CUTLASS_DIR)/examples/common
CUTLASS_COMPUTE_FLAGS := --expt-relaxed-constexpr --expt-extended-lambda
PEAK_TFLOPS := 1032
