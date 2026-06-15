# CUDA Tile Benchmark Makefile
include env.mk

CXXFLAGS   := -std=c++17 -O3 -arch=sm_110a \
              $(CUTLASS_INCLUDES) \
              $(CUTLASS_COMPUTE_FLAGS) \
              -DCUTLASS_COMMIT=\"$(CUTLASS_COMMIT)\"

LDFLAGS    := -lcudart

# Core benchmarks
BENCHMARKS := bench_nvfp4_fp4 bench_nvfp4_fp4_bf16 bench_bf16_cutlass

.PHONY: all clean

all: $(BENCHMARKS)

bench_nvfp4_fp4: bench_nvfp4_fp4.cu helper.h
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

bench_nvfp4_fp4_bf16: bench_nvfp4_fp4_bf16.cu helper.h
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

bench_bf16_cutlass: bench_bf16_cutlass.cu helper.h
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

# Custom tile compilation: make bench_nvfp4_fp4.m128n128 TILES="-DTILE_M=128 -DTILE_N=128"
bench_nvfp4_fp4.m%: bench_nvfp4_fp4.cu helper.h
	$(NVCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(TILES)

clean:
	rm -f $(BENCHMARKS) bench_nvfp4_fp4.* bench_bf16* *.bak

# Note: bench_bf16.cu requires -enable-tile + -std=c++20 which hangs nvcc on CUDA 13.3
# bench_bf16_min is in legacy/ and was pre-compiled

# Override by creating env.local.mk with custom NVCC / CUTLASS_DIR.
