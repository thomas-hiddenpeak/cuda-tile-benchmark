# CUDA Tile Benchmark — FP4 & BF16 GEMM on Blackwell (SM110)

[![CI](https://github.com/thomas-hiddenpeak/cuda-tile-benchmark/actions/workflows/benchmark-ci.yml/badge.svg)](https://github.com/thomas-hiddenpeak/cuda-tile-benchmark/actions/workflows/benchmark-ci.yml)

NVFP4 and BF16 Block-Scaled GEMM tuning on NVIDIA Jetson AGX Thor (sm_110a) using CUTLASS. Searches over tile shape, cluster layout, SF vector size, and problem size.

## Hardware

| Parameter | Value |
|---|---|
| GPU | NVIDIA Thor (sm_110a) |
| SM count | 20 |
| Clock | 1,575 MHz (MAXN) |

## Measured Performance

| Type | Config | TFLOPS |
|---|---|---|
| **FP4→FP4** | M256×N128×K256 + C2×2×1 | **579** |
| BF16→BF16 (CUTLASS)`*` | M256×N128×K64 + C2×2×1 | 491 |
| BF16→BF16 (cuda_tile.h, ⚠️ old) | C2×2×1 | 458 |
| FP4→BF16 | SFD bottleneck | 464 |

`*` CUTLASS CollectiveBuilder (`OpClassTensorOp`, `bfloat16_t`) — replaces the unbuildable `bench_bf16.cu`. TFLOPS pending on-target verification.

**Summary**: FP4→FP4 is limited by SFD (Scale Factor Decompression) overhead, roughly 32% more than the equivalent BF16→BF16 path. The BF16 baseline has moved to CUTLASS native BF16 tensor core (`bench_bf16_cutlass.cu`), pending on-target validation. Further tuning within CUTLASS yields diminishing returns; improving beyond the current 579 TFLOPS would likely require handwritten PTX or architecture-level changes.

## Quick Start

### Prerequisites

- CUDA 13.3 (CUDA Toolkit 13.3)
- CUTLASS (master branch, with `sm_110a` support)

### Environment Setup

Edit `env.mk` (for Makefile) or `env.sh` (for shell scripts) to set local paths:

```bash
# Option 1: edit env.mk directly
NVCC       ?= /usr/local/cuda-13.3/bin/nvcc
CUTLASS_DIR ?= /path/to/cutlass

# Option 2 (recommended): create env.local.mk, which won't pollute git
$ cp env.mk env.local.mk
$ vi env.local.mk   # modify NVCC / CUTLASS_DIR
```

### Building

```bash
cd cuda_tile_benchmark

# Build all benchmarks
make

# Build a single binary
make bench_nvfp4_fp4

# Custom tile/cluster via Makefile pattern rule
make bench_nvfp4_fp4.m128n128 TILES="-DTILE_M=128 -DTILE_N=128 -DTILE_K=128"

# Or use a shell script (auto-sources env.sh)
./run_tile_test.sh 256 128 256 2 1
```

> **Note**: `-arch=sm_110a` requires the `a` suffix (enables TMA, tcgen05.mma.blockscaled, TMEM). `--expt-relaxed-constexpr` is also required. See [NVFP4_BREAKTHROUGH.md](NVFP4_BREAKTHROUGH.md).

### Running

```bash
./bench_nvfp4_fp4 --m=4096 --n=4096 --k=4096 --iterations=5
```

Example output:
```
GPU: NVIDIA Thor | SMs: 20 | Clock: 2601 MHz | Peak: 1032 TF (dense @ 1575 MHz)
  Disposition: Passed
  Problem Size: 4096x4096x4096
  Avg runtime: 0.241 ms (min: 0.238, max: 0.246, stddev: 0.002)
  GFLOPS: 569805.7
  TFLOPS: 569.806
```

JSON output:
```bash
./bench_nvfp4_fp4 --m=4096 --n=4096 --k=4096 --json
```

### CLI Options

| Option | Default | Description |
|---|---|---|
| `--m=N` | 1024 | M dimension |
| `--n=N` | 1024 | N dimension |
| `--k=N` | 1024 | K dimension |
| `--iterations=N` | 50 | Performance measurement iterations |
| `--warmup=N` | 5 | Warmup iterations (not timed) |
| `--seed=N` | 42 | Random seed |
| `--json` | off | Output structured JSON |

## Project Structure

### Core Benchmarks

| File | Type | Description |
|---|---|---|
| `bench_nvfp4_fp4.cu` | **FP4→FP4** | Primary FP4 benchmark, CUTLASS CollectiveBuilder implementation |
| `bench_nvfp4_fp4_bf16.cu` | FP4→BF16 | Mixed-precision SFD bottleneck analysis |
| `bench_bf16_cutlass.cu` | **BF16→BF16** | `OpClassTensorOp` + `bfloat16_t`, CUTLASS native tensor core baseline |
| `bench_bf16.cu` | BF16→BF16 | Based on cuda_tile.h (⚠️ not buildable with CUDA 13.3, requires `-enable-tile`) |
| `legacy/bench_bf16_min.cu` | BF16→BF16 | Runnable simplified version (single tile config) |
| `legacy/bench_nvfp4_cutlass.cu` | FP4→BF16 | CUTLASS 72a port (early experiments) |
| `legacy/bench_nvfp4_ptx.cu` | Handwritten PTX | PTX-level NVFP4 experiments |
| `legacy/bench_nvfp4_ultra.cu` | Experimental | Aggressive optimization attempts |

### Search Scripts

Configurations are switched via compiler `-D` flags (no source modification). **Each config takes ~1-2 minutes to compile** (CUTLASS template instantiation overhead).

| Script | Purpose | Configs |
|---|---|---|
| `build_nvfp4_cutlass.sh` | CUTLASS 72a port build | 1 |
| `run_fp4_m256.sh` | M256 tile series + cluster | 6 |
| `run_fp4_search.sh` | M×N × SF Vector full search | 36 |
| `run_fp4_asymmetric.sh` | Asymmetric tiles (M128/M256) | 32 |
| `run_sf_search.sh` | SF vector size | 8 |
| `run_tile_search.sh` | Tile shape × cluster | 40 |
| `run_tile_test.sh` | Single config test | 1 |

Results are saved to `results/results_YYYYMMDD_HHMMSS.jsonl`.

### Supporting Files

| File | Description |
|---|---|
| `env.mk` | Makefile environment (NVCC, CUTLASS_DIR, PEAK_TFLOPS), supports `env.local.mk` overrides |
| `env.sh` | Shell environment (same variables, sourced by search scripts) |
| `run_isolated.sh` | GPU clock locking / MAXN power mode for reproducible benchmarks |
| `helper.h` | GpuTimer, CUDA/CUTLASS CHECK macros |
| `FP4_OPTIMIZATION_SPEC.md` | Optimization history and search records |
| `NVFP4_BREAKTHROUGH.md` | Issues encountered on sm_110a |
| `METHODOLOGY.md` | Measurement methodology: timing approach, outlier handling, statistics, known limitations |
| `analyze_results.py` | Result aggregation: top-N extraction, grouped stats, LaTeX table generation, CSV export, cross-implementation comparison |
| `plot_results.py` | Visualization: bar charts, scaling line plots, tile×cluster heatmaps, grouped comparison charts |
| `benchmark_suite.py` | Multi-run harness: N-run aggregation (grand mean ± 95% CI), JSON report output |

## Search Results

### FP4→FP4 Tile Search

| M×N | K | Cluster | TF | Status |
|---|---|---|---|---|
| 128×128 | 256 | C2×2×1 | 492 | ✅ |
| 128×192 | 256 | C2×1×1 | 559 | ✅ |
| 128×256 | 256 | C1×1×1 | — | ❌ smem |
| 256×128 | 256 | C2×2×1 | **579** | ✅ |
| 256×192 | 256 | C2×1×1 | 570 | ✅ |
| 256×192 | 256 | C2×2×1 | 527 | ✅ |
| 256×256 | 256 | C2×1×1 / C2×2×1 | — | ❌ smem |
| 256×64 | 256 | C2×1×1 / C2×2×1 | — | ❌ hang |

### FP4→FP4 Cluster Search (M128×128×K256)

| Cluster | TF |
|---|---|
| C2×2×1 | 492 ✅ |
| C4×1×1 / C2×1×1 / C1×2×1 / C1×4×1 | ~287 |
| C1×1×1 | 231 |
| C4×2×1 / C2×4×1 / C4×4×1 / C2×2×2 | ❌ unavailable |

### BF16→BF16 Cluster Search (C2×2×1)

| TF |
|---|
| 454 |
| 458 |

### Problem Size Scaling (M256×N128, C2×2×1)

| M=N=K | TF |
|---|---|
| 1024 | 60 |
| 2048 | 260 |
| 4096 | 579 |

Full search history in [FP4_OPTIMIZATION_SPEC.md](FP4_OPTIMIZATION_SPEC.md).

## Directory Layout

```
├── env.mk                      # Makefile environment (NVCC/CUTLASS_DIR)
├── env.sh                      # Shell environment (sourced by search scripts)
├── run_isolated.sh             # GPU clock locking / isolation tool
├── bench_nvfp4_fp4.cu          # FP4→FP4 primary benchmark
├── bench_nvfp4_fp4_bf16.cu     # FP4→BF16 mixed precision
├── bench_bf16_cutlass.cu       # BF16→BF16 CUTLASS native tensor core baseline
├── bench_bf16.cu               # BF16→BF16 based on cuda_tile.h (old baseline, not buildable)
├── helper.h                    # GpuTimer, CUDA/CUTLASS CHECK macros
├── build_nvfp4_cutlass.sh      # Build script (CUTLASS 72a port)
├── run_fp4_*.sh                # Search scripts (compiler -D flags)
├── run_sf_search.sh
├── run_tile_*.sh
├── FP4_OPTIMIZATION_SPEC.md    # Optimization spec and search records
├── NVFP4_BREAKTHROUGH.md       # Issues encountered on sm_110a
├── tile_search_results.md
├── analyze_results.py          # Result aggregation and analysis
├── plot_results.py             # Visualization tool
├── benchmark_suite.py          # Multi-run aggregation harness
├── Makefile                    # Build entry point
├── README.md
├── METHODOLOGY.md              # Measurement methodology documentation
├── .gitignore
├── .editorconfig               # Editor format settings
├── .github/workflows/          # CI pipeline config
├── legacy/                     # Historical / experimental files
├── probes/                     # Hardware probe tools
├── tests/                      # Test files
└── results/                    # Search results (JSONL) + README
```

## Key Observations

1. **SFD is the main bottleneck**: Scale Factor Decompression overhead is the primary factor limiting FP4→FP4 performance vs the BF16 CUTLASS baseline.
2. **BF16 baseline migrated**: `bench_bf16.cu` (cuda_tile.h, not buildable) remains for reference; `bench_bf16_cutlass.cu` (CUTLASS `OpClassTensorOp` + `bfloat16_t`) is the new baseline.
3. **C2×2×1 is the most effective cluster**: Confirmed for both FP4 and BF16. C4×2×1, C2×4×1, C4×4×1 don't compile.
4. **M256 tiles help modestly**: Going from M128→M128 to M256→N128 gives about 18% (492→579 TF). M256×N256 hits shared memory limits.
5. **CUTLASS tuning headroom is limited**: At ~579 TFLOPS the framework is near its ceiling for this architecture. Further gains would require handwritten PTX or architecture-level optimization.

## Next Steps

Completed:
- [x] Centralized path config: `env.mk` / `env.sh` with `env.local.mk` override support
- [x] Result tracking: `results/*.jsonl` ignored, metadata tracked
- [x] Search scripts use compiler `-D` flags (no source modification)
- [x] Benchmark outputs structured stats (per-iteration, min/max/stddev/median)
- [x] JSON output format + automatic result archiving
- [x] BF16 baseline migrated to CUTLASS native tensor core (`bench_bf16_cutlass.cu`)
- [x] Directory reorganization (`legacy/`, `probes/`, `tests/`, `results/`)
- [x] Build infrastructure: Makefile + .editorconfig + .gitignore
- [x] Reproducible benchmark tooling: `run_isolated.sh` (GPU clock locking)
- [x] Statistical rigor: 95% CI, CV, SEM embedded in benchmark output
- [x] CUTLASS commit + CUDA version pinned at compile time (`-DCUTLASS_COMMIT`)
- [x] Measurement methodology documentation: `METHODOLOGY.md`
- [x] Result analysis pipeline: `analyze_results.py` (top-N / grouped / LaTeX / CSV / comparison)
- [x] Visualization: `plot_results.py` (bar, line, heatmap, comparison charts)
- [x] Multi-run aggregation: `benchmark_suite.py` (N runs + grand mean ± 95% CI)
- [x] CI pipeline: `.github/workflows/benchmark-ci.yml` (lint + Python check + compile check)
- [x] Editor config: `.editorconfig` (2-space indent, LF line endings)

Pending:
- [ ] Handwritten PTX kernel to bypass CUTLASS framework overhead
- [ ] Explore M512 / N512 tile sizes
- [ ] SFD pipeline / asynchronous decompression

## Pitfalls

See [NVFP4_BREAKTHROUGH.md](NVFP4_BREAKTHROUGH.md) — 3 wasted directions with documented solutions.

## License

BSD-3-Clause (CUTLASS sample code)
