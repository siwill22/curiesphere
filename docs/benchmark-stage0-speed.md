# Stage 0 benchmark: columns, Gram matrix, precision (2026-10-08)

Stage 0 of the [inversion toolbox spec](inversion-toolbox-spec.md): where the time goes in a global fit (GCM-GLC, 22,591 cells, LCS-1 degrees 16–133, 17,700 coefficients), and whether the Apple GPU and Apple's Accelerate library pay off under the [precision policy](adr/0005-precision-policy.md).

Machine: M2 Max (12 CPU cores, 38-core GPU, 96 GB). CPU runs at `nice -n 15` with 1 or 3 BLAS threads. Scripts: `benchmarks/stage0/`.

## Results

**Cell columns** (all 3.32M continental grid nodes; extrapolated from 202 random cells, 36,699 nodes, including the cells nearest each pole):

| Method | Time, 1 CPU thread | Max relative difference from the reference |
|---|---|---|
| `basis.direct_columns` (reference: a Python loop over nodes) | 22 min | — |
| Vectorised: one matrix product per (cell, grid row), numpy float64 on Accelerate | 4.0 min | 1.9e-14 |
| The same in torch, CPU float64 | 5.2 min | 1.9e-14 |
| The same in torch, GPU (MPS) float32 | 1.5 min | 3.9e-7 |

70% of the GPU time is the per-node preparation on the CPU (radial factors, cosines). Moving it to the GPU would bring all columns to well under a minute.

**Gram matrix BBᵀ** (17,700² from 17,700 × 22,591), eigendecomposition and Cholesky solve:

| | OpenBLAS (pygmt17), 1 thread | Accelerate, 1 thread | Accelerate, 3 threads | GPU float32 |
|---|---|---|---|---|
| BBᵀ | 144 s | 25 s | 14 s | 4.2 s (including copying back) |
| Cholesky solve at one damping | 40 s | 36 s | 27 s | — |
| `eigh` (17,700²) | ~13 min (from the GCM-GLC run) | 16.4 min | 15.3 min | not available |

**Effect of float32 on the answer.** Forming BBᵀ in float32 on the GPU, then solving in float64:
- BBᵀ differs from float64 by 1e-5 relative;
- v0 is unchanged (0.75061);
- VIS changes by a median of 1.4e-6 and a maximum of 1.6e-5 posterior standard deviations.

That is three orders of magnitude inside the ADR 0005 test (1% of a standard deviation). The float64 solve reproduces the stored GCM-GLC result to 6e-7 standard deviations.

## Conclusions

1. **Vectorising the node sums is the main win for columns:** 5× in exact float64 on one thread, with no new dependencies. The GPU adds another 3×, more once the preparation runs there too.
2. **Accelerate speeds matrix products 6× over OpenBLAS** (25 s against 144 s for BBᵀ), and the GPU 34×. Neither speeds up `eigh` or Cholesky. The full eigendecomposition (~15 min, single- or 3-threaded) is now the bottleneck of a global fit. It runs once per data set, band and error spectrum and is cached.
3. **Float32 on the GPU passes the precision test** for both columns and BBᵀ by a wide margin.
4. **PyTorch in pygmt17 is unusable as installed.** The pip wheel brings its own `libomp.dylib`, which clashes with conda's, and `import torch` aborts. The benchmark used a separate virtual environment (numpy 2.5 and scipy 1.18 on Accelerate, torch 2.14).

## Environment (decided)

The toolbox gets a new, clean conda environment, `curiesphere` (`environment.yml`, all from conda-forge):
- numpy 2.5 on Accelerate (`libblas *newaccelerate`);
- pytorch 2.13 sharing conda's OpenMP runtime, so it imports alongside numpy and MPS works;
- pygplates 1.0, pyshtools 4.14, pygmt 0.19.

pygmt17 stays as it is, for the paper notebooks. The test suite passes in both environments; the one numpy-2 incompatibility (`np.trapz`, in `remit/earthvim.py`) now works under both.

The benchmark rerun in the new environment matches the throwaway one within run-to-run variation:

| | Time |
|---|---|
| columns, numpy float64 | 5.5 min |
| columns, GPU float32 | 2.3 min |
| BBᵀ float64, 1 thread | 31 s |
| BBᵀ GPU float32 | 4.6 s |

The float32 effect is identical: at most 1.6e-5 posterior standard deviations.

## Open decision

- **eigh:** whether a faster route is worth it (a Cholesky per damping value with the log-determinant from its diagonal; a partial eigendecomposition), or whether ~15 minutes once per configuration is acceptable.
