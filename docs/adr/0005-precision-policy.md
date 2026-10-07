# Float64 on the CPU by default; float32 on the GPU only where tested

All computation is float64 on the CPU by default, so results never depend on the backend unless a config asks. Float32 on the Apple GPU (PyTorch MPS, which has no float64) is allowed only for building cell columns and for large matrix products (GᵀG, BBᵀ, batched forward runs), and only once a test against float64 shows the final VIS changes by much less than its posterior standard deviation (e.g. under 1%). Eigendecompositions, Cholesky factorisations and sampler acceptance steps stay float64 on the CPU: the eigenvalues span about fourteen orders of magnitude, and MPS has no float64 linear algebra anyway. A benchmark (columns on CPU vs MPS; BBᵀ with OpenBLAS, Accelerate and MPS float32; the float32 effect on GCM-GLC) comes first, and also decides whether numpy is switched to Apple's Accelerate.

## Considered Options

- GPU wherever it runs: rejected, because results would depend on precision in the ill-conditioned steps.
- CPU only: rejected, because column building and large products are well suited to the GPU and moving them there also keeps load off the shared CPU.
