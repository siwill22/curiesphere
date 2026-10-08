# Inversion toolbox: specification (agreed 2026-10-07)

The next version of curiesphere gathers every method that estimates crustal VIS from satellite field models into one installable toolbox, `remit.inversion`. It replaces the scripts in `notebooks/` (the `domains/` framework, the frozen legacy scripts and `run_global_continents.py`) with one engine, one data path and one result type. The focus is continental crust and induced magnetisation; ocean remanence is deferred but designed for.

Terms are defined in [CONTEXT.md](../CONTEXT.md). Decisions that are hard to reverse are in [docs/adr/](adr/).

## Decisions

| | Decision | Record |
|---|---|---|
| Estimated quantity | VIS (SI·km) for induced sources; VIM only derived. *Source strength* is the general term, so remanent sources fit in later. | CONTEXT.md |
| Parameter unit | Cells (HEALPix, NSIDE 64 by default). Every parameterisation is a grouping of cells, G_param = G_cells S; no polygon-exact domains. | [ADR 0001](adr/0001-cells-are-the-smallest-parameter-unit.md) |
| Source geometry | A list of layers, each a (top, base) pair of constant depths or grids. This version runs with one layer, default top of basement to 30 km below the ellipsoid. Geometries are compared, never estimated. | CONTEXT.md |
| Fit types | *Local fit*: Br at data points within a buffer of the target area, fixed background (H&M + slabs allowed outside the target area, never inside). *Global fit*: Gauss coefficients, outer cells on all other continental crust, only the ocean model fixed. | [ADR 0002](adr/0002-local-fit-on-points-global-fit-on-coefficients.md) |
| Error model | Noise + model error, each a per-degree spectrum with an amplitude. Each amplitude is fixed (from field-model differences or a reference fit) or fitted with the model by the evidence; every result records which. Diagonal for coefficients, dense for points. | CONTEXT.md |
| Methods | All current methods, rebuilt on one linear-Gaussian core, in stages (below). | this spec |
| Location | `remit.inversion` in the installable package; the forward depth code moves into `remit`; `notebooks/depth_models.py` becomes a re-export so the paper notebooks run unchanged. | [ADR 0003](adr/0003-inversion-toolbox-in-the-package.md) |
| Running | One TOML file fully specifies an inversion, with inheritance for variants (no environment variables). One entry point, `python -m remit.inversion run config.toml`; the same steps as a Python API. Results carry their resolved config. | this spec |
| Caching | Content-addressed, shared, outside the repo (default `~/.cache/curiesphere`). Cell columns computed once for all continental crust per NSIDE, layer and direction model. | [ADR 0004](adr/0004-content-addressed-cache-global-columns.md) |
| Precision and GPU | Float64 on the CPU by default. Float32 on the GPU (PyTorch MPS) only for column building and large matrix products, each after a test against float64. Decompositions and sampler acceptance stay float64 on the CPU. | [ADR 0005](adr/0005-precision-policy.md) |
| Outputs | One result type (NetCDF), a 0.1° grid export, a standard figure set (VIS with domains, uncertainty, data Br, model Br, residual; plain-English captions). | this spec |
| Review | The browser viewer is the main review tool, fed only by standard results through one exporter, for any target area. Its detailed design is a later topic. | this spec |
| Testing | Synthetic data is a data option (truth library, noise from the error model); results from synthetic data store the truth. Three tiers: unit tests, synthetic recovery per method, acceptance on real data. | this spec |
| Modes | `mode = explore` (fast defaults) or `final` (robust defaults), each default overridable; results record and display their mode. Each method has a mode-agreement test in its acceptance. | [ADR 0006](adr/0006-explore-and-final-modes.md) |
| Old code | Tag `pre-toolbox` first; delete the code a stage replaces once that stage passes. Key result files (numbers, CSVs, figures) kept, caches not. | this spec |

## Package layout (proposed)

```
remit/
  depth.py              source depth, layers, slab columns by transform, LCS-1 filter   (from notebooks/depth_models.py)
  inversion/
    __main__.py         python -m remit.inversion run config.toml
    config.py           load, inheritance, validation, resolved config, content hash
    cache.py            content-addressed store
    geometry.py         grid nodes, cells, layers, direction models, target areas and outlines
    columns.py          cell columns by node sum (vectorised; CPU or MPS) and by transform (reference)
    data.py             field models, filter, band, background, local (points) and global (coefficients) data, synthetic data
    truths.py           truth library for synthetic data
    errors.py           noise and model-error spectra, amplitude policies, whitening (diagonal or dense)
    parameterise.py     S builders: cells, coarse cells, domains (polygons or labels), covariate designs, uniform level
    core.py             linear-Gaussian core: MAP, posterior, evidence, discrepancy; by eigendecomposition or Cholesky; bounds
    methods/            gcm, cellmap (GMRF/L2/TV), polygon_domains, regression, bounds, merge, voronoi, lvoronoi, partition
    results.py          result type, NetCDF read/write, grid export
    figures.py          standard figure set (PyGMT)
    viewer.py           exporter for the browser viewer (design later)
```

What exists now and where it goes:

| Now | Goes to |
|---|---|
| `notebooks/depth_models.py` (forward parts) | `remit/depth.py` |
| `domains/basis.py`, `cell_basis.py`, columns in `run_global_continents.py` | `geometry.py`, `columns.py`, `data.py` |
| `domains/errors.py`, `error_model.py`, the whitening in `problem.load` and `linear.whiten`, the diagonal weights in `run_global_continents.py` | `errors.py` |
| `domains/linear.py`, `domains/gmrf.py` (Tikhonov, discrepancy, evidence), `run_global_continents.py` (eigen solve, evidence) | `core.py`, `methods/cellmap.py`, `methods/gcm.py` |
| `domains/design.py`, `domains/covariates.py` | `parameterise.py`, `methods/regression.py` |
| `domains/merge.py`, `voronoi.py`, `lvoronoi.py`, `partition.py`, `graph.py`, `outputs.py` | `methods/`, `results.py` |
| `domains/plotting.py`, the region plot scripts, `plot_global_continents.py` | `figures.py` |
| `export_viewer_inversion.py`, `viewer_info.py` | `viewer.py` |
| `build_region.py`, `build_bands.py`, `problem.py`, the `run_*.py` scripts | config files + `__main__.py` |

## Configuration (sketch)

```toml
inherit = "base_australia.toml"        # variants override only what differs

[target]
area = "outlines/australia.gpkg"       # or "continental_crust" for GLC
nside = 64

[geometry]
layers = [{ top = "basement", base = 30.0 }]   # km below the ellipsoid, or a grid path
direction = "induced_igrf13"

[fit]
type = "local"                         # "local" (points + fixed background) or "global" (coefficients + outer cells)
background = "hm05_slabs+gk07_nr"      # local fits only
buffer_km = 300

[data]
model = "lcs1"                         # or "mf7", "2e", or "synthetic"
band = [16, 133]
filter = "lcs1"
# truth = { kind = "provinces_hm_stats", seed = 1 }   # synthetic only

[errors]
noise = { spectrum = "lcs1-2e", amplitude = "fixed" }
model_error = { spectrum = "lcs1", amplitude = "evidence" }

[parameterisation]
kind = "domains"                       # cells, coarse_cells, domains, covariates, uniform
polygons = "hasterok2022"

[method]
name = "pd"
bounds = [0.0, 4.0]
```

## Stages and acceptance

Each stage passes when it reproduces the listed current results under the same settings (data, band, error model). It also passes the mode-agreement test for its methods: explore and final runs of the same configs, with the differences reported in final posterior standard deviations (and partition agreement for domain methods). The results are then reviewed in the viewer next to the old ones, and the replaced code is deleted.

| Stage | Contents | Reproduces |
|---|---|---|
| 0 (done, [report](benchmark-stage0-speed.md)) | Benchmark: vectorised node-sum columns (CPU vs MPS), BBᵀ (OpenBLAS, Accelerate, MPS float32), float32 effect on GCM-GLC. Decides GPU use and whether numpy moves to Accelerate. | — |
| 1 | `depth`, config, cache, geometry, columns, data (local and global, synthetic), errors, core, results, figures; methods GCM and CM. Tag `pre-toolbox` before starting. | GCM-GLC lcs1_16-133: v0 = 0.751 ± 0.011 SI·km, noise × 1.53, prior s.d. 0.51 (evidence); v0 = 0.743 (discrepancy). Synthetic (one seeded noise draw, evidence): uniform truths 1.0 and 1.5 recovered as v0 = 1.000 and 1.500; uniform 1.0 with an ocean-model error (true HM05_NR, GK07_NR subtracted) gives v0 = 0.963; 0.04 × Crust1.0 truth (mean 1.352) gives v0 = 0.879, RMS error 0.61 SI·km. CM-AUS: `regions/australia/cache/map_sweep.npz`. Columns: the existing Australian cell columns to rounding. |
| 2 | PD, CR, bounds. | PD-AUS: `notebooks/inversion/real_LCS1.csv` (39 domain values). CR: `linear_models.npz`, `null_regression.npz`. Bounds: `bounds.npz`. |
| 3 | FD-merge. | `merge_points.npz`, `merge_points_sweep.npz` (FD-merge-AUS). |
| 4 | FD-vor, FD-lvor, FD-part. | The `*_synthetic.npz` and `partition_diagnostics.npz` results in `regions/australia/cache/`. |

## Provisions for later work

Built in now, because they would be expensive to retrofit:

- Cell columns are keyed by (cell, layer, direction model), so remanent directions and extra layers are new cache entries, not new code paths.
- Each parameter carries its own bounds; the core makes no positivity assumption (a signed RVIM amplitude would be unbounded).
- A parameterisation may group cells by any label, not only spatially (e.g. ocean age bins per basin).

Deferred: ocean remanence (cells, signed amplitudes, polarity), multiple VIS layers, the viewer's detailed design.

## Spherical geometry

All of the toolbox works on the sphere: spherical harmonics, equal-area HEALPix cells, great-circle buffers, source radii on the WGS84 ellipsoid (grid latitudes treated as geocentric, as now). No step assumes a flat Earth. Inputs and outputs are lon/lat on WGS84.
