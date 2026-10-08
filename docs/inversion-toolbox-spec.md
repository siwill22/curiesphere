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
| Old code | Tag `pre-toolbox` first; delete the code a stage replaces once that stage passes. Key result files (numbers, CSVs, figures) kept, caches not. Done after stage 4: all old code, including work never committed, is on branch `legacy-snapshot` (tag `pre-toolbox-final`); the samplers, GCM-GLC and the first-generation domain scripts are deleted. Still kept, because work not yet moved to the toolbox uses them: the future-data test (`build_bands`, `compare_bands`, `compare_future`, `make_multiscale_truth`, which run `run_linear_models`, `run_realistic_synthetic` and `run_merge`) and the viewer exporter (`export_viewer_inversion`, after `run_bounds`, `run_map_sweep`, `run_null_regression`, `run_covariates`), with the `notebooks/domains/` modules they import. The acceptance scripts and dated show-me reports run against the tag. | this spec |

## Package layout (proposed)

```
remit/
  depth.py              source depth, layers, slab columns by transform, LCS-1 filter   (from notebooks/depth_models.py)
  inversion/
    __main__.py         python -m remit.inversion run config.toml
    config.py           load, inheritance, validation, resolved config, content hash
    cache.py            content-addressed store
    geometry.py         grid nodes, cells, layers, direction models, target areas and outlines
    graph.py            the cell graph (edges, bridges), Laplacian and difference operator on the sphere
    columns.py          cell columns by node sum (vectorised; CPU or MPS) and by transform (reference)
    data.py             field models, filter, band, background, local (points) and global (coefficients) data, synthetic data
    truths.py           truth library for synthetic data
    errors.py           noise and model-error spectra, amplitude policies, whitening (diagonal or dense)
    problem.py          the problem of a local fit: cells, G at the points, data, compressed and whitened error model
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
| `domains/graph.py`, `outputs.cell_edges`, `region.py` | `graph.py`, `geometry.py` |
| `domains/merge.py`, `voronoi.py`, `lvoronoi.py`, `partition.py`, the rest of `outputs.py` | `methods/`, `results.py` |
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
| 1 (done) | `depth`, config, cache, geometry, columns, data (local and global, synthetic), errors, core, results, figures; methods GCM and CM. Tag `pre-toolbox` before starting. | GCM-GLC lcs1_16-133: v0 = 0.751 ± 0.011 SI·km, noise × 1.53, prior s.d. 0.51 (evidence); v0 = 0.743 (discrepancy). Synthetic (one seeded noise draw, evidence): uniform truths 1.0 and 1.5 recovered as v0 = 1.000 and 1.500; uniform 1.0 with an ocean-model error (true HM05_NR, GK07_NR subtracted) gives v0 = 0.963; 0.04 × Crust1.0 truth (mean 1.352) gives v0 = 0.879, RMS error 0.61 SI·km. CM-AUS: `regions/australia/cache/map_sweep.npz` and the GMRF fit in `linear_models.npz`. Result: problem identical (mask, cells, graph, points; G to 2e-14, background 4e-10, same 845-dimensional data subspace); at the first build's h, s, ell the GMRF mean and s.d. agree to 1e-6 s.d.; L2 and TV maps at the same beta to 1e-12. The toolbox's evidence maximum is 0.28 log units higher, at a length of 1554 km rather than 2040 (a flat surface); end to end the GMRF mean differs by at most 0.07 first-build s.d., the maps by 0.01 SI·km RMS (the field model is no longer filtered a second time: 0.10 nT RMS of the data). Explore against final: median 0.14, max 0.77 explore s.d. Columns: the existing Australian cell columns to rounding. |
| 2 (done) | PD, CR, bounds; `gmrf.py` (shared by CM and CR), `covariates.py`, `parameterise.py` (domains, designs, summaries); the core takes a design of several undamped columns, with the evidence complete with its constants and a box prior. | PD-AUS: `notebooks/inversion/real_LCS1.csv` (39 domain values). CR: `linear_models.npz`, `null_regression.npz`. Bounds: `bounds.npz`. Result: PD's estimator on the first inversion's polygon-exact G reproduces its values to 2e-14; end to end, with domains on cell edges, within 1.33 noise s.d. (median 0.50), noise s.d. ratio 1.02. CR: coefficients within 0.013 s.d., class statistics and hypothesis ranking as before (evidences within 0.33), null test spreads as before (age and undated slopes exceed 2 reported s.d. in 70-77% of replicates, in both). Bounds: identical to two decimals. Explore against final: CR coefficients within 2.2 s.d.; PD values differed by up to 6.5 noise s.d. with coarse cells, which change the domains themselves, so domain methods now keep their cells in explore (ADR 0006 amendment): PD values then identical, noise s.d. within 1%. |
| 3 (done) | FD-merge (`methods/merge.py`); CR scores FD-merge partitions as hypotheses. | `merge_points.npz`, `merge_points_sweep.npz` (FD-merge-AUS). Result: on the first build's data and noise realisations every partition, lambda_95, the null check and both bootstraps reproduce exactly. End to end the partitions agree only loosely (ARI 0.26-0.92, similar K): the toolbox's own null draws give lambda_95 = 1304 (first 1255), and its data differ by 0.10 nT RMS (the field model is no longer filtered twice); each change alone moves the partitions as much (ARI 0.14-1.0), so single FD-merge partitions are fragile to perturbations far below the noise. Merged partitions still beat every covariate hypothesis in CR by ~320 log units (first build ~280). Explore against final: with coarse cells lambda_95 806 against 1304, partitions ARI 0.09-0.57; with the cells kept and 10/10 null and bootstrap runs (ADR 0006 amendment) explore takes 5 min against 13, lambda_95 1435 against 1304, partitions ARI 0.25-1.0, about the spread between two final runs with different noise draws. |
| 4 (done) | FD-vor, FD-lvor, FD-part (`methods/voronoi.py`, `lvoronoi.py`, `partition.py`, `sampling.py`); `core.box_evidence` shared by all samplers. | The `*_synthetic.npz` and `partition_diagnostics.npz` results in `regions/australia/cache/`. These cannot be reproduced draw for draw: run_voronoi_synthetic.py seeded its chains with Python's hash() of a string (different in every interpreter), and the chains had not converged (two chains on the same data disagree on K). Accepted instead (acceptance/stage4.py): on the first build's data the old and new samplers, from the same seed, pass through identical states (FD-vor sampling and annealing, FD-lvor with guided births, the boundary penalty and a start from a merge partition, FD-part); the prior tests give uniform P(K), P(M) and P(U given M), and FD-part's exact test over the 203 partitions of a 2 x 3 grid gives the first build's own value (total variation 0.0163). The synthetic suites (hours of chains) were not rerun. Real-data configs run in explore mode (cells kept, shorter chains): FD-vor 4 min, FD-lvor 3 min, FD-part 1.5 min. |

## Provisions for later work

Built in now, because they would be expensive to retrofit:

- Cell columns are keyed by (cell, layer, direction model), so remanent directions and extra layers are new cache entries, not new code paths.
- Each parameter carries its own bounds; the core makes no positivity assumption (a signed RVIM amplitude would be unbounded).
- A parameterisation may group cells by any label, not only spatially (e.g. ocean age bins per basin).

Deferred: ocean remanence (cells, signed amplitudes, polarity), multiple VIS layers, the viewer's detailed design.

After stage 4 (agreed 2026-10-08): *domains with internal variation*. Each domain has a mean VIS and a random (fractal) field inside it, described by an amplitude, a spectral slope and an outer scale. The fields are integrated out exactly, so the means come with uncertainties that include variation inside the domains, and the evidence says whether the data support it. Planned order: per-domain means with one shared amplitude and slope; then an amplitude per domain; then a slope per class; fields independent between domains, compared with fields crossing domain edges. The slope is shared because the data's band (about 300-2500 km) spans too few scales inside one domain to fix it. It extends `gmrf.py` (classes become domains; a fractional power of the precision makes the slope free) and turns the first polygon-domain inversion's T3 simulations into the inference model.

## Spherical geometry

All of the toolbox works on the sphere: spherical harmonics, equal-area HEALPix cells, great-circle buffers, source radii on the WGS84 ellipsoid (grid latitudes treated as geocentric, as now). No step assumes a flat Earth. Inputs and outputs are lon/lat on WGS84.
