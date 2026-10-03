# Domain inversion framework: design (agreed 2026-10-03)

This framework generalises the Australia and North America free-domain work (`docs/inversion-australia-domains.md`). It inverts a satellite lithospheric field for uniform-VIS domains within any continent-scale area. Three methods share one engine and one output format, so their results can be compared directly.

## Goals

- **Simple models:** a small number of domains of uniform VIS, with compact, not spindly, geometry.
- **Boundary confidence:** how confident each boundary is.
- **Trade-off:** between simplicity (number of domains K) and fit.
- **Speed:** blocky models are acceptable if they are much faster.

## Scope (v1)

- **Area:** one continent-scale area at a time; global runs come later.
- **Region:** any outline file (GeoPackage or GeoJSON, lon/lat WGS84). Cells with an ocean age are removed. A helper builds outlines from Hasterok et al. (2022). The outline sets only the outer edge of the area, never boundaries inside it. Point-in-polygon tests are done on the sphere (pygplates), so outlines may cross 180° or contain a pole.
- **Outside the area:** everything is fixed: H&M + slabs on oceanic crust, plus GK07_NR remanence. A low-priority option is free scale factors on the ocean model in sectors of the 300 km buffer.
- **Sources:** VIS spread uniformly from the top of basement to 30 km below the ellipsoid; IGRF-13 inducing field.
- **Data:** LCS-1 Br, degrees 16–133, LCS-1 filter F(l), HEALPix points (NSIDE 128) within 300 km of the area. The mean of LCS-1, MF7 and MLI_SHA_2D is used only as a check on the final models.

## Shared engine

- **Cell basis:** HEALPix NSIDE 64 cells (about 100 km), one column each (computed once per region). An NSIDE 32 basis (about 200 km) is made by summing the four nested children of each cell, with no new transforms.
- **Error model:** a correlated covariance C = C_noise + s²·C_model.
  - C_noise comes from the LCS-1 − MF7 spectrum.
  - The shape of C_model comes from the residual spectrum of a reference fit (the shape only, never the residual map).
  - The data and columns are whitened once with a Cholesky factor of C.
  - The amplitude s is estimated from the data: hierarchically in the Bayesian method, and by a fit in the others.
- **VIS range:** 0–4 SI·km. H&M's global maximum is 3.38 as a surface sheet, about 4 for our slab. H&M statistics may set priors and bounds; H&M maps are never inputs.

## Methods

- **(a) Merging:** agglomerative merging with the score misfit + λK + γL, as now, on whitened data. λ_95 is recalibrated on null data under the new error model. Merging runs at 200 km, then the boundaries are refined at 100 km (only boundary cells move). This becomes the default only after checking it against a full 100 km run on Australia.
- **(b) Voronoi optimisation:** simulated annealing with the sampler's moves, at fixed K = 1…30, from several starts. Gives the misfit-against-K curve.
- **(c) Voronoi rjMCMC:**
  - One seed per domain; single Voronoi domains are convex.
  - Units are reported by merging neighbouring domains whose values are indistinguishable. Labelled Voronoi (seeds sharing values) is the possible second step.
  - Priors: K uniform on 1…K_max; seeds uniform over the area; VIS uniform on 0–4 (H&M-shaped prior as the one sensitivity test). The width of the value prior sets the price of a domain (the Occam factor).
  - VIS values are integrated out analytically for each layout.
  - Moves update the fit incrementally; parallel tempering helps escape local traps.
  - Implemented twice, our own sampler and BayesBay, compared on a small synthetic case before one is chosen.

Dropped: TV regularisation (it is the convex relaxation of the same problem, but shrinks contrasts and does not count domains) and smooth-then-cluster approaches.

## Common outputs

- **Boundary probability for each cell edge:**
  - (c): from the posterior;
  - (a) and (b): from about 30 parametric-bootstrap reruns (best-fit prediction + a realisation of the full error model); (b) also uses several starts per dataset.
- **One trade-off figure per region:** misfit against K for (a) at several γ and for (b), P(K) for (c), and the K picked by λ_95.
- **Maps:** VIS, model Br and residual, in the existing layout.

## Validation before real data (all methods)

- **Truths:**
  - null (uniform);
  - Hasterok-group shapes, used as realistic non-convex shapes, not as truth about the Earth;
  - each method's real-data model inverted by the others.
- **Realism:** lognormal variation within domains, plus errors drawn from the full error model.
- **Metrics:** ARI; true boundaries found and false ones reported; K recovered; calibration (whether boundaries at 80% probability are real about 80% of the time).

## Code

A small module `notebooks/domains/` of plain functions (region, basis, errors, merge, voronoi, outputs, figures), with thin run scripts and one config file per region. The existing scripts (`merge_domains.py` and the others) are frozen, so the Australia and North America results stay reproducible.

## Order, with a check-in after each step

1. **Module:** region, basis (including the 200 km sums) and outputs. Check: reproduces the existing Australian inputs and merging result.
2. **Error model and whitening.** Check: null runs give K = 1; the fitted amplitude is plausible.
3. **Merging upgrade.** Check: 200 km + refine agrees with 100 km on Australia.
4. **Own Voronoi sampler and annealing.** Check: recovers simple synthetic cases.
5. **BayesBay comparison.**
6. **Common test suite on all methods.**
7. **Real data:** Australia, then North America.

## Progress

### Step 1: module, region and basis (2026-10-03)

**Code**
- `notebooks/domains/`: `region.py` (outline → extent, Hasterok helper), `basis.py` (cells, columns, data, background, noise, coarse basis, reuse of unchanged cells) and `outputs.py` (cell graph, boundary probability, trade-off records).
- Run script: `notebooks/build_region.py <config>`.
- Region: `notebooks/regions/australia/` (`config.toml`, `outline.gpkg`); the cache is not tracked.

**Check against the old pipeline**
- The old Australian extent was imposed, and three cell columns were recomputed.
- The result is identical: cells, data points, background, LCS-1, MF7 and noise are bit-identical, and so are G and the recomputed columns.
- The cell graph equals that of `merge_domains.py`.
- The old 8-domain layout refitted from the new arrays gives misfit/σ 1.9912, as before.

**Coarse basis**
- 381 NSIDE 32 cells; 319 have all four children, and the rest are partial cells at the coast.
- Every child centre lies in its parent pixel.

**Extent correction**
- The old code clipped the polygons to a box ending at 5°S, which cut 394 nodes of the North Australian margin polygon (4.3–5°S, 134–140°E).
- The new region follows the outline: 1,406 cells (1,392 columns reused, 14 computed) and 7,681 points.

### Step 2: error model and whitening (2026-10-03)

**Code:** `notebooks/domains/errors.py` and `notebooks/error_model.py <config>`, which writes `cache/errors.npz` and `<region>/error_model.png`.

**Effective information**
- At Australia's 7,681 points, 99% of the variance of the noise covariance lies in about 730 eigen-directions, and 99.9% in about 850.
- The points (about 55 km apart) oversample a field resolved to about 150 km, so treating them as independent overstates the information about tenfold.
- The data are therefore compressed to the 845 leading directions of the total covariance (99.9% of the trace) and whitened there. Every method works with 845 numbers instead of 7,681.

**Shape of the model error**
- Taken from the covariance-against-distance of the residual of a reference fit: the old 8-domain model, residual 40.8 nT RMS.
- A spectrum shaped like LCS-1 itself fits well (curve misfit 23.7 nT²), as does a fitted power law (21.2; exponent 0.38).
- The LCS-1 − MF7 shape is too narrow (51.0).
- The LCS-1 shape is used: it has no fitted exponent, and unmodelled magnetisation should produce a field that looks like the field.

**Checks**
- **Covariance code:** 100 random-phase noise realisations, an independent synthesis, whiten to variance 1.00 per direction (5–95% range 0.76–1.23, as expected from 100 samples).
- **Amplitudes:** with σ_n fixed at RMS(LCS-1 − MF7) = 20.4 nT, the maximum-likelihood model error is s = 40.9 nT, and χ²/dimension = 1.04 overall. By scale, though, it ranges from 1.49 where noise-shaped errors dominate to 0.74 where field-shaped errors dominate.
- **Both amplitudes estimated:** σ_n = 31.4 nT and s = 32.5 nT, with χ² by fifths of 1.14, 1.11, 0.72, 0.97, 1.07. The short-wavelength error is larger than the data noise alone, i.e. part of the model error is fine-scale.
- **Design change:** both amplitudes are estimated from the data, not just s. The whitening is exact for any pair of amplitudes.

**Deferred to step 3:** the null check (noise-only data return K = 1) needs the merging code in the module.
