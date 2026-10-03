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

### Step 3: merging on whitened data (2026-10-03)

**Code:** `notebooks/domains/merge.py` (the merging of `merge_domains.py`/`merge_compact.py`, with weighted edges and the coarse-then-refine driver) and `notebooks/run_merge.py <config>`, which writes `cache/merge.npz`. Quick-look figure: `notebooks/regions/australia/merge_check.png`.

**Units**
- γ is per fine edge (about 100 km) and given relative to λ_95.
- The old γ 10 was about 0.008 λ_95.

**Null calibration and check (the step 2 check)**

| Mode | λ_95 (whitened χ²) | Fresh null data giving K = 1 |
|---|---|---|
| Fine (100 km) | 118 | 95% (19 of 20) |
| Coarse then refine | 68.9 | 85% (17 of 20) |

Coarse then refine is biased: λ is calibrated on the coarse path, but the fine clean-up adds flexibility afterwards.

**Real data, Australia** (K and whitened χ², with 845 dimensions; one domain gives χ² 1693)

| Setting | Fine | Coarse then refine | ARI between them |
|---|---|---|---|
| λ×1, γ 0 | 2 domains, 1041 | 3 domains, 1089 | 0.05 |
| λ×1, γ 0.01 λ_95 | 1, 1693 | 3, 1379 | 0 |
| λ×1/3, γ 0 | 8, 823 | 10, 785 | 0.22 |
| λ×1/3, γ 0.01 λ_95 | 2, 1616 | 6, 1192 | 0.05 |
| γ ≥ 0.03 λ_95 | 1–2 | 1–2 | – |

- **Fine at λ×1:** two domains. High VIS covers the Yilgarn, the south coast, the Gawler and a band into the centre; the rest of the continent is low. This is the robust structure of the earlier runs.
- **Bootstrap (30 sets, λ×1, γ 0):**
  - fine: K = 2–3; the south-coast boundary and the boundary around the south-western block have probability 0.6–0.9;
  - coarse then refine: more diffuse;
  - the two maps correlate at only 0.41.

**Conclusions**
- **Coarse then refine fails the check against fine.** The partitions differ (ARI ≤ 0.22), the fit is worse at equal K, and the null calibration is biased. Fine stays the default.
- **Fine is fast enough now.** The whitened data have 845 dimensions instead of 7,681, so a full run (3 restarts + clean-up) takes about 50 s for Australia.
- **The honest error model supports far fewer domains:** 2 at λ_95, against 7–8 before. Correlated, field-shaped model error is no longer counted as signal.
- **γ:** even 0.01 λ_95 collapses the model to 1–2 domains, so the useful γ range is lower (about 0.001–0.005 λ_95).
- **Open issue:** the error amplitudes come from the residual of an 8-domain fit. A 2-domain model leaves more residual (χ²/dimension 1.23), so re-estimating the amplitudes would lower K further. The hierarchical Bayesian method (c) estimates them jointly with the model.

**Correction (same day): the "2 domains" conclusion is withdrawn.** Maps (`show-me/2026-10-03-whitened-merging/`):
- **Residuals:** the uniform model leaves 64 nT RMS, 2 domains 48 nT, 8 domains 41 nT. The fitted data (LCS-1 minus background) are 71 nT RMS.
- **The error model is the cause.** It was sized from the residual of the 8-domain fit (45 nT RMS in total), and half of it has the spectrum of LCS-1. It counts unexplained geology as error, so every model beyond about 2 domains looks insignificant. The arithmetic is consistent: the compressed space holds 99.9% of the data energy. The error is in the modelling choice.
- **The 2-domain model is not simple.** Its high domain is a convoluted, interfingering shape whose edges reproduce many anomalies. The domain count understates its complexity at γ = 0.
- **Consequence:** the model-error term needs an estimate independent of our own fits, or must be left out of the stopping rule. The step 2 and step 3 results that depend on the fitted amplitudes are on hold.

**Option A: data noise only, two weightings (run_merge.py WEIGHTING=points/noise; maps in `show-me/2026-10-03-whitened-merging/` 02–03)**
- **A2, point weighting** (σ 20.4 nT, compressed data, λ calibrated on correlated noise):
  - λ_95 = 1,255; fresh null data give K = 1 in 90%.
  - It reproduces the earlier results: 9 domains at γ 0 (38.8 nT), and 7 compact domains at γ 0.01 λ_95 (12.5 per edge; 43.6 nT; L = 536, against the old 511).
- **A1, noise-whitened:**
  - λ_95 = 74.4; fresh null data give K = 1 in 100%.
  - 19–21 spindly domains (37–39 nT), and the γ grid is too weak in these units.
  - Its weight sits on the largest scales.
- **γ does not transfer between weightings as a fraction of λ_95,** because the fit gain per boundary edge scales differently.

**A2 γ sweep, bootstrap and trade-off (2026-10-04; `show-me/2026-10-04-a2-sweep/`)**
- **λ×1:**

  | γ (λ_95) | K | L (edges) | Residual |
  |---|---|---|---|
  | 0.005 | 7 | 653 | 41.1 nT |
  | 0.01 | 7 | 536 | 43.6 nT |
  | 0.015 | 5 | 375 | 46.1 nT |
  | 0.02 | 7 | 369 | 45.7 nT |

  At λ×1/3, γ 0.01 gives 18 domains (40.1 nT) and γ 0.02 gives 12 (43.7 nT).
- **Bootstrap at λ×1, γ 0.01:**
  - K = 3–6 (median 5).
  - Stable boundaries (above 80%): the south coast, the southern edge of the central block, and parts of the Gawler body.
  - The eastern domain is not stable (20–40%).
- **Trade-off:** no elbow. The residual falls steadily, from 64 nT at K = 1 to about 45 at K = 7 and about 32 at K = 100. K is set by λ.
- **Decision:** option A2 is the merging baseline, with γ about 0.01 λ_95. Coarse-then-refine is dropped.
