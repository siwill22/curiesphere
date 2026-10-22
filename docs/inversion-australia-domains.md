# Domain VIS inversion: Australia (method record)

*Decisions agreed on 2026-10-01, before any code was written. The code is `notebooks/invert_domains.py`; the general background is in `docs/inversion-ideas.md` (question Q2, formulation (c)).*

## Purpose

A shakedown of the method. The deliverables are working inversion machinery and a statement of which domain parameters LCS-1 can and cannot resolve. Any geological reading of the fitted values is a bonus. The code is meant to generalise to other continents and other polygon sets.

## Set-up

| Item | Decision |
|---|---|
| Domains | Hasterok et al. (2022) provinces, from `gprm.datasets.Geology.fetch_GlobalTectonicMap()`, with `continent == 'Australia'`. The PNG and Bismarck units also carry that tag and are excluded. |
| Mapping | `notebooks/inversion/australia_domains.csv` maps each polygon to a parameter. The script writes the table once and then only reads it, so it can be edited by hand. Lumping and unlumping are table edits, not code changes. An empty `param` keeps the background. The key is the GeoDataFrame index, because the Hasterok `id` is not unique. |
| Margins | Passive margins and offshore plateaus are lumped into four parameters: `margin_W` (W rifted margin, Exmouth, Zenith, Quokka, Batavia, Gulden Draak), `margin_N`, `margin_E` (E rifted margin, Queensland–Marion Plateau) and `margin_S` (S rifted margin, Bass Strait). We are prepared to unlump them. |
| Parameter | One uniform VIS v_k (SI·km) per parameter. No H&M values are used inside the domains. |
| Ocean overlap | Cells with an ocean age keep the ocean model; the age grid wins over the polygons. The overlap is at most 8% of any polygon (Zenith Plateau), and 0% for most. |
| Source depth | v_k is spread uniformly in radius from the top of basement (ellipsoid − water − sediment) to 30 km below the ellipsoid. The 30 km base is a stand-in, not a claim. A Moho-following base is a question for later (magnetic-layer thickness). |
| Inducing field | IGRF-13, the pyshtools default epoch, evaluated on the r0 sphere, as in the rest of the model. |
| Background (fixed) | The current depth model everywhere else: H&M + slabs VIS on the other continents and PNG (on the ellipsoid), HM05 oceanic-crust VIS, and GK07_NR remanence. |

## Data and misfit

| Item | Decision |
|---|---|
| Target | LCS-1 Br at r0 = 6371.0 km, regional. |
| Points | HEALPix NSIDE 128 (about 55 km spacing), inside the domains plus a 300 km great-circle buffer. The buffer includes the continent–ocean edge anomalies, which tie the absolute level. |
| Band | Degrees 16–133. F(l) is applied to the model. With l_h = 186.8, F ≥ 0.99 up to degree 133, so the filter has little effect in this band. Degree 150 is a sweep. |
| Error model | σ = RMS(LCS-1 − MF7) at the points, uniform and diagonal. The formal covariance is therefore too optimistic, and the error bars come from a noise ensemble: random-phase realisations with the global per-degree spectrum of LCS-1 − MF7. MF7 is read like LCS-1 (r0 = 6371.0 km). |

## Solve

1. **v_ref:** a one-parameter fit, the single uniform VIS for all parameters. It also tests whether the common mode is resolvable. If it is not, v_ref falls back to a literature value of crustal susceptibility × thickness chosen independently of H&M, and every v_k is then relative to it.
2. **Diagnostic solve:** bounded least squares (v ≥ 0), undamped, with the SVD of G/σ, the resolution matrix and the posterior covariance.
3. **Working solve:** the same, with Tikhonov damping towards v_ref·**1**. The damping strength comes from the synthetic tests and an L-curve, and v_ref is varied to show which v_k depend on it. Smoothness across boundaries is not used, because it would damp the contrasts that the data resolve.

## Synthetic tests (before real data)

| # | Truth | Tests |
|---|---|---|
| T1 | Random v_k (log-normal, median 0.6 SI·km), the inversion's own G, plus noise | Conditioning, noise propagation, which domains resolve |
| T2 | Equal v_k (0.3, 0.6, 1.0 SI·km) | The common mode and the v_ref fit |
| T3 | Lognormal internal variation from first principles (no H&M), inverted as uniform | Error from assuming uniform domains |
| T4 | Sheet on the ellipsoid, or slab base at 15 or 45 km | Depth bias |
| T5 | Background altered (constant VIS on the other continents; another ocean model) | Background leakage |
| T6 | T1 plus a compact unmodelled anomaly | Leakage into neighbours; whether a robust misfit is needed |

Sweeps: buffer width (0, 300, 600 km), band (133 or 150), damping, and full against aggregated domains. The criteria for aggregating domains are to be decided after T1–T2.

**Check-ins with the user:** after T1–T2, which settle granularity, and after T3–T6, before any real-data fit.

## Spherical geometry

- **Point in polygon:** pygplates, with great-circle edges. Polygon edges (up to 2° long in the source) are densified to 0.1° steps in lon/lat first, so the great-circle edges follow the GIS straight lines.
- **Buffer:** great-circle distance.
- **Data points:** equal-area HEALPix.
- **Source radii:** taken from the WGS84 ellipsoid.
- **Latitudes:** the grid latitudes are used as geocentric, as in `depth_models.py`.

## Implementation notes

- `depth_models.slab_vis_coeffs` computes the slab with 8-point Gauss–Legendre quadrature in radius inside the depth series. It agrees with 280 slices of 100 m to within 2×10⁻⁷ at degree 133.
- `depth_models.induced_to_gmm` is bit-identical to `GlobalVIS.vim()`, but expands the inducing field only once.
- A 2–30 km slab is 22% weaker than a sheet on the 6371 km sphere at degree 133, and 11% weaker at degree 60.
- One column takes about 19 s.

## Results

### Set-up as built (2026-10-01)

- **Domains:** 46 polygons map to 39 parameters (35 onshore and 4 margin lumps), covering 13.3 M km² of domain cells.
  - About 2,700 non-ocean cells in the box belong to no domain.
  - They are all outside Australia: PNG, the Java and Banda forearcs, and slivers of Hasterok's oceanic polygons where the age grid has no age, along the continent–ocean boundary. They keep the background.
- **Data points:** 7,649 in total, 5,160 inside the domains and 2,489 in the 300 km buffer.
- **Field scale:** σ = RMS(LCS-1 − MF7) at the points = 20.4 nT. For comparison, LCS-1 RMS is 61.8 nT, the background 34.3 nT, and uniform 0.6 SI·km domains 29.6 nT.
- **Noise realisations:** RMS 16.4 nT at the points, so the global LCS-1 − MF7 spectrum underestimates the Australian difference by about 20%. From the second run on (2026-10-02), the realisations are scaled to σ = 20.4 nT. The T1 numbers below come from the first run; with the scaled noise, the RMS std is 0.052 instead of 0.042 SI·km, and v_ref is ±0.013 instead of ±0.011.

### T1 (random v_k, median 0.6 SI·km, 100 noise realisations) and T2 (common mode)

- **Conditioning:** the weighted G is well conditioned, with condition number 8.2. With the exact G and background, every one of the 39 parameters is recovered without bias.
- **Noise std:** 0.017–0.076 SI·km (3–13% of 0.6). It scales with domain size (log–log correlation −0.80).
  - Smallest errors: the margin lumps.
  - Largest: Hay-Booligal (40,000 km²), Mossman, Georgetown–Coen and Darling Basin.
- **Ensemble against formal error:** the ensemble std is 1.7–2.7× the formal (diagonal-σ) error, as expected for correlated noise. The error bars must come from the ensemble.
- **Damping:** not needed in T1. α/s_max ≤ 0.01 changes nothing, 0.03 adds a small bias, and 0.1 adds a large bias (RMS 0.06).
- **Common mode:**
  - It is the weakest direction, as expected (Runcorn): 83% of it lies on the smallest singular vector.
  - It is still resolved. The single-parameter v_ref is recovered to ±0.011 SI·km (2%) at levels of 0.3, 0.6 and 1.0. For a finite continent, the edges and the non-dipole inducing field break the annihilator.
- **What v_ref measures:** it is the best single level weighted by G, not the area mean. On T1 data it returns 0.575, against an area-weighted truth of 0.67.

**Reading:** with exact physics, noise alone does not force any aggregation. The limits will come from model error: non-uniform domains (T3), depth (T4) and the background (T5). Granularity should be decided after T3.

### T3: non-uniform domains (2026-10-02)

**Truth.** H&M is not used.
- **Formula:** inside each domain, v = v_k · exp(σ_ln·ε − σ_ln²/2), where v_k are the T1 values.
- **Why lognormal:** rock susceptibility is roughly lognormal.
- **The random field ε:** one field across the whole continent, zero mean and unit variance, Gaussian, on the sphere. It is self-similar: power per degree ∝ l^(1−β), from degree 2 to 600 (about 30 km).
- **Cases:**
  - σ_ln = 0.25, 0.5 or 1.0;
  - β = 2 (rough) or 3 (smoother);
  - 5 random fields per case, each inverted with 20 noise realisations, uniform domains, undamped.
- **Score:** recovered v_k against the actual area-weighted mean of the truth in each domain.

| σ_ln | β | RMS error | RMS bias | noise std | RMS relative error | misfit/σ |
|---|---|---|---|---|---|---|
| 0.25 | 2 | 0.122 | 0.110 | 0.053 | 0.30 | 1.44 |
| 0.25 | 3 | 0.120 | 0.107 | 0.053 | 0.26 | 1.17 |
| 0.5 | 2 | 0.213 | 0.207 | 0.052 | 0.66 | 2.43 |
| 0.5 | 3 | 0.219 | 0.213 | 0.052 | 0.56 | 1.64 |
| 1.0 | 2 | 0.396 | 0.393 | 0.051 | 2.78 | 5.22 |
| 1.0 | 3 | 0.461 | 0.458 | 0.051 | 2.27 | 3.18 |

Errors are in SI·km. For comparison, T1 with uniform truth gives misfit/σ = 0.98.

**Reading**
- **Internal variation dominates the error.** It is the main error source even when it is mild: at σ_ln = 0.25, the bias is twice the noise and the relative error is 26–30%. At σ_ln ≥ 0.5, uniform-domain values are not reliable for most domains.
- **Domain size.** Small domains suffer most (log–log correlation of bias with area −0.3 to −0.6), but size is not the whole story. Some large onshore domains are poor too (Tennant Creek, Capricorn, Darling Orogen, Madura), probably by trading off with their neighbours. The margin lumps (S, E, W) are the most robust.
- **The misfit gives it away.** Misfit/σ rises clearly with σ_ln, so the real-data misfit can be used to estimate how much internal variation there is.

### Real data: LCS-1, degrees 16–133 (2026-10-02)

The fit is bounded least squares, undamped, with the background fixed.

| Model | misfit/σ | Correlation with LCS-1 |
|---|---|---|
| Background only (no VIS in the domains) | 3.46 | – |
| One uniform VIS for all domains (v_ref = 0.60 SI·km) | 3.14 | – |
| 39 uniform domains | 2.70 (onshore 2.68, buffer 2.73) | 0.49 |
| Reference only: existing model (H&M VIS in Australia, GK07_NR) | 3.71 | 0.19 |

- **The 39-domain model fits better than the existing model in this band.** That is expected, since the domain model has 39 parameters fitted to these data.
- **The residual is large and has structure** at the scale of the anomalies inside the domains. This is the T3 situation: the data hold a great deal of variation within the domains, which uniform domains cannot represent.
- **The variation the misfit implies:** comparing it with T3 gives σ_ln ≈ 0.55 (β = 2) or 0.84 (β = 3). This is an upper limit, because depth, background and remanence errors are in the misfit too.
- **Recovered values:**
  - Precambrian west and centre: about 1.0–1.4 SI·km (Pilbara, Gawler, Coompana, Madura, Curnamona, Albany-Fraser, W Yilgarn, Tennant Creek, Mount Isa).
  - Phanerozoic east: about 0.1–0.4 SI·km (Lachlan, Delamerian, Thomson, Adelaide Rift, Darling Basin).
  - No domain is at the v = 0 bound.
- **Error bars:** noise ensemble combined with the T3 bias at the implied σ_ln, taking the larger of the two β values.
  - Below 30% for margin_E (14%), margin_S (27%) and Sydney–Surat (28%).
  - 40–50% for Cape York, margin_W and Pilbara.
  - Most onshore domains: 60% to over 100%.
  - The per-domain T3 bias rests on only 5 random fields per case, so these error bars are rough.

**Reading.** Individual domain values are not reliable at this granularity, because internal variation dominates. The west–east contrast is large, though, and may survive at a coarser grouping. That is the next thing to test with the T3 machinery.

### Merged groupings (2026-10-02, `notebooks/invert_domain_groups.py`)

**How it works:**
- The columns of the members are summed, so no new forward runs are needed.
- The truth for a group is the area-weighted mean of the T3 truth over the group.
- The model error is read from T3 at the σ_ln implied by the 39-domain fit (0.55 for β = 2, 0.84 for β = 3), taking the larger of the two.
- The groupings are in `inversion/australia_groups.csv`:
  - **13 groups:** Hasterok's `prov_group`, plus Central (Arunta, Musgrave), East (New England, Sydney–Surat, Mossman) and Tasmania, with Albany-Fraser and Darling Orogen in the WAC, and the 4 margins.
  - **6 groups:** Precambrian, Phanerozoic and the 4 margins.

| Scheme | Parameters | LCS-1 misfit/σ | Relative error |
|---|---|---|---|
| Domains | 39 | 2.70 | Median about 80%; only margin_E, margin_S and Sydney–Surat below 30% |
| Groups | 13 | 3.00 | 19–80%; Tasmania 141%, margin_N 243% |
| Groups | 6 | 3.04 | Phanerozoic 0.48 ± 0.17 (35%); Precambrian 0.81 ± 0.62 (77%); margin_E 16%, margin_S 31% |

**Reading**
- Merging helps only modestly. The anomalies inside the domains are still in the data, and fewer parameters are still forced to fit them.
- On these error bars, the Precambrian–Phanerozoic difference (0.81 against 0.48 SI·km) is not significant.
- The error bars are conservative: they take the worse of the two β values, and the lognormal at σ_ln ≈ 0.8 has a heavy tail.
- The E and S margins are robust at every granularity.

## Free domains: formed from the data (2026-10-02)

**Motivation.** Fixed polygons impose surface-geology boundaries, and the field sees the whole magnetic lithosphere. Polygons are a comparison, never a truth. The aim is the smallest number of domains, with sharp boundaries and geologically plausible shapes, that the data require.

**Basis (`notebooks/cell_basis.py`).**
- **Cells:** 1,402 HEALPix NSIDE-64 cells (equal area, about 100 km) over the same outer extent as the fixed-polygon inversion. Each cell's column is a 1 SI·km slab from the top of basement to 30 km, at degree 150.
- **Check:** the cells summed over all the domains of the fixed-polygon inversion reproduce those domains' coefficients to 3×10⁻¹⁵.
- **Evaluation at the data points:** done with an explicit synthesis matrix, checked against pyshtools.

**Merging (`notebooks/merge_domains.py`).**
- **Start:** every cell is a domain.
- **Merging:** merging two touching domains has an exact closed-form cost in least squares. The merge path is guided by a small smoothness term (μ ∈ {0.03, 0.1, 0.3}, as restarts), but layouts are scored by the plain data misfit.
- **Number of domains K:** minimises χ² + λK, with λ calibrated on null data (one uniform domain + correlated noise) so that K = 1 in 95% of realisations: λ_95 = 1313.
- **Clean-up:** boundary cells move between domains if that lowers the score.
- **Connectivity:** two bridges join Tasmania and the offshore plateaus to the mainland graph.
- **Version 1** (merging from the unregularised cell fit) failed: the starting cell values are mostly noise.

**Boundary length and floor (`notebooks/merge_compact.py`).**
- **Shape cost:** γ per cell edge on a domain boundary, in the merge order, in the choice of K and in the clean-up.
- **VIS floor:** 0.45 SI·km, the H&M (+ slabs) area-mean over Hasterok's Phanerozoic group (10th percentile 0.33). H&M's values are a surface sheet, so for our slab this is a gentle floor.

### Results

| Model (LCS-1) | K | Boundary edges | misfit/σ |
|---|---|---|---|
| 39 fixed polygons | 39 | – | 2.70 |
| Free, λ_95, γ 0, v ≥ 0 | 8 | 1139 | 1.99 (two domains at v = 0) |
| Same layout, v ≥ 0.45 | 8 | 1139 | 2.18 |
| Free, λ_95, γ 10, v ≥ 0.45 | 7 | 511 | 2.12 |
| Free, λ_95/3, γ 10, v ≥ 0.45 | 17 | 714 | 2.00 |
| Free, λ_95/3, γ 30, v ≥ 0.45 | 14 | 296 | 2.32 |

- **Synthetic random 8-block truths** (3 cases): only 2–5 domains are recovered, even without noise.
  - The stopping rule, not the search, is the limit: the true layout always scores worse than the one found (`show-me/2026-10-02-free-domains/13-…`).
  - Blocks whose signal improves χ² by less than λ_95 are below detection at this noise level.
  - The boundary penalty helps in one case, loses a real block in another, and is unstable in the third.
- **Self-consistency.** Real-data models were used as truths, each inverted with its own setting, with noise at the LCS-1 − MF7 level or at the real-fit misfit level (about 2σ):

| Truth | Median ARI, noise | Median ARI, misfit level | Domains recovered |
|---|---|---|---|
| 8 domains, γ 0 | 0.36 | 0.23 | 5–10 |
| **7 domains, γ 10** | **0.83** | **0.70** | 5–8, value error 13–15% |
| 17 domains, λ/3, γ 10 | 0.34 | 0.32 | 9–16 |

**Reading**
- **The reproducible model** is the 7-domain compact one: high-VIS domains in the south-west and along the south coast (Yilgarn, Gawler, south coast, about 1.0–1.8 SI·km), within a uniform background. The background sits at the floor (0.45), so its level comes from the floor, not the data.
- **The data support about 6–8 compact domains** at the current noise estimate. The spindly unpenalised model and the 17-domain model are not reproducible.
- **This tests stability, not truth.** A method that prefers compact shapes reproducing a compact model is expected. Compactness is an assumption, a geologically reasonable one.

**Open:**
- the sensitivity of the 7-domain model to the floor, the ocean background south of Australia (its stripes are stronger than in LCS-1) and the slab base;
- the noise level, which sets the detection threshold;
- a Bayesian (transdimensional) version.

## Noise level: comparison with Swarm models (2026-10-02, `notebooks/noise_models.py`)

**Models compared:** LCS-1, MF7, and two ESA Swarm lithospheric models, MLI_SHA_2D 0501 (Thébault, degrees 16–133) and MLI_SHA_2C 1101 (comprehensive inversion, degrees 16–120). The MLI files are kept outside the repo in `~/Data/Swarm/MLI`, and their coefficients are converted from 6371.2 to 6371.0 km.

**RMS Br differences at the 7,649 Australian data points, degrees 16–133:**

| Pair | RMS difference (nT) |
|---|---|
| LCS-1 − MF7 | 20.4 |
| LCS-1 − MLI_2D | 20.2 |
| MF7 − MLI_2D (independent data: CHAMP against Swarm) | 15.5 |

For reference, LCS-1 itself is 61.9 nT RMS.

**Three-cornered hat** (LCS-1, MF7, MLI_2D): LCS-1 17.1, MF7 11.2, MLI_2D 10.8 nT. In degrees 16–120, with MLI_2C included, the LCS-1 error is 11.5–15.9 nT, depending on which trio is used.

**Spectra** (global): the differences are below 1 nT² per degree up to about degree 50, then rise to about 10 nT² per degree at degrees 100–133, roughly a third of the signal power.

**Maps** (`inversion/noise_difference_maps.png`): LCS-1 − MF7, and LCS-1 − MLI_2D in the west, are dominated by north–south stripes, the signature of satellite-track artefacts. MF7 − MLI_2D is much less striped. A pattern shared by the two LCS-1 differences but missing from MF7 − MLI_2D points to LCS-1 itself. Shared damping between MF7 and MLI_2D (both have less power above degree 60) would also make LCS-1 look worse than it is.

**Reading**
- **The LCS-1 error over Australia** is probably about 11–17 nT RMS in degrees 16–133, against the 20.4 nT used so far. That makes σ about 1.2–1.9× too large.
- **The error is anisotropic**, with north–south track stripes. The isotropic random-phase noise model does not represent it.
- **Model error dominates the real fit.** The fits leave about 40 nT RMS, roughly 2× the old σ and 2.5–3.5× these estimates. Reducing σ would therefore lower the detection threshold for data noise, but it would not reduce that model error.

### Three-model mean as the target (2026-10-02)

**Target:** the equal-weight mean of LCS-1, MF7 and MLI_SHA_2D (`inversion/target_mean3.npz`).
- **Estimated error:** 7.7 nT, i.e. √0.5 × the RMS deviation of the models from their mean, assuming independent errors of equal size.
- **Power relative to LCS-1:** 1.01 in degrees 16–60, 0.95 in 60–100 and 0.85 in 100–133. Part of the drop is noise averaging out; part is the other models' extra damping.

**Switches:** `TARGET=mean3` and `SIGMA=<nT>` in `merge_domains.py`; outputs get a suffix.

**Results**
- **Same σ as LCS-1 (20.4 nT):** the domains are almost unchanged. λ×1, γ 10 gives 6 domains with a residual of 41.3 nT RMS, against 7 domains and 43.3 nT for LCS-1. λ/3, γ 10 gives 13 domains against 17.
- **σ = 7.7 nT** (with λ_95 unchanged in normalised units): 21–69 domains, with residuals of about 30–38 nT RMS, i.e. misfit/σ of about 4–5. Model error, not data noise, dominates the residual.

**Reading**
- The robust structure (high VIS in the south-west and along the south coast and Gawler, against a uniform background) does not depend on the target model.
- The stopping rule has to allow for model error, not just data noise, for example with σ_total² = σ_noise² + σ_model², with σ_model estimated from the data. Without that, a smaller σ produces more domains.

## North America: quick test of the free-domain method (2026-10-03)

**Scripts:** `notebooks/region_setup.py`, then `cell_basis.py`, `merge_domains.py` (with `SYN_CASES=0 N_REAL=0`) and `merge_compact.py`, each with `REGION=north_america`. Outputs go to `notebooks/inversion_north_america/`.

**Set-up**
- **Method:** as for Australia.
- **Extent:** Hasterok North America polygons on continental and transitional crust, west of 180°. Central America, the Caribbean arcs, the Aleutian arc, the Arctic Ocean ridges and the Siberian units are excluded (`domains.csv`, editable).
- **Size:** 33 M km², 3,446 cells and 17,204 points.
- **Noise:** σ = 22.4 nT.
- **VIS floor:** 0.50 SI·km, the H&M area-mean over the Phanerozoic continental provinces (10th percentile 0.26).
- **Stopping rule:** λ_95 = 2122.
- **Run time:** cells 2 h; null calibration 25 min; compact settings 17 min.

**Results**

| Model | K | Residual (nT) | misfit/σ |
|---|---|---|---|
| H&M (existing) | – | 69.0 | 3.08 |
| Free, λ_95, v ≥ 0 | 10 | 39.9 | 1.78 |
| λ×1, γ 10, v ≥ 0.5 | 7 | 43.7 | 1.95 |
| λ×1/3, γ 10, v ≥ 0.5 | 19 | 37.6 | 1.68 |
| λ×1/3, γ 30, v ≥ 0.5 | 12 | 48.0 | 2.15 |
| λ×1, γ 30, v ≥ 0.5 | 3 | 52.5 | 2.35 |

**The 7-domain model (λ×1, γ 10)**
- low (0.56 SI·km): the Cordillera, Alaska, the Arctic islands, Greenland, the Appalachians, and a central band from Hudson Bay through the Great Lakes into the central US;
- 1.16: the northern Canadian Shield and the Grenville/Labrador east;
- 1.01: the southern US and Mexico;
- 1.89: the southern midcontinent;
- 2.1–2.2: two Great Bear/Wopmay bodies.

The floor is not active for this model: v ≥ 0 gives the same misfit.

**Not run:** the synthetic blocks, the noise-added runs and the self-consistency test. Figures: `show-me/2026-10-02-north-america/`.
