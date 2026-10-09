# Experiment: how much do the spherical-harmonic degrees matter? (2026-10-07 to 10-09)

*Background research for planning, not a method decision. The question: how do the VIS maps of Australia depend on the range of spherical-harmonic degrees we invert? In particular:*
- *What do the low degrees (below 16, hidden by the core field) do to absolute levels and boundaries?*
- *What do higher degrees add, up to 300, from MLI_SHA_2E and from a future satellite mission?*

## Summary

1. **Low degrees (1–15) set the regional levels, not the detail.** In a synthetic test, adding them would halve the uncertainty of regional means (e.g. a craton's mean VIS) and remove their bias. Maps and boundary positions barely change. This is Runcorn's theorem in practice: a uniform layer has almost no external field, so absolute levels sit in the degrees the core field hides.
2. **With today's data, degrees above 133 add little, because of the noise.**
   - Degrees 133 to 185 improve the province-scale maps by about 15% on the ~100 km cells, and less on the ~50 km cells.
   - Degrees 186–300 from MLI_SHA_2E add almost nothing under our noise rule.
   - On real 2E data to degree 300, merging collapses to 1–2 domains.
3. **The noise is what limits us, not the method.** With low noise at high degrees (future data, noise 10% of the signal amplitude), degree 300 transforms the maps:
   - fine-scale (50–200 km) recovery rises from 0.06 to 0.89 (TV map, province truth);
   - map error falls from 0.22 to 0.06 SI·km;
   - all bodies 117–150 km across are found, and three-quarters of those 83–117 km across;
   - merged boundaries come to lie within about 50 km of the true ones, instead of about 400 km.
4. **Stopping early with future data loses most of the gain.** Lower noise at degree 133 improves broad levels only. Most fine detail needs degrees 186–300. Even at 30% noise, degree 300 is far better than today's data.
5. **Prior models that allow sharp edges recover the detail; smooth Gaussian priors don't.**
   - **Smooth Gaussian priors:** the GP never models the fine scale. A two-scale GP (broad + detail) gave the same maps as one scale in every case.
   - **TV** (edges allowed) recovers the fine scale.
   - **Merging** (uniform domains) does too, given enough domains.
6. **Uncertainties on broad quantities do not shrink with more degrees.**
   - Bounds on regional means widen as degrees are added: each degree adds noise, and regional means are low-degree quantities.
   - A single-scale GP becomes overconfident and biased on broad levels as degrees are added (covariate synthetic).

## Set-up

| Item | Choice |
|---|---|
| Problem | VIS per cell, uniform from the top of basement to 30 km. Forward model: exact VSH-based cell columns, induced by IGRF-13. Data: Br at 7,681 points (HEALPix NSIDE 128, ~55 km apart), Australia plus a 300 km buffer. Background (other continents, oceans, remanence) is fixed, as in [inversion-australia-domains.md](inversion-australia-domains.md). |
| Cells | Stage 1: NSIDE 64 (~100 km, 1,406 cells). Stages 2–3: NSIDE 128 (~50 km, 5,358 cells), enough for degree 300, whose half-wavelength is 67 km. The cell columns are summed directly over each cell's 0.1° nodes (`basis.cell_matrix_direct`); they match the global transform to about 1e-14. |
| Data models | **LCS-1** (to degree 185, damped above ~130; its fitted damping filter is applied to data, G and noise alike). **MLI_SHA_2E** version 0901 (degrees 16–1500, includes aeromagnetic data; no filter). Over Australia the two correlate at 0.98 (degrees 16–60), 0.92 (61–100), 0.81 (101–133) and 0.65 (134–185). |
| Noise, real data | The disagreement of the two models: LCS-1 − F·2E, per degree. Above degree 185 there is no second model, so the noise is held at the noise-to-signal ratio of degrees 150–185. That ratio is 1.26× the signal power, which is pessimistic. |
| Noise, future data | At every degree, today's noise or a fraction f of the 2E signal amplitude, whichever is smaller (f = 10% or 30%). The level is anchored to today's measured noise over Australia. Data compression keeps the directions that carry signal, not only those that carry noise. |
| Synthetic truths | **Provinces:** one value per Hasterok et al. (2022) province (53), from the H&M statistics only. **Provinces + variation:** plus 25% variation with ~200 km correlation. **Multiscale:** plus 20% variation at ~50 km and 40 magnetic bodies (+0.8 SI·km, 50–150 km across). **Covariate:** class means + crust and heat-flow terms + correlated variation. The multiscale truth's field up to degree 133 is the same as without the detail (75 nT against 76 nT); the detail is all at higher degrees. |
| Methods | GP map (one and two scales, hyperparameters by evidence), L2 (smooth) and TV (edges allowed) maps at the GP's misfit, merging (domains, refined on the finer cells), and extremal bounds on regional means. |

## Results

### Stage 1: ~100 km cells, degrees 16–100, 133, 185 (LCS-1 and 2E), low-degree tests

- **LCS-1 against 2E (16–133):**
  - maps correlate at 0.94;
  - regional means agree within ±0.14 SI·km;
  - the choice of noise reference makes no difference.
- **Degree 100 → 133:** synthetic map errors fall about 12%.
- **Degree 133 → 185:** another ~15% on the province truths (LCS-1 to 185: GP 0.194, TV 0.173 SI·km). On the covariate truth the GP gets worse and overconfident (NAC mean 0.60 ± 0.09 against a true 1.00).
- **Starting at degree 20 or 25 instead of 16:** slightly worse.
- **Adding degrees 1–15 (synthetic only):** halves the regional uncertainty and removes the level bias; barely changes the detail.
- **Bounds on regional means:** with positivity alone they span 0–4 SI·km in every band. The data do not fix absolute levels.

### Stage 2: ~50 km cells, real data to degree 300 (today's noise)

Map error against the truth (RMS, SI·km):

| Run | Provinces: GP | Provinces: TV | +Variation: GP | +Variation: TV | Covariate: GP |
|---|---|---|---|---|---|
| LCS-1 16–133 | 0.223 | 0.205 | 0.247 | 0.235 | 0.251 |
| LCS-1 16–185 | 0.224 | 0.199 | 0.246 | 0.225 | 0.342 |
| 2E 16–185 | 0.229 | 0.216 | 0.255 | 0.234 | 0.381 |
| 2E 16–300 | 0.226 | 0.205 | 0.242 | 0.247 | 0.360 |

Real data, GP regional means (SI·km):

| Run | Area mean | NAC | WAC | SAC |
|---|---|---|---|---|
| LCS-1 16–133 | 0.53 ± 0.08 | 0.37 ± 0.14 | 0.69 ± 0.14 | 0.75 ± 0.14 |
| 2E 16–300 | 0.62 ± 0.06 | 0.46 ± 0.10 | 0.82 ± 0.10 | 0.86 ± 0.10 |

- **Real GP maps, degree 133 against 300:** correlation 0.88 (TV 0.76).
- **Bounds widen with degree.** On the provinces + variation truth, the NAC mean is 0.43–1.35 at 133 and 0.18–1.87 at 300 (roughness-limited bounds).
- **Merging the real data at degree 300:** 1–2 domains (bootstrap 1–2), against 5 at degree 133. The misfit is 1.6× the noise, and pure-noise data merge to one domain only 85% of the time.
- **Merging synthetic provinces at degree 300:** 6 domains, capturing 0.69 of the true variation (degree 133: 7 domains, 0.63).

So real 2E data above degree 185 hold detail that a few uniform domains cannot fit: small sources, or error in the model.

### Stage 3: future data (synthetic only, ~50 km cells)

Share of the true VIS recovered, by scale (1 = perfect, 0 = no better than a flat map). The scales are defined by Gaussian smoothing along great circles:
- **broad:** larger than ~400 km;
- **middle:** about degrees 100–200;
- **fine:** about degree 200 down to the 50 km cell size.

TV map, provinces truth:

| Run | Broad | Middle | Fine | RMS error (SI·km) |
|---|---|---|---|---|
| Today, to 133 | 0.93 | 0.59 | 0.06 | 0.221 |
| Today, to 185 | 0.89 | 0.71 | 0.24 | 0.217 |
| Future 10%, to 133 | 0.99 | 0.76 | 0.19 | 0.171 |
| Future 10%, to 185 | 0.99 | 0.92 | 0.50 | 0.126 |
| Future 10%, to 300 | 1.00 | 0.99 | 0.89 | 0.061 |
| Future 30%, to 300 | 0.97 | 0.95 | 0.80 | 0.104 |

Multiscale truth: bodies found (more than half the true contrast recovered), GP map. TV is similar and finds 46% of the smallest bodies at degree 300:

| Run | 50–83 km (13 bodies) | 83–117 km (16) | 117–150 km (11) |
|---|---|---|---|
| Today, to 133 | 0% | 12% | 36% |
| Today, to 185 | 8% | 38% | 82% |
| Future 10%, to 185 | 0% | 56% | 82% |
| Future 10%, to 300 | 31% | 75% | 100% |

Merging, provinces truth: domains, and the mean distance from each true boundary to the nearest merged one:

| Run | Domains | Distance to nearest merged boundary |
|---|---|---|
| Today, to 133 | 7 | 382 km |
| Future 10%, to 185 | 56 | 87 km |
| Future 10%, to 300 | 58 | 63 km |

On the multiscale truth, merging reaches the code's 100-domain cap.

Maps for the multiscale truth: `notebooks/regions/australia_fine/future_maps.png`. Each row shows a run's TV map, merged domains, map error and Br misfit.

## Lessons

- **The noise model decides what extra degrees can do.** Holding the noise-to-signal ratio of degrees 150–185 made degrees 186–300 useless by construction. A planning study must state its noise assumption up front.
- **Choose measures by scale.** RMS over the whole region, regional means and bounds are dominated by broad levels, which high degrees cannot improve. Their gain shows only in fine-scale recovery, boundary distances and small-body detection.
- **Compression must keep the signal directions.** Compressing on the noise covariance alone drops exactly the low-noise, informative directions.
- **Smooth Gaussian priors waste high-resolution data.** Methods for future data should allow sharp edges (TV, merging, domain samplers). Merging needs its 100-domain cap raised.
- **Absolute levels need low degrees or prior information,** whatever the high degrees do.
- **Cell size and data spacing must follow the degree.** 50 km cells and ~55 km points carry degree 300. Degree 500 needs ~25 km cells (~21,000 over Australia), denser points and a matrix-free solver.

## Caveats

- **Stage 3 is synthetic.** It assumes Gaussian noise with a smooth spectrum and a forward model with no error (no depth or remanence error).
- **The future noise levels are assumptions,** not a mission specification.
- **The truths are built from province polygons,** which reflect surface geology. They test resolution, not geology.

## Reproduce

`notebooks/run_degree_experiment.sh stage1 stage2 stage3` reruns everything as it was run: every step, its settings and its log file. It runs one job at a time with ≤3 workers, `nice -n 15` and single-threaded BLAS, in the conda env `pygmt17`. Stages 2–3 first need `build_region.py regions/australia_fine/config.toml`.

| What | Where |
|---|---|
| The runs | `[bands]` in `regions/australia/config.toml` and `regions/australia_fine/config.toml` |
| Building the band variants | `build_bands.py` |
| Comparing them | `compare_bands.py` (stages 1–2), `compare_future.py` (stage 3) |
| Maps | `regions/australia_fine/plot_future.py` |
| Results (not in git) | `regions/<region>/cache/bands/<run>/`, `bands_summary.npz`, `future_summary.npz` |

The inversion explorer (GeodeViewers/CuriesphereInversion, Degrees tab) shows stage 1 only. Stages 2–3 use the ~50 km cells, and the viewer holds one region.

## Follow-up

1. **Raise merging's domain cap.** `K_SAVE = 100` in `domains/merge.py` limits merging to 100 domains. The multiscale truth reached it with future data to degree 300. Raise it and rerun the stage 3 merging.
2. **Regression, class statistics and hypotheses on the ~50 km cells.** They were skipped there (`PARTS=gp`), because refitting the correlation length many times on 5,358 cells is slow. They need a faster recomputation of the correlation blocks.
3. **Degree 500.** This needs ~25 km cells (~21,000 over Australia), denser data points and a matrix-free solver: the dense G and the data-space covariances no longer fit.
4. **Stages 2–3 in the viewer.** This would need a second region (the ~50 km cells).
