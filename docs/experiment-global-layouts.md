# Experiment plan: do the global coefficients support polygon layouts? (planned 2026-10-10)

*The question: does a global fit to the Gauss coefficients support polygon domains? Two separate claims are tested: that domains differ in level, and that VIS patterns break at domain boundaries. A better global VIS map than GCM-GLC is a secondary aim. Local fits can't answer this well: each continent sees only its own data, and needs a ring error for what lies outside it.*

Decisions: [ADR 0008](adr/0008-gaussian-prior-on-domain-levels.md) (domain levels), [ADR 0009](adr/0009-ocean-model-error-in-global-fits.md) (ocean-model error), [ADR 0010](adr/0010-df-in-a-global-fit.md) (DF-GLC). Terms: [CONTEXT.md](../CONTEXT.md) (layout, background error, GCM, DF).

## Set-up

| Item | Choice |
|---|---|
| Problem | As GCM-GLC ([gcm_glc_lcs1_16-133.toml](../inversions/gcm_glc_lcs1_16-133.toml)): all continental crust, NSIDE 64 (22,591 cells), LCS-1 degrees 16–133 (17,700 coefficients), one layer from top of basement to 30 km, induced by IGRF-13. |
| Layouts | The ladder: **one level** → **continents** (7) → **groups** (`prov_group`, else the province) → **provinces** (~750, the `param` column of the seven continent tables). Plus a **class** layout: CR's 3 classes (cratons+shields, orogens+arcs, margins+rifts+oceanic). Cells outside every polygon get one rest level per continent. Cells go to the domain covering most of their area, as now. |
| GCM with a layout (idea 1) | Domain levels ~ N(v0, tau^2), cells damped toward their domain's level. tau, the damping and the error amplitudes are fitted by the evidence, via Woodbury on the cached eigendecomposition. |
| DF-GLC (idea 2) | A fractal pattern per domain (dense covariance, one block per domain), with β in 2–3 and a shared patch size. Strength shared by all domains, then one per class. Evidence maximised, error bars from its curvature. ~1–1.5 min per evaluation, 1–2 h per layout. |
| Error model | Measured noise (LCS-1 − MLI_SHA_2E) + the ocean-model error (spectrum from GK07_NR − HM05_NR, amplitude fitted). For GCM with a layout, a fitted noise amplitude is also run as a check. |
| Compute | One run at a time, `nice -n 15`, 1 BLAS thread. B (3.2 GB) and one dense 22,591² matrix (4.1 GB) in memory. |

## Synthetic checks

Each check uses one seeded noise draw, plus an ocean-model error (truth computed with HM05_NR, fitted with GK07_NR).

| # | Truth | Expected | Idea |
|---|---|---|---|
| S1 | One level, 1.0 SI·km | "One level" wins (false-positive check) | 1 |
| S2 | Group levels, drawn with the tau fitted to LCS-1 | Groups win; provinces do not beat groups | 1 |
| S3 | Group levels + a fractal pattern with breaks at group boundaries | Breaks beat one pattern over the whole area | 2 |
| S4 | 0.04 × Crust1.0 (as in GCM-GLC) | No layout wins strongly (realistic false-positive check) | 1 |

## Steps, with a check-in after 3, 6 and 7

1. **Code.** Add a layout option to GCM (domain levels as a Woodbury update) and the ocean-model error term. Then confirm that the "one level" layout reproduces GCM-GLC (v0 = 0.751 ± 0.011 SI·km).
2. **Synthetic checks** S1, S2 and S4.
3. **LCS-1, GCM with a layout,** on the whole ladder and the class layout, under both error models. Report the evidence table, tau, and maps of VIS + domains, model Br, data Br and residual.
4. **Code.** DF-GLC, with a smoke test at NSIDE 32.
5. **Synthetic check** S3.
6. **LCS-1, DF-GLC, shared strength,** on the ladder. Compare with the local DF results per continent.
7. **LCS-1, DF-GLC, per-class strengths** (3 classes). The `prov_type` classes come only if the 3 classes clearly beat one strength.

**Gate:** if step 3 finds no layout better than one level, or step 2 shows the evidence can't tell layouts apart, steps 6–7 run only on "one level" and the best-ranked layout.

## Progress

**2026-10-10, steps 1–2.** GCM with layouts reproduces GCM-GLC on the one-level layout (v0 0.751, noise ×1.53). Real data, fitted noise scale: provinces beat groups by 215 log units, and one level by 570. Synthetic checks (fitted noise scale):

| Check | Best layout | Gain over one level |
|---|---|---|
| S1 one level | continents | 9 (passes: nothing gains much) |
| S2 group levels | groups (provinces −251) | 2427 (passes) |
| S4 0.04 × Crust1.0 | groups (provinces −55) | 1368 (**fails** as a false-positive check) |

S4 shows that grouping neighbouring cells into domains wins for any truth that is smooth over hundreds of km, whether or not the domain edges are geological. Beating one level therefore doesn't test the polygons. **Added: control layouts blind to geology**: HEALPix tiles (NSIDE 8 and 16, ~800 and ~400 km) and random Voronoi tiles, as many as the groups (437) and the provinces (724). The polygons count only if they beat the controls clearly on real data and not on S4. The random tiles are more even in size than the polygons (provinces 41,000–660,000 km² between the 10th and 90th percentiles; the matching tiles 124,000–540,000 km²).

The ocean-cell columns of ADR 0009 (29,627 cells, basement to 6 km below, magnetised with the ocean model's own VIM) reproduce the ocean model's field when summed (correlation 1.000, amplitude within 1% up to degree 133).

**2026-10-11, overnight (step 3 and the control layouts).** Results and figures: [reports/2026-10-11-layouts](reports/2026-10-11-layouts/README.md). Crust1.0 (S4) is not a null, since crustal thickness follows the provinces. Added **S5**, a smooth random truth (beta 2.5, ~500 km, s.d. 0.4 SI·km), as the null.

Globally, with the ocean-model error (best 16 nT; +3300 log evidence over noise alone), the polygons beat random tiles of the same number on LCS-1: provinces +217, groups +156. On S5 they lose: −460 and −311. Locally (DF, Gaussian means), the polygons beat random tiles in EUR +257, NAM +127, ANT +63 and AUS +34 (nulls: NAM −55, ANT −9, AUS −23), but not in SAM +6, AFR +4 or ZEA −4. The spread of the domain means falls to its lower bound in AFR, NAM, ANT and EUR: where the polygons help locally, they do so through independent patterns, not levels.

## Deferred

- Soft boundaries: a shared global pattern plus the domain patterns, with the split fitted by the evidence (ADR 0010).
- Null tests over many noise draws: only once a result is worth calibrating.
- FD-merge layouts globally: merge partitions exist only for Australia.

## Caveat to watch

Degrees below 16 are missing from the data, so the mean level of a domain larger than about 2500 km is poorly constrained, and the prior on levels (tau) will carry it ([experiment-sh-degree-range.md](experiment-sh-degree-range.md), point 1). The continents rung is the most exposed to this.
