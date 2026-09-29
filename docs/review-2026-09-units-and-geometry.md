# remit review: forward-model amplitudes, units and geometry

*Written 2026-09-29 during work on the Geode Curiesphere viewer (Geode `docs/plans/curiesphere-viewer.md`). Nothing in remit has been changed. This document records what was found, how, and what to do next. The follow-up work is meant to happen on its own branch in a separate session.*

*Update: A, B and C are now fixed on branch `fix/forward-amplitudes` (uncommitted at time of writing), with regression tests in `tests/test_forward_transform.py`. Results in §11.*

*Update 2: the fixes were checked against the maths of Gubbins et al. (2011, GJI 187, 99–117), the formalism `vhtools.py` implements (§13). A and C agree with it exactly. The paper's maths is silent on B; its Fig. 7 shows that the Leeds group used the same VIS × B_nT shortcut as remit's old code, which is not evidence about H&M's calibration, so B stands.*

*Revised the same day after a second review: A confirmed by an independent dipole sum (Appendix A5); B's open question settled from Hemant & Maus (2005) itself; C upgraded, because it is what the Runcorn residual actually measures; §10 re-prioritised.*

Line numbers refer to `main` at commit "Update models.py", plus the uncommitted `return ValueError` → `raise ValueError` edits in `earthvim.py` and `vhtools.py`. Those edits don't change any line numbers.

---

## 1. Summary

| # | Finding | Effect on forward models | How sure |
|---|---|---|---|
| A | The conversion from remit's internal coefficients to standard Gauss coefficients leaves every order-m > 0 coefficient √2 too small | Forward fields about **0.71×** their intended strength | **Confirmed** three ways: a convention-free rotation test, a closed form, and an independent point-dipole sum sharing no code with remit (ratio 0.707 for m > 0, 1.000 after the fix, tangential components included). |
| B | The induced magnetization multiplies susceptibility-thickness by **B in nT**; it should use **H = B/μ₀** with km→m | Induced part about **1.257×** too strong | **Confirmed.** Hemant & Maus (2005) define VIS in SI·km and use M̃ = χ̃B/μ₀ (their Eq. 3), and tuned their susceptibilities to MF3 under that convention. Gubbins et al. (2011) Fig. 7, made at Leeds from the same VIS, appears to use the old VIS × B_nT shortcut, so it inherits the same factor (§13.3). The fix applies to the continental grid and `_noLIPs` alike. |
| A+B | Combined | Induced about **0.89×**, remanent about **0.71×**; the continent/ocean balance is off by about 26% | Follows from A and B |
| C | Leakage into odd-degree zonal (m = 0) terms in the numerical integration | Negligible in the real models (~1e-4 of the field, §11), though it is ~93% of the Runcorn residual and grows with degree up to lmax | Measured; cause **confirmed** (the trapezium end-corrections: exact quadrature removes it, §11) |
| D | remit treats input latitudes as geocentric. The inputs are probably geodetic | Forward sources may be misplaced north–south by up to 0.19° (21 km) at 45° | Code behaviour **verified**; the datum of the input grids is **inferred** (the files don't say) |
| E | Sphere vs WGS84 radius: "0 km" is a sphere, not the Earth's surface | A display/definition question for the viewer; also affects where forward-model sources sit | Measured on LCS-1 |

What checked out: ocean remanence units; the overall μ₀·10⁹/r₀ scaling; the field-direction maths; Runcorn's theorem (a uniform-susceptibility shell in the Earth's own field gives almost no external field; what is left over is mostly C).

---

## 2. Background in plain terms

- **Degree and order.** A spherical-harmonic model is a sum of patterns. Each *degree* l has 2l + 1 patterns, labelled by *order* m. The m = 0 pattern varies only with latitude, like bands round the globe (a "zonal" term). All the others (m > 0) vary with longitude too. A real anomaly map is almost entirely m > 0: at degree 100, 200 of the 201 patterns are.
- **VIM and VIS.** VIM is *vertically integrated magnetization*: magnetization (A/m) × layer thickness (m) = amperes. It is what remit's thin-shell transform (`forward_transform`) turns into Gauss coefficients. VIS is *vertically integrated susceptibility*: susceptibility (dimensionless SI) × thickness. remit's VIS is in **SI·km** (see §4).
- **B and H.** Magnetization induced in a rock is susceptibility × **H**, the magnetizing field in A/m. Field models such as IGRF give **B** in nT. The two are related by H = B/μ₀, with μ₀ = 4π×10⁻⁷ T·m/A.
- **Geodetic and geocentric latitude.** Because the Earth is flattened, a point has two latitudes. *Geodetic* is the angle of the local vertical (perpendicular to the WGS84 ellipsoid); it is what ordinary maps, GPS, gravity grids and coastlines use. *Geocentric* is the angle of the line from the Earth's centre. They agree at the equator and the poles and differ by at most 0.19° (about 21 km north–south) at 45°. Spherical-harmonic field models (LCS-1, MF7, the Swarm models, IGRF) and pyshtools use geocentric coordinates.

---

## 3. Problem A: m > 0 coefficients are √2 too small

### The test

1. **Rotation test (convention-free).** Magnetise a thin shell radially with the pattern "cosine of the angle from an axis", once with the axis at the north pole (z), once pointing to (0°N, 0°E) (x), and once to (0°N, 90°E) (y). These are the same physical source, only rotated, so the resulting dipole must have the same strength in all three cases. remit's `forward_transform`, with 1000 A amplitude, gives:

   | Source axis | g₁₀ | g₁₁ | h₁₁ | Dipole strength |
   |---|---|---|---|---|
   | z (pole) | +6.564e-02 | 0 | 0 | 6.564e-02 nT |
   | x (equator, 0°E) | ~0 | +4.649e-02 | ~0 | 4.649e-02 nT |
   | y (equator, 90°E) | ~0 | ~0 | +4.649e-02 | 4.649e-02 nT |

   The ratio is 0.7082. After removing the m = 0 case's own 0.16% integration error it is 1/√2 = 0.7071.
2. **Closed form.** For a purely radial VIM M₀·S_lm on a thin shell of radius a (S_lm a Schmidt semi-normalised harmonic), the external coefficient is g_lm = μ₀·M₀·l / ((2l + 1)·a) × 10⁹ nT, for every m. Results:

   | Pattern | remit / closed form |
   |---|---|
   | (1, 0) | 0.998407 |
   | (3, 2) | 0.707106 |
   | (20, 7) | 0.707107 |

   So m = 0 is right and every m > 0 is 1/√2 low.
3. **Independent point-dipole sum (Appendix A5).** The shell is sampled with 1.5 million equal-area points; each carries a dipole of moment VIM × dS, and Br at 300 km altitude is summed directly. No remit or pyshtools code is involved on this side. The comparison is with remit's coefficients expanded by pyshtools at 12 random points (least-squares ratio):

   | Source | remit ÷ direct, as is | After ×√2 on m > 0 |
   |---|---|---|
   | S₁₁ radial | 0.7070 | 0.9999 |
   | Degree 20, m > 0 only, random | 0.7072 | 1.0001 |
   | Degree 20, m = 0 only | 0.9973 | 0.9973 |
   | Tilted uniform vector × (1 + S₃₂), all three components | 0.863 (12% misfit) | 0.998 (0.4% misfit) |

   The last row shows the fix is also right for the tangential (θ, φ) terms, which the tests above do not exercise. The remaining 0.3% on the m = 0 rows is C.

**Why it happens.** With Schmidt P_lm, the FFT-plus-latitude integral already gives 4π/(2l+1) for every m, so no m-dependent factor is needed to get Gauss coefficients. `fnorm` converts to complex-normalised vector harmonics (the convention of the VSH formalism), and line 309 never converts back.

### Where in the code

- `remit/vhtools.py:286`: `fnorm = np.sqrt(1.0/(2-kronecker))`, which is 1 for m = 0 and 1/√2 for m > 0.
- `vhtools.py:302–304`: `fnorm` multiplies Elm, Ilm and Tlm, remit's internal vector-harmonic coefficients.
- `vhtools.py:309`: `clm = mu0*Ilm*1e9/fi/gmm.r0`, the conversion to Gauss coefficients. `fnorm` is **not** undone here.
- `vhtools.py:410–419`: `inverse_transform` applies the same `fnorm` again.

### Why nobody would have noticed

The forward and inverse transforms use `fnorm` consistently, so converting a magnetization grid to vector harmonics and back returns the original grid. A round-trip test passes. The error only appears in the Gauss coefficients from line 309, which are what get expanded with pyshtools and compared with LCS-1.

### Effect

Every forward model's field is about 0.71× its intended strength. It's a near-uniform weakening. The m = 0 bands are relatively 41% stronger, but they carry about 1/(2l + 1) of the power, so that is invisible at these degrees.

### Fix (verified by A5)

At line 309, divide out `fnorm` (equivalently, multiply clm by √2 for m > 0) when forming glm and hlm. Leave the internal VSH normalisation and `inverse_transform` alone. Verified by A5. Note that the returned `vsh` object stays complex-normalised, so anything computing spectra from `vsh.Ilm` directly must allow for that (no notebook currently does: they all use `coeffs`).

---

## 4. Problem B: induced VIM uses B instead of H

### The code

- `remit/earthvim.py:222`: `mrad = self.vis * inducing_field.rad.data`, and the same for theta and phi. `inducing_field` is IGRF-13 expanded with pyshtools, in **nT**.
- The same convention predates remit: `~/GIT/OceanVIM/notebooks/VIM_tools.py:397` (a separate repo), `Ocean_IVIM_Mr = OceanVIS*Br`.

### The units of VIS

The grid files (`remit/data/continents/suscp_sphint.nc`, `Slabs_VIS_FlatModel_rect.nc`) carry no unit metadata. The evidence is `notebooks/basis_models.py:116`:

```python
# ... replace by the VIS value stated by H&M 2005 for oceanic crust (--> no LIPs)
model_vis.vis[ind] = 0.066 * 2.11 + 0.049 * 4.97
```

2.11 km and 4.97 km are the standard oceanic layer 2 and layer 3 thicknesses, so this is susceptibility × thickness in **km**. The grid values (0–3.38) fit that too: 0.01–0.05 SI × 20–40 km of crust.

### The arithmetic

VIM [A] = χ·t [m] × H [A/m] = VIS [SI·km] × 1000 × B [nT] × 10⁻⁹ / μ₀ = **0.796** × VIS × B_nT.

The code uses VIS × B_nT, so the induced part is 1/0.796 = **1.257×** (that is, μ₀×10⁶) too strong.

### What Hemant & Maus (2005) say (resolves the earlier caveat)

The earlier draft left open whether H&M built their VIS map under the same "VIS × F" shortcut, in which case using it that way would reproduce their model. The paper rules this out:

- **Units.** Figs. 2 and 4 label VIS "SI.km". Fig. 4's colour scale tops out at 3.38, which is exactly the maximum of `suscp_sphint.nc` (3.3829), so the grid is H&M's VIS map.
- **Physics.** Eq. 3: dm = M̃ ds = (1/μ₀) χ̃ B ds, carried into Eq. 4. They use H = B/μ₀.
- **Calibration.** Susceptibilities are scaled by a global 0.55, chosen to minimise the RMS difference between predicted and MF3 Gauss coefficients (§2.2), and described as the only link between the predicted and observed fields, adjusting "the overall amplitude" (§4). So the VIS values are calibrated against data under χB/μ₀. Using them as VIS × B_nT overpredicts by 1.257×.
- **`_noLIPs`.** The constant 0.066 SI × 2.11 km + 0.049 SI × 4.97 km is H&M's own oceanic model (§2.1.2, after White et al. 1992). It matches the grid's normal-ocean values (0.385 and 0.405) to within 6%, so it is in the same convention as the grid. The distinction the earlier draft drew between the grid and "literature" constants does not exist: one fix covers everything remit induces.
- **Remanence.** Eq. 5 uses μ₀·I₀ with I₀ in A, and Fig. 6 is labelled "Amperes", consistent with §7.

Two typos in the paper could mislead: the dipole-intensity factor is printed √(1 + 3cos³θ) (should be cos²), and the paleolatitude relation is printed tan Θ = tan I/2 (read as tan I = 2 tan Θ). `utils/grid.py` has both right.

Not stated in the paper: whether Fig. 4 is the initial or the first-iteration VIS, and which main-field model and epoch were used as the inducing field (remit uses IGRF-13).

---

## 5. Combined effect

| Part | Problem A | Problem B | Net factor |
|---|---|---|---|
| Remanent (oceans) | 0.707 | — | **0.71** |
| Induced (continents, slabs, `_noLIPs` oceans) | 0.707 | 1.257 | **0.89** |

So forward models are too weak overall, and the induced part is about 26% too strong relative to the remanent part.

For the continents this is now settled: H&M's VIS values are fitted to data under correct physics (§4), so remit's HM05-based induced fields are about 0.89× their calibrated amplitude. Fixing A alone would leave them 1.257× too strong, so A and B should be fixed together.

For the ocean remanence it is also settled: the parameters (MagMax, layer weights) are taken directly from Gee & Kent (2007) (preset `GK07`, `notebooks/basis_models.py:41`), not tuned to LCS-1. So the magnetization values are right and should not change. The published forward maps' remanent part is about 0.71× what those values imply, and their induced part about 0.89×. The fix is in the code, not the parameters.

---

## 6. Issue C: zonal leakage

A pure degree-1, m = 0 radial source should give only g₁₀. remit also returns small **odd-degree zonal** terms:

| Coefficient | Value (nT) |
|---|---|
| g₁₀ | 0.0656 (the intended one) |
| g₃₀ | −3.1e-4 |
| g₅₀ | −5.0e-4 |
| g₁₁₀ | −9.4e-4 |
| g₁₅₀ | −1.09e-3 |

The leakage grows with degree, reaching about 1.7% of g₁₀ per degree by l = 15, and totals about 0.021 nT up to l = 30. m > 0 sources leak far less (3e-9 for (20,7)).

Suspected cause: the trapezium end-corrections at `vhtools.py:239–253` (9/8 and 7/8 on the first two and last two rows) assume a grid with both pole rows, but the DH2 grid has the north pole and not the south pole (`utils/grid.py:142 DH2`). That asymmetry would favour odd zonal terms. Not confirmed.

**Where it shows up.** The Runcorn test (§7) leaves a residual of max 0.08 nT. 93% of that residual (by summed |coefficient|) is in m = 0 terms, and its largest coefficients are g₃₀,₀, g₂₈,₀, g₂₆,₀, …, growing towards lmax. For a band-limited uniform shell in an internal field the exact answer is zero, so the residual is a measure of C, not a physical limit. Induced models have a large zonal part (VIS × a mostly dipolar field), which is where C acts. *Measured after the fix, though, C changes the real models by only ~0.0003 nT RMS at 300 km (corrected m = 0 terms equal 0.796 × published for VIS and 1.000 × for the remanent part, to that precision). It dominates the Runcorn test only because the uniform-shell field cancels there.*

**Better fix than patching the end-corrections:** replace the home-made trapezium weights with proper Driscoll–Healy quadrature weights (`pyshtools.expand.DHaj`), which are exact for band-limited functions on this grid. That also removes the `cos(0.000001)` workaround for the pole row in `_setup_transform`.

---

## 7. What checked out

- **Ocean remanence units.** `SeafloorAgeProfile` gives RVIM = Σ M [A/m] × dz [m] = **A**. MagMax and the layer weights are in A/m, depths in m (`earthvim.py:352–424`, `utils/profile.py:239–270`).
- **Scale of the transform.** `clm = mu0*Ilm*1e9/fi/r0` gives nT from a VIM in A (μ₀ [T·m/A] × A / m = T). g₁₀ matches the closed form to within 0.16%.
- **Directions.** `utils/grid.py:16 paleoIncDec2field`: inclination from tan I = 2 tan λ; Br = −Bz (z down, Blakely's convention); Bθ = −Bx (north); Bφ = By (east). The amplitude factor √(1 + 3 sin²λ) (`grid.py:65 vim2magnetisation`) is the dipole-field intensity relative to the equator.
- **Runcorn's theorem.** A shell of *uniform* susceptibility magnetised by an *internal* field produces no external field. With VIS = 1 × IGRF (degrees ≤ 30), remit returns max |g| = 0.08 nT, against 3.86 nT for the radial part alone. The cancellation works to about 2%. So the internal-field projection and the radial/tangential balance are consistent. (This test can't detect problem A, because A scales both parts equally.) The 2% residual left over is almost all odd/even zonal leakage (C, §6), so it should shrink substantially once C is fixed.

---

## 8. Latitude convention (geodetic vs geocentric)

### Verified

- remit takes each input grid's latitude value as the sphere's latitude: `vhtools.py:27`, `self.colat = np.radians(90-lat)`. There is no conversion anywhere: a search of `remit/` and the notebook helpers finds no mention of geodetic, geocentric, the ellipsoid or WGS84.
- pyshtools, and every SH field model, work in geocentric coordinates.

### Inferred, not verified

The input grids (age, paleolatitude/declination, VIS) are probably geodetic. Their files label the axis only as "latitude, degrees_north" with no datum. Their sources (ship-navigated magnetic picks, geological maps) are normally geodetic, but the processing in between (GPlates, GMT `sphinterpolate`) treats the Earth as a sphere and passes the numbers straight through.

### How big the effect is

Sliding a map north–south by δ changes each degree-l component by 2(1 − J₀((l + ½)δ)) in mean square, averaged over anomaly orientations. For small shifts that is about (l + ½)δ/√2 in RMS terms, so it grows roughly in proportion to degree. The measured change, LCS-1 Br at 3000 random points between 40° and 50° (mean shift 0.191° = 21.3 km), evaluated at the geodetic versus the geocentric latitude of each point:

| Degrees | Altitude | Measured RMS change / RMS field | Predicted | Degree doing most of the work |
|---|---|---|---|---|
| 16–60 | 0 km | 12% | 10% | ~44 |
| 16–100 | 0 km | 18% | 16% | ~68 |
| 16–133 | 0 km | 24% | 21% | ~86 |
| 16–185 | 0 km | 28% | 25% | ~107 |
| 16–133 | 100 km | 16% | 14% | ~59 |
| 16–133 | 300 km | 8% | 8% | ~32 |
| 16–185 | 300 km | 8% | 8% | ~32 |
| 50–133 | 0 km | 25% | 22% | ~94 |
| 50–133 | 300 km | 16% | 15% | ~62 |

By latitude, for degrees 16–133 at 0 km: 2% at 0–5°, 17% at 20–25°, 23% at 40–50°, 16% at 60–65°, 7% at 75–80°. The simple formula comes out 10–15% below the measurements, which suggests LCS-1 has more north–south variation at these latitudes than an isotropic field would.

### Implications

- **Viewer overlays (firm).** Gravity grids, coastlines, isochrons and picks are geodetic; SH maps are geocentric. Drawn together without conversion, magnetic anomalies sit up to 21 km towards the equator from where they belong. This affects the Geode viewer and the paper-style PyGMT figures alike.
- **Forward vs observed (unmeasured).** If the inputs are geodetic, remit's forward sources are shifted relative to the true (geocentric) observed field in the same way. Difference maps and correlations would then include a mid-latitude residual. How large it is for an actual forward model **has not been measured**; §10 step 7 does that.

---

## 9. Radius: sphere vs WGS84

The viewer and the notebooks treat "0 km" as the sphere r = 6371.2 km. The WGS84 surface is +6.9 km out at the equator and −14.4 km in at the poles. Because degree-l fields scale as (a/r)^(l+2), that matters at high degree. LCS-1 Br, RMS at the ellipsoid surface ÷ RMS at the sphere:

| Latitude band | Surface − 6371.2 km | Ratio, degrees 16–133 | Ratio, degrees 16–185 |
|---|---|---|---|
| 0–5° | +6.88 km | 0.909 | 0.887 |
| 20–25° | +3.82 km | 0.949 | 0.939 |
| 40–50° | −3.69 km | 1.055 | 1.067 |
| 60–65° | −9.85 km | 1.128 | 1.151 |
| 75–80° | −13.43 km | 1.152 | 1.196 |

Two separate consequences:
- **Display definition (viewer).** Comparing two SH models at the same point is unaffected by this choice. What changes is what "0 km" and the altitude slider mean. Satellite altitudes are normally quoted above the ellipsoid.
- **Forward-model source geometry (remit).** remit puts its VIM shell on a 6371.0 km sphere. The real sources lie below the ellipsoid, and oceanic ones also below the seafloor. At high degree that gives a latitude-dependent amplitude bias of the same order as the table. Correcting it properly is modelling work, not a bug fix.
  - Source depth is a bigger amplitude effect than the ellipsoid: continental sources centred ~15 km down give roughly (a/(a − d))^(l+2) ≈ 1.27 at l = 100, comparable to B.
  - Hemant & Maus (2005) put their sheet on the ellipsoid ("r′ is adjusted for the ellipticity of the Earth", "thin ellipsoidal sheet", §3). So remit's sphere differs from the geometry their 0.55 calibration was done in. At their comparison setting (400 km, degrees 16–90) that is a few percent at most, but it belongs in any attempt to reproduce HM05 exactly.

---

## 10. Proposed work for the branch

1. ~~**Independent confirmation of A.**~~ **Done** (§3 item 3, Appendix A5). Cases (a) S₁₁ and (b) random degree 20 were run, plus an m = 0 case and a three-component case; GK07 at lmax 40 was not needed.
2. ~~**Check Hemant & Maus (2005)'s VIS definition.**~~ **Done** (§4): B applies to everything remit induces.
3. **Fix A and B together** (fixing only A leaves induced sources 1.257× too strong relative to H&M's calibration):
   - `vhtools.py:309`: undo `fnorm` in the Gauss-coefficient conversion.
   - `earthvim.py:222`: multiply by 1e-6/μ₀ (SI·km × nT → A).
   - Consider a `legacy=True` option (or similar) that keeps the old factors, so the published models can still be reproduced; this ties in with the viewer decision in §12.
4. **Fix C:** switch the latitude quadrature to Driscoll–Healy weights (§6). Do this before setting test tolerances.
5. **Regression tests:**
   - the closed-form radial-VIM case;
   - the rotation-invariance case (ratio 1.000);
   - the point-dipole sum (A5), including the three-component case, to 0.5% or better;
   - Runcorn, with a tolerance set *after* C is fixed. Don't use "about 2%, as now": that would lock C in;
   - a zonal-leakage check (C) with a tight bound.
6. **Measure the effect.** For GK07, DAH981 and HM05 at 0 km and 300 km, make before/after maps next to LCS-1, plus an RMS table.
7. **Latitude experiment (low priority).** Build GK07 twice, once with its input grids regridded from geodetic to geocentric latitude first, and measure the change in the GK07 − LCS-1 difference and degree correlation. Forward-vs-observed correlations are well below 1, so a ≤ 21 km shift is likely buried in model misfit. If the regrid is cheap, simply applying it may be simpler than running the experiment. For the viewer overlays the fix is a plotting-time geocentric→geodetic conversion.

## 11. Status after the fixes (branch `fix/forward-amplitudes`)

**Code changes.**
- `remit/vhtools.py`: Gauss coefficients divided by `fnorm` (A). The trapezium end-corrections are replaced by exact Driscoll–Healy weights (`_dh_weights`), which raises `ValueError` for grids that are not DH2 (C).
- `remit/earthvim.py`: induced VIM multiplied by 1e3·1e-9/μ₀ (B).

**Tests** (`tests/test_forward_transform.py`, 10 tests, ~5 s):
- the closed form for four (l, m) pairs, to 1e-10;
- rotation invariance, to 1e-10;
- Runcorn: residual below 1e-10 of the radial-only field. It is now 4e-14 nT, down from 0.08 nT;
- a forward/inverse round trip with all three components;
- an independent point-dipole sum at 300 km, to 1e-5;
- the induced-VIM unit conversion;
- rejection of non-DH grids.

With the quadrature fixed, the closed-form checks are exact to about 1e-15 and the zonal leakage has gone. That confirms the end-corrections as the cause of C. They also caused the 0.16% error on g₁₀.

**Effect on the real models** (lmax 185, same inputs, old code run from a worktree of `main`). Amplitude ratio, new ÷ old, of Br RMS over degrees 16–185:

| Model | 0 km | 300 km |
|---|---|---|
| VIS (induced only) | 1.12 | 1.10 |
| GK07 remanent part (GK07 − VIS) | 1.41 | 1.41 |
| GK07 | 1.22 | 1.16 |
| HM05 | 1.18 | 1.19 |
| DAH981 | 1.19 | 1.13 |

The induced ratio is slightly below the 1.125 expected for m > 0 content, because B also scales down the m = 0 terms, which A leaves alone.

Comparison with LCS-1, degrees 16–133, old → new:

| Model | RMS model ÷ RMS LCS-1, 0 km | same, 300 km | Correlation at 300 km |
|---|---|---|---|
| VIS | 0.69 → 0.78 | 0.63 → 0.70 | 0.539 → 0.540 |
| GK07 | 0.81 → 0.97 | 0.71 → 0.82 | 0.588 → 0.587 |
| HM05 | 0.77 → 0.91 | 0.75 → 0.90 | 0.586 → 0.578 |
| DAH981 | 0.76 → 0.89 | 0.66 → 0.75 | 0.581 → 0.586 |

- **Amplitudes.** The published forward models were weaker than LCS-1 everywhere. The corrected ones are closer and still do not exceed it.
- **Correlations.** These barely change. A is almost a uniform scale factor; the small shifts come from the 1.26× change in the induced/remanent balance.
- **RMS misfit rises slightly** (for example GK07 at 300 km: 0.817 → 0.843). This is expected and is not evidence against the fix. With correlation around 0.58, the misfit-minimising prediction is one damped to about 0.58× the observed amplitude, so any model that gets closer to the true amplitude increases the plain RMS difference.

---

## 12. Decisions left to Simon

- **What the viewer ships.** Corrected forward models; as published, labelled "as in Williams et al. 2025"; or both as separate models.
- **Viewer geometry.** Keep the sphere; a geodetic grid with ellipsoidal height (each row evaluated at its geocentric colatitude and radius, components rotated to the local vertical); or fix only the latitude registration.
- **Forward-model geometry scope.** Geodetic→geocentric regrid of the inputs only; also source radius (ellipsoid or seafloor); or leave it and document it.
- **The paper.** Whether any of this needs revisiting there.

---

## Appendix: scripts used (run in the `pygmt17` env from `~/GIT/curiesphere`)

### A1. Closed-form radial VIM

```python
import sys, numpy as np, pyshtools
sys.path.insert(0, '.')
from remit.vhtools import GlobalMagnetizationModel
from pyshtools.legendre import PlmSchmidt
mu0 = 4e-7*np.pi; a = 6371000.; L = 30
N = 2*(L+1)
lat = 90 - np.arange(N)*180/N; lon = np.arange(2*N)*360/(2*N)
LON, LAT = np.meshgrid(lon, lat); th = np.radians(90-LAT); ph = np.radians(LON)
z = np.zeros_like(th)
M0 = 1000.
for l, m in [(1, 0), (3, 2), (20, 7)]:
    pl = np.array([PlmSchmidt(l, np.cos(t))[l*(l+1)//2+m] for t in th[:,0]])
    pat = pl[:,None] * np.cos(m*ph)
    _, c = GlobalMagnetizationModel(lon, lat, M0*pat, z, z, a).transform(lmax=L)
    expect = mu0*M0*l/((2*l+1)*a)*1e9
    got = c.coeffs[0, l, m]
    print(f'radial VIM {M0:.0f} A * S_{l},{m}: remit g = {got:.6e} nT, closed form {expect:.6e} nT, ratio {got/expect:.6f}, leakage {np.abs(c.coeffs).sum()-abs(got):.1e}')
igrf = pyshtools.datasets.Earth.IGRF_13()
G = igrf.expand(a=a, lmax=L, extend=False)
Br, Bt, Bp = G.rad.data, G.theta.data, G.phi.data
_, c = GlobalMagnetizationModel(lon, lat, Br, Bt, Bp, a).transform(lmax=L)
_, c2 = GlobalMagnetizationModel(lon, lat, Br, z, z, a).transform(lmax=L)
print('uniform VIS=1 x IGRF (nT): max |g| =', f'{np.abs(c.coeffs).max():.3e}', '; radial part alone:', f'{np.abs(c2.coeffs).max():.3e}')
```

### A2. Rotation invariance and zonal leakage

```python
import sys, numpy as np
sys.path.insert(0, '.')
from remit.vhtools import GlobalMagnetizationModel
a = 6371000.; L = 30; N = 2*(L+1)
lat = 90 - np.arange(N)*180/N; lon = np.arange(2*N)*360/(2*N)
LON, LAT = np.meshgrid(lon, lat); th = np.radians(90-LAT); ph = np.radians(LON)
z = np.zeros_like(th); M0 = 1000.
ux = np.sin(th)*np.cos(ph); uy = np.sin(th)*np.sin(ph); uz = np.cos(th)
for name, pat in [('along z (m=0)', uz), ('along x (g11)', ux), ('along y (h11)', uy)]:
    _, c = GlobalMagnetizationModel(lon, lat, M0*pat, z, z, a).transform(lmax=L)
    g10, g11, h11 = c.coeffs[0,1,0], c.coeffs[0,1,1], c.coeffs[1,1,1]
    print(f'{name:15s} g10 {g10:+.5e}  g11 {g11:+.5e}  h11 {h11:+.5e}  |dipole| {np.sqrt(g10**2+g11**2+h11**2):.5e}')
_, c = GlobalMagnetizationModel(lon, lat, M0*uz, z, z, a).transform(lmax=L)
big = np.argwhere(np.abs(c.coeffs) > 1e-4)
print('non-negligible coefficients for the z case (i, l, m, value):', [(int(i),int(l),int(m), float(c.coeffs[i,l,m])) for i,l,m in big][:12])
```

### A3. Geodetic vs geocentric shift, by band and altitude

```python
import numpy as np, pyshtools
from scipy.special import j0
clm,_ = pyshtools.shio.shread('remit/data/shc/LCS_mod.cof', lmax=185)
base = pyshtools.SHMagCoeffs.from_array(clm, r0=6371200.)
A, B = 6378137.0, 6356752.314245; e2 = 1 - (B/A)**2
def geocentric_lat(phid):
    p = np.radians(phid); N = A/np.sqrt(1-e2*np.sin(p)**2)
    return np.degrees(np.arctan2(N*(1-e2)*np.sin(p), N*np.cos(p)))
rng = np.random.default_rng(1); n = 3000
phid = rng.uniform(40, 50, n)*rng.choice([-1,1], n); lon = rng.uniform(-180, 180, n); phic = geocentric_lat(phid)
delta = np.radians(np.mean(np.abs(phid - phic)))
print(f'mean shift in band: {np.degrees(delta):.3f} deg = {delta*6371:.1f} km')
for lmin, lmax in [(16,60),(16,100),(16,133),(16,185),(50,133)]:
    for alt in (0, 100, 300):
        c = base.copy(); c.coeffs[:, :lmin, :] = 0
        r = 6371200. + alt*1e3
        a = np.asarray(c.expand(a=r, lat=phid, lon=lon, lmax_calc=lmax))[:,0]
        b = np.asarray(c.expand(a=r, lat=phic, lon=lon, lmax_calc=lmax))[:,0]
        meas = np.sqrt(((a-b)**2).mean()/(b**2).mean())
        l = np.arange(lmin, lmax+1)
        P = (l+1)**2*(6371200./r)**(2*l+4)*(c.coeffs[:, l, :]**2).sum(axis=(0,2))/(2*l+1)
        pred = np.sqrt((P*2*(1-j0((l+0.5)*delta))).sum()/P.sum())
        print(lmin, lmax, alt, round(meas, 3), round(pred, 3), round(pred*np.sqrt(2)/delta - 0.5))
```

### A4. Ellipsoid radius and latitude, by latitude band

```python
import numpy as np, pyshtools
clm,_ = pyshtools.shio.shread('remit/data/shc/LCS_mod.cof', lmax=185)
c = pyshtools.SHMagCoeffs.from_array(clm, r0=6371200.); c.coeffs[:, :16, :] = 0
A, B = 6378137.0, 6356752.314245; e2 = 1 - (B/A)**2
def geocentric(phi_d_deg, h=0.0):
    p = np.radians(phi_d_deg); N = A/np.sqrt(1 - e2*np.sin(p)**2)
    x = (N + h)*np.cos(p); z = (N*(1 - e2) + h)*np.sin(p)
    return np.degrees(np.arctan2(z, x)), np.hypot(x, z)
br = lambda a, la, lo, L: np.asarray(c.expand(a=a, lat=np.atleast_1d(la), lon=np.atleast_1d(lo), lmax_calc=L)).reshape(-1, 3)[:, 0]
rng = np.random.default_rng(1)
for band in [(0,5),(20,25),(40,50),(60,65),(75,80)]:
    n = 3000
    phid = rng.uniform(*band, n) * rng.choice([-1,1], n); lon = rng.uniform(-180,180,n)
    phic, r = geocentric(phid)
    ratios = []
    for L in (133, 185):
        m = 400
        e = np.array([br(ri, pc, lo, L)[0] for ri, pc, lo in zip(r[:m], phic[:m], lon[:m])])
        ratios.append(np.sqrt((e**2).mean()/(br(6371200., phic[:m], lon[:m], L)**2).mean()))
    sc = br(6371200., phic, lon, 133); sd = br(6371200., phid, lon, 133)
    rel = np.sqrt(((sd-sc)**2).mean()/(sc**2).mean())
    print(band, round(np.mean(r)/1e3-6371.2, 2), [round(x, 3) for x in ratios], round(rel, 3))
```

### A5. Independent point-dipole sum (confirms A, and the fix)

Set `APPLY_FIX = True` to multiply remit's m > 0 coefficients by √2 before comparing. The last block prints the composition of the Runcorn residual (§6). Takes a few minutes.

```python
import sys, numpy as np, pyshtools
sys.path.insert(0, '.')
from remit.vhtools import GlobalMagnetizationModel
from pyshtools.legendre import PlmSchmidt
mu0 = 4e-7*np.pi; a = 6371000.; L = 30; N = 2*(L+1)
lat = 90 - np.arange(N)*180/N; lon = np.arange(2*N)*360/(2*N)
rng = np.random.default_rng(0)
APPLY_FIX = False

def schmidt_field(th, ph, spec):
    """sum of c*P_lm(cos th)*cos(m ph) + s*P_lm*sin(m ph) for (l,m,c,s) in spec"""
    out = np.zeros_like(th)
    lmax = max(s[0] for s in spec)
    for i, (t, p) in enumerate(zip(th.ravel(), ph.ravel())):
        P = PlmSchmidt(lmax, np.cos(t))
        out.ravel()[i] = sum(P[l*(l+1)//2+m]*(c*np.cos(m*p)+s*np.sin(m*p)) for l, m, c, s in spec)
    return out

def comps(th, ph, case):
    """return Mr, Mtheta, Mphi (A) at colat th, lon ph"""
    if case == 'S11 radial':
        return 1000*np.sin(th)*np.cos(ph), 0*th, 0*th
    if case == 'deg20 radial, m>0 only':
        return 1000*schmidt_field(th, ph, spec20), 0*th, 0*th
    if case == 'deg20 radial, m=0 only':
        return 1000*schmidt_field(th, ph, [(20, 0, 1., 0.)]), 0*th, 0*th
    if case == 'tilted uniform x (1+S32), 3-comp':
        f = 1000*(1 + 0.5*schmidt_field(th, ph, [(3, 2, 0.7, -0.4)]))
        v = np.array([0.3, -0.5, 0.8])
        rh = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
        thh = np.stack([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)])
        phh = np.stack([-np.sin(ph), np.cos(ph), 0*ph])
        return [f*np.tensordot(v, e, 1) for e in (rh, thh, phh)]

spec20 = [(20, m, rng.normal(), rng.normal()) for m in range(1, 21)]

# equal-area Fibonacci source points on the shell
Ns = 1_500_000
k = np.arange(Ns) + 0.5
sth = np.arccos(1 - 2*k/Ns); sph = np.mod(np.pi*(1 + 5**0.5)*k, 2*np.pi)
rh = np.stack([np.sin(sth)*np.cos(sph), np.sin(sth)*np.sin(sph), np.cos(sth)], 1)
thh = np.stack([np.cos(sth)*np.cos(sph), np.cos(sth)*np.sin(sph), -np.sin(sth)], 1)
phh = np.stack([-np.sin(sph), np.cos(sph), 0*sph], 1)
dS = 4*np.pi*a**2/Ns

# observation points at 300 km
no = 12; r_obs = a + 300e3
olat = np.degrees(np.arcsin(rng.uniform(-1, 1, no))); olon = rng.uniform(0, 360, no)
ot, op = np.radians(90-olat), np.radians(olon)
ro = r_obs*np.stack([np.sin(ot)*np.cos(op), np.sin(ot)*np.sin(op), np.cos(ot)], 1)

LON, LAT = np.meshgrid(lon, lat); gth, gph = np.radians(90-LAT), np.radians(LON)
for case in ['S11 radial', 'deg20 radial, m>0 only', 'deg20 radial, m=0 only', 'tilted uniform x (1+S32), 3-comp']:
    Mr, Mt, Mp = comps(sth, sph, case)
    mom = (Mr[:, None]*rh + Mt[:, None]*thh + Mp[:, None]*phh)*dS   # dipole moments, A m^2
    Br_d = []
    for x in ro:
        R = x - a*rh; Rn = np.linalg.norm(R, axis=1)
        B = mu0/(4*np.pi)*(3*(np.sum(mom*R, 1)/Rn**2)[:, None]*R - mom)/Rn[:, None]**3
        Br_d.append(B.sum(0) @ (x/np.linalg.norm(x))*1e9)
    Br_d = np.array(Br_d)
    gMr, gMt, gMp = comps(gth, gph, case)
    _, c = GlobalMagnetizationModel(lon, lat, gMr, gMt, gMp, a).transform(lmax=L)
    if APPLY_FIX: c.coeffs[:, :, 1:] *= np.sqrt(2)
    Br_r = np.asarray(c.expand(a=r_obs, lat=olat, lon=olon, lmax_calc=L))[:, 0]
    ratio = np.sum(Br_r*Br_d)/np.sum(Br_d**2)
    print(f'{case:35s} rms direct {np.sqrt(np.mean(Br_d**2)):.4e} nT   remit/direct (lsq) {ratio:.4f}   misfit {np.sqrt(np.mean((Br_r-ratio*Br_d)**2)/np.mean(Br_d**2)):.1e}')

# Runcorn residual: what is it made of?
igrf = pyshtools.datasets.Earth.IGRF_13(); G = igrf.expand(a=a, lmax=L, extend=False)
_, c = GlobalMagnetizationModel(lon, lat, G.rad.data, G.theta.data, G.phi.data, a).transform(lmax=L)
co = c.coeffs; idx = np.dstack(np.unravel_index(np.argsort(-np.abs(co).ravel())[:10], co.shape))[0]
print('Runcorn residual, 10 largest (i,l,m,nT):', [(int(i), int(l), int(m), round(float(co[i, l, m]), 4)) for i, l, m in idx])
z = np.abs(co[0, :, 0]).sum(); allsum = np.abs(co).sum()
print(f'fraction of |residual| in m=0 terms: {z/allsum:.2f};  in degrees >13: {np.abs(co[:, 14:, :]).sum()/allsum:.2f}')
```

---

## 13. Cross-check against Gubbins et al. (2011)

`forward_transform` and `inverse_transform` implement the vector spherical harmonic (VSH) formalism of Gubbins, Ivers, Masterton & Winch (2011), *GJI* 187, 99–117. Each fix was checked term by term against that paper.

### 13.1 Problem A agrees with Eqs. 32–34

The paper gives g_lm = (μ₀/r_E)·√(l·ε_m)·Re(I_lm) and h_lm = −(μ₀/r_E)·√(l·ε_m)·Im(I_lm), with ε_m = 2 − δ_m0 (Eqs. 32–34). In the code, `1/fi[l-1]` = √l and `1/fnorm` = √ε_m. The old line had √l but not √ε_m. The fixed line `clm = mu0*Ilm*1e9/fi/r0/fnorm` is Eq. 32 exactly, and `hlm = -imag(clm)` has the sign of Eq. 33.

The √ε_m is there because the paper's harmonics are complex and mean-normalised: Y_lm = √((2l+1)/ε_m)·P_lm·e^{imφ} (A7), so the Schmidt real harmonic is √(ε_m/(2l+1))·Re Y_lm (A9). Putting A9 into Eq. 10 gives Eq. 32, √ε_m included. So the paper itself carries the factor the old code left out.

Also checked, and unchanged by the fixes:
- **E, I and T coefficients (Eqs. 29–31).** Each of Er/Et/Ep, Ir/It/Ip and Tt/Tp times `fe`, `fi`, `ft` and `fnorm/4π` equals (1/4π)∮M̄·(Y)*dΩ, with the components of Eqs. 19–27 and the conjugate of A7. The FFT sign (e^{−imφ}), dP/dθ = −sin θ·dP/dx, and the 1/sin θ in the φ and T_θ terms all match. E00 = −(1/4π)∮M_r dΩ matches Y_{0,1} = −r̂.
- **Inverse transform (Eq. 28).** Each term in `inverse_transform` is coefficient × the Eq. 19–27 component with the e^{imφ} removed, and `irfft` supplies the conjugate −m half. Keeping `fnorm` there is correct.
- The `vsh` object stays in the paper's normalisation, so the paper's I/E/T energy ratios can be compared directly with remit's.

### 13.2 Problem C agrees with §4.2 and §5.1

The paper used FFT in φ and the trapezium rule in θ on a 0.25° grid. It warns that E and T should integrate to zero external field but "will be subject to numerical error, producing leakage into spurious magnetic fields", which matters because E dominates. That is issue C. The Driscoll–Healy weights evaluate Eqs. 29–31 exactly for band-limited input, and the Runcorn residual (the paper's §3.3 argument for a uniform shell in an internal field) is now 4e-14 nT.

### 13.3 Problem B: the paper's maths is silent; Fig. 7 shows the Leeds convention, not H&M's

Eq. 37 writes M = χ·B₀ with no μ₀. It is a schematic statement for the null-space argument, not a units convention, so the maths neither supports nor contradicts B.

remit's `suscp_sphint.nc` came from Hemant and is VIS. It is taken to be the same model that Gubbins et al. used. (Their acknowledgement thanks Hemant "for providing us with his model of VIM", read here as the VIS model.) The VIM in their Figs. 2 and 7 was therefore computed at Leeds, from this VIS and IGRF-11 truncated at degree 13. Its colour-bar ticks are Mr −120 000 to 80 000 A, Mθ −100 000 to 0 A and Mφ −24 000 to 16 000 A. The same VIS (0.5° resample) in IGRF-13, also truncated at degree 13, gives these grid extremes:

| Component | Epoch | Fixed (VIS × H) | Old (VIS × B_nT) | Fig. 7 ticks |
|---|---|---|---|---|
| Mr min (43°N, 88°W) | 2005 / 2020 | −101k / −98k | −128k / −123k | −120k |
| Mr max (20°S, 140°E) | 2005 / 2020 | 88k / 87k | 110k / 110k | 80k |
| Mθ min (7°N, 21°E, Bangui) | 2005 / 2020 | −85k / −85k | −107k / −107k | −100k |
| Mθ max | 2005 / 2020 | 12k / 11k | 15k / 14k | 0 (bar runs slightly above) |
| Mφ min / max | 2005 / 2020 | −23k…22k / −24k…22k | −29k…27k / −30k…28k | −24k…16k |

The minima of Mr and Mθ, the two largest components, fit the old convention; Mφ and the Mr maximum fit the fixed one. Colour-bar ticks are only a rough guide, but on balance Gubbins et al. appear to have formed VIM as VIS[SI·km] × B[nT]. That is the same shortcut as remit's old code and `OceanVIM/notebooks/VIM_tools.py` (§4), which share the Leeds lineage.

**What this means for B.** Fig. 7 records how the Leeds code converted VIS to VIM. It says nothing about the convention Hemant & Maus used when they calibrated the VIS values against MF3, so it is not independent evidence against B. That question is answered by H&M's own text (Eq. 3, M̃ = χ̃B/μ₀, and the 0.55 calibration, §4), so B stands.

**Consequence for Gubbins et al. (2011).** Their induced VIM (Figs. 2, 7–10) is probably 1.257× too strong relative to their remanent VIM (Masterton 2010, Figs. 11–14, in A). The energy splits for the induced part alone (89:8:3) and the remanent part alone (42:26:32) do not depend on overall scale. The combined split (88:8:4) is weighted slightly too far towards the induced part.

An independent check is still possible and cheap: fit the least-squares amplitude of remit's induced field (after fixing A) to LCS-1 or MF7 over the continents, degrees 16–90, as H&M did to get 0.55. A factor near 1 with B confirms it; near 1.257 would mean H&M's calibration was done under VIS × B_nT after all.

Script (run in `pygmt17` from the repo root):

```python
import sys, numpy as np, pyshtools
sys.path.insert(0, '.')
from remit.data.models import load_vis_model
vis = load_vis_model(name='Hemant2005'); vis.resample(resolution=0.5)
k = 1e3*1e-9/pyshtools.constants.mu0.value   # the B fix
for yr in (2005, 2020):
    igrf = pyshtools.datasets.Earth.IGRF_13(year=yr); igrf.coeffs[:, 14:, :] = 0
    g = vis.vim(inducing_field=igrf)            # fixed convention
    for lab, a in [('Mr', g.mrad), ('Mt', g.mtheta), ('Mp', g.mphi)]:
        print(yr, lab, 'fixed', np.nanmin(a), np.nanmax(a), 'old', np.nanmin(a)/k, np.nanmax(a)/k)
```
