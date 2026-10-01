# Benchmark: remit forward models against a direct dipole sum

*2026-09-30, branch `fix/forward-amplitudes`. Script: `benchmarks/dipole_sum.py`. Results: `benchmarks/results/dipole_sum.csv` and `benchmarks/results/dipole_sum.png`.*

## Purpose

This benchmark checks remit's vector-spherical-harmonic forward model, including the fixes A–C of `docs/review-2026-09-units-and-geometry.md` and the depth series of `notebooks/depth_models.py`. The check uses an independent equivalent-source calculation: one point dipole per grid node, with the field summed directly in spherical coordinates. The dipole calculation shares no code with remit's transform.

## Method

**Dipole side.** The field is

    B(x) = μ0/4π Σ_j [3(m_j·R)R/|R|⁵ − m_j/|R|³],  with R = x − s_j.

- There is one dipole per node of the 1800 × 3600 Driscoll–Healy grid (0.1°).
- Each moment is m_j = VIM_j · r_s² · w_j · Δλ. Here w_j are the Driscoll & Healy (1994) latitude weights, computed from their formula in the script.
- The sum runs in float64 with numba, at about 1.5×10⁹ dipole–observer pairs per second on 12 cores.

**Inputs.**
- The remanent VIM is taken from remit (`SeafloorGrid.vim` and `depth_models.remanent_to_gmm`), because it is the model definition rather than the code under test.
- The induced VIM is built independently, as VIS [SI·km] × 10³ × B_IGRF [nT] × 10⁻⁹/μ0.

**Observation points.**
- HEALPix NSIDE 16: 3072 equal-area points, including the polar caps.
- Altitudes 100 and 300 km above r0 = 6371.0 km.
- Components Br, Bθ and Bφ.
- All degrees from 1 upwards (no lmin).

**Metric.** RMS and maximum of (remit − dipole sum), each divided by the RMS of the dipole-sum field.

## Cases and results

| Case | What is tested | Result |
|---|---|---|
| 0. Induced units | remit's induced VIM grid against the one built independently (fix B) | Max relative difference **3×10⁻¹⁶** |
| 1. Exact, band-limited | GK07 + VIS on the r0 sphere, with the magnetisation low-passed to degree 149 in Cartesian components. The external field is then band-limited to degree 150, so remit at lmax 150 should reproduce the dipole sum exactly. Tests `forward_transform`, fixes A and C, and all three components. | RMS **1–3×10⁻¹³**, max **≤ 7×10⁻¹²**, at both altitudes and in all components |
| 2. Real inputs, sphere | GK07 remanent, induced and combined, unfiltered. remit at lmax 150, 300 and 400. | Differences equal remit's own truncation (next table) |
| 3. Real inputs, depth | GK07_NR with realistic source depth. There are 13 remanent and 14 induced slices of about 500 m, identical for both methods. Tests `depth_weighted_coeffs`. | Differences equal remit's own truncation (next table) |

For cases 2 and 3 the inputs contain all wavelengths the 0.1° grid can hold, while remit stops at lmax. The remit − dipole difference at lmax L is therefore compared with the field that remit itself puts into degrees L+1 to 400 (the "truncation").

Combined model, Br, RMS relative values:

| Case | Altitude | lmax | remit − dipole | remit degrees L+1..400 |
|---|---|---|---|---|
| 2 sphere | 300 km | 150 | 3.4×10⁻³ | 3.4×10⁻³ |
| | | 300 | 4.2×10⁻⁶ | 4.2×10⁻⁶ |
| | | 400 | 4.9×10⁻⁸ | — |
| | 100 km | 150 | 0.19 | 0.19 |
| | | 300 | 2.3×10⁻² | 2.3×10⁻² |
| | | 400 | 5.7×10⁻³ | — |
| 3 depth | 300 km | 150 | 2.9×10⁻³ | 2.9×10⁻³ |
| | | 300 | 3.1×10⁻⁶ | 3.1×10⁻⁶ |
| | | 400 | 3.1×10⁻⁸ | — |
| | 100 km | 150 | 0.17 | 0.17 |
| | | 300 | 1.8×10⁻² | 1.7×10⁻² |
| | | 400 | 3.7×10⁻³ | — |

The remanent and induced parts and the other two components behave the same way (see the CSV).

**What remains at lmax 400 is the field above degree 400.** From lmax 300 to 400, the residual falls by 4× at 100 km and by 86× at 300 km. The ratio of those two factors is 21, which is exactly the extra upward-continuation attenuation between the two altitudes over 100 degrees, ((6371+100)/(6371+300))¹⁰⁰ = 1/21.

The depth series is consistent across lmax: a separate run at lmax 150 equals the lmax-400 run truncated to degree 150, to within 2×10⁻¹⁷.

## Conclusions

- **The code is exact to rounding.** remit's forward transform (with fixes A–C), the induced-unit conversion (fix B) and the depth series all agree with an independent dipole sum to rounding, 10⁻¹³, whenever the comparison is exact (case 1). On real inputs they agree to the level of spectral truncation (cases 2 and 3). No residual is left that would point to an error.
- **Truncation errors on real inputs.** For the full-resolution age and VIS grids, stopping at degree 150 changes Br by 0.3% RMS at 300 km and by 17–19% RMS at 100 km. Stopping at 185, as in the paper figures, falls between these.
- **Quadrature of the reference method.** With plain sin θ Δθ Δλ weights, the dipole sum itself is only accurate to about 10⁻⁵ RMS and 10⁻³ maximum (case 1 is also run this way). The Driscoll–Healy weights remove that error.

## Found along the way

- **Bug fixed in `depth_weighted_coeffs`.** Above about 20 series terms, `factorial(k)` overflows numpy's int64 and produces an object array. Only lmax ≥ about 390 needs that many terms, so no earlier result (lmax ≤ 185) was affected.
- **VIS precision.** The VIS grid is stored as float32. The benchmark converts it to float64 before building the independent induced VIM. remit's own product is float64 because the IGRF grid is float64.

## Not covered

- The thin-shell approximation itself: magnetic tesseroids were ruled out for now.
- An independent spherical-harmonic quadrature, coefficient by coefficient: see `docs/benchmark-sh-quadrature.md` (benchmark 2).
- Altitudes below 100 km.
- The latitude registration (review item D).
