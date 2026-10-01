# Benchmark 2: remit's Gauss coefficients against an independent quadrature

*2026-10-01, branch `fix/forward-amplitudes`.*

**Files**
- Reference implementation: `benchmarks/sh_reference.py`
- Runner: `benchmarks/sh_quadrature.py`
- Coarse unit test: `tests/test_sh_reference.py`
- Results: `benchmarks/results/sh_quadrature.csv`, `sh_quadrature_per_degree.csv` and `sh_quadrature.png`

This complements the dipole-sum benchmark (`docs/benchmark-dipole-sum.md`):

| | Benchmark 1: dipole sum | Benchmark 2: independent quadrature |
|---|---|---|
| Compared | the field at points | every Gauss coefficient g_lm, h_lm |
| Limitation | truncation, except for band-limited input | none: real, unfiltered inputs give an exact comparison |
| Errors that depend on (l, m) | not located | located directly |

## Method

For VIM **M** per unit area on a shell at radius r_s = ρa, the external potential is V = μ0/4π ∫ **M**·∇′(1/|r − r′|) dS′. Expanding with the Schmidt addition theorem and matching to the Gauss form gives

    g_lm = μ0/(4πa) ∫ ρ^(l+1) [ l M_r P_lm cos mφ + M_θ ∂θP_lm cos mφ − M_φ (m/sinθ) P_lm sin mφ ] dΩ
    h_lm = μ0/(4πa) ∫ ρ^(l+1) [ l M_r P_lm sin mφ + M_θ ∂θP_lm sin mφ + M_φ (m/sinθ) P_lm cos mφ ] dΩ

This form has four exact properties:
- a radial VIM M0·S_lm gives g = μ0 M0 l/((2l+1)a);
- a poloidal tangential VIM M0·∇_h S_lm gives μ0 M0 l(l+1)/((2l+1)a);
- a toroidal tangential VIM gives 0;
- a magnetisation proportional to an internal potential field gives 0 (Runcorn's theorem).

**What is independent of remit**
- The integral form itself. remit uses the internal/external/toroidal decomposition of Gubbins et al. (2011), which is different algebra.
- Longitude sums: explicit cos mφ and sin mφ products, where remit uses an FFT.
- Latitude quadrature: Driscoll & Healy (1994) weights computed from their formula.
- For variable radius, the factor ρ^(l+1) is applied exactly at each node, where remit's depth series expands it as a power series.

**What is shared with remit**
- The Legendre functions, from pyshtools `PlmSchmidt_d1`. remit is under test, not pyshtools.
- The input VIM grids, which are the model definition.

The induced VIM is built independently, from VIS × IGRF/μ0, as in benchmark 1.

## Results

| Case | Agreement |
|---|---|
| 1. Closed forms: radial, poloidal and toroidal VIM at (l, m) = (150, 77), (185, 185) and (400, 250) | **remit** and the **reference** both reproduce the analytic values: radial to 3–5×10⁻¹⁵, poloidal to 1×10⁻¹³ – 1×10⁻¹². Both give zero for the toroidal case, and every other coefficient is ≤ 10⁻¹¹ of the radial value. |
| 2. Real inputs on the r0 sphere: GK07 remanent, induced and combined, lmax 400 | Per-degree RMS difference **≤ 2.5×10⁻¹³**. Worst single coefficient ≤ 2.5×10⁻¹² of its degree's RMS. |
| 3. Realistic depth: GK07_NR, 27 slices of 500 m, lmax 185. remit's depth series against the exact per-node factor | Per-degree RMS difference **≤ 9×10⁻¹⁴**. Worst single coefficient ≤ 9×10⁻¹³. |
| Unit test (lmax 30): random, not band-limited three-component VIM; closed forms; random source radii | Agreement to 8×10⁻¹⁵ (VIM) and 7×10⁻¹⁵ (depth); tests set at 10⁻¹³ |

Every comparison agrees to rounding error, at every degree and order. remit's forward transform (fixes A–C), its handling of the tangential components, the induced units (fix B) and the depth series all agree with the independent quadrature for real inputs, with no truncation caveat.

**Small structure at rounding level.** The per-degree difference rises slowly with l, from about 10⁻¹⁵ to about 10⁻¹³. It also steps up by about 3× near l ≈ 155, and the (l, m) map shows faint bands at particular orders (around m ≈ 160, 270 and 320). This is three orders of magnitude below anything physically relevant. The cause hasn't been traced; candidates are pyshtools' Legendre scaling regimes, or the different rounding of an FFT and explicit sums.

## Not covered

- The thin-shell approximation itself (tesseroids, set aside).
- The production 100 m slices: the depth case uses the same 500 m slices as benchmark 1. The code path is the same, with a different slice list.
- Degrees above 400 for remit, which is limited by its memory use for Legendre tables. The reference itself works to the grid limit (lmax 899).
