# Inversion with the VSH forward model: ideas and issues

*Notes from a discussion on 2026-10-01, written up as is so they can be picked up later. Nothing here has been implemented. The references are from memory; check them before citing.*

---

## 1. Feasibility in general

An inversion using the vector-spherical-harmonic (VSH) forward model is feasible. The forward model is linear and fast, which is what an inversion needs. The hard part is not computation but non-uniqueness, and the VSH formulation is useful precisely because it states that non-uniqueness exactly.

**What makes it feasible**

- **Linearity.** The field depends linearly on the magnetisation (VIM), so the inverse problem is linear. The depth series keeps it linear in magnetisation for a fixed source geometry.
- **Speed.** One forward transform takes about 2 s at lmax 185. An iterative least-squares solver (CG or LSQR) with hundreds of iterations therefore takes minutes.
- **What is missing.** It needs the **adjoint** of `forward_transform`, which remit does not have. `inverse_transform` is not the adjoint, because the quadrature weights differ. The adjoint is straightforward to write and should be verified with the dot-product test.
- **The forward map is diagonal in the right basis.** In the VSH basis, each Gauss coefficient depends only on the internal coefficient I_lm of the same degree and order. That gives an exact separation between what the data see and what they cannot see.

**Main issues**

1. **Annihilators, the central problem.** Of the three VSH components of a magnetisation (internal I, external E, toroidal T), only I produces an external field. Roughly two-thirds of a general vector magnetisation is therefore invisible, and an unconstrained vector inversion is hopeless. Prior structure is needed:
   - **Remanence:** the direction is known from age and the palaeopole, so only a scalar magnitude is inverted for. Annihilators remain, because the direction varies over the globe.
   - **Induced magnetisation:** its direction follows the present field. It has its own annihilators: by Runcorn's theorem, a uniform susceptibility in a dipole field is invisible (Maus & Haak 2003 discuss the general case). So the mean susceptibility level of a region is poorly constrained, which is exactly the amplitude question.
   - **Remanent versus induced:** in the oceans the two trade off unless their spatial patterns differ. They mostly do (stripes against smooth).
2. **The lost long wavelengths.** Degrees below about 16 are hidden by the core field. Any long-wavelength magnetisation (regional VIS levels, the superchron, broad age trends) is unconstrained and must come from the prior.
3. **Depth and amplitude trade off.** Source depth and spectral slope substitute for each other. This is the classic depth ambiguity of potential fields; the degree-dependent amplitude tilt found against LCS-1 is an instance of it. Depth has to be fixed (bathymetry + sediments) or given strong priors.
4. **Using a model as data.** Inverting LCS-1 or MF7 coefficients means inheriting their damping (the F(l) question) and their unstated, correlated errors. Inverting along-track data avoids that, but requires synthesis at about 10⁶ points per iteration (feasible) and taking on the data processing (core field, ionosphere, magnetosphere).
5. **Underdetermination.** A 0.1° grid has about 6.5M unknowns against about 34k coefficients at lmax 185. Regularisation decides the answer: minimum-norm departure from a prior, smoothness, or L1 as in LCS-1. Results must state that choice, ideally with resolution kernels.
6. **Regional versus global.** Global harmonics with regional weighting leak power. Slepian localisation (as in the paper) or a regional misfit with a global forward model deals with this, at some cost in complexity.

**Three formulations, from easy to hard**

- **(a) Correct only the visible part of a prior model.** Replace the prior's I_lm with values that fit the data, degree by degree. This is trivial and unique, but the correction has no physically constrained direction, so it is a diagnostic rather than a geological model.
- **(b) Scalar magnitude with known direction, on a grid.** For example, a remanence scale factor on a coarse grid plus regional VIS adjustments, solved iteratively with forward and adjoint operators and regularisation. This is the real inversion: feasible, but it needs the adjoint and careful annihilator analysis.
- **(c) Low-dimensional parametric inversion.** Extend the tuning scripts with layer magnetisations, P and λ, plus a smooth spatial scale field (low-degree SH, or age and region bins). It is linear, well conditioned and interpretable, and almost everything needed already exists.

The intention is to do both (b) and (c).

---

## 2. Science questions

### Q1. Ocean remanence amplitude as a function of space (directions known)

**Formulation.** VIM_rem(x) = s(x) · VIM_prior(x). The prior (for example GK07_NR) fixes direction, polarity pattern and depth profile; s(x) is the amplitude field to find. The problem is linear in s.
- **(c):** s on a coarse basis, such as low-degree SH or age-band and region bins.
- **(b):** s on a grid, with smoothness regularisation and positivity (s ≥ 0).

**What is resolvable**

- **Stripe modulation (good news).** A smooth s(x) modulates the stripes, so its effect appears at stripe wavelengths (about degree 40–150), which the data see. s should be resolvable at a few hundred to about 1000 km scale, even though its own wavelength is long.
- **Quiet zones (KQZ, JQZ).** With no reversals and uniform polarity, s behaves like a uniform induced layer: large annihilators and the loss of degrees below 16. Expect s to be poorly constrained there.
- **Trade-off with oceanic VIS.** Fix the oceanic induced part, or invert it jointly with a much smoother parameterisation.
- **Trade-off with mixing and depth.** A smooth s cannot produce a degree-dependent tilt. If the real cause is reduced short-wavelength coherence (largest at slow ridges with narrow stripes), the inversion will map it into lower s in those regions. This is an interpretation trap, not a computational problem.

### Q2. Continental domains from polygons (e.g. cratons against mobile belts)

**Formulation.** One parameter per domain: VIM = Σ_k c_k · 1_k(x) · (something). Two choices answer different questions:

- **(i) Uniform VIS per domain** (piecewise-constant susceptibility × thickness). This is the cleanest test of whether cratons have a different mean level, but it runs into Runcorn's theorem.
  - A domain's uniform interior produces almost no external field. The signal comes from its edges and from the non-dipole part of the inducing field.
  - The data therefore constrain contrasts between neighbouring domains much better than absolute levels.
  - Continent–ocean margins help tie the absolute level, because oceanic crust's VIS is relatively well known and the crust is thin.
  - With degrees below 16 lost, very large domains lose their mean entirely.
- **(ii) A scale factor per domain multiplying the H&M VIS.** This keeps the internal structure, which provides signal from inside the domain. It is better determined, but it answers whether internal variability is stronger, not whether the mean level is higher.
- **Recommendation:** fit both, and compare.

**Practical points**

- **Problem size.** Tens to hundreds of domains is a classic (c) problem. Build the full G matrix (one forward transform per domain, a few minutes in total) and solve directly. That gives the posterior covariance and resolution matrix. The singular values of G show directly which domains, or which domain contrasts, are unresolved.
- **Continental remanence** (Bangui, Kursk and others) is not in the model and will bias neighbouring domains. Use a robust (L1 or Huber) misfit, or mask known remanent anomalies.
- **Rasterising the polygons.** Polygons go onto the grid with pygplates point-in-polygon, as for the ocean masks.

### Q3. Thickness of the magnetic layer: is there information in the data, or do geological variations dominate?

- **In the thin-shell limit, thickness cannot be separated from magnetisation.** The field depends only on VIM = M × t, so a "thickness" inversion with fixed susceptibility is really a VIS inversion, relabelled.
- **Thickness information comes only through the depth of the layer base.** A layer from z_top to z_bot has a field that scales as (e^{−k·z_top} − e^{−k·z_bot})/k. This separates from M·t only when k·t ≳ 1, i.e. at wavelengths shorter than about 2π·t.
  - For t of about 30 km that means wavelengths below about 200 km (degree above about 200).
  - Those wavelengths are strongly attenuated at satellite altitude, and LCS-1 is damped there (it is unconstrained above about degree 150–185).
- **Expectation.** The data probably contain little independent thickness information, and susceptibility variations will dominate. That is the substance of the criticism of satellite-based Curie-depth and heat-flow estimates, the best known being Fox Maule et al. (2005) for Antarctica.

**A clean test, with no inversion needed first**

1. Build pairs of models with identical VIM: one with t varying at constant χ, one with χ varying at constant t. Place each on depth-resolved slices; the depth series in `notebooks/depth_models.py` handles this exactly.
2. Compute the field difference between each pair, degree by degree, at the LCS-1 altitudes and in its band.
3. Compare that difference with the LCS-1 uncertainty. Estimate the uncertainty empirically, for example as LCS-1 − MF7 per degree.
4. If the difference is below the uncertainty, the data cannot see thickness, whatever an inversion returns.

---

## 3. Shared infrastructure

- **For (c), a basis-column operator.** One forward run per parameter, cached as in `notebooks/tune_ocean_model.py`. Solve by direct least squares, with covariance, resolution and synthetic recovery tests.
- **For (b), an adjoint** of `forward_transform` and of the depth series, verified by the dot-product test, plus an iterative solver (LSQR, or projected gradient for positivity).
- **Data target.** One of:
  - LCS-1 coefficients in a degree band, with F(l) applied to the model;
  - LCS-1 Br at equal-area points, which makes regional weighting easy.
- **Error model.** Needed for either target. LCS-1 provides no per-coefficient errors, so differences such as LCS-1 − MF7 or LCS-1 − Swarm MLI 2D are the practical choice.
- **Validation before real data.** Checkerboard and recovery tests for each problem, and an annihilator analysis (null-space projections) to state what the data cannot constrain.

## 4. Decisions left open

1. **Order.** Suggested: the Q3 thickness-sensitivity test first (cheap, and it decides whether Q3 is worth an inversion), then Q2 with formulation (c), then Q1, starting with (c) and building the adjoint for (b).
2. **Data target.** LCS-1 coefficients or gridded LCS-1, and which model (MF7 or Swarm) to use for the error estimate.
3. **Polygons for Q2.** Which domain data set to use (for example, a global tectonic-province compilation).
4. **Regularisation for (b).** Minimum departure from the prior, smoothness, or sparsity (L1).

## Related

- `docs/review-2026-09-units-and-geometry.md`, §14: source depth, the LCS-1 filter, and near-ridge tuning.
- `docs/benchmark-dipole-sum.md`: remit's forward model checked against an independent dipole sum.
