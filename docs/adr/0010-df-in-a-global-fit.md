# DF in a global fit: shared strengths, evidence maximised at 100 km cells

DF-GLC keeps DF's model (ADR 0007): a mean per domain, plus an independent fractal pattern inside each domain with a shared exponent and patch size, as a dense covariance in one block per domain. The domain means follow ADR 0008. It differs from the local DF in three ways:

1. **Strengths are shared.** A strength per domain would mean ~750 hyperparameters at about a minute per evaluation. The strength is shared by all domains, or by all domains of one CR class (3 classes); a finer class set comes only if 3 classes clearly beat one.
2. **No sampling.** The hyperparameters maximise the evidence, with error bars from its curvature. MCMC at the full size would take days.
3. **100 km cells (NSIDE 64) for results.** At NSIDE 32 the cells (~200 km) approach the shortest wavelengths in the data (degree 133, ~300 km), so the evidence would charge the coarse cells' misfit to the model. Small provinces would also lose their cells (as in the ADR 0006 amendment). NSIDE 32 is used only for smoke tests.

Soft boundaries (a shared global pattern plus the domain patterns, with the split fitted) stay in the same model family, and are deferred until sharp boundaries have been tested.

## Considered Options

- A GMRF with links cut at domain boundaries: sparse and cheaper to build, but only roughly fractal, so comparisons with local DF results would mix two changes. The block-diagonal dense covariance costs about the same per evaluation, because the blocks are independent.
- MCMC at NSIDE 32: feasible (~1 s per evaluation), but rejected for the coarse-cell misfit above.
