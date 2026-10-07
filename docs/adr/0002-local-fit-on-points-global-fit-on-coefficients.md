# Local fits use Br at points, global fits use Gauss coefficients

An inversion either holds everything outside its target area fixed (a local fit: Hemant & Maus on other continents allowed, never inside the target area) or estimates outer cells on all other continental crust (a global fit: only the ocean model fixed). We pair each with one data form: local fits use Br at data points within a buffer of the target area, global fits use the field model's Gauss coefficients. In a coefficient fit a fixed Hemant & Maus background leaves its misfit worldwide in the residual, inflating the fitted noise and the target-area uncertainties; points near the target keep that out. A global fit has no such misfit to exclude, and in coefficients its isotropic noise is diagonal and cheap. Columns, error spectra, the filter and the estimators are shared by both.

## Considered Options

- Coefficients only (every problem global): rejected as the only option, because a fixed background's misfit elsewhere would set the noise level for the target area.
- Points only: rejected for global problems, where dense point covariances are expensive and the coefficient form is exact and diagonal.
