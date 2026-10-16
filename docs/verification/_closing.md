# Earlier checks {#sec-earlier}

Two earlier benchmarks (September–October 2026) check remit coefficient by coefficient, and are still valid:

- **Benchmark 1** (`docs/benchmark-dipole-sum.md`): the same dipole sum at 3072 points, for the real-geology
  cases. It first confirmed the fixes to remit's amplitudes (the √2 of the order m > 0 coefficients and the
  induced units; `docs/review-2026-09-units-and-geometry.md`).
- **Benchmark 2** (`docs/benchmark-sh-quadrature.md`): every Gauss coefficient to degree 400 against an
  independent quadrature of the potential integral, plus closed-form cases. They agree to 2.5×10⁻¹³ per degree.
  This is the coefficient-level counterpart of the maps here.

# Found along the way {#sec-found}

- **Cell columns crashed for cells at the north pole.** `remit.inversion.columns` asked pyshtools for Legendre
  derivatives on the pole row, where they are undefined. That row carries zero weight in the sums, so the fix
  gives it zero Legendre rows. Production runs were not affected, because the north-pole node has an ocean age
  and is never continental crust. A target area there (or a different age grid) would have failed.
- **Rounding next to the poles is larger and coherent**, about 2×10⁻¹² of the field rather than 10⁻¹⁴, because
  the grid weights of the rows next to the poles are tiny numbers (@sec-dipoles). Harmless.

# Not covered yet

- The inversions themselves: recovering known truths from synthetic data. This is the next part.
- The thin-layer source model itself, against volume sources such as magnetic tesseroids.
- Altitudes below 100 km. The inversions evaluate degrees 16–133 at the surface, which these checks cover through
  the coefficients.

# Rerunning the checks {#sec-rerun}

The code is in `benchmarks/verification/`:

| File | What it does |
|---|---|
| `reference.py` | The dipole sum (numba, 3 threads by default; `VERIFY_THREADS`), Driscoll–Healy grids, map synthesis |
| `models.py` | The test models |
| `forward.py` | remit's transform against the dipole sum: about 1.5 h for all cases |
| `columns.py` | The cell columns against the dipole sum: about 1.5 h |
| `tests/test_verification.py` | Fast versions for pytest (10 s): planted dipoles, checkerboards, random magnetisation and the uniform shell on a 2° grid, observed 3000 km up where remit to degree 89 must match the dipole sum exactly |

From the repository root, in the `curiesphere` environment:

```bash
export PYTHONPATH=benchmarks/verification:benchmarks:notebooks:$PYTHONPATH OMP_NUM_THREADS=3
nice -n 15 conda run -n curiesphere python benchmarks/verification/forward.py     # or name cases
nice -n 15 conda run -n curiesphere python benchmarks/verification/columns.py
conda run -n curiesphere python docs/verification/make_figures.py
conda run -n riftsclimate quarto render docs/verification/verification.qmd
conda run -n curiesphere python -m pytest tests/test_verification.py
```

Every dipole sum and transform is cached in `benchmarks/verification/cache/`, so a rerun only recomputes what is
missing; delete a case's cache files to recompute it. Results (NetCDF maps and CSV summaries) go to
`benchmarks/verification/results/`; only the CSV files are kept in git.
