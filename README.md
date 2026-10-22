# curiesphere
Python code to perform forward modelling of lithospheric magnetization using vector spherical harmonics

[![DOI](https://zenodo.org/badge/859125725.svg)](https://doi.org/10.5281/zenodo.14854132)


## Summary
This repository contains a python implementation of the method originally described in:
```
"Analysis of lithospheric magnetization in vector spherical harmonics"
Gubbins et al, 2011, Geophysical Journal International
```

Release v1.0.0 of this code was used to create the results in:
```
"Magnetization of oceanic lithosphere from modelling of satellite observations"
Williams et al, 2025, Journal of Geophysical Research

preprint here: https://doi.org/10.22541/essoar.174371621.15559464/v1
```

The python interface allows creation of global magnetization models from inputs defined on regular lat-long grids. Included in the repository are input data required to generate results for Earth using global susceptibility models for the continents (Hemant and Maus, 2005) and subduction zones (Williams and Gubbins, 2019) and models for the remanent magnetization of the oceans (Williams et al, 2025). The notebooks folder contains the code used for the analysis of Williams et al (2025).

## Changes in v2.0.0
- **Bug fixes.** Errors in the forward transform and in the units of the induced magnetization have been corrected (details in `docs/review-2026-09-units-and-geometry.md`).
- **Ocean depth.** Optional forward models now place the oceanic sources at a realistic depth: below the sea floor and sediments, on the WGS84 ellipsoid, with each layer of the oceanic lithosphere at its own depth (`notebooks/depth_models.py`).
- **Benchmarks.** The code has been benchmarked against an independent equivalent-source (dipole sum) calculation and an independent spherical-harmonic quadrature, and agrees with both to rounding error (`docs/benchmark-dipole-sum.md`, `docs/benchmark-sh-quadrature.md`).
- **Outputs differ from v1.0.0.** As a consequence, forward-model amplitudes are typically 10-15% higher than with release v1.0.0 (global RMS of Br); pattern correlations are almost unchanged. To reproduce the published figures and numbers of Williams et al (2025), use release v1.0.0 (https://doi.org/10.5281/zenodo.14854133).

## Inversion for crustal susceptibility
`remit.inversion` inverts a satellite field model (LCS-1, degrees 16-133) for VIS, the vertically integrated susceptibility of the crust, either over all continents (a global fit to the Gauss coefficients) or over one target area such as Australia (a local fit to field values). Five methods are verified:

- **DF** (domain fractal) and its global form **DF-GLC**: a level per domain plus a fractal pattern inside each domain;
- **CM** (cell map): a smooth random field about one level;
- **PD** (polygon domains): one value per geological polygon;
- **GCM** (global cell map): one value per cell, pulled toward one level.

Each run is set by a config in `inversions/`:
```
python -m remit.inversion run inversions/df_glc_lcs1_16-133_groups.toml
```
Design decisions are recorded in `docs/adr/`. The verification reports in `docs/verification/` (Quarto; start with `inversion-overview.qmd`) check that each method is coded correctly and show its resolution and how far its uncertainties can be trusted.

## Python Requirements
- numpy
- scipy
- matplotlib
- xarray
- pandas
- geopandas
- rasterio
- pyshtools
- pygmt
- astropy_healpix
- pygplates
- pytest (to run the tests: `python -m pytest tests`)
