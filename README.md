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
To reproduce the figures and numbers of that paper, use release v1.0.0.

## Changes since v1.0.0
v2.0.0 corrects three errors in the forward transform and the induced magnetization, which change the amplitudes of all forward models (details and tests in `docs/review-2026-09-units-and-geometry.md` and `tests/`):
- Gauss coefficients of order m > 0 were a factor sqrt(2) too small.
- Induced magnetization used the field B in nT where H = B/mu0 is required, so the induced part was about 1.26x too strong.
- The latitude quadrature leaked a small amount of power into zonal terms; it now uses exact Driscoll-Healy weights, and `forward_transform` raises `ValueError` for grids that are not Driscoll-Healy (see `remit.utils.grid.DH2`).

Together these increase forward-model Br by roughly 1.2-1.4x relative to v1.0.0, with pattern correlations almost unchanged.

The notebooks also add optional extensions (not used by the paper notebooks unless selected):
- `notebooks/depth_models.py`: forward models with realistic source depth (WGS84 ellipsoid, bathymetry and sediments) and an approximate LCS-1 resolution filter.
- `GK07_NR` in `notebooks/basis_models.py`: GK07 with the near-ridge enhancement retuned against LCS-1 (P = 0.94, lambda = 3 Ma), from `notebooks/tune_near_ridge.py`.

The python interface allows creation of global magnetization models from inputs defined on regular lat-long grids. Included in the repository are input data required to generate results for Earth using global susceptibility models for the continents (Hemant and Maus, 2005) and subduction zones (Williams and Gubbins, 2019) and models for the remanent magnetization of the oceans (Williams et al, 2025). The notebooks folder contains code necessary to reproduce the analysis of Williams et al (2025).

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
