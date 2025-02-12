# curiesphere
Python code to perform forward modelling of lithospheric magnetization using vector spherical harmonics

[![DOI](https://zenodo.org/badge/859125725.svg)](https://doi.org/10.5281/zenodo.14854132)


## Summary
This repository contains a python implementation of the method originally described in:
```
"Analysis of lithospheric magnetization in vector spherical harmonics"
Gubbins et al, 2011, Geophysical Journal International
```

This version was used to create the results in:
```
"Magnetization of oceanic lithosphere from modelling of satellite observations"
Williams et al, 2025, submitted to Journal of Geophysical Research
```

The python interface allows creation of global magnetization models from inputs defined on regular lat-long grids. Included in the repository are input data required to generate results for Earth using global susceptibility models for the continents (Hemant and Maus, 2005) and subduction zones (Williams and Gubbins, 2019) and models for the remanent magnetization of the oceans (Williams et al, submitted). The notebooks folder contains code necessary to reproduce the analysis of Williams et al (submitted).

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
