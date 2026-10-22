"""
Fast versions of the forward-model verification (benchmarks/verification, docs/verification): remit's transform
against the direct dipole sum on a coarse 2 deg grid.

remit's coefficients are the exact coefficients of the grid's node dipoles at every degree it computes, so remit
to degree 89 must equal the dipole sum wherever the field above degree 89 is negligible. 3000 km above the sphere
the field above degree 89 is attenuated by (6371/9371)^90 = 1e-15, so the two must agree to rounding for any
magnetisation, band-limited or not.

Run with:  conda run -n curiesphere python -m pytest tests/test_verification.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks', 'verification'))
import models                                                       # noqa: E402
from reference import R0, DipoleSum, dh_grid, obs_grid, remit_br    # noqa: E402
from remit.vhtools import GlobalMagnetizationModel                  # noqa: E402

G = dh_grid(90)
OBS = obs_grid(10.)
H = [3000e3]
L = 89


def both(vim):
    ds = DipoleSum(OBS, H)
    ds.add(G, *vim, R0)
    c = GlobalMagnetizationModel(G.lon, G.lat, *vim, R0).transform(lmax=L)[1].coeffs
    return ds.br()[0], remit_br(c, OBS, H, L)[0]


def planted(lat, lon, u):
    vim = [np.zeros(G.colat.shape) for _ in range(3)]
    i, j = int(np.argmin(np.abs(G.lat - lat))), int(np.argmin(np.abs(G.lon - lon)))
    for k in range(3):
        vim[k][i, j] = 1e3*u[k]
    return vim


@pytest.mark.parametrize('lat, lon, u', [
    (0., 0., (1, 0, 0)),          # radial, on the lon 0 seam
    (30., 90., (0, -1, 0)),       # north
    (-30., 180., (0, 0, 1)),      # east, on the dateline
    (88., 45., (0, 0, 1)),        # east, the row next to the north pole
    (-88., 270., (0, -1, 0)),     # north, the last row (next to the south pole)
])
def test_planted_dipole(lat, lon, u):
    dip, rem = both(planted(lat, lon, u))
    assert np.abs(rem - dip).max() < 1e-11*np.abs(dip).max()


@pytest.mark.parametrize('pattern', ['checkerboard', 'healpix', 'random'])
def test_sharp_patterns(pattern):
    if pattern == 'random':
        rng = np.random.default_rng(1)
        vim = [rng.normal(size=G.colat.shape)*40. for _ in range(3)]
    else:
        vis = models.checkerboard(G) if pattern == 'checkerboard' else models.healpix_tiles(G)
        vim = models.induced(G, vis)
    dip, rem = both(vim)
    assert np.abs(rem - dip).max() < 1e-11*np.abs(dip).max()


def test_uniform_shell_has_no_external_field():
    """uniform VIS induced by IGRF-13 (an internal field): Runcorn's theorem, no field outside"""
    dip, rem = both(models.induced(G, np.ones(G.colat.shape)))
    ref = np.abs(both(models.induced(G, models.checkerboard(G)))[0]).max()
    assert np.abs(rem).max() < 1e-12*ref
    assert np.abs(dip).max() < 1e-11*ref
