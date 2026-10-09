"""
The cell columns by direct summation over a cell's nodes (notebooks/domains/basis.direct_columns) against
the global transform (depth_models.slab_vis_coeffs) for a small patch of 0.1 deg nodes; and the direct G
against synthesis of those coefficients.

Run with:  conda run -n pygmt17 python -m pytest tests/test_direct_columns.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
import depth_models as dm                                    # noqa: E402
from domains import basis                                    # noqa: E402

LMAX = 40


def test_direct_columns_equal_global_transform():
    ocean = basis.ocean_grid()[0]
    glat = np.asarray(ocean.lat)
    cell = -np.ones((len(glat), len(np.asarray(ocean.lon))), dtype=int)
    cell[1100:1104, 1300:1305] = 0                           # 4 x 5 nodes near 20 S, 130 E
    S = basis.direct_setup(ocean, 30.0, LMAX)
    gh = basis.direct_columns(cell, [0], S)
    ref = dm.slab_vis_coeffs(np.where(cell == 0, 1., 0.), S['r_top'], S['r_base'],
                             dm.induced_to_gmm(np.asarray(ocean.lon), glat), lmax=LMAX).coeffs
    refp = np.stack([ref[0, S['L'], S['M']], ref[1, S['L'], S['M']]])
    assert np.abs(gh[:, 0] - refp).max() < 1e-11*np.abs(refp).max()

    # direct G at a few points = Br synthesis of the same coefficients (br_points)
    plon, plat = np.array([129., 131.5, 133.]), np.array([-19., -20.5, -24.])
    Pc, Ps = basis.synthesis_parts(plon, plat, 16, LMAX)
    G = Pc @ gh[0, 0] + Ps @ gh[1, 0]
    c = np.zeros((2, LMAX+1, LMAX+1))
    c[0, S['L'], S['M']], c[1, S['L'], S['M']] = gh[0, 0], gh[1, 0]
    assert np.allclose(G, basis.br_points(c, plon, plat, 16, LMAX), rtol=1e-10, atol=1e-12)
