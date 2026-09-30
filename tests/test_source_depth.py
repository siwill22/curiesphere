"""
Tests for the depth-resolved forward model (notebooks/depth_models.py).

Run with:  conda run -n pygmt17 python -m pytest tests
"""
import os
import sys

import numpy as np
import pyshtools
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
from depth_models import depth_weighted_coeffs, n_terms, lcs1_filter, ellipsoid_radius, R0
from remit.vhtools import GlobalMagnetizationModel
from test_forward_transform import schmidt, lon, lat, TH, PH, L, M0, mu0

V = np.array([0.3, -0.5, 0.8])


def unit_vectors(th, ph, axis=0):
    rh = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)], axis)
    thh = np.stack([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)], axis)
    phh = np.stack([-np.sin(ph), np.cos(ph), 0*ph], axis)
    return rh, thh, phh


def strength(th, ph):
    return M0*(1 + 0.5*schmidt(3, 2, th, ph, 0.7, -0.4) + 0.3*schmidt(8, 5, th, ph, -0.2, 0.9))


def directions(th, ph):
    return [np.tensordot(V, e, 1) for e in unit_vectors(th, ph)]


def to_gmm(w):
    return GlobalMagnetizationModel(lon, lat, *[w*d for d in directions(TH, PH)], R0)


def source_radius(th):
    # sources between 10 km above and 30 km below the reference sphere
    return R0 + 10e3 - 20e3*(1 - np.cos(th))


def test_variable_source_radius_matches_point_dipole_sum():
    ns = 400_000
    k = np.arange(ns) + 0.5
    sth = np.arccos(1 - 2*k/ns)
    sph = np.mod(np.pi*(1 + 5**0.5)*k, 2*np.pi)
    rs = source_radius(sth)
    rh, thh, phh = unit_vectors(sth, sph, 1)
    f = strength(sth, sph)
    mr, mt, mp = [f*d for d in directions(sth, sph)]
    # VIM is per unit area at the source radius
    moment = (mr[:, None]*rh + mt[:, None]*thh + mp[:, None]*phh)*4*np.pi*rs[:, None]**2/ns
    pos = rs[:, None]*rh

    rng = np.random.default_rng(1)
    r_obs = R0 + 300e3
    olat = np.degrees(np.arcsin(rng.uniform(-1, 1, 6)))
    olon = rng.uniform(0, 360, 6)
    ot, op = np.radians(90-olat), np.radians(olon)
    xo = r_obs*np.stack([np.sin(ot)*np.cos(op), np.sin(ot)*np.sin(op), np.cos(ot)], 1)
    br_direct = []
    for x in xo:
        R = x - pos
        Rn = np.linalg.norm(R, axis=1)
        B = mu0/(4*np.pi)*(3*(np.sum(moment*R, 1)/Rn**2)[:, None]*R - moment)/Rn[:, None]**3
        br_direct.append(B.sum(0) @ (x/r_obs)*1e9)

    u = np.log(source_radius(TH)/R0)
    coeffs = depth_weighted_coeffs([(strength(TH, PH), u)], to_gmm, L, n_terms(np.abs(u).max(), L))
    br = np.asarray(coeffs.expand(a=r_obs, lat=olat, lon=olon, lmax_calc=L))[:, 0]
    # the source radius is band-limited but strength*radius^(l+1) is not exactly,
    # so allow a slightly looser tolerance than the r0 test
    assert np.allclose(br, br_direct, rtol=1e-4, atol=1e-6*np.abs(br_direct).max())


def test_constant_depth_equals_transform_at_source_radius():
    rs = R0 - 12e3
    w = strength(TH, PH)
    u = np.full_like(TH, np.log(rs/R0))
    series = depth_weighted_coeffs([(w, u)], to_gmm, L, n_terms(abs(u[0, 0]), L))
    _, direct = GlobalMagnetizationModel(lon, lat, *[w*d for d in directions(TH, PH)], rs).transform(lmax=L)
    direct = direct.change_ref(r0=R0)
    assert np.allclose(series.coeffs, direct.coeffs, rtol=1e-10, atol=1e-12*np.abs(direct.coeffs).max())


def test_slices_add_linearly():
    w = strength(TH, PH)
    u1, u2 = np.full_like(TH, -1e-3), np.full_like(TH, -2e-3)
    n = n_terms(2e-3, L)
    both = depth_weighted_coeffs([(w, u1), (0.5*w, u2)], to_gmm, L, n)
    sep = (depth_weighted_coeffs([(w, u1)], to_gmm, L, n).coeffs
           + depth_weighted_coeffs([(0.5*w, u2)], to_gmm, L, n).coeffs)
    assert np.allclose(both.coeffs, sep, rtol=1e-12, atol=1e-14*np.abs(sep).max())


def test_filter_and_ellipsoid():
    assert lcs1_filter(150, 150) == pytest.approx(0.5)
    assert lcs1_filter(0, 150) == pytest.approx(1, abs=1e-5)
    assert ellipsoid_radius(0.) == pytest.approx(6378137.0)
    assert ellipsoid_radius(90.) == pytest.approx(6356752.314, abs=1e-3)
