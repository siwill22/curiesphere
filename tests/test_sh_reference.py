"""
remit's forward transform against an independent quadrature of the potential
integral (benchmarks/sh_reference.py), coefficient by coefficient, on a coarse grid.

Run with:  conda run -n pygmt17 python -m pytest tests
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
from sh_reference import gauss_coeffs, gauss_coeffs_at_radius, _legendre
from depth_models import depth_weighted_coeffs, n_terms
from remit.vhtools import GlobalMagnetizationModel
from test_forward_transform import lon, lat, TH, PH, L, M0, mu0, a

ZERO = np.zeros_like(TH)


def remit(mr, mt, mp):
    _, c = GlobalMagnetizationModel(lon, lat, mr, mt, mp, a).transform(lmax=L)
    return c.coeffs


def harmonic(l, m):
    """S_lm = P_lm cos m phi and its horizontal gradient on the unit sphere (pole row 0)"""
    S, dS_th, dS_ph = np.zeros_like(TH), np.zeros_like(TH), np.zeros_like(TH)
    k = l*(l + 1)//2 + m
    for i in range(1, TH.shape[0]):
        p, dp = _legendre(l, TH[i, 0])
        S[i] = p[k]*np.cos(m*PH[i])
        dS_th[i] = dp[k]*np.cos(m*PH[i])
        dS_ph[i] = -m*p[k]*np.sin(m*PH[i])/np.sin(TH[i, 0])
    return S, dS_th, dS_ph


@pytest.mark.parametrize('l, m', [(1, 0), (4, 3), (17, 9), (30, 30)])
def test_closed_forms_tangential(l, m):
    S, dth, dph = harmonic(l, m)
    poloidal = (ZERO, M0*dth, M0*dph)                 # M0 grad_h S_lm
    toroidal = (ZERO, -M0*dph, M0*dth)                # M0 r_hat x grad_h S_lm
    expected = mu0*M0*l*(l + 1)/((2*l + 1)*a)*1e9
    for fn in (remit, lambda *v: gauss_coeffs(lat, lon, *v, a, L)):
        c = fn(*poloidal)
        assert c[0, l, m] == pytest.approx(expected, rel=1e-10)
        c[0, l, m] = 0
        assert np.abs(c).max() < 1e-10*expected
        assert np.abs(fn(*toroidal)).max() < 1e-10*expected


def test_matches_reference_for_random_magnetization():
    # not band-limited: both evaluate the same quadrature on the same samples
    rng = np.random.default_rng(7)
    m = [M0*rng.normal(size=TH.shape) for _ in range(3)]
    for v in m:
        v[0] = v[0, 0]                                # one value at the pole
    ref = gauss_coeffs(lat, lon, *m, a, L)
    assert np.abs(remit(*m) - ref).max() < 1e-13*np.abs(ref).max()


def test_depth_series_matches_exact_radius_factor():
    rng = np.random.default_rng(8)
    m = [M0*rng.normal(size=TH.shape) for _ in range(3)]
    rho = 1 + 0.004*rng.uniform(-1, 1, size=TH.shape)  # sources within ~25 km of r0
    u = np.log(rho)
    to_gmm = lambda w: GlobalMagnetizationModel(lon, lat, *[w*v for v in m], a)
    series = depth_weighted_coeffs([(np.ones_like(TH), u)], to_gmm, L, n_terms(np.abs(u).max(), L)).coeffs
    exact = gauss_coeffs_at_radius(lat, lon, *m, rho, a, L)
    assert np.abs(series - exact).max() < 1e-13*np.abs(exact).max()
