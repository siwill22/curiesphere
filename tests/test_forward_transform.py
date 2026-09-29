"""
Regression tests for the thin-shell forward transform (remit.vhtools) and the
induced-magnetization units (remit.earthvim.GlobalVIS).

See docs/review-2026-09-units-and-geometry.md for the background to each test.
Run with:  conda run -n pygmt17 python -m pytest tests
"""
import numpy as np
import pyshtools
import pytest
from pyshtools.legendre import PlmSchmidt

from remit.vhtools import GlobalMagnetizationModel, inverse_transform
from remit.earthvim import GlobalVIS

mu0 = pyshtools.constants.mu0.value
a = 6371000.
L = 30
N = 2*(L+1)
lat = 90 - np.arange(N)*180/N           # DH2 grid: north pole in, south pole out
lon = np.arange(2*N)*360/(2*N)
LON, LAT = np.meshgrid(lon, lat)
TH, PH = np.radians(90-LAT), np.radians(LON)
ZERO = np.zeros_like(TH)
M0 = 1000.                               # VIM amplitude, A


def schmidt(l, m, th, ph, c=1., s=0.):
    """Schmidt semi-normalised surface harmonic P_lm(cos th)(c cos m ph + s sin m ph)"""
    th = np.asarray(th)
    p = np.array([PlmSchmidt(l, np.cos(t))[l*(l+1)//2+m] for t in th.ravel()]).reshape(th.shape)
    return p*(c*np.cos(m*ph) + s*np.sin(m*ph))


def transform(mr, mt=ZERO, mp=ZERO):
    _, c = GlobalMagnetizationModel(lon, lat, mr, mt, mp, a).transform(lmax=L)
    return c.coeffs


@pytest.mark.parametrize('l, m', [(1, 0), (3, 2), (20, 7), (30, 30)])
def test_radial_vim_matches_closed_form(l, m):
    # radial VIM M0*S_lm on a shell of radius a gives g_lm = mu0*M0*l/((2l+1)a), for every m
    coeffs = transform(M0*schmidt(l, m, TH, PH))
    expected = mu0*M0*l/((2*l+1)*a)*1e9
    assert coeffs[0, l, m] == pytest.approx(expected, rel=1e-10)
    # and nothing else
    coeffs[0, l, m] = 0
    assert np.abs(coeffs).max() < 1e-10*expected


def test_rotated_dipole_sources_give_equal_dipoles():
    # the same radial source pointed along z, x and y must give the same dipole strength
    ux, uy, uz = np.sin(TH)*np.cos(PH), np.sin(TH)*np.sin(PH), np.cos(TH)
    strengths = [np.sqrt((transform(M0*u)[:, 1, :2]**2).sum()) for u in (uz, ux, uy)]
    assert strengths[1] == pytest.approx(strengths[0], rel=1e-10)
    assert strengths[2] == pytest.approx(strengths[0], rel=1e-10)


def test_uniform_shell_in_internal_field_has_no_external_field():
    # Runcorn's theorem: uniform susceptibility x an internal field -> no external field
    G = pyshtools.datasets.Earth.IGRF_13().expand(a=a, lmax=L, extend=False)
    full = transform(G.rad.data, G.theta.data, G.phi.data)
    radial_only = transform(G.rad.data)
    assert np.abs(full).max() < 1e-10*np.abs(radial_only).max()


def test_round_trip_recovers_band_limited_magnetization():
    rng = np.random.default_rng(3)
    ux, uy, uz = np.sin(TH)*np.cos(PH), np.sin(TH)*np.sin(PH), np.cos(TH)
    f = M0*(1 + 0.5*schmidt(4, 3, TH, PH, 0.7, -0.4))
    v = rng.normal(size=3)
    mr = f*(v[0]*ux + v[1]*uy + v[2]*uz)
    mt = f*(v[0]*np.cos(TH)*np.cos(PH) + v[1]*np.cos(TH)*np.sin(PH) - v[2]*np.sin(TH))
    mp = f*(-v[0]*np.sin(PH) + v[1]*np.cos(PH))
    vsh, _ = GlobalMagnetizationModel(lon, lat, mr, mt, mp, a).transform(lmax=L)
    back = inverse_transform(lon, lat, vsh, L)
    # compare away from the pole row, where the inverse divides by sin(theta)
    for orig, new in [(mr, back.mrad), (mt, back.mtheta), (mp, back.mphi)]:
        assert np.allclose(new[1:], orig[1:], atol=1e-8*M0)


def test_matches_independent_point_dipole_sum():
    # Br at 300 km from a direct sum of point dipoles (no remit or pyshtools code),
    # compared with remit's Gauss coefficients; the source has all three components
    rng = np.random.default_rng(0)

    def source(th, ph):
        f = M0*(1 + 0.5*schmidt(3, 2, th, ph, 0.7, -0.4) + 0.3*schmidt(8, 5, th, ph, -0.2, 0.9))
        v = np.array([0.3, -0.5, 0.8])
        rh = np.stack([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
        thh = np.stack([np.cos(th)*np.cos(ph), np.cos(th)*np.sin(ph), -np.sin(th)])
        phh = np.stack([-np.sin(ph), np.cos(ph), 0*ph])
        return [f*np.tensordot(v, e, 1) for e in (rh, thh, phh)]

    ns = 400_000
    k = np.arange(ns) + 0.5
    sth = np.arccos(1 - 2*k/ns)
    sph = np.mod(np.pi*(1 + 5**0.5)*k, 2*np.pi)
    rh = np.stack([np.sin(sth)*np.cos(sph), np.sin(sth)*np.sin(sph), np.cos(sth)], 1)
    thh = np.stack([np.cos(sth)*np.cos(sph), np.cos(sth)*np.sin(sph), -np.sin(sth)], 1)
    phh = np.stack([-np.sin(sph), np.cos(sph), 0*sph], 1)
    mr, mt, mp = source(sth, sph)
    moment = (mr[:, None]*rh + mt[:, None]*thh + mp[:, None]*phh)*4*np.pi*a**2/ns

    r_obs = a + 300e3
    olat = np.degrees(np.arcsin(rng.uniform(-1, 1, 6)))
    olon = rng.uniform(0, 360, 6)
    ot, op = np.radians(90-olat), np.radians(olon)
    xo = r_obs*np.stack([np.sin(ot)*np.cos(op), np.sin(ot)*np.sin(op), np.cos(ot)], 1)
    br_direct = []
    for x in xo:
        R = x - a*rh
        Rn = np.linalg.norm(R, axis=1)
        B = mu0/(4*np.pi)*(3*(np.sum(moment*R, 1)/Rn**2)[:, None]*R - moment)/Rn[:, None]**3
        br_direct.append(B.sum(0) @ (x/r_obs)*1e9)

    coeffs = pyshtools.SHMagCoeffs.from_array(transform(*source(TH, PH)), r0=a)
    br_remit = np.asarray(coeffs.expand(a=r_obs, lat=olat, lon=olon, lmax_calc=L))[:, 0]
    assert np.allclose(br_remit, br_direct, rtol=1e-5, atol=1e-7*np.abs(br_direct).max())


def test_induced_vim_uses_H_not_B():
    # VIS [SI.km] x B [nT] -> VIM [A] needs 1e3*1e-9/mu0 (Hemant and Maus 2005, Eq. 3)
    dipole = pyshtools.SHMagCoeffs.from_zeros(lmax=1, r0=6371200.)
    dipole.coeffs[0, 1, 0] = -30000.
    B = dipole.expand(lmax=L, extend=True)
    vis = GlobalVIS(B.rad.lons(), B.rad.lats(), 2.*np.ones_like(B.rad.data))
    gmm = vis.vim(inducing_field=dipole)
    expected = 2.*B.rad.data[:-1, :-1]*1e-6/mu0
    assert np.allclose(gmm.mrad, expected, rtol=1e-12)


def test_rejects_non_dh_grid():
    lat_bad = np.linspace(90, -90, N)     # includes the south pole
    with pytest.raises(ValueError):
        GlobalMagnetizationModel(lon, lat_bad, ZERO, ZERO, ZERO, a).transform(lmax=L)
