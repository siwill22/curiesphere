"""
Forward models with realistic source depth, and an LCS-1 resolution filter.

`load_vim_models` is a drop-in replacement for basis_models.load_vim_models, so the
paper notebooks can be rerun unchanged apart from their import line. The variant is
chosen with the environment variable DEPTH_VARIANT:

    'none'          corrected remit code, sources on the r0 sphere (= basis_models)
    'depth'         sources at their geocentric radius (see below)
    'filter'        'none' passed through the LCS-1 resolution filter F(l)
    'depth+filter'  both
    'depth+nr'      'depth' with the tuned near-ridge enhancement (P = 0.94, lambda = 3 Ma,
                    notebooks/tune_near_ridge.py) applied to every remanent model

Source radius
    r_s = r_WGS84(lat) - water depth - sediment thickness - depth below basement
  - remanence: each model's own magnetisation-vs-depth cross-section, in 100 m slices
  - oceanic VIS: split over the Hemant & Maus (2005) oceanic crust, layer 2
    (0-2.11 km, 0.066 SI) and layer 3 (2.11-7.08 km, 0.049 SI)
  - continental VIS: on the ellipsoid (depth 0)
  water depth: SRTM15 v2.7 (GMT earth_relief_06m, downloaded by pygmt);
  sediments: NGDC total sediment thickness merged with CRUST2.0 (2011, 5' grid), read
  from $CURIESPHERE_SEDIMENT_GRID (default ~/Data/GMTdata/GlobalGrids/NGDCcrust2_SedThick_2011.nc).
  Grid latitudes are used as geocentric latitudes (spherical-Earth treatment of the
  grid), only the radius follows the ellipsoid.

  A source at radius r_s with VIM per unit area there contributes to the Gauss
  coefficients referenced to r0 with the factor (r_s/r0)^(l+1) (transform at r_s,
  then change_ref to r0). This is applied exactly: with u = ln(r_s/r0),
      (r_s/r0)^(l+1) = sum_k (l+1)^k u^k / k!
  so the model is sum_k (l+1)^k/k! * transform(VIM * u^k), one ordinary remit
  transform per term.

LCS-1 filter
  For uniform data at radius a+h and L2 damping of Br at radius a, each Gauss
  coefficient is multiplied by
      F(l) = 1 / (1 + ((a+h)/a)^(2(l - l_h)))
  h = 300 km (CHAMP 2006-2010, Olsen et al. 2017 Fig. 2). l_h is fitted to the
  LCS-1 / EMM2015 power ratio for degrees 133-185 (Olsen et al. 2017 Fig. 8), with
  EMM2015 taken as flat at 32 nT^2 (read off that figure) and LCS-1 from LCS_mod.cof.
"""
import os
from math import factorial

import numpy as np
import pyshtools
import xarray as xr
from scipy.optimize import minimize_scalar

import basis_models
from basis_models import MODEL_LIST
from remit.earthvim import SeafloorAgeProfile, GlobalVIS
from remit.vhtools import GlobalMagnetizationModel
from remit.data.models import load_ocean_age_model, load_vis_model, load_lcs
from remit.utils.profile import DEFAULT_P
from remit.utils.grid import agearray2magnetisation, paleoIncDec2field, DH2, coeffs2map

R0 = 6371000.
LMAX = 185
WGS84_A, WGS84_F = 6378137.0, 1/298.257223563
SEDIMENT_GRID = os.path.expanduser(os.environ.get(
    'CURIESPHERE_SEDIMENT_GRID', '~/Data/GMTdata/GlobalGrids/NGDCcrust2_SedThick_2011.nc'))
# Hemant & Maus (2005) oceanic crust: (top, bottom) in m, susceptibility in SI
HM05_OCEAN_VIS_LAYERS = [(0., 2110., 0.066), (2110., 7080., 0.049)]
SLICE = 100.                                  # m, vertical slice for the VIS layers
LCS1_H, LCS1_A, EMM2015_POWER = 300e3, 6371.2e3, 32.
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'depth', 'cache')


# ---------------------------------------------------------------- geometry

def ellipsoid_radius(lat):
    """Geocentric radius (m) of the WGS84 ellipsoid at latitude lat (degrees)"""
    a, b = WGS84_A, WGS84_A*(1-WGS84_F)
    c, s = np.cos(np.radians(lat)), np.sin(np.radians(lat))
    return np.sqrt(((a*a*c)**2 + (b*b*s)**2)/((a*c)**2 + (b*s)**2))


def _to_grid(da, lon, lat, xname, yname):
    """Sample a -180/180 lon-lat grid at the model grid nodes (lon 0-360)"""
    da = da.rename({xname: 'x', yname: 'y'})
    lat = np.clip(lat, float(da.y.min()), float(da.y.max()))   # extend edge rows
    lon180 = ((np.asarray(lon) + 180) % 360) - 180
    return da.interp(x=xr.DataArray(lon180, dims='lon'),
                     y=xr.DataArray(lat, dims='lat')).transpose('lat', 'lon').values


def top_of_crust_radius(ocean):
    """Radius (m) of the top of the magnetic crust on the ocean-age grid:
    ellipsoid - water depth - sediment thickness (sea floor and sediments only
    where there is ocean-age crust; elsewhere the ellipsoid)"""
    import pygmt
    relief = pygmt.datasets.load_earth_relief(resolution='06m', registration='gridline')
    elev = _to_grid(relief, ocean.lon, ocean.lat, 'lon', 'lat')
    sed = _to_grid(xr.open_dataarray(SEDIMENT_GRID), ocean.lon, ocean.lat, 'x', 'y')
    depth = np.maximum(-elev, 0.) + np.nan_to_num(sed)
    depth[np.isnan(ocean.age)] = 0.
    return ellipsoid_radius(ocean.lat)[:, None] - depth


# ---------------------------------------------------------------- depth-weighted transform

def n_terms(umax, lmax, tol=1e-12):
    """Number of series terms so that the remainder of exp((l+1)u) is below tol"""
    x = (lmax+1)*umax
    k = 0
    while x**(k+1)/factorial(k+1)*np.exp(x) > tol:
        k += 1
    return k + 1


def depth_weighted_coeffs(slices, to_gmm, lmax, nterm):
    """
    Gauss coefficients (r0 = R0) of sources distributed over radius.

    slices : iterable of (w, u) grids; w is the source strength of one slice (the
             scalar that to_gmm turns into a VIM), u = ln(r_s/R0) of that slice
    to_gmm : scalar grid -> GlobalMagnetizationModel on the R0 sphere
    """
    S = None
    for w, u in slices:
        if S is None:
            S = [np.zeros_like(w) for _ in range(nterm)]
        t = w.copy()
        for k in range(nterm):
            S[k] += t
            t *= u
    l = np.arange(lmax+1)
    total = np.zeros((2, lmax+1, lmax+1))
    for k in range(nterm):
        _, c = to_gmm(S[k]).transform(lmax=lmax)
        total += c.coeffs*((l+1.)**k/factorial(k))[None, :, None]
    return pyshtools.SHMagCoeffs.from_array(total, r0=R0)


# ---------------------------------------------------------------- remanence and VIS

def profile_slices(params):
    """Magnetisation profile, mid-depths (m below basement) and VIM of each slice
    (rows of RVIM per slice vs age) for a MODEL_LIST entry"""
    p = params.copy()
    dims = p.pop('seafloor_layer')
    if dims == '2d':
        prof = SeafloorAgeProfile.layer2d(**p)
        dz = prof.depth[1] - prof.depth[0]
        z, rm = prof.depth + dz/2, prof.RM          # RM is already x dz (stratified_vim)
    else:
        prof = SeafloorAgeProfile.layer1d(**p)
        thickness = prof.layer_thickness
        if not np.isscalar(thickness):
            raise NotImplementedError('age-dependent layer thickness')
        n = int(np.ceil(thickness/SLICE))
        z = (np.arange(n) + 0.5)*thickness/n
        rm = np.repeat(prof.RVIM[None, :]/n, n, axis=0)
    keep = np.abs(rm).max(axis=1) > 0
    assert np.allclose(rm.sum(0), prof.RVIM)
    return prof, z[keep], rm[keep]


def remanent_to_gmm(ocean):
    Br, Bt, Bp = paleoIncDec2field(ocean.paleolatitude, ocean.declination)
    C = np.sqrt(1 + 3*np.sin(np.radians(ocean.paleolatitude))**2)

    def to_gmm(M):
        grids = [np.nan_to_num(M*C*B) for B in (Br, Bt, Bp)]
        lon, lat, grids = DH2(ocean.lon, ocean.lat, grids)
        return GlobalMagnetizationModel(lon, lat, *grids, R0)
    return to_gmm


def remanent_coeffs(ocean, r_top, params, lmax=LMAX, zero_depth=False):
    prof, z, rm = profile_slices(params)
    u_all = [np.log((r_top - zi)/R0) for zi in (z.min(), z.max())]
    umax = 0. if zero_depth else max(np.nanmax(np.abs(u)) for u in u_all)

    def slices():
        for zi, rmi in zip(z, rm):
            w = agearray2magnetisation(ocean.age, prof.age, rmi)
            u = np.zeros_like(r_top) if zero_depth else np.log((r_top - zi)/R0)
            yield w, u
    return depth_weighted_coeffs(slices(), remanent_to_gmm(ocean), lmax, n_terms(umax, lmax)), prof


def vis_coeffs(ocean, vis, r_top, lmax=LMAX, zero_depth=False):
    """Induced part: oceanic VIS spread over the H&M 2005 oceanic crust below r_top,
    continental VIS on the ellipsoid"""
    is_ocean = ~np.isnan(ocean.age)
    r_ell = ellipsoid_radius(ocean.lat)[:, None]*np.ones_like(r_top)
    layers = []
    for top, bottom, chi in HM05_OCEAN_VIS_LAYERS:
        n = int(round((bottom-top)/SLICE))
        dz = (bottom-top)/n
        layers += [(top + (i+0.5)*dz, chi*dz) for i in range(n)]
    zs = np.array([zl for zl, _ in layers])
    frac = np.array([f for _, f in layers])
    frac /= frac.sum()

    def u_of(zi):
        if zero_depth:
            return np.zeros_like(r_top)
        return np.log(np.where(is_ocean, r_top - zi, r_ell)/R0)

    umax = 0. if zero_depth else max(np.abs(u_of(zs.max())).max(), np.abs(u_of(0.)).max())

    def slices():
        for zi, f in zip(zs, frac):
            yield np.where(is_ocean, f, 1./len(zs))*vis.vis, u_of(zi)

    def to_gmm(v):
        return GlobalVIS(vis.lon, vis.lat, v).vim()
    return depth_weighted_coeffs(slices(), to_gmm, lmax, n_terms(umax, lmax))


# ---------------------------------------------------------------- LCS-1 filter

def lcs1_filter(l, l_h, h=LCS1_H, a=LCS1_A):
    return 1./(1. + ((a+h)/a)**(2.*(np.asarray(l, float) - l_h)))


def lowes_spectrum(coeffs):
    """Lowes-Mauersberger R_n at the reference radius (nT^2)"""
    c = coeffs.coeffs
    l = np.arange(c.shape[1])
    return (l+1)*(c**2).sum(axis=(0, 2))


def fit_lh(emm_power=EMM2015_POWER, h=LCS1_H, degrees=(133, 185)):
    R = lowes_spectrum(load_lcs(lmin=16, lmax=185))
    n = np.arange(degrees[0], degrees[1]+1)
    y = np.log(R[n]/emm_power)
    res = minimize_scalar(lambda lh: np.sum((y - 2*np.log(lcs1_filter(n, lh, h)))**2),
                          bounds=(100, 300), method='bounded')
    return res.x


def apply_filter(coeffs, l_h):
    l = np.arange(coeffs.lmax+1)
    out = coeffs.copy()
    out.coeffs = coeffs.coeffs*lcs1_filter(l, l_h)[None, :, None]
    return out


# ---------------------------------------------------------------- models

NEAR_RIDGE = dict(P=0.94, lmbda=3.)


def with_near_ridge(params, P=NEAR_RIDGE['P'], lmbda=NEAR_RIDGE['lmbda']):
    """A MODEL_LIST entry with the near-ridge enhancement replaced. Models that
    fix the ridge value with MagMax (DAH981, M12) have MagMax rescaled so that
    old crust keeps its magnetisation and only the enhancement changes."""
    p = dict(params)
    P_old = p.get('P', DEFAULT_P)
    if p.get('MagMax') is not None and p.get('blocking_temperatures') is None:
        p['MagMax'] = p['MagMax']*(1 + P)/(1 + P_old)
    p['P'], p['lmbda'] = P, lmbda
    return p


def depth_model_coeffs(model_names, lmax=LMAX, zero_depth=False, near_ridge=False):
    """Depth-resolved Gauss coefficients (lmax 185, r0 = R0) for MODEL_LIST names,
    cached in notebooks/depth/cache"""
    os.makedirs(CACHE, exist_ok=True)
    tag = '_zero' if zero_depth else ''
    ocean = vis = r_top = None
    out = {}

    def inputs():
        nonlocal ocean, vis, r_top
        if ocean is None:
            ocean = load_ocean_age_model()
            vis = load_vis_model(name='Hemant2005+slabs', match=(ocean.lon, ocean.lat, ocean.age))
            assert np.allclose(vis.lat, ocean.lat) and np.allclose(vis.lon, ocean.lon)
            r_top = top_of_crust_radius(ocean)
        return ocean, vis, r_top

    fvis = os.path.join(CACHE, f'VIS{tag}_{lmax}.npy')
    if not os.path.exists(fvis):
        np.save(fvis, vis_coeffs(*inputs(), lmax=lmax, zero_depth=zero_depth).coeffs)
    cvis = np.load(fvis)

    for name in model_names:
        params = MODEL_LIST[name]
        if params['seafloor_layer'] is None:
            out[name] = (pyshtools.SHMagCoeffs.from_array(cvis.copy(), r0=R0), None)
            continue
        if near_ridge:
            params = with_near_ridge(params)
        f = os.path.join(CACHE, f'{name}{tag}{"_nr" if near_ridge else ""}_{lmax}.npy')
        if not os.path.exists(f):
            ocean_, _, r_top_ = inputs()
            c, _ = remanent_coeffs(ocean_, r_top_, params, lmax=lmax, zero_depth=zero_depth)
            np.save(f, c.coeffs)
        out[name] = (pyshtools.SHMagCoeffs.from_array(np.load(f) + cvis, r0=R0), None)
    return out


def load_vim_models(model_names, lmax=185, altitude=0):
    """Same interface and output as basis_models.load_vim_models"""
    variant = os.environ.get('DEPTH_VARIANT', 'depth+filter')
    assert variant in ('none', 'depth', 'filter', 'depth+filter', 'depth+nr'), variant

    if variant in ('none', 'filter'):
        models = basis_models.load_vim_models(model_names, lmax=lmax, altitude=altitude)
        coeffs = {name: m['coeffs'] for name, m in models.items()}
    else:
        coeffs = {name: c.pad(lmax) for name, (c, _) in
                  depth_model_coeffs(model_names, near_ridge=(variant == 'depth+nr')).items()}

    if 'filter' in variant:
        l_h = fit_lh()
        print(f'LCS-1 filter: h = {LCS1_H/1e3:.0f} km, l_h = {l_h:.1f}')
        coeffs = {name: apply_filter(c, l_h) for name, c in coeffs.items()}

    vim_models = {}
    for name in model_names:
        c = coeffs[name]
        model_rad = coeffs2map(c, altitude=altitude, lmax=lmax, lmin=16)
        vim_models[name] = {'totalvim': None, 'profile': None, 'coeffs': c, 'rad': model_rad}
    return vim_models
