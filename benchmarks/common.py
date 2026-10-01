"""
Inputs shared by the benchmarks (dipole_sum.py, sh_quadrature.py): the 0.1 deg
Driscoll-Healy grid, GK07 remanent and VIS induced magnetisation, an induced VIM built
independently of remit, and the depth slices of GK07_NR.

Importing this module loads the age and VIS grids (a few seconds); the depth set-up is
built on first use of depth_setup().
"""
import os
import time
from types import SimpleNamespace

import numpy as np
import pyshtools

import depth_models as dm
from basis_models import MODEL_LIST
from remit.data.models import load_ocean_age_model, load_vis_model
from remit.earthvim import GlobalVIS, SeafloorAgeProfile
from remit.utils.grid import agearray2magnetisation
from sh_reference import dh_weights

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'cache')
RESULTS = os.path.join(HERE, 'results')
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

R0 = dm.R0
MU0 = pyshtools.constants.mu0.value
SLICE_BENCH = 500.                       # m, vertical slice thickness for the depth case


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def cached(name, fn):
    f = os.path.join(CACHE, name + '.npy')
    if not os.path.exists(f):
        t = time.time()
        np.save(f, fn())
        log(f'  {name}: computed in {time.time()-t:.0f} s')
    return np.load(f)


def unit_vectors(colat, lon):
    st, ct, sp, cp = np.sin(colat), np.cos(colat), np.sin(lon), np.cos(lon)
    rh = np.stack([st*cp, st*sp, ct], -1)
    th = np.stack([ct*cp, ct*sp, -st], -1)
    ph = np.stack([-sp, cp, np.zeros_like(sp)], -1)
    return rh, th, ph


# ---------------------------------------------------------------- grid and inputs

log('loading inputs')
ocean = load_ocean_age_model()
vis = load_vis_model(name='Hemant2005+slabs', match=(ocean.lon, ocean.lat, ocean.age))
assert np.allclose(vis.lat, ocean.lat) and np.allclose(vis.lon, ocean.lon)
LAT, LON = ocean.lat[:-1], ocean.lon[:-1]              # DH2: drop the south pole and lon 360
assert LAT[0] == 90 and len(LAT) == 1800 and len(LON) == 3600
COLAT, LONR = np.meshgrid(np.radians(90 - LAT), np.radians(LON), indexing='ij')
RH, TH, PH = unit_vectors(COLAT, LONR)
W_DH, _ = dh_weights(1800)
assert np.isclose(W_DH.sum(), 2.) and np.isclose((W_DH*np.cos(COLAT[:, 0])**2).sum(), 2/3)

# independent induced VIM: VIS [SI km] x 1e3 [m/km] x B [nT] x 1e-9 [T/nT] / mu0
igrf = pyshtools.datasets.Earth.IGRF_13().expand(lmax=899, extend=True)
assert np.allclose(igrf.rad.lats(), ocean.lat) and np.allclose(igrf.rad.lons(), ocean.lon)
IGRF = [g.data[:-1, :-1] for g in (igrf.rad, igrf.theta, igrf.phi)]
VIS = vis.vis[:-1, :-1].astype(float)                 # the VIS grid is stored as float32


def induced_vim(vis_grid):
    return [vis_grid*1e3*b*1e-9/MU0 for b in IGRF]


gv = vis.vim()                                         # remit's induced VIM
_p = dict(MODEL_LIST['GK07'])
_p.pop('seafloor_layer')
rem_gmm = ocean.vim(SeafloorAgeProfile.layer2d(**_p))  # GK07 remanent VIM (remit model definition)


# ---------------------------------------------------------------- depth slices (GK07_NR)

_depth = None


def depth_setup():
    """Source radii and 500 m slices of GK07_NR (remanent) and the H&M oceanic crust
    (induced) on the full 1801 x 3601 grid; [:-1, :-1] gives the DH2 grid."""
    global _depth
    if _depth is not None:
        return _depth
    d = SimpleNamespace()
    d.r_top_full = dm.top_of_crust_radius(ocean)
    d.r_top = d.r_top_full[:-1, :-1]
    d.is_ocean_full = ~np.isnan(ocean.age)
    d.r_ell_full = dm.ellipsoid_radius(ocean.lat)[:, None]*np.ones_like(d.r_top_full)

    # remanent slices: the 100 m cross-section of GK07_NR summed into 500 m bins
    prof, z100, rm100 = dm.profile_slices(MODEL_LIST['GK07_NR'])
    bins = np.floor(z100/SLICE_BENCH).astype(int)
    d.z_rem = (np.unique(bins) + 0.5)*SLICE_BENCH
    rm_rem = np.array([rm100[bins == b].sum(0) for b in np.unique(bins)])
    d.rem_slices = [(zi, agearray2magnetisation(ocean.age, prof.age, rmi)) for zi, rmi in zip(d.z_rem, rm_rem)]
    d.to_gmm_rem = dm.remanent_to_gmm(ocean)

    # induced slices: H&M 2005 oceanic layers in ~500 m slices; continents on the ellipsoid
    layers = []
    for top, bottom, chi in dm.HM05_OCEAN_VIS_LAYERS:
        n = max(1, int(round((bottom - top)/SLICE_BENCH)))
        dz = (bottom - top)/n
        layers += [(top + (i + 0.5)*dz, chi*dz) for i in range(n)]
    d.z_ind = np.array([z for z, _ in layers])
    f_ind = np.array([f for _, f in layers])
    d.f_ind = f_ind/f_ind.sum()

    def ind_weight(k):
        return np.where(d.is_ocean_full, d.f_ind[k], 1./len(d.z_ind))*vis.vis.astype(float)

    def ind_radius(k):
        return np.where(d.is_ocean_full, d.r_top_full - d.z_ind[k], d.r_ell_full)

    def remit_coeffs(part, lmax):
        """remit's depth-weighted Gauss coefficients (depth_models.depth_weighted_coeffs)"""
        if part == 'remanent':
            slices = [(w, np.log((d.r_top_full - zi)/R0)) for zi, w in d.rem_slices]
            to_gmm = d.to_gmm_rem
        else:
            slices = [(ind_weight(k), np.log(ind_radius(k)/R0)) for k in range(len(d.z_ind))]
            to_gmm = lambda v: GlobalVIS(vis.lon, vis.lat, v).vim()
        umax = max(np.nanmax(np.abs(u)) for _, u in slices)
        return dm.depth_weighted_coeffs(slices, to_gmm, lmax, dm.n_terms(umax, lmax)).coeffs

    d.ind_weight, d.ind_radius, d.remit_coeffs = ind_weight, ind_radius, remit_coeffs
    log(f'  {len(d.z_rem)} remanent slices ({d.z_rem.min():.0f}-{d.z_rem.max():.0f} m), '
        f'{len(d.z_ind)} induced slices ({d.z_ind.min():.0f}-{d.z_ind.max():.0f} m)')
    _depth = d
    return d
