"""
Invert LCS-1 for a uniform VIS in each of a set of continental domains: Australia first,
as a shakedown of the method. Decisions: docs/inversion-australia-domains.md.

Method
  - domains: Hasterok et al. (2022) provinces (gprm), continent == 'Australia', PNG and
    Bismarck units excluded. inversion/australia_domains.csv maps each polygon to a parameter
    (passive margins and offshore plateaus lumped; an empty param keeps the background).
    The table is written once and then read, so it can be edited by hand.
  - parameter: one uniform VIS v_k (SI.km) per parameter, no H&M values inside the domains.
    Cells with ocean age keep the ocean model (age grid wins over polygons).
  - sources: v_k spread uniformly in radius from the top of basement (ellipsoid - water -
    sediment) to BASE_DEPTH below the ellipsoid; inducing field IGRF-13.
  - background (fixed): the depth model elsewhere: H&M + slabs VIS (continental VIS on the
    ellipsoid, oceanic VIS over the HM05 oceanic crust) with the domain cells removed, plus
    GK07_NR remanence (depth_models cache).
  - data: Br at r0 = 6371.0 km, degrees LMIN-LMAX_FIT, LCS-1 filter F(l) applied to the
    model, on HEALPix points (NSIDE) within BUFFER_KM (great circle) of a domain cell.
  - error model: sigma = RMS(LCS-1 - MF7) at the points; noise realisations are random-phase
    fields with the global per-degree spectrum of LCS-1 - MF7.
  - solve: bounded least squares (v >= 0), optionally damped towards v_ref * 1.

Synthetic tests in this script: T1 (random v_k, recovery and noise ensemble) and T2 (common
mode, single uniform v_ref). T3-T6 and the real-data fit come after the first check-in.

Run from notebooks/ (conda env pygmt17):
    PYTHONPATH=..:$PYTHONPATH python invert_domains.py
Outputs go to notebooks/inversion/ (the .npz caches are not tracked).
"""
import os
import time

import numpy as np
import pandas as pd
import pygmt
import pygplates
import pyshtools
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree
from gprm.datasets import Geology
from ptt.utils.points_in_polygons import find_polygons

import depth_models as dm
from remit.data.models import load_ocean_age_model, load_vis_model, load_lcs, DATA_DIR
from remit.utils.region import generate_healpix_points

OUT = 'inversion'
os.makedirs(OUT, exist_ok=True)
TABLE = f'{OUT}/australia_domains.csv'

R0 = dm.R0
LMIN, LMAX_FIT, LMAX = 16, 133, dm.LMAX
BASE_DEPTH = 30e3                 # m below the ellipsoid
BUFFER_KM = 300.
NSIDE = 128                       # HEALPix, ~55 km spacing
BBOX = (95., 160., -50., -5.)     # lon, lon, lat, lat: the region rasterised for the domains
DENSIFY_DEG = 0.1                 # max lon/lat step along polygon edges before the great-circle test
N_NOISE = 100
SEED = 20261001
PNG_UNITS = ['Papua', 'Woodlark', 'Trobriand', 'San Christobal', 'Bismar', 'Mamberamo']
MARGIN_LUMPS = {                  # transitional polygons -> lumped margin parameter
    'Western Australian Rifted Margin': 'margin_W', 'Exmouth Plateau': 'margin_W',
    'Zenith Plateau': 'margin_W', 'Quokka Rise': 'margin_W', 'Batavia Knoll': 'margin_W',
    'Gulden Draak': 'margin_W',
    'North Australian Rifted Margin': 'margin_N',
    'East Australian Rifted Margin': 'margin_E', 'Queensland-Marion Plateau': 'margin_E',
    'South Australia Rifted Margin': 'margin_S', 'Bass Straight': 'margin_S'}


# ============================================================ 1. domain table
provinces = Geology.fetch_GlobalTectonicMap()
aus = provinces[(provinces.continent == 'Australia')
                & ~provinces.prov_name.str.contains('|'.join(PNG_UNITS), na=False)].copy()
aus['key'] = aus.index                      # Hasterok 'id' is not unique (five polygons share 406)

if not os.path.exists(TABLE):
    def default_param(row):
        if row.crust_type == 'continental':
            return row.prov_name.replace(' ', '_')
        return MARGIN_LUMPS.get(row.prov_name, '')
    table = aus[['key', 'id', 'prov_name', 'prov_type', 'crust_type', 'prov_group']].copy()
    table['param'] = aus.apply(default_param, axis=1)
    table.to_csv(TABLE, index=False)
    print(f'wrote draft {TABLE}')
table = pd.read_csv(TABLE, keep_default_na=False)
table = table[table.param != '']
params = sorted(set(table.param), key=lambda p: (p.startswith('margin'), p))
n_par = len(params)
print(f'{len(table)} polygons -> {n_par} parameters')


# ============================================================ 2. parameter grid (0.1 deg ocean-age grid)
ocean = load_ocean_age_model()
glon, glat = np.asarray(ocean.lon), np.asarray(ocean.lat)          # lon 0 -> 360, lat 90 -> -90
age = np.asarray(ocean.age)
cache = f'{OUT}/param_grid.npz'
if os.path.exists(cache) and list(np.load(cache)['params']) == params:
    pgrid = np.load(cache)['pgrid']
else:
    def densified_polygon(ring):
        """PolygonOnSphere through the ring's vertices, with points added along each edge so that
        the great-circle edges follow the straight lon/lat lines of the GIS source"""
        xy = np.asarray(ring.coords)
        pts = []
        for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
            n = max(1, int(np.ceil(max(abs(x1-x0), abs(y1-y0))/DENSIFY_DEG)))
            t = np.arange(n)/n
            pts += list(zip(y0 + t*(y1-y0), x0 + t*(x1-x0)))
        return pygplates.PolygonOnSphere(pts)

    polygons, proxies = [], []
    for _, row in table.iterrows():
        geom = aus.loc[row.key].geometry
        assert all(len(p.interiors) == 0 for p in getattr(geom, 'geoms', [geom])), 'holes not handled'
        for part in getattr(geom, 'geoms', [geom]):
            polygons.append(densified_polygon(part.exterior))
            proxies.append(params.index(row.param))
    ilon = np.where((glon >= BBOX[0]) & (glon <= BBOX[1]))[0]
    ilat = np.where((glat >= BBOX[2]) & (glat <= BBOX[3]))[0]
    II, JJ = np.meshgrid(ilat, ilon, indexing='ij')
    points = [pygplates.PointOnSphere(float(glat[i]), float(glon[j])) for i, j in zip(II.ravel(), JJ.ravel())]
    t = time.time()
    hit = find_polygons(points, polygons, proxies, subdivision_depth=6)
    print(f'point-in-polygon on {len(points)} cells: {time.time()-t:.0f} s')
    pgrid = np.full(age.shape, -1, dtype=np.int16)
    pgrid[II.ravel(), JJ.ravel()] = [-1 if h is None else h for h in hit]
    pgrid[~np.isnan(age)] = -1                                        # age grid wins
    np.savez(cache, pgrid=pgrid, params=np.array(params))

cell_area = np.cos(np.radians(glat))[:, None]*np.ones_like(age)      # relative cell areas
free = pgrid >= 0
km2 = (np.radians(0.1)*R0/1e3)**2
par_area = np.array([cell_area[pgrid == k].sum()*km2 for k in range(n_par)])
# coverage check: non-ocean cells inside the bounding box of the domains not in any parameter
LO, LA = np.meshgrid(glon, glat)
in_box = (LO > 112) & (LO < 154) & (LA > -44) & (LA < -10.5)
gaps = in_box & np.isnan(age) & ~free
print(f'domain cells: {free.sum()} ({cell_area[free].sum()*km2/1e6:.2f} M km2); '
      f'non-ocean cells in 112-154E, 10.5-44S not in any domain: {gaps.sum()}')


# ============================================================ 3. data points
hlon, hlat = generate_healpix_points(NSIDE)
hlon = np.mod(hlon, 360.)

def unit(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], -1)

chord, _ = cKDTree(unit(LO[free], LA[free])).query(unit(hlon, hlat))
dist_km = 2*np.arcsin(np.clip(chord/2, 0, 1))*R0/1e3                # great-circle distance to a domain cell
keep = dist_km <= BUFFER_KM
plon, plat, pdist = hlon[keep], hlat[keep], dist_km[keep]
onshore = pdist < 10.                                               # inside a domain (within half a cell)
print(f'points: {keep.sum()} ({onshore.sum()} inside the domains, {(~onshore).sum()} in the {BUFFER_KM:.0f} km buffer)')


# ============================================================ 4. forward columns
l = np.arange(LMAX+1)
F = dm.lcs1_filter(l, dm.fit_lh())

def br_at_points(c):
    """Br (nT) at the data points from Gauss coefficients (r0 = R0): degrees LMIN-LMAX_FIT, F(l) applied"""
    c = np.array(c)[:, :LMAX_FIT+1, :LMAX_FIT+1]*F[None, :LMAX_FIT+1, None]
    c[:, :LMIN] = 0
    x = pyshtools.SHMagCoeffs.from_array(c, r0=R0)
    return np.asarray(x.expand(a=R0, lat=plat, lon=plon, lmax_calc=LMAX_FIT))[:, 0]

def load_cof(name):
    clm, _ = pyshtools.shio.shread(f'{DATA_DIR}/shc/{name}', lmax=LMAX_FIT)
    return clm                                                     # r0 taken as 6371.0 km, as for LCS-1

cache = f'{OUT}/columns.npz'
if os.path.exists(cache) and list(np.load(cache)['params']) == params:
    z = np.load(cache)
    assert np.array_equal(z['plon'], plon), 'point set changed; delete the cache'
    G, b, d_lcs, d_mf7 = z['G'], z['b'], z['d_lcs'], z['d_mf7']
else:
    t = time.time()
    coeff_cache = f'{OUT}/column_coeffs.npz'          # Gauss coefficients, independent of the point set
    if os.path.exists(coeff_cache) and list(np.load(coeff_cache)['params']) == params:
        C_cols, C_bg = np.load(coeff_cache)['C_cols'], np.load(coeff_cache)['C_bg']
    else:
        r_top = dm.top_of_basement_radius(ocean)
        r_base = dm.ellipsoid_radius(glat)[:, None]*np.ones_like(r_top) - BASE_DEPTH
        assert (r_top[free] > r_base[free]).all(), 'basement below the slab base'
        to_gmm = dm.induced_to_gmm(glon, glat)
        C_cols = np.array([dm.slab_vis_coeffs(np.where(pgrid == k, 1., 0.), r_top, r_base, to_gmm).coeffs
                           for k in range(n_par)])
        print(f'{n_par} columns: {time.time()-t:.0f} s', flush=True)
        # background: depth-model VIS with the domain cells removed, plus GK07_NR remanence
        vis = load_vis_model(name='Hemant2005+slabs', match=(ocean.lon, ocean.lat, ocean.age))
        vis.vis = np.where(free, 0., vis.vis)
        C_bg = dm.vis_coeffs(ocean, vis, dm.top_of_crust_radius(ocean)).coeffs
        C_bg += np.load(f'{dm.CACHE}/GK07_nr_{LMAX}.npy')
        np.savez(coeff_cache, params=np.array(params), C_cols=C_cols, C_bg=C_bg)
    G = np.stack([br_at_points(c) for c in C_cols], 1)              # nT per SI.km
    b = br_at_points(C_bg)
    d_lcs = br_at_points(load_lcs(lmin=LMIN, lmax=LMAX).coeffs)
    d_mf7 = br_at_points(load_cof('MF7_mod.cof'))
    np.savez(cache, params=np.array(params), plon=plon, G=G, b=b, d_lcs=d_lcs, d_mf7=d_mf7)
    print(f'columns cached: {time.time()-t:.0f} s')


# ============================================================ 5. error model
sigma = np.sqrt(np.mean((d_lcs - d_mf7)**2))
diff = load_lcs(lmin=LMIN, lmax=LMAX_FIT).coeffs - load_cof('MF7_mod.cof')
diff[:, :LMIN] = 0
R_l = (l[:LMAX_FIT+1]+1)*(diff**2).sum((0, 2))                    # Lowes spectrum of LCS-1 - MF7

rng = np.random.default_rng(SEED)
cache = f'{OUT}/noise.npz'
if os.path.exists(cache) and np.load(cache)['noise'].shape == (N_NOISE, len(plon)):
    noise = np.load(cache)['noise']
else:
    ll = l[:LMAX_FIT+1]
    var = np.where(ll >= LMIN, R_l/((ll+1)*(2*ll+1)), 0.)          # variance per coefficient
    noise = []
    for _ in range(N_NOISE):
        c = rng.standard_normal((2, LMAX_FIT+1, LMAX_FIT+1))*np.sqrt(var)[None, :, None]
        c[1, :, 0] = 0
        c *= np.arange(LMAX_FIT+1)[None, None, :] <= ll[None, :, None]       # m <= l only
        noise.append(np.asarray(pyshtools.SHMagCoeffs.from_array(c, r0=R0).expand(
            a=R0, lat=plat, lon=plon, lmax_calc=LMAX_FIT))[:, 0])
    noise = np.array(noise)
    np.savez(cache, noise=noise)
print(f'sigma = RMS(LCS-1 - MF7) at the points = {sigma:.2f} nT; '
      f'noise realisations RMS = {noise.std():.2f} nT; LCS-1 RMS = {d_lcs.std():.2f} nT')
# the global spectrum underestimates the difference over Australia: scale the realisations to sigma
noise *= sigma/np.sqrt(np.mean(noise**2))
print(f'noise realisations scaled to RMS {np.sqrt(np.mean(noise**2)):.2f} nT')


# ============================================================ 6. operator diagnostics
Gw = G/sigma
U, s, Vt = np.linalg.svd(Gw, full_matrices=False)
ones = np.ones(n_par)/np.sqrt(n_par)
common = Vt @ ones                                                 # common mode in the singular basis
print(f'\nsingular values (weighted, per SI.km): max {s[0]:.3g}, min {s[-1]:.3g}, condition {s[0]/s[-1]:.3g}')
print(f'common mode: weight on the 5 weakest singular vectors {np.sum(common[-5:]**2):.2f}, '
      f'on the 5 strongest {np.sum(common[:5]**2):.2f}')

def solve(dw, alpha=0., v_ref=0.):
    """bounded least squares (v >= 0), damped towards v_ref * 1 with weight alpha"""
    if alpha == 0.:
        return lsq_linear(Gw, dw, bounds=(0., np.inf)).x
    A = np.vstack([Gw, alpha*np.eye(n_par)])
    return lsq_linear(A, np.r_[dw, alpha*v_ref*np.ones(n_par)], bounds=(0., np.inf)).x

def resolution_and_cov(alpha):
    """linearised (unbounded) model resolution matrix and covariance (SI.km^2) at damping alpha"""
    H = Gw.T @ Gw + alpha**2*np.eye(n_par)
    Hi = np.linalg.inv(H)
    return Hi @ Gw.T @ Gw, Hi @ Gw.T @ Gw @ Hi

g1 = Gw @ np.ones(n_par)                                           # field of v = 1 SI.km everywhere

def fit_vref(dw):
    """single uniform VIS for all parameters: least squares, with formal 1-sigma"""
    return (g1 @ dw)/(g1 @ g1), 1/np.sqrt(g1 @ g1)

ALPHAS = s[0]*np.array([0., 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])


# ============================================================ 7. T1: random v_k
V0 = 0.6                                                           # SI.km, a generic continental level
v_true = V0*np.exp(0.5*np.random.default_rng(SEED+1).standard_normal(n_par))
y0 = G @ v_true                                                    # background is exact in T1, so omitted
rows = []
est = np.zeros((len(ALPHAS), N_NOISE, n_par))
vrefs = np.zeros(N_NOISE)
for j in range(N_NOISE):
    dw = (y0 + noise[j])/sigma
    vrefs[j] = fit_vref(dw)[0]
    for ia, a in enumerate(ALPHAS):
        est[ia, j] = solve(dw, a, vrefs[j])
noise_free = [solve(y0/sigma, a, fit_vref(y0/sigma)[0]) for a in ALPHAS]

print('\nT1: random v_k (median 0.6 SI.km), 100 noise realisations')
print(f'  {"alpha/s_max":>11s} {"RMS err":>8s} {"RMS bias":>9s} {"RMS std":>8s} {"noise-free err":>15s} {"median R_kk":>12s}')
summary = []
for ia, a in enumerate(ALPHAS):
    err = est[ia] - v_true
    R, C = resolution_and_cov(a)
    summary.append(dict(alpha_rel=a/s[0], rms_err=np.sqrt((err**2).mean()), rms_bias=np.sqrt((err.mean(0)**2).mean()),
                        rms_std=np.sqrt((est[ia].std(0)**2).mean()),
                        noise_free_err=np.sqrt(((noise_free[ia]-v_true)**2).mean()),
                        median_Rkk=np.median(np.diag(R))))
    r = summary[-1]
    print(f'  {r["alpha_rel"]:11.0e} {r["rms_err"]:8.3f} {r["rms_bias"]:9.3f} {r["rms_std"]:8.3f} '
          f'{r["noise_free_err"]:15.2e} {r["median_Rkk"]:12.3f}')
pd.DataFrame(summary).to_csv(f'{OUT}/T1_summary.csv', index=False)

per_par = pd.DataFrame(dict(param=params, area_kkm2=(par_area/1e3).round(0), v_true=v_true))
for ia, a in enumerate(ALPHAS):
    R, C = resolution_and_cov(a)
    tag = f'a{a/s[0]:.0e}'
    per_par[f'mean_{tag}'] = est[ia].mean(0)
    per_par[f'std_{tag}'] = est[ia].std(0)
    per_par[f'formal_{tag}'] = np.sqrt(np.diag(C))
    per_par[f'Rkk_{tag}'] = np.diag(R)
per_par.to_csv(f'{OUT}/T1_per_parameter.csv', index=False)


# ============================================================ 8. T2: common mode
print('\nT2: the same v in every domain; single-parameter v_ref fit and full inversion')
t2 = []
for level in [0.3, 0.6, 1.0]:
    y = G @ (level*np.ones(n_par))
    fits = np.array([fit_vref((y + noise[j])/sigma)[0] for j in range(N_NOISE)])
    full0 = np.array([solve((y + noise[j])/sigma) for j in range(N_NOISE)])
    t2.append(dict(level=level, vref_mean=fits.mean(), vref_std=fits.std(), vref_formal=fit_vref(y/sigma)[1],
                   full_mean_of_means=full0.mean(), full_rms_std=np.sqrt((full0.std(0)**2).mean()),
                   full_spread_between_domains=full0.mean(0).std()))
    r = t2[-1]
    print(f'  v = {level:.1f}: v_ref = {r["vref_mean"]:.3f} +- {r["vref_std"]:.3f} (formal {r["vref_formal"]:.3f}); '
          f'full inversion: mean {r["full_mean_of_means"]:.3f}, per-domain std {r["full_rms_std"]:.3f}')
pd.DataFrame(t2).to_csv(f'{OUT}/T2_summary.csv', index=False)
# v_ref against T1 truth: what the single uniform VIS returns when domains differ
print(f'  T1 data: v_ref = {vrefs.mean():.3f} +- {vrefs.std():.3f}; area-weighted mean of v_true = '
      f'{(v_true*par_area).sum()/par_area.sum():.3f}, plain mean {v_true.mean():.3f}')


# ============================================================ 9. T3: non-uniform domains (first principles)
# Truth inside each domain: v = v_k * exp(s*eps - s^2/2), with eps a zero-mean, unit-variance
# Gaussian random field on the sphere. Lognormal because rock susceptibility is roughly lognormal;
# eps is self-similar (2D power spectrum ~ k^-beta, i.e. power per degree ~ l^(1-beta)) from
# degree 2 to EPS_LMAX (~30 km). The -s^2/2 keeps the expected domain mean at v_k; the score is
# against the actual area-weighted mean of the truth in each domain.
T3_SIGMAS, T3_BETAS = [0.25, 0.5, 1.0], [2., 3.]
N_EPS, N_NOISE_T3, EPS_LMAX = 5, 20, 600

cache = f'{OUT}/T3_truths.npz'
if os.path.exists(cache) and np.load(cache)['y'].shape == (len(T3_SIGMAS), len(T3_BETAS), N_EPS, len(plon)):
    z = np.load(cache)
    y_t3, m_t3 = z['y'], z['m']
else:
    t = time.time()
    r_top = dm.top_of_basement_radius(ocean)
    r_base = dm.ellipsoid_radius(glat)[:, None]*np.ones_like(r_top) - BASE_DEPTH
    to_gmm = dm.induced_to_gmm(glon, glat)
    w_free = cell_area[free]
    y_t3 = np.zeros((len(T3_SIGMAS), len(T3_BETAS), N_EPS, len(plon)))
    m_t3 = np.zeros((len(T3_SIGMAS), len(T3_BETAS), N_EPS, n_par))
    eps_rng = np.random.default_rng(SEED+3)
    for ib, beta in enumerate(T3_BETAS):
        ll = np.arange(EPS_LMAX+1)
        power = np.where(ll >= 2, np.maximum(ll, 1.)**(1-beta), 0.)
        for ie in range(N_EPS):
            eps = pyshtools.SHCoeffs.from_random(power, seed=int(eps_rng.integers(2**31))).expand(
                grid='DH2', lmax=int((len(glat)-1)/2 - 1), extend=True).data
            assert eps.shape == pgrid.shape
            e = eps[free]
            e = (e - np.average(e, weights=w_free))/np.sqrt(np.average((e - np.average(e, weights=w_free))**2, weights=w_free))
            for i_s, sig in enumerate(T3_SIGMAS):
                v = np.zeros(pgrid.shape)
                v[free] = v_true[pgrid[free]]*np.exp(sig*e - sig**2/2)
                m_t3[i_s, ib, ie] = [np.average(v[pgrid == k], weights=cell_area[pgrid == k]) for k in range(n_par)]
                y_t3[i_s, ib, ie] = br_at_points(dm.slab_vis_coeffs(v, r_top, r_base, to_gmm).coeffs)
            print(f'  T3 truths beta {beta:.0f}, field {ie+1}/{N_EPS}: {time.time()-t:.0f} s', flush=True)
    np.savez(cache, y=y_t3, m=m_t3)

print('\nT3: lognormal internal variation (truth mean per domain = T1 values), inverted as uniform')
print(f'  {"sigma_ln":>8s} {"beta":>5s} {"RMS err":>8s} {"RMS bias":>9s} {"noise std":>9s} '
      f'{"max |bias|":>10s} {"RMS rel err":>11s} {"misfit/sigma":>13s}')
t3_rows, t3_est = [], np.zeros((len(T3_SIGMAS), len(T3_BETAS), N_EPS, N_NOISE_T3, n_par))
for i_s, sig in enumerate(T3_SIGMAS):
    for ib, beta in enumerate(T3_BETAS):
        chi = []
        for ie in range(N_EPS):
            for j in range(N_NOISE_T3):
                dw = (y_t3[i_s, ib, ie] + noise[j])/sigma
                t3_est[i_s, ib, ie, j] = solve(dw)
                chi.append(np.sqrt(np.mean((Gw @ t3_est[i_s, ib, ie, j] - dw)**2)))
        e = t3_est[i_s, ib]                                         # [eps, noise, par]
        m = m_t3[i_s, ib][:, None, :]
        bias = e.mean(1) - m_t3[i_s, ib]                            # [eps, par]
        t3_rows.append(dict(sigma_ln=sig, beta=beta, rms_err=np.sqrt(((e - m)**2).mean()),
                            rms_bias=np.sqrt((bias**2).mean()), noise_std=np.sqrt((e.std(1)**2).mean()),
                            max_abs_bias=np.abs(bias).max(), rms_rel_err=np.sqrt((((e - m)/m)**2).mean()),
                            misfit_over_sigma=np.mean(chi)))
        r = t3_rows[-1]
        print(f'  {sig:8.2f} {beta:5.0f} {r["rms_err"]:8.3f} {r["rms_bias"]:9.3f} {r["noise_std"]:9.3f} '
              f'{r["max_abs_bias"]:10.3f} {r["rms_rel_err"]:11.3f} {r["misfit_over_sigma"]:13.3f}')
pd.DataFrame(t3_rows).to_csv(f'{OUT}/T3_summary.csv', index=False)
chi_t1 = np.mean([np.sqrt(np.mean((Gw @ est[0, j] - (y0 + noise[j])/sigma)**2)) for j in range(N_NOISE)])
print(f'  (T1, uniform truth: misfit/sigma = {chi_t1:.3f})')

# per-domain bias (relative to the true domain mean), RMS over the random fields
t3_par = pd.DataFrame(dict(param=params, area_kkm2=(par_area/1e3).round(0)))
for i_s, sig in enumerate(T3_SIGMAS):
    for ib, beta in enumerate(T3_BETAS):
        bias = t3_est[i_s, ib].mean(1) - m_t3[i_s, ib]
        t3_par[f'relbias_s{sig}_b{beta:.0f}'] = np.sqrt(((bias/m_t3[i_s, ib])**2).mean(0))
t3_par.to_csv(f'{OUT}/T3_per_parameter.csv', index=False)


# ============================================================ 10. figures
# domains and data points
fig = pygmt.Figure()
region = [95, 160, -50, -5]
proj = 'M16c'
pg = np.where(free, pgrid, np.nan).astype(float)
grid = xr.DataArray(pg[::-1], coords=dict(lat=glat[::-1], lon=glon), dims=('lat', 'lon')).sel(
    lon=slice(region[0], region[1]), lat=slice(region[2], region[3]))
pygmt.makecpt(cmap='categorical', series=[0, n_par-1, 1])
fig.grdimage(grid, region=region, projection=proj, cmap=True, nan_transparent=True, frame=['af', '+tParameters and data points'])
fig.coast(shorelines='0.3p,gray30', resolution='i', area_thresh=1000)
fig.plot(x=plon[onshore], y=plat[onshore], style='c0.04c', fill='black')
fig.plot(x=plon[~onshore], y=plat[~onshore], style='c0.04c', fill='gray50')
for k, p in enumerate(params):
    sel = pgrid == k
    fig.text(x=float(np.average(LO[sel], weights=cell_area[sel])), y=float(np.average(LA[sel], weights=cell_area[sel])),
             text=str(k), font='7p,Helvetica-Bold,white', fill='black@40')
fig.savefig(f'{OUT}/01-domains-points.png', dpi=200)

# singular values and the common mode
fig2, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].semilogy(s/s[0], 'o-', ms=3)
ax[0].set(xlabel='index', ylabel='s / s_max', title='Singular values of G / sigma')
ax[1].bar(np.arange(n_par), common**2)
ax[1].set(xlabel='singular-vector index', ylabel='weight', title='Common mode (v = 1 everywhere)')
fig2.tight_layout()
fig2.savefig(f'{OUT}/02-singular-values.png', dpi=150)

# T1 recovery
fig3, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, ia in zip(axes, [0, 2, 4]):
    m, e = est[ia].mean(0), est[ia].std(0)
    margin = np.array([p.startswith('margin') for p in params])
    ax.errorbar(v_true[~margin], m[~margin], e[~margin], fmt='o', ms=3, label='onshore')
    ax.errorbar(v_true[margin], m[margin], e[margin], fmt='s', ms=4, color='C3', label='margins')
    lim = [0, max(v_true.max(), m.max())*1.15]
    ax.plot(lim, lim, 'k-', lw=0.5)
    ax.set(xlabel='true v (SI.km)', title=f'alpha / s_max = {ALPHAS[ia]/s[0]:.0e}', xlim=lim)
axes[0].set_ylabel('recovered v, mean +- std over noise')
axes[0].legend()
fig3.tight_layout()
fig3.savefig(f'{OUT}/03-T1-recovery.png', dpi=150)

# T3 recovery against the true domain means
fig6, axes = plt.subplots(len(T3_BETAS), len(T3_SIGMAS), figsize=(13, 8.4), sharex=True, sharey=True)
for ib, beta in enumerate(T3_BETAS):
    for i_s, sig in enumerate(T3_SIGMAS):
        ax = axes[ib, i_s]
        m, e = m_t3[i_s, ib], t3_est[i_s, ib]
        ax.errorbar(m.ravel(), e.mean(1).ravel(), e.std(1).ravel(), fmt='o', ms=2, alpha=0.6)
        lim = [0, 3]
        ax.plot(lim, lim, 'k-', lw=0.5)
        ax.set(xlim=lim, ylim=lim, title=f'sigma_ln = {sig}, beta = {beta:.0f}')
        if ib == len(T3_BETAS) - 1:
            ax.set_xlabel('true domain mean (SI.km)')
        if i_s == 0:
            ax.set_ylabel('recovered (mean +- std over noise)')
fig6.tight_layout()
fig6.savefig(f'{OUT}/06-T3-recovery.png', dpi=150)

# T1 per-domain uncertainty and resolution maps, at alpha = 0 and the middle damping
for ia, name in [(0, '04-T1-std-alpha0'), (3, '05-T1-std-alpha1e-2')]:
    R, _ = resolution_and_cov(ALPHAS[ia])
    fig = pygmt.Figure()
    for panel, (vals, label, series) in enumerate([(est[ia].std(0)/V0, 'noise std / 0.6 SI.km', [0, 0.15, 0.01]),
                                                    (np.diag(R), 'R@-kk@-', [0.8, 1, 0.01])]):
        g = np.full(pg.shape, np.nan)
        g[free] = vals[pgrid[free]]
        grd = xr.DataArray(g[::-1], coords=dict(lat=glat[::-1], lon=glon), dims=('lat', 'lon')).sel(
            lon=slice(region[0], region[1]), lat=slice(region[2], region[3]))
        pygmt.makecpt(cmap='viridis', series=series, reverse=(panel == 1))
        fig.grdimage(grd, region=region, projection=proj, cmap=True, nan_transparent=True,
                     frame=['af', f'+tT1, alpha/s@-max@- = {ALPHAS[ia]/s[0]:.0e}: {label}'])
        fig.coast(shorelines='0.3p,gray30', resolution='i', area_thresh=1000)
        fig.colorbar(frame=f'af+l{label}')
        fig.shift_origin(xshift='18c')
    fig.savefig(f'{OUT}/{name}.png', dpi=200)


# ============================================================ 11. real data: LCS-1
# Fit the domain values to LCS-1 with the background fixed. The misfit is compared with T3 to
# estimate how much internal variation (plus any other model error) the data imply; the error
# bar per domain combines the noise ensemble and the T3 bias at that level.
print('\nReal data: LCS-1, degrees 16-133')
dw_real = (d_lcs - b)/sigma
v_real = solve(dw_real)
v_ref_real, v_ref_formal = fit_vref(dw_real)
res = d_lcs - b - G @ v_real
chi_real = np.sqrt(np.mean(res**2))/sigma
original = br_at_points(np.load(f'{dm.CACHE}/VIS_{LMAX}.npy') + np.load(f'{dm.CACHE}/GK07_nr_{LMAX}.npy'))
rms = lambda x, sel=slice(None): np.sqrt(np.mean(x[sel]**2))/sigma
print(f'  misfit / sigma: background only {rms(d_lcs - b):.2f}; uniform v_ref = {v_ref_real:.3f} SI.km '
      f'{rms(d_lcs - b - v_ref_real*G @ np.ones(n_par)):.2f}; {n_par} domains {chi_real:.2f} '
      f'(onshore points {rms(res, onshore):.2f}, buffer {rms(res, ~onshore):.2f})')
print(f'  reference only, existing model (H&M VIS in Australia, GK07_NR): {rms(d_lcs - original):.2f}')
print(f'  correlation with LCS-1: domains {np.corrcoef(d_lcs, b + G @ v_real)[0, 1]:.3f}, '
      f'existing model {np.corrcoef(d_lcs, original)[0, 1]:.3f}')
print(f'  domains at the v = 0 bound: {[params[k] for k in np.where(v_real < 1e-6)[0]]}')

# noise spread: refit with each noise realisation added to the real data's best-fit prediction
noise_std = np.array([solve((G @ v_real + noise[j])/sigma) for j in range(N_NOISE)]).std(0)

# internal-variation level implied by the misfit, per beta, by interpolating T3's misfit curve
t3 = pd.DataFrame(t3_rows)
sig_est, model_err = {}, {}
for beta in T3_BETAS:
    tb = t3[t3.beta == beta].sort_values('sigma_ln')
    xs = np.r_[0., tb.sigma_ln]
    ys = np.r_[chi_t1, tb.misfit_over_sigma]
    sig_est[beta] = np.interp(chi_real, ys, xs)
    relb = np.vstack([np.zeros(n_par)] + [t3_par[f'relbias_s{sg}_b{beta:.0f}'] for sg in tb.sigma_ln])
    model_err[beta] = np.array([np.interp(sig_est[beta], xs, relb[:, k]) for k in range(n_par)])*v_real
    flag = ' (beyond the tested range)' if chi_real > ys.max() else ''
    print(f'  beta = {beta:.0f}: misfit implies sigma_ln ~ {sig_est[beta]:.2f}{flag}')
err_model = np.maximum(model_err[T3_BETAS[0]], model_err[T3_BETAS[1]])     # the larger of the two
err_total = np.sqrt(noise_std**2 + err_model**2)

real = pd.DataFrame(dict(param=params, area_kkm2=(par_area/1e3).round(0), v=v_real, noise_std=noise_std,
                         model_err=err_model, total_err=err_total))
real['rel_err'] = real.total_err/np.maximum(real.v, 1e-6)
real.to_csv(f'{OUT}/real_LCS1.csv', index=False)
print(real.round(3).sort_values('v', ascending=False).to_string(index=False))

# maps: recovered v and relative error; observed, modelled and residual Br at the points
fig = pygmt.Figure()
for panel, (vals, label, cmap, series) in enumerate([
        (v_real, 'v (SI.km)', 'viridis', [0, np.ceil(v_real.max()*10)/10, 0.1]),
        (np.minimum(real.rel_err.values, 1.), 'error / v (capped at 1)', 'hot', [0, 1, 0.05])]):
    g = np.full(pg.shape, np.nan)
    g[free] = vals[pgrid[free]]
    grd = xr.DataArray(g[::-1], coords=dict(lat=glat[::-1], lon=glon), dims=('lat', 'lon')).sel(
        lon=slice(region[0], region[1]), lat=slice(region[2], region[3]))
    pygmt.makecpt(cmap=cmap, series=series, reverse=(cmap == 'hot'))
    fig.grdimage(grd, region=region, projection=proj, cmap=True, nan_transparent=True,
                 frame=['af', f'+tLCS-1 fit: {label}'])
    fig.coast(shorelines='0.3p,gray30', resolution='i', area_thresh=1000)
    fig.colorbar(frame=f'af+l{label}')
    fig.shift_origin(xshift='18c')
fig.savefig(f'{OUT}/07-real-v.png', dpi=200)

fig = pygmt.Figure()
lim = np.ceil(np.percentile(np.abs(d_lcs), 99)/10)*10
pygmt.makecpt(cmap='polar', series=[-lim, lim], background=True)
for title, vals in [('LCS-1 Br', d_lcs), ('model Br (domains + background)', b + G @ v_real),
                    ('residual', res), ('existing model residual (reference)', d_lcs - original)]:
    fig.plot(x=plon, y=plat, style='c0.09c', fill=vals, cmap=True, region=region, projection='M11c',
             frame=['af', f'+t{title}'])
    fig.coast(shorelines='0.3p,gray30', resolution='i', area_thresh=1000)
    fig.shift_origin(xshift='12.5c')
fig.colorbar(frame='af+lnT', position='JMR+o1c/0c')
fig.savefig(f'{OUT}/08-real-Br.png', dpi=200)
print(f'\nfigures and tables in {OUT}/')
