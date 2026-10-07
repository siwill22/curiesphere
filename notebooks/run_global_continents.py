"""
Global continental VIS from LCS-1 Gauss coefficients: a simplified Olsen, Ravat & Purucker (2023, EPSL).

  unknowns   VIS (SI.km) of the HEALPix cells over all crust without an ocean age (continents, shelves and
             un-aged plateaus), spread uniformly from the top of basement to base_depth below the ellipsoid,
             induced by IGRF-13 (basis.direct_columns). Not dipoles at 70 km as in the paper.
  oceans     the existing model, held fixed and subtracted: ocean VIS + slabs under ocean-age crust + GK07_NR
             remanence (basis.background_coeffs with the continents removed)
  data       LCS-1 Gauss coefficients, degrees lmin-lmax; the LCS-1 filter F(l) applied to G and background
  weights    an isotropic noise field is diagonal in Gauss coefficients: var = R_noise(l)/((l+1)(2l+1)),
             R_noise the Lowes spectrum of LCS-1 - F 2E (as build_bands.py 'lcs1-2e')
  solution   min |W^1/2 (g - G m)|^2 + alpha^2 sum_k a_k (m_k - v0)^2, a_k the cell area / mean area, v0 free
             (no thickness prior). With x = sqrt(a)(m - v0) and B = W^1/2 G A^-1/2, one eigendecomposition of
             B'B (or of BB' when there are fewer data than cells) gives every alpha; v0 is profiled in closed
             form. Posterior s.d. per cell (given v0) with prior s.d. 1/alpha for x.
  alphas     'disc'  chi2 = number of data (discrepancy principle)
             'evid'  largest evidence, with the noise scale sigma fitted too (x ~ N(0, tau^2), alpha^2 = sigma^2/tau^2):
                     the discrepancy principle is weak here, chi2 changes little over many decades of alpha because
                     much of the misfit is field over the oceans that continental sources cannot fit
             'min'   RMS Br misfit at the surface (unweighted, in the band) within minimal_rms_nT of its floor, the
                     paper's criterion (fit to 1 nT) adapted: continental sources alone cannot fit the field over
                     the oceans, so the misfit has a floor well above 1 nT
             alpha^2 is searched above ALPHA2_FLOOR x the largest eigenvalue, where eigenvalues are above rounding
  cases      real data, and synthetic truths with one noise draw: Crust1.0 thickness x chi; uniform 1.0 and
             1.5 SI.km; uniform 1.0 with an ocean-model error (true remanence HM05_NR, GK07_NR subtracted)
Since the oceans are fixed, a uniform continental VIS is not an annihilator: the data see the level through
the continent-ocean contrast, conditional on the ocean model.
Spherical geometry: spherical harmonics and equal-area HEALPix cells; source radii on the WGS84 ellipsoid.

Run from notebooks/ (conda env pygmt17):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 15 \
    env PYTHONPATH=..:$PYTHONPATH python run_global_continents.py regions/global_continents/config.toml
env BAND=<name> picks a [bands] entry (default the first). Every step is cached in <region>/cache/.
"""
import os
import sys
import time
import tomllib
import multiprocessing

import numpy as np
import xarray as xr
from astropy_healpix import healpy as hp
from scipy.linalg import eigh

import depth_models as dm
from domains import basis, errors

cfg_path = sys.argv[1]
with open(cfg_path, 'rb') as f:
    cfg = tomllib.load(f)
REG = os.path.dirname(os.path.abspath(cfg_path))
CACHE = f'{REG}/cache'
BAND = os.environ.get('BAND', next(iter(cfg['bands'])))
OUT = f'{CACHE}/{BAND}'
os.makedirs(f'{CACHE}/G', exist_ok=True)
os.makedirs(OUT, exist_ok=True)
lmin, lmax = cfg['bands'][BAND]['lmin'], cfg['bands'][BAND]['lmax']
LMAX_CELL = cfg['lmax_cell']
assert lmax <= LMAX_CELL
t0 = time.time()


def log(msg):
    print(f'[{(time.time() - t0)/60:5.1f} min] {msg}', flush=True)


ocean, glon, glat, age = basis.ocean_grid()

# 1. continental cells: every 0.1 deg node without an ocean age, less the south-pole row and the 360 deg
#    column (not in the Driscoll-Healy sums of direct_columns; the 360 column repeats 0)
f = f'{CACHE}/cells.npz'
if not os.path.exists(f):
    ext = np.isnan(age)
    ext[-1, :] = False
    ext[:, -1] = False
    cell, cell_ids, area = basis.make_cells(ext, glon, glat, cfg['nside_cell'])
    clon, clat = (np.asarray(x, float) for x in hp.pix2ang(cfg['nside_cell'], cell_ids, lonlat=True))
    np.savez_compressed(f, cell=cell, cell_ids=cell_ids, area_km2=area, clon=clon, clat=clat, nside=cfg['nside_cell'])
z = np.load(f)
cell, cell_ids, area, clon, clat = z['cell'], z['cell_ids'], z['area_km2'], z['clon'], z['clat']
ext = cell >= 0
K = len(cell_ids)
log(f'{K} continental cells (NSIDE {cfg["nside_cell"]}), {area.sum()/1e6:.1f} M km2')

# 2. cell columns: packed Gauss coefficients (g, h) of 1 SI.km in each cell to LMAX_CELL, in blocks (resumable)
BLOCK = 200
blocks = [np.arange(k, min(k + BLOCK, K)) for k in range(0, K, BLOCK)]
_job = {}


def _block(b):
    fb = f'{CACHE}/G/{b:04d}.npy'
    if not os.path.exists(fb):
        np.save(fb + '.tmp.npy', basis.direct_columns(_job['cell'], blocks[b], _job['S']))
        os.replace(fb + '.tmp.npy', fb)
    return b


todo = [b for b in range(len(blocks)) if not os.path.exists(f'{CACHE}/G/{b:04d}.npy')]
if todo:
    log(f'{len(todo)} of {len(blocks)} column blocks to compute with {cfg["n_workers"]} workers')
    _job.update(cell=cell, S=basis.direct_setup(ocean, cfg['base_depth_km'], LMAX_CELL))
    with multiprocessing.get_context('fork').Pool(cfg['n_workers']) as pool:
        for n, _ in enumerate(pool.imap_unordered(_block, todo), 1):
            if n % 10 == 0 or n == len(todo):
                log(f'  {n}/{len(todo)} blocks')

# forward check: direct columns against the global transform (dm.slab_vis_coeffs) for a mid-latitude cell
# and the cell nearest the south pole
f = f'{CACHE}/forward_check.txt'
if not os.path.exists(f):
    L, M = basis.packed_lm(LMAX_CELL)
    r_top = dm.top_of_basement_radius(ocean)
    r_base = dm.ellipsoid_radius(glat)[:, None]*np.ones_like(r_top) - cfg['base_depth_km']*1e3
    to_gmm = dm.induced_to_gmm(glon, glat)
    lines = []
    for i in (int(np.argmin(np.abs(clat + 25))), int(np.argmin(clat))):
        ref = dm.slab_vis_coeffs(np.where(cell == i, 1., 0.), r_top, r_base, to_gmm, lmax=LMAX_CELL).coeffs
        col = np.load(f'{CACHE}/G/{i//BLOCK:04d}.npy')[:, i % BLOCK]
        err = max(np.abs(col[0] - ref[0, L, M]).max(), np.abs(col[1] - ref[1, L, M]).max())/np.abs(ref).max()
        lines.append(f'cell {cell_ids[i]} (lon {clon[i]:.1f}, lat {clat[i]:.1f}): max relative difference {err:.2e}')
        assert err < 1e-6, lines[-1]
    open(f, 'w').write('\n'.join(lines) + '\n')
log('forward check:\n' + open(f).read().rstrip())

# 3. background (the oceans, held fixed) to LMAX_CELL
f = f'{CACHE}/C_bg.npy'
if not os.path.exists(f):
    log('background coefficients (ocean VIS + slabs under ocean crust + GK07_NR)')
    np.save(f, basis.background_coeffs(ext, ocean, lmax=LMAX_CELL))
C_bg = np.load(f)
C_ocean_err = (np.load(f'{dm.CACHE}/HM05_nr_{dm.LMAX}.npy') - np.load(f'{dm.CACHE}/GK07_nr_{dm.LMAX}.npy'))

# 4. band: rows (g then h), filter, data, noise weights
L, M = basis.packed_lm(LMAX_CELL)
sel = (L >= lmin) & (L <= lmax)
selh = sel & (M > 0)
row_l = np.concatenate([L[sel], L[selh]])
row_m = np.concatenate([M[sel], M[selh]])
row_gh = np.concatenate([np.zeros(sel.sum(), int), np.ones(selh.sum(), int)])
NDATA = len(row_l)
F = dm.lcs1_filter(np.arange(LMAX_CELL + 1), dm.fit_lh())


def rows(c):
    """band rows of a (2, l, m) coefficient array"""
    return np.where(row_gh == 0, c[0, row_l, row_m], c[1, row_l, row_m])


c_data = basis.field_model(cfg['data'], LMAX_CELL)
ref = basis.read_shc(cfg['noise_ref'], LMAX_CELL)*F[None, :, None]
diff = c_data - ref
diff[:, :16] = 0
R_noise = errors.lowes(diff)
sigma_l = np.sqrt(R_noise/((np.arange(LMAX_CELL + 1) + 1)*(2*np.arange(LMAX_CELL + 1) + 1)))
w = 1/sigma_l[row_l]                                          # whitening, per row
brw = (row_l + 1.)**2/(2*row_l + 1)                           # mean square Br at r0 per squared coefficient
g_data = rows(c_data) - F[row_l]*rows(C_bg)
log(f'band {BAND}: degrees {lmin}-{lmax}, {NDATA} data, {K} cells; '
    f'noise RMS Br {np.sqrt(np.sum(brw*sigma_l[row_l]**2)):.2f} nT')

# whitened, area-scaled matrix B = W^1/2 F G A^-1/2 (G itself is not kept)
a = area/area.mean()
B = np.empty((NDATA, K))
for b, idx in enumerate(blocks):
    col = np.load(f'{CACHE}/G/{b:04d}.npy')
    B[:, idx] = np.concatenate([col[0][:, sel], col[1][:, selh]], axis=1).T
B *= (w*F[row_l])[:, None]
B /= np.sqrt(a)[None, :]
u = B @ np.sqrt(a)                                            # whitened field of a uniform 1 SI.km

# 5. eigendecomposition of the row space: Q (cells x r) orthonormal, lam > 0; the other K - r directions have lam = 0
f = f'{OUT}/eig.npz'
if not os.path.exists(f):
    log(f'eigendecomposition ({"BB^T" if NDATA <= K else "B^TB"}, {min(NDATA, K)}^2)')
    if NDATA <= K:
        lam, U = eigh(B @ B.T, overwrite_a=True, check_finite=False)
        keep = lam > lam.max()*1e-13
        lam, Q = lam[keep], (B.T @ U[:, keep])/np.sqrt(lam[keep])
        del U
    else:
        lam, Q = eigh(B.T @ B, overwrite_a=True, check_finite=False)
        keep = lam > lam.max()*1e-13
        lam, Q = lam[keep], Q[:, keep]
    np.save(f'{OUT}/Q.npy', Q)
    np.savez(f, lam=lam)
lam = np.load(f)['lam']
Q = np.load(f'{OUT}/Q.npy')
log(f'{len(lam)} constrained directions, {K - len(lam)} unseen; eigenvalues {lam.min():.2e} - {lam.max():.2e}')
q = Q.T @ (B.T @ u)


def solve(y, alpha2):
    """map for whitened data y at alpha^2: m, v0, chi2, RMS Br misfit (nT), s.d. of v0"""
    p = Q.T @ (B.T @ y)
    den = lam + alpha2
    prec_v0 = u @ u - np.sum(q*q/den)
    v0 = (u @ y - np.sum(p*q/den))/prec_v0
    x = Q @ ((p - v0*q)/den)
    r = y - v0*u - B @ x
    return dict(m=v0 + x/np.sqrt(a), v0=v0, chi2=float(r @ r), rms=float(np.sqrt(np.sum(brw*(r/w)**2))),
                sd_v0=float(1/np.sqrt(prec_v0)), r=r/w, J=float(r @ r + alpha2*(x @ x)))


def posterior_sd(alpha2, chunk=2000):
    """s.d. of m per cell given v0: var x_k = 1/alpha2 - sum_i Q_ki^2 lam_i/(alpha2 (lam_i + alpha2))"""
    s = lam/(alpha2*(lam + alpha2))
    var = np.concatenate([1/alpha2 - (Q[k:k + chunk]**2) @ s for k in range(0, K, chunk)])
    return np.sqrt(np.maximum(var, 0)/a)


ALPHA2_FLOOR = 1e-10


def find_alpha2(y, key, target, n=50):
    """alpha^2 (bisection in log) at which solve(y)[key] equals target (both increase with alpha)"""
    lo, hi = np.log10(lam.max()*ALPHA2_FLOOR), np.log10(lam.max()) + 4
    for _ in range(n):
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if solve(y, 10**mid)[key] < target else (lo, mid)
    return 10**(0.5*(lo + hi))


def log_evidence(y, alpha2):
    """log marginal likelihood (up to a constant) of y = v0 u + B x + e, x ~ N(0, tau^2), e ~ N(0, sigma^2),
    alpha^2 = sigma^2/tau^2, with sigma^2 at its maximum (J/N) and v0 at its estimate"""
    return -0.5*(NDATA*np.log(solve(y, alpha2)['J']/NDATA) + np.sum(np.log1p(lam/alpha2)))


def find_evidence(y):
    """alpha^2 of the largest evidence: a grid in log alpha^2, then a bounded search around its best point"""
    from scipy.optimize import minimize_scalar
    grid = np.linspace(np.log10(lam.max()*ALPHA2_FLOOR), np.log10(lam.max()) + 4, 29)
    k = int(np.argmax([log_evidence(y, 10**g) for g in grid]))
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    return 10**minimize_scalar(lambda g: -log_evidence(y, 10**g), bounds=(lo, hi), method='bounded').x


# 6. cases
rng = np.random.default_rng(cfg['seed'])
noise = rng.standard_normal(NDATA)
crust = xr.open_dataarray(os.path.expanduser(cfg['crust']))
h_crust = crust.sel(lon=xr.DataArray(((clon + 180) % 360) - 180), lat=xr.DataArray(clat), method='nearest').values
assert np.isfinite(h_crust).all()
y_ocean_err = w*F[row_l]*rows(C_ocean_err)
truths = {'syn_crust': cfg['synthetic_chi']*h_crust, 'syn_u1.0': np.ones(K), 'syn_u1.5': np.full(K, 1.5),
          'syn_u1.0_oceanerr': np.ones(K)}
cases = {'real': w*g_data}
for name, mt in truths.items():
    cases[name] = B @ (np.sqrt(a)*mt) + noise + (y_ocean_err if name.endswith('oceanerr') else 0)

weak = np.flatnonzero(lam > lam.max()*ALPHA2_FLOOR)[:4]                # weakest directions above rounding
res = dict(cell_ids=cell_ids, clon=clon, clat=clat, area_km2=area, lam=lam, n_unseen=K - len(lam),
           weak=Q[:, weak]/np.sqrt(a)[:, None], weak_lam=lam[weak], row_l=row_l, row_m=row_m, row_gh=row_gh, g_data=g_data,
           sigma_l=sigma_l, lmin=lmin, lmax=lmax, ndata=NDATA, h_crust=h_crust)
sweep = 10**np.linspace(np.log10(lam.max()*ALPHA2_FLOOR), np.log10(lam.max()) + 2, 25)
for name, y in cases.items():
    floor = solve(y, lam.max()*ALPHA2_FLOOR)['rms']
    res[f'{name}_rms_floor'] = floor
    for kind, a2 in (('disc', find_alpha2(y, 'chi2', NDATA)), ('evid', find_evidence(y)),
                     ('min', find_alpha2(y, 'rms', floor + cfg['minimal_rms_nT']))):
        s = solve(y, a2)
        key = f'{name}_{kind}'
        sig = np.sqrt(s['J']/NDATA) if kind == 'evid' else 1.          # noise scale: fitted for 'evid'
        res.update({f'{key}_m': s['m'], f'{key}_v0': s['v0'], f'{key}_sd_v0': s['sd_v0']*sig, f'{key}_alpha2': a2,
                    f'{key}_chi2': s['chi2'], f'{key}_rms': s['rms'], f'{key}_sigma': sig,
                    f'{key}_tau': sig/np.sqrt(a2)})
        extra = ''
        if name == 'real':
            res[f'{key}_sd'] = posterior_sd(a2)*sig
            res[f'{key}_resid'] = s['r']
        else:
            res[f'{name}_truth'] = truths[name]
            extra = f', RMS(m - truth) {np.sqrt(np.average((s["m"] - truths[name])**2, weights=a)):.3f}, ' \
                    f'true mean {np.average(truths[name], weights=a):.3f}'
        log(f'{key:22s} RMS floor {floor:.2f} nT, alpha2 {a2:.2e} (noise x {sig:.2f}, prior s.d. {sig/np.sqrt(a2):.2f} SI km): '
            f'chi2/N {s["chi2"]/NDATA:.2f}, RMS Br misfit {s["rms"]:.2f} nT, '
            f'v0 {s["v0"]:.3f} +- {s["sd_v0"]*sig:.3f}, mean VIS {np.average(s["m"], weights=a):.3f}, '
            f'negative area {a[s["m"] < 0].sum()/a.sum():.1%}{extra}')
    if name == 'real':
        tab = np.array([[a2, (s := solve(y, a2))['chi2']/NDATA, s['rms'], s['v0']] for a2 in sweep])
        res['real_sweep'] = tab
        log('real data: alpha2, chi2/N, RMS Br misfit, v0\n' +
            '\n'.join(f'  {t[0]:.2e}  {t[1]:8.3f}  {t[2]:7.2f}  {t[3]:.3f}' for t in tab))
np.savez(f'{OUT}/results.npz', **res)
log(f'wrote {OUT}/results.npz')
