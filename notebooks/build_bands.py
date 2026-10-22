"""
Variants of the cached problem: other degree ranges and data models ([bands] in the region config).
Each run gets cache/bands/<name>/ with data.npz, noise.npz, G.npz and errors.npz, made as build_region.py
and error_model.py make the first build (same cells, data points, background and cell columns), so that
every run script works on it with env BAND=<name>.

  data       the run's model (LCS-1 or MLI 2E) as Br at the points, degrees lmin-lmax, its filter applied;
             the background (H&M + slabs outside the extent, GK07_NR) in the same band and filter
  filter     LCS-1: the filter fitted to its damping (as the first build), applied to data, G and noise;
             MLI 2E: none (its spectrum is flat to beyond degree 300)
  noise      'mf7'      LCS-1 - MF7, as the first build (degrees <= 133)
             'lcs1-2e'  the disagreement of the two models, LCS-1 - F 2E (F the LCS-1 filter, so that its
                        damping is not counted as error), as coefficients of the run's model: for LCS-1
                        data the reference is F 2E, for 2E data it is LCS-1 / F. Same spectrum for both.
             Above LCS-1's last degree (185) there is no second model: the noise is held at the ratio of
             noise to signal (the run's model) of degrees 150-185, R_noise(l) = ratio x R_model(l).
             sigma = RMS(data - reference) at the points (plus that ratio x the data's mean square above
             185); realisations with the same spectrum and filter
  synthetic  no real data (d = b = 0) and no model-error term (noise-only error model). Below degree 16
             the noise spectrum is held at its degree-16 value, and sigma is raised so that the noise at
             degrees >= 16 is exactly that of the same run from degree 16 (same random draws).
  noise_fraction = f (future data, synthetic runs): at every degree the noise is today's ('lcs1-2e'
             below 186) or f times the signal amplitude (the run's model), whichever is smaller:
             R_noise(l) = min(R_today(l), f^2 R_model(l)); above 185, f^2 R_model(l). sigma = today's
             sigma times the RMS ratio of realisations with the two spectra (same random draws)
  compress = "signal": the data compression keeps 99.9% of the trace of noise + signal (the run's model)
             rather than of the noise alone (needed when the noise is low at high degrees)
Spherical geometry: everything is in spherical harmonics or at points on the sphere, as in build_region.py.

Run from notebooks/ (conda env pygmt17), after build_region.py (and the cell columns to [bands] lmax_cell):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 15 \
    env PYTHONPATH=..:$PYTHONPATH python build_bands.py regions/australia/config.toml
env RUNS=a,b builds only those runs; existing files are kept (delete a run's directory to rebuild it).
Cell columns beyond the first build's lmax_cell are computed first with N_WORKERS (default 3) processes;
with columns = "direct" in the config, G is computed straight from the cells' nodes instead (no column files).
"""
import os
import sys
import tomllib

import numpy as np

import depth_models as dm
import error_model
from domains import basis, errors

cfg_path = sys.argv[1]
with open(cfg_path, 'rb') as f:
    cfg = tomllib.load(f)
REG = os.path.dirname(os.path.abspath(cfg_path))
CACHE = f'{REG}/cache'
B = cfg['bands']
RUNS = os.environ.get('RUNS', ','.join(B['runs'])).split(',')
LMIN_REAL = 16                          # the lowest degree of MF7 and MLI 2E
LMAX_LCS1 = dm.LMAX                     # the last degree of LCS-1 (185)
RATIO_DEGREES = slice(150, LMAX_LCS1 + 1)
DIRECT = cfg.get('columns', 'global') == 'direct'

z0 = np.load(f'{CACHE}/data.npz')
plon, plat = z0['plon'], z0['plat']
cells = np.load(f'{CACHE}/cells.npz')
C_bg = np.load(f'{CACHE}/C_bg.npy')
if max(B['runs'][r]['lmax'] for r in RUNS) > C_bg.shape[1] - 1:      # background to the highest degree of the runs
    lmax_bg = max(B['runs'][r]['lmax'] for r in RUNS)
    f = f'{CACHE}/C_bg_l{lmax_bg}.npy'
    if not os.path.exists(f):
        np.save(f, basis.background_coeffs(np.load(f'{CACHE}/extent.npz')['ext'], basis.ocean_grid()[0], lmax_bg))
    C_bg = np.load(f)
L_H = dm.fit_lh()


def filt(kind, lmax):
    return dm.lcs1_filter(np.arange(lmax+1), L_H) if kind == 'lcs1' else np.ones(lmax+1)


def coeffs(model, lmax):
    return basis.field_model(os.path.expanduser(B['models'][model]['coeffs']) if model in B['models'] else model, lmax)


# cell columns to [bands] lmax_cell when a run goes beyond the first build's (resumable, as build_region.py)
ocean = basis.ocean_grid()[0]
if not DIRECT and max(B['runs'][r]['lmax'] for r in RUNS) > cfg['lmax_cell']:
    assert max(B['runs'][r]['lmax'] for r in RUNS) <= B['lmax_cell']
    basis.compute_cell_columns(cells['cell'], cells['cell_ids'], ocean, f'{REG}/{B["cell_dir"]}', cfg['base_depth_km'],
                               B['lmax_cell'], int(os.environ.get('N_WORKERS', 3)))


for name in RUNS:
    run = B['runs'][name]
    out = f'{CACHE}/bands/{name}'
    os.makedirs(out, exist_ok=True)
    lmin, lmax, data_model = run['lmin'], run['lmax'], run['data']
    synthetic = run.get('synthetic', False)
    noise_kind = run.get('noise', B['noise'])
    F = filt(B['models'][data_model]['filter'], lmax)
    c = coeffs(data_model, lmax)
    if noise_kind == 'mf7':
        assert data_model == 'lcs1' and lmax <= 133, 'MF7 stops at degree 133'
        ref = coeffs('mf7', lmax)
    else:
        other = '2e' if data_model == 'lcs1' else 'lcs1'
        F_lcs = filt('lcs1', lmax)
        ref = coeffs(other, lmax)*(F_lcs if data_model == 'lcs1' else 1/F_lcs)[None, :, None]
    ref[:, LMAX_LCS1+1:] = 0                                       # no second model above LCS-1's last degree
    diff = c - ref
    diff[:, :LMIN_REAL] = 0
    diff[:, LMAX_LCS1+1:] = 0
    R_noise, R_field = errors.lowes(diff), errors.lowes(c)
    R_noise[1:LMIN_REAL] = R_noise[LMIN_REAL]                      # used by synthetic runs only (lmin < 16)
    ratio = float(R_noise[RATIO_DEGREES].sum()/R_field[RATIO_DEGREES].sum()) if lmax > LMAX_LCS1 else 0.
    R_noise[LMAX_LCS1+1:] = ratio*R_field[LMAX_LCS1+1:]
    f_noise = run.get('noise_fraction')
    R_today = R_noise.copy()
    if f_noise is not None:
        assert synthetic, 'noise_fraction: synthetic runs only'
        R_noise[LMAX_LCS1+1:] = np.inf
        R_noise[LMIN_REAL:] = np.minimum(R_noise[LMIN_REAL:], f_noise**2*R_field[LMIN_REAL:])
    F_noise = None if noise_kind == 'mf7' else F                   # the first build's realisations are unfiltered
    print(f'== {name}: {data_model}, degrees {lmin}-{lmax}, noise {noise_kind}' + (' (synthetic only)' if synthetic else '') +
          (f', noise at most {f_noise:g} x the signal amplitude' if f_noise is not None else ''), flush=True)

    f = f'{out}/data.npz'
    if not os.path.exists(f):
        lo = max(lmin, LMIN_REAL)
        top = min(lmax, LMAX_LCS1)
        sigma = float(np.sqrt(np.mean((basis.br_points(c, plon, plat, lo, top, F) -
                                       basis.br_points(ref, plon, plat, lo, top, F))**2)))
        if lmax > LMAX_LCS1:
            sigma = float(np.sqrt(sigma**2 + ratio*np.mean(basis.br_points(c, plon, plat, LMAX_LCS1 + 1, lmax, F)**2)))
            print(f'  noise above degree {LMAX_LCS1}: {ratio:.2f} x the signal power (ratio of degrees 150-{LMAX_LCS1})')
        if f_noise is not None:                                    # today's sigma x the ratio of the spectra (same draws)
            ms = lambda R: np.mean(basis.noise_realisations(plon, plat, lmin, lmax, 20, cfg['seed'], R, F_noise)**2)
            sigma *= float(np.sqrt(ms(R_noise)/ms(R_today)))
        if synthetic:
            d = b = d_ref = np.zeros(len(plon))
            if lmin < LMIN_REAL:
                n_all = basis.noise_realisations(plon, plat, lmin, lmax, 20, cfg['seed'], R_noise, F_noise)
                n_hi = basis.noise_realisations(plon, plat, LMIN_REAL, lmax, 20, cfg['seed'], R_noise, F_noise)
                sigma *= float(np.sqrt(np.mean(n_all**2)/np.mean(n_hi**2)))
        else:
            d = basis.br_points(c, plon, plat, lmin, lmax, F)
            d_ref = basis.br_points(ref, plon, plat, lmin, lmax, F)
            b = basis.br_points(C_bg, plon, plat, lmin, lmax, F)
        np.savez(f, plon=plon, plat=plat, b=b, d=d, d_ref=d_ref, sigma=sigma, lmin=lmin, lmax=lmax, F=F,
                 R_noise=R_noise, R_field=R_field, data_model=data_model, noise_kind=noise_kind, synthetic=synthetic,
                 noise_fraction=np.nan if f_noise is None else f_noise)
    z = np.load(f)
    print(f'  {len(plon)} points; sigma {float(z["sigma"]):.2f} nT; RMS data - background '
          f'{np.sqrt(np.mean((z["d"] - z["b"])**2)):.1f} nT', flush=True)

    f = f'{out}/noise.npz'
    if not os.path.exists(f):
        np.savez(f, noise=basis.noise_realisations(plon, plat, lmin, lmax, cfg['n_noise'], cfg['seed'], R_noise, F_noise))

    f = f'{out}/G.npz'
    if not os.path.exists(f):
        if DIRECT:
            G = basis.cell_matrix_direct(cells['cell'], len(cells['cell_ids']), ocean, cfg['base_depth_km'], plon, plat,
                                         lmin, lmax, F, n_workers=int(os.environ.get('N_WORKERS', 3)))
        else:
            cell_dir = f'{CACHE}/cells' if lmax <= cfg['lmax_cell'] else f'{REG}/{B["cell_dir"]}'
            G = basis.cell_matrix(cell_dir, cells['cell_ids'], plon, plat, lmin, lmax, F)
        np.savez(f, G=G, plon=plon, plat=plat, cell_ids=cells['cell_ids'])
    G = np.load(f)['G']

    if not os.path.exists(f'{out}/errors.npz'):
        r = None
        if not synthetic:
            r, K = error_model.reference_residual(G, z['d'], z['b'], cells)
            print(f'  reference fit: {K} domains')
        s2_c = None
        if run.get('compress') == 'signal':
            s2_c = float(np.mean(basis.br_points(c, plon, plat, max(lmin, LMIN_REAL), lmax, F)**2))
        error_model.build(out, plon, plat, lmin, lmax, R_noise, R_field, float(z['sigma']),
                          np.load(f'{out}/noise.npz')['noise'], r=r, choice=cfg.get('model_error_shape'), F=F,
                          plot=f'{out}/error_model.png', label=noise_kind, s2_compress=s2_c)
    print(f'  done: G {G.shape}, {np.load(f"{out}/errors.npz")["T"].shape[0]} dimensions', flush=True)
