"""
GCM (global cell map, CONTEXT.md): VIS per cell on all continental crust from a global fit (ADR 0002): the Gauss
coefficients of a field model in a band, the ocean model fixed in the background, damped toward one fitted uniform
level v0 with no thickness prior; the damping by the evidence, the discrepancy principle, or given.

Config keys used:
  [target] area, nside;  [geometry] layers (one), direction, lmax;  [compute] columns_backend, products_backend,
  solver, coarsen;  [fit.background] vis_model, remanence, near_ridge;
  [data] model ('lcs1', 'mf7', a .shc path, or 'synthetic'), band, filter, and for synthetic data: based_on (the
         model whose filter and noise spectrum are used), truth, seed, ocean_error (a remanence model taken as the
         truth in the oceans while the background keeps its own);
  [errors.noise] reference (a field model), amplitude ('evidence': a noise scale fitted with the model; 'fixed');
  [method] damping ('evidence', 'discrepancy' or alpha^2)
"""
import os
import time

import numpy as np
from scipy import sparse

from .. import columns, core, data, errors, geometry, results, truths
from ..cache import Cache, save_atomic

EIG_VERSION = 2                     # 2: the eigenvectors of the decomposed side (data or cells), not always cells


def run(cfg, cache=None, log=print):
    cache = cache or Cache()
    t0 = time.time()
    say = lambda msg: log(f'[{(time.time() - t0)/60:5.1f} min] {msg}')
    tg, geo, comp, d = cfg['target'], cfg['geometry'], cfg['compute'], cfg['data']
    if len(geo['layers']) != 1:
        raise NotImplementedError('one layer')
    mask = geometry.target_mask(tg['area'])
    cells = geometry.make_cells(mask, tg['nside'])
    store = columns.ColumnStore(cells, tg['area'], geo['layers'][0], geo['direction'], geo['lmax'],
                                comp['columns_backend'], cache).build(say)
    lo, hi = d['band']
    band = data.Band(lo, hi)
    synthetic = d['model'] == 'synthetic'
    model = d.get('based_on', 'lcs1') if synthetic else d['model']
    F = data.resolution_filter(d.get('filter', 'lcs1'), geo['lmax'])
    c_model = data.field_coeffs(model, geo['lmax'])
    noise = cfg['errors']['noise']
    R = errors.noise_spectrum(c_model, data.field_coeffs(noise['reference'], geo['lmax']), F)
    w = 1/np.sqrt(errors.coefficient_variance(R)[band.l])
    say(f'{len(cells)} cells (NSIDE {cells.nside}), band {lo}-{hi}: {len(band)} data; '
        f'noise RMS Br {np.sqrt(np.sum(band.mean_square_br()/w**2)):.2f} nT')

    # whitened, area-scaled matrix B = W^1/2 F G A^-1/2, built block by block from the column store
    fine, parent = cells, None
    for _ in range(comp.get('coarsen', 0)):
        cells, p = geometry.coarsen(cells)
        parent = p if parent is None else p[parent]
    G = np.zeros((len(band), len(cells)))
    for b, idx in enumerate(store.blocks):
        Gb = band.columns(store.get(idx))
        if parent is None:
            G[:, idx] = Gb
        else:                                                 # coarse columns: sums of their children's
            S = sparse.csr_matrix((np.ones(len(idx)), (parent[idx], np.arange(len(idx)))), shape=(len(cells), len(idx)))
            G += (S @ Gb.T).T
    a = cells.area_km2/cells.area_km2.mean()
    B = G*(w*F[band.l])[:, None]/np.sqrt(a)[None, :]
    del G
    u = B @ np.sqrt(a)

    # data
    bg = cfg['fit']['background']
    C_bg = data.background(cache, mask, tg['area'], geo['lmax'], bg['vis_model'], bg['remanence'], bg['near_ridge'])
    truth = None
    if synthetic:
        truth_fine = truths.truth(d['truth'], fine)
        truth = truth_fine if parent is None else np.bincount(parent, truth_fine*fine.area_km2)/cells.area_km2
        y = B @ (np.sqrt(a)*truth) + np.random.default_rng(d['seed']).standard_normal(len(band))
        if d.get('ocean_error'):
            err = (data.background(cache, mask, tg['area'], geo['lmax'], bg['vis_model'], d['ocean_error'],
                                   bg['near_ridge']) - C_bg)
            y = y + w*F[band.l]*band.rows(err)
    else:
        y = w*(band.rows(c_model) - F[band.l]*band.rows(C_bg))

    # solve
    route = comp['solver']
    eig = None
    if route == 'eigh':
        spec = dict(columns=store.spec, coarsen=comp.get('coarsen', 0), band=[lo, hi], filter=d.get('filter', 'lcs1'),
                    weights=[model, noise['reference']], products=comp['products_backend'], version=EIG_VERSION)
        d_eig = cache.entry('eig', spec)
        if os.path.exists(f'{d_eig}/E.npy'):
            eig = (np.load(f'{d_eig}/lam.npy'), np.load(f'{d_eig}/E.npy'))
        else:
            say(f'eigendecomposition ({min(B.shape)}^2)')
    solver = core.LevelRidge(B, u, route, comp['products_backend'], eig)
    if route == 'eigh' and eig is None:
        save_atomic(f'{d_eig}/lam.npy', lambda p: np.save(p, solver.lam))
        save_atomic(f'{d_eig}/E.npy', lambda p: np.save(p, solver.E))
    fit_sigma = noise.get('amplitude', 'evidence') == 'evidence'
    damping = cfg['method'].get('damping', 'evidence')
    if damping == 'evidence':
        alpha2 = solver.by_evidence(y, fit_sigma)
    elif damping == 'discrepancy':
        alpha2 = solver.by_target(y, lambda s: s['chi2'], len(band))
    else:
        alpha2 = float(damping)
    sol = solver.solve(y, alpha2)
    sigma = solver.sigma(sol, fit_sigma)
    m = sol['v0'] + sol['x']/np.sqrt(a)
    sd = solver.posterior_sd_x(alpha2, sigma)/np.sqrt(a) if route == 'eigh' or solver.side == 'cells' else None
    resid = sol['r']/w
    rms = float(np.sqrt(np.sum(band.mean_square_br()*resid**2)))
    attrs = dict(v0=sol['v0'], v0_sd=sigma/np.sqrt(sol['prec_v0']), damping=str(damping), alpha2=alpha2,
                 sigma=sigma, tau=sigma/np.sqrt(alpha2), chi2_per_datum=sol['chi2']/len(band), rms_br_misfit=rms,
                 log_evidence=solver.log_evidence(y, alpha2, fit_sigma), n_data=len(band), route=route,
                 columns_precision=store.spec['precision'],
                 products_precision='float32' if comp['products_backend'] == 'mps' else 'float64',
                 negative_area=float(cells.area_km2[m < 0].sum()/cells.area_km2.sum()))
    say(f'alpha2 {alpha2:.3g} (noise x {sigma:.2f}, prior s.d. {sigma/np.sqrt(alpha2):.2f} SI.km): '
        f'chi2/N {attrs["chi2_per_datum"]:.2f}, RMS Br misfit {rms:.2f} nT, v0 {sol["v0"]:.3f} +- {attrs["v0_sd"]:.3f}, '
        f'negative area {attrs["negative_area"]:.1%}')
    cell_vars = dict(vis=m)
    if sd is not None:
        cell_vars['vis_sd'] = sd
    if truth is not None:
        cell_vars['truth'] = truth
    row_vars = dict(data=y/w, model=y/w - resid, residual=resid)
    return results.make(cfg, cells, cell_vars, row_vars, band, **attrs)
