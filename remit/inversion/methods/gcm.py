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
  [errors.background] (with layouts) the ocean-model error of ADR 0009: kind = 'ocean_cells', layer (of the ocean
         cells, default basement to 6 km below it), rms_nT = [values]: each ocean cell's share of the background field
         scaled by an independent error of s.d. s, s set by the error's RMS Br in the band; every value on the grid is
         fitted (one whitening and one eigendecomposition each, cached) and the evidence picks among them
  [parameterisation] kind = 'layouts', tables = [continent domain tables]; [method] layouts = names from
         parameterise.LAYOUTS (default all): with layouts the cells are damped toward a level per domain, the levels
         scattering about v0 with a spread fitted by the evidence (ADR 0008); damping must be 'evidence'. Every layout
         is fitted and ranked by the evidence; the best one gives vis, vis_sd and vis_domain, and each layout's map is
         kept as layout_vis_<name>.
"""
import os
import time

import numpy as np
from scipy import sparse

from .. import columns, core, data, errors, geometry, parameterise, results, truths
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
        lays = None
        if d['truth']['kind'] == 'layout_levels':
            lays = parameterise.layouts(fine, cfg['parameterisation']['tables'], cache, tg['area'],
                                        [d['truth']['layout']])[0]
        truth_fine = truths.truth(d['truth'], fine, lays)
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
    if cfg.get('parameterisation', {}).get('kind') == 'layouts':
        eig_spec = spec if route == 'eigh' else None
        return _run_layouts(cfg, cells, B, u, a, y, w, F, band, solver, fit_sigma, truth, cache, say, eig_spec)
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


def _ocean_covariance(cfg, band, w, F, cache, say):
    """C = Bo Bo' (whitened, data x data) of the ocean cells' background columns (ADR 0009), and the RMS Br (nT) in the
    band of an error with C as its covariance"""
    eb, tg, geo, bg = cfg['errors']['background'], cfg['target'], cfg['geometry'], cfg['fit']['background']
    if eb.get('kind') != 'ocean_cells':
        raise ValueError(f'[errors.background] kind {eb.get("kind")!r}')
    layer = eb.get('layer', {'top': 'basement', 'base': 'basement+6'})
    direction = f'background:{bg["remanence"]}:{bg["vis_model"]}:{"nr" if bg["near_ridge"] else ""}'
    cells = geometry.make_cells(geometry.ocean_mask(), tg['nside'])
    store = columns.ColumnStore(cells, 'ocean', layer, direction, geo['lmax'], cfg['compute']['columns_backend'],
                                cache).build(say)
    import hashlib
    spec = dict(columns=store.spec, band=[band.lmin, band.lmax],
                weights=hashlib.sha256(np.ascontiguousarray(w*F[band.l]).tobytes()).hexdigest()[:16], version=1)
    d = cache.entry('background_cov', spec)
    f = os.path.join(d, 'C.npy')
    if not os.path.exists(f):
        say(f'ocean-model error covariance from {len(cells)} ocean cells')
        C = np.zeros((len(band), len(band)))
        for idx in store.blocks:
            Bo = band.columns(store.get(idx))*(w*F[band.l])[:, None]
            C += Bo @ Bo.T
        save_atomic(f, lambda p: np.save(p, C))
    C = np.load(f)
    rms1 = float(np.sqrt(np.sum(band.mean_square_br()*np.diag(C)/w**2)))
    return C, rms1, dict(columns=store.spec, version=1)


def _run_layouts(cfg, cells, B, u, a, y, w, F, band, base, fit_sigma, truth, cache, say, eig_spec):
    """GCM with layouts (ADR 0008), each fitted by the evidence over (alpha^2, beta), with the ocean-model error
    (ADR 0009) on a grid of amplitudes when [errors.background] is given; the best (layout, amplitude) mapped"""
    from scipy.linalg import cholesky, solve_triangular
    if cfg['method'].get('damping', 'evidence') != 'evidence':
        raise ValueError('GCM with layouts takes damping = "evidence"')
    par, tg = cfg['parameterisation'], cfg['target']
    names = cfg['method'].get('layouts', list(parameterise.LAYOUTS))
    lays, coverage = parameterise.layouts(cells, par['tables'], cache, tg['area'], names)
    say(f'{len(lays)} layouts ({coverage:.1%} of the target nodes in a polygon): '
        + ', '.join(f'{n} ({len(d)})' for n, (_, d) in lays.items()))
    eb = cfg['errors'].get('background')
    grid = [0.]
    if eb:
        C_oc, rms1, oc_spec = _ocean_covariance(cfg, band, w, F, cache, say)
        grid = [float(r) for r in eb['rms_nT']]
        say(f'ocean-model error grid (RMS Br, nT): {grid}')
    fits = {}                                   # (layout, rms) -> row, map
    for rms in grid:
        if rms == 0:
            Bw, uw, yw, Lc, ll0, solver = B, u, y, None, 0., base
        else:
            Lc = cholesky(np.eye(len(y)) + (rms/rms1)**2*C_oc, lower=True)
            ll0 = -float(np.sum(np.log(np.diag(Lc))))       # the whitening's Jacobian, part of the evidence
            Bw = solve_triangular(Lc, B, lower=True)
            uw, yw = solve_triangular(Lc, u, lower=True), solve_triangular(Lc, y, lower=True)
            eig = None
            if base.route == 'eigh':
                d_eig = cache.entry('eig', dict(eig_spec, background=oc_spec, rms_nT=rms))
                if os.path.exists(f'{d_eig}/E.npy'):
                    eig = (np.load(f'{d_eig}/lam.npy'), np.load(f'{d_eig}/E.npy'))
                else:
                    say(f'eigendecomposition with the ocean-model error at {rms:g} nT')
            solver = core.LevelRidge(Bw, uw, base.route, eig=eig)
            if base.route == 'eigh' and eig is None:
                save_atomic(f'{d_eig}/lam.npy', lambda p: np.save(p, solver.lam))
                save_atomic(f'{d_eig}/E.npy', lambda p: np.save(p, solver.E))
        for name, (label, dnames) in lays.items():
            fits[name, rms] = _fit_layout(solver, Bw, uw, yw, a, label, dnames, cells, band, w, fit_sigma, Lc, ll0)
            r = fits[name, rms][0]
            say(f'{name} (ocean error {rms:g} nT): {r["n_domains"]} domains, log evidence {r["log_evidence"]:.1f}; '
                f'noise x {r["sigma"]:.2f}, cell s.d. {r["tau_cells"]:.2f}, domain s.d. {r["tau_domains"]:.2f} SI.km, '
                f'v0 {r["v0"]:.3f}, RMS Br misfit {r["rms_br_misfit"]:.2f} nT')
        if rms != 0:
            del Bw, solver
    rows = {}
    for name in lays:                           # each layout at its best amplitude, with the profile over the grid
        best_r = max(grid, key=lambda r: fits[name, r][0]['log_evidence'])
        rows[name] = dict(fits[name, best_r][0], ocean_error_rms_nT=best_r,
                          ocean_error_profile={str(r): fits[name, r][0]['log_evidence'] for r in grid})
    best = max(rows, key=lambda n: rows[n]['log_evidence'])
    for n in rows:
        rows[n]['delta_log_evidence'] = rows[n]['log_evidence'] - rows[best]['log_evidence']
    say('ranking: ' + ', '.join(f'{n} {rows[n]["delta_log_evidence"]:+.1f} (ocean error {rows[n]["ocean_error_rms_nT"]:g} nT)'
                                for n in sorted(rows, key=lambda n: -rows[n]['log_evidence'])))
    m, sd, level, label, resid = fits[best, rows[best]['ocean_error_rms_nT']][1]
    cell_vars = dict(vis=m, vis_domain=level, domain=label.astype(float))
    if sd is not None:
        cell_vars['vis_sd'] = sd
    for n in rows:
        mn, _, _, ln, _ = fits[n, rows[n]['ocean_error_rms_nT']][1]
        cell_vars[f'layout_vis_{n}'] = mn
        cell_vars[f'layout_domain_{n}'] = ln.astype(float)
    if truth is not None:
        cell_vars['truth'] = truth
    r = rows[best]
    attrs = dict(layout=best, layouts={n: {k: v for k, v in rw.items() if k != 'domains'} for n, rw in rows.items()},
                 domains=r['domains'], v0=r['v0'], v0_sd=r['v0_sd'], damping='evidence', alpha2=r['alpha2'],
                 beta=r['beta'], sigma=r['sigma'], tau=r['tau_cells'], tau_domains=r['tau_domains'],
                 ocean_error_rms_nT=r['ocean_error_rms_nT'], chi2_per_datum=r['chi2_per_datum'],
                 rms_br_misfit=r['rms_br_misfit'], log_evidence=r['log_evidence'], n_data=len(band), route=base.route,
                 node_coverage=coverage, negative_area=r['negative_area'],
                 domain_names={n: d for n, (_, d) in lays.items()},
                 panel_titles={'vis_domain': f'Domain levels ({best} layout)'})
    row_vars = dict(data=y/w, model=y/w - resid, residual=resid)
    return results.make(cfg, cells, cell_vars, row_vars, band, **attrs)


def _fit_layout(base, B, u, y, a, label, dnames, cells, band, w, fit_sigma, Lc=None, ll0=0.):
    """one layout on one (whitened) problem: (row of fitted quantities, (vis, vis_sd, domain level, label, residual
    in nT)); Lc undoes the ocean-model whitening of the residual, ll0 is its Jacobian"""
    sa = np.sqrt(a)
    k = len(dnames)
    D = np.eye(k)[label] if k > 1 else np.zeros((len(cells), 0))
    if k > 1:
        Ud = B @ (sa[:, None]*D)                            # whitened field of a unit level in each domain
        solver = core.LinearRidge(B, np.hstack([u[:, None], Ud]), base.route,
                                  eig=(base.lam, base.E) if base.route == 'eigh' else None,
                                  prior=np.r_[False, np.ones(k, bool)])
        if base.route == 'eigh':
            solver._Q = base._Q
        if not hasattr(base, '_alpha2_one'):
            base._alpha2_one = base.by_evidence(y, fit_sigma)
        alpha2, beta = solver.by_evidence_beta(y, fit_sigma, base._alpha2_one)
    else:
        solver, beta = base, 0.
        alpha2 = base._alpha2_one = base.by_evidence(y, fit_sigma)
    sol = solver.solve(y, alpha2, beta)
    sigma = solver.sigma(sol, fit_sigma)
    theta = sol['theta']
    level = theta[0] + (D @ theta[1:] if k > 1 else 0.)
    m = level + sol['x']/sa
    sd = (solver.cell_sd(alpha2, sigma, sa, np.hstack([np.ones((len(cells), 1)), D]), beta)
          if base.route == 'eigh' or base.side == 'cells' else None)
    if base.route == 'eigh':
        base._Q = solver._Q
    Pi = np.linalg.inv(sol['P'])*sigma**2
    resid = (sol['r'] if Lc is None else Lc @ sol['r'])/w
    ll = solver.log_evidence(y, alpha2, fit_sigma, beta=beta) + ll0
    dom = {}
    if k > 1:
        area = np.bincount(label, cells.area_km2, k)
        for j, dn in enumerate(dnames):              # each domain's level v0 + delta_j, its s.d. and its area
            e = np.zeros(k + 1)
            e[0] = e[j + 1] = 1.
            dom[dn] = dict(level=float(theta[0] + theta[j + 1]), level_sd=float(np.sqrt(e @ Pi @ e)),
                           area_km2=float(area[j]))
    row = dict(n_domains=k, log_evidence=float(ll), alpha2=float(alpha2), beta=float(beta), sigma=sigma,
               tau_cells=sigma/np.sqrt(alpha2), tau_domains=sigma/np.sqrt(beta) if k > 1 else 0.,
               v0=float(theta[0]), v0_sd=float(np.sqrt(Pi[0, 0])), chi2_per_datum=sol['chi2']/len(band),
               rms_br_misfit=float(np.sqrt(np.sum(band.mean_square_br()*resid**2))),
               negative_area=float(cells.area_km2[m < 0].sum()/cells.area_km2.sum()), domains=dom)
    return row, (m, sd, level, label, resid)
