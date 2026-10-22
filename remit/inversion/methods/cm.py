"""
CM (cell map, CONTEXT.md): VIS per cell in a target area from a local fit (ADR 0002), as a smooth random field
about one fitted level, and optionally as penalised maps with bounds.

GMRF   v = v0 + u, u a Gaussian Markov random field on the cell graph (gmrf.py) with standard deviation s and
       correlation length ell; whitened data yw = Gw v + e, e ~ N(0, h^2 I); v0 integrated out (flat prior). h and s
       maximise the evidence at each ell; ell is profiled or given. Output: the posterior mean and s.d. of v (no
       bounds).
maps   min |yw - Gw v|^2 + beta (|diag(sqrt(a))/ell (v - v_ref)|^2 + |D (v - v_ref)|^2) with lo <= v <= hi:
       'l2', or 'tv' (iteratively reweighted from the L2 map: weights rms/sqrt((Dv)^2 + eps^2), eps = 0.01 rms).
       beta at the misfit N h^2 of the GMRF fit (bisection in log beta), times beta_scale; or a number.
       v_ref: 'hm_mean' (the area mean of the H&M model over the target cells, a statistic, never a map), 'gmrf'
       (the GMRF's v0), or a number.

Config keys used, besides those of problem.local:
  [method] length ('evidence' or km), reference, bounds ([lo, hi]),
           maps (a list of tables: label, penalty 'l2' or 'tv', length 'gmrf' or km, beta 'discrepancy' or a number,
           beta_scale)
The GMRF always uses the eigendecomposition: after compression the data side is small (~1000).
"""
import time

import numpy as np
from scipy.sparse import diags, vstack

from .. import core, data, gmrf, graph, problem, results
from ..cache import Cache


def penalty_rows(a, D, ell, beta, w=None):
    """rows (sparse) of beta (|diag(sqrt(a))/ell v|^2 + |W^1/2 D v|^2), a scaled to mean 1 (TV: IRLS weights w)"""
    Dw = D if w is None else diags(np.sqrt(w)) @ D
    return (np.sqrt(beta)*vstack([diags(np.sqrt(a/np.mean(a))/ell), Dw])).tocsr()


def penalised(Gw, yw, a, D, ell, beta, v_ref, penalty='l2', lo=0., hi=np.inf, n_irls=15, eps_rel=0.01, GtG=None):
    GtG = Gw.T @ Gw if GtG is None else GtG
    v0 = np.full(Gw.shape[1], v_ref) if np.ndim(v_ref) == 0 else v_ref
    v = core.bounded_map(Gw, yw, penalty_rows(a, D, ell, beta), v0, lo, hi, AtA=GtG)
    if penalty == 'l2':
        return v
    rms = np.sqrt(np.mean((D @ v)**2))
    eps = eps_rel*rms
    for _ in range(n_irls):
        w = rms/np.sqrt((D @ v)**2 + eps**2)
        v_new = core.bounded_map(Gw, yw, penalty_rows(a, D, ell, beta, w), v0, lo, hi, v_start=v, AtA=GtG)
        done = np.max(np.abs(v_new - v)) < 1e-4
        v = v_new
        if done:
            break
    return v


def discrepancy(solve, Gw, yw, chi2_target, log_beta=(0., 21.), n_bisect=16):
    """beta (bisection on log beta) at which the misfit of solve(beta) equals chi2_target; (beta, v, chi2)"""
    lo, hi = log_beta
    sol = {}
    for _ in range(n_bisect):
        mid = 0.5*(lo + hi)
        v = solve(np.exp(mid))
        chi2 = float(np.sum((Gw @ v - yw)**2))
        sol[mid] = (v, chi2)
        lo, hi = (mid, hi) if chi2 < chi2_target else (lo, mid)
    best = min(sol, key=lambda k: abs(sol[k][1] - chi2_target))
    return float(np.exp(best)), sol[best][0], sol[best][1]


def run(cfg, cache=None, log=print):
    cache = cache or Cache()
    t0 = time.time()
    say = lambda msg: log(f'[{(time.time() - t0)/60:5.1f} min] {msg}')
    if cfg['fit'].get('type', 'local') != 'local':
        raise NotImplementedError('CM with a global fit')
    pb = problem.local(cfg, cache, say)
    cells, me = pb.cells, cfg['method']
    Gw, yw, area = pb.Gw, pb.yw, cells.area_km2
    N = len(yw)
    edges, bridges = graph.cell_edges(cells)
    Lw = graph.laplacian(edges, 1/graph.edge_lengths_km(edges, cells.clon, cells.clat)**2, len(cells))
    D = graph.difference_operator(edges, cells.clon, cells.clat)

    one = np.ones((len(cells), 1))
    fit = gmrf.GMRF(Gw, yw, area, Lw).fit(one, length=me.get('length', 'evidence'))
    solver, alpha2, h, ell, ll = fit['ridge'], fit['alpha2'], fit['h'], float(fit['ell'][0]), fit['ll']
    sol = solver.solve(yw, alpha2)
    s = float(fit['s'][0])
    v = sol['v0'] + fit['L'] @ sol['x']
    sd = solver.posterior_sd_map(alpha2, h, fit['L'])
    attrs = dict(v0=sol['v0'], v0_sd=h/np.sqrt(sol['prec_v0']), noise_scale_h=h, prior_sd_s=s, length_km=ell,
                 log_evidence=ll, chi2_per_datum=sol['chi2']/N, negative_area=float(area[v < 0].sum()/area.sum()),
                 n_edges=len(edges), n_bridges=len(bridges), **pb.info)
    say(f'GMRF: noise x {h:.2f}, VIS s.d. {s:.2f} SI.km, length {ell:.0f} km, v0 {sol["v0"]:.3f} +- {attrs["v0_sd"]:.3f}; '
        f'negative area {attrs["negative_area"]:.1%}')
    cell_vars = dict(vis=v, vis_sd=sd)

    ref = me.get('reference', 'hm_mean')
    v_ref = (float(np.average(data.hm_cell_vis(cells), weights=area)) if ref == 'hm_mean' else
             sol['v0'] if ref == 'gmrf' else float(ref))
    lo, hi = me.get('bounds', [0., np.inf])
    GtG = Gw.T @ Gw
    maps = {}
    for m in me.get('maps', []):
        ell_m = ell if m.get('length', 'gmrf') == 'gmrf' else float(m['length'])
        solve = lambda b: penalised(Gw, yw, area, D, ell_m, b, v_ref, m['penalty'], lo, hi, GtG=GtG)
        vm = None
        if m.get('beta', 'discrepancy') == 'discrepancy':
            beta, vm, _ = discrepancy(solve, Gw, yw, N*h**2, n_bisect=16 if m['penalty'] == 'l2' else 12)
        else:
            beta = float(m['beta'])
        if m.get('beta_scale', 1.) != 1. or vm is None:
            beta *= m.get('beta_scale', 1.)
            vm = solve(beta)
        chi2 = float(np.sum((Gw @ vm - yw)**2))
        cell_vars[f'vis_{m["label"]}'] = vm
        maps[m['label']] = dict(penalty=m['penalty'], length_km=ell_m, beta=beta, chi2_per_datum=chi2/N)
        say(f'map {m["label"]}: {m["penalty"]}, length {ell_m:.0f} km, beta {beta:.3g}, chi2/N {chi2/N:.2f} '
            f'(GMRF noise level {h**2:.2f})')
    attrs.update(reference=v_ref, bounds=[lo, hi], maps=maps)
    if pb.truth is not None:
        cell_vars['truth'] = pb.truth
    model = pb.G @ v
    point_vars = dict(data=pb.y, model=model, residual=pb.y - model)
    return results.make(cfg, cells, cell_vars, points=(pb.plon, pb.plat), point_vars=point_vars, **attrs)
