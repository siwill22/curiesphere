"""
Extremal bounds (CONTEXT.md; Parker's ideal-body idea on the cells), from a local fit: the smallest and largest value
of each summary c'v (a craton's mean VIS, the craton - rest contrast) over all models that fit the data and the bounds,
    min / max c'v  subject to  |Gw v - yw|^2 <= h^2 chi2_q(N),  lo <= v <= hi,
h the noise scale of the GMRF fit (CM) and chi2_q(N) the q quantile of chi^2 with N degrees of freedom. With the
bounds alone the interval is what the data force, whatever the regularisation; it is repeated with roughness limits
|D v| <= k x the roughness of the L2 map (D: the gradient between cells), to show how much smoothness it takes to pin
a summary down. The L2 map is CM's: length of the GMRF fit, beta at the misfit N h^2, v_ref the H&M mean, bounded.

Config keys used, besides those of problem.local and [covariates]:
  [method] quantities (names from parameterise.quantities), bounds ([lo, hi]), quantile, roughness (list of k)
"""
import time

import numpy as np
from scipy.stats import chi2 as chi2_dist

from .. import core, covariates, data, gmrf, graph, parameterise, problem, results
from ..cache import Cache
from ..geometry import area_name
from .cm import discrepancy, penalised


def run(cfg, cache=None, log=print):
    cache = cache or Cache()
    t0 = time.time()
    say = lambda msg: log(f'[{(time.time() - t0)/60:5.1f} min] {msg}')
    pb = problem.local(cfg, cache, say)
    cells, me = pb.cells, cfg['method']
    Gw, yw, area = pb.Gw, pb.yw, cells.area_km2
    N = len(yw)
    cov = covariates.load(cells, cfg['covariates'], cache, area_name(cfg['target']['area']))
    edges, _ = graph.cell_edges(cells)
    Lw = graph.laplacian(edges, 1/graph.edge_lengths_km(edges, cells.clon, cells.clat)**2, len(cells))
    D = graph.difference_operator(edges, cells.clon, cells.clat)
    fit = gmrf.GMRF(Gw, yw, area, Lw).fit(np.ones((len(cells), 1)))
    h, ell = fit['h'], float(fit['ell'][0])
    lo, hi = me.get('bounds', [0., 4.])
    v_ref = float(np.average(data.hm_cell_vis(cells), weights=area))
    GtG = Gw.T @ Gw
    _, l2, _ = discrepancy(lambda b: penalised(Gw, yw, area, D, ell, b, v_ref, 'l2', 0., np.inf, GtG=GtG), Gw, yw, N*h**2)
    rough_l2 = float(np.linalg.norm(D @ l2))
    target = h**2*chi2_dist.ppf(me.get('quantile', 0.95), N)
    Q = parameterise.quantities(cov, area)
    names = me.get('quantities', [q for q in Q if q != 'area mean'])
    say(f'GMRF noise x {h:.2f}, length {ell:.0f} km; misfit limit {target:.0f} for {N} data; L2 roughness {rough_l2:.3g}')
    table, cell_vars = {}, dict(vis=l2)
    for q in names:
        table[q] = {}
        for k in [None] + list(me.get('roughness', [4., 2., 1.])):
            t = time.time()
            o = core.extremes(Gw, yw, Q[q], target, lo, hi, D=D, rough_max=None if k is None else k*rough_l2)
            tag = 'positivity' if k is None else f'roughness {k:g} x L2'
            table[q][tag] = [o['min']['value'], o['max']['value']]
            if k is None:
                cell_vars[f'vis_min_{q}'], cell_vars[f'vis_max_{q}'] = o['min']['v'], o['max']['v']
            say(f'{q:14s} {tag:22s} {o["min"]["value"]:6.2f} to {o["max"]["value"]:5.2f} SI.km ({time.time() - t:.0f} s)')
    if pb.truth is not None:
        cell_vars['truth'] = pb.truth
        for q in names:
            table[q]['truth'] = float(Q[q] @ pb.truth)
    attrs = dict(bounds_table=table, misfit_limit=target, noise_scale_h=h, length_km=ell, roughness_l2=rough_l2,
                 bounds=[lo, hi], **pb.info)
    model = pb.G @ l2
    return results.make(cfg, cells, {k.replace(' ', '_'): v for k, v in cell_vars.items()}, points=(pb.plon, pb.plat),
                        point_vars=dict(data=pb.y, model=model, residual=pb.y - model), **attrs)
