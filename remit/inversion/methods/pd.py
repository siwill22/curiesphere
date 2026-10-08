"""
PD (polygon domains, CONTEXT.md): one VIS per domain fixed in advance from geological polygons, from a local fit.
Domains are groupings of cells (parameterise.domains, ADR 0001): v = S theta.

  fit        bounded least squares: min |yw - Gw S theta|^2 with lo <= theta <= hi (box QP)
  noise      the noise scale: 'fixed' (1: the error model as it is) or 'evidence' (chi^2/(N - K), fitted)
  formal     s.d. of theta from (S'Gw'Gw S)^-1 at that scale (no bounds)
  draws      s.d. over refits of the model's prediction plus error-model realisations (noise_draws of them, bounds
             kept), the spread the first polygon-domain inversion reported as its noise s.d.; the map's vis_sd when
             draws are made (with diagonal weighting the formal s.d. ignores correlated errors and is too small)

Config keys used, besides those of problem.local:
  [parameterisation] kind = 'domains', table (a CSV: key of the Hasterok polygon, param)
  [method] bounds ([lo, hi]), noise ('fixed' or 'evidence'), noise_draws, seed
"""
import time

import numpy as np

from .. import core, parameterise, problem, results
from ..cache import Cache
from ..geometry import area_name


def bounded_lsq(A, y, lo, hi):
    return core.box_qp(A.T @ A, A.T @ y, lo, hi, tol=1e-12)


def run(cfg, cache=None, log=print):
    cache = cache or Cache()
    t0 = time.time()
    say = lambda msg: log(f'[{(time.time() - t0)/60:5.1f} min] {msg}')
    pb = problem.local(cfg, cache, say)
    cells, me, par = pb.cells, cfg['method'], cfg['parameterisation']
    if par.get('kind') != 'domains':
        raise ValueError('PD needs [parameterisation] kind = "domains"')
    label, params, coverage = parameterise.domains(cells, par['table'], cache, area_name(cfg['target']['area']))
    used = np.unique(label)
    if len(used) < len(params):
        say(f'domains holding no cell (left out): {[params[k] for k in set(range(len(params))) - set(used)]}')
    params = [params[k] for k in used]
    label = np.searchsorted(used, label)
    S = parameterise.indicator(label)
    Gw, yw = pb.Gw, pb.yw
    A = Gw @ S
    N, K = A.shape
    lo, hi = me.get('bounds', [0., np.inf])
    theta = bounded_lsq(A, yw, lo, hi)
    r = yw - A @ theta
    chi2 = float(r @ r)
    sigma = np.sqrt(chi2/(N - K)) if me.get('noise', 'fixed') == 'evidence' else 1.
    sd_formal = sigma*np.sqrt(np.diag(np.linalg.inv(A.T @ A)))
    rng = np.random.default_rng(me.get('seed', 0))
    n_draws = int(me.get('noise_draws', 0))
    draws = np.array([bounded_lsq(A, A @ theta + pb.W @ pb.draw(rng), lo, hi) for _ in range(n_draws)])
    sd_draws = draws.std(0) if n_draws else np.full(K, np.nan)
    area = np.bincount(label, cells.area_km2)
    say(f'{K} domains ({coverage:.1%} of the target nodes in a domain), {N} data: chi2/N {chi2/N:.2f}; '
        f'{np.sum(theta <= lo + 1e-9)} at the lower bound' + (f'; {n_draws} noise draws' if n_draws else ''))
    table = {p: dict(v=float(theta[k]), sd_formal=float(sd_formal[k]), sd_draws=float(sd_draws[k]),
                     area_km2=float(area[k])) for k, p in enumerate(params)}
    cell_vars = dict(vis=S @ theta, vis_sd=S @ (sd_draws if n_draws else sd_formal), domain=label.astype(float))
    if pb.truth is not None:
        cell_vars['truth'] = pb.truth
    model = pb.G @ (S @ theta)
    attrs = dict(domains=table, chi2_per_datum=chi2/N, noise_scale=sigma, n_domains=K, node_coverage=coverage,
                 bounds=[lo, hi], **pb.info)
    return results.make(cfg, cells, cell_vars, points=(pb.plon, pb.plat),
                        point_vars=dict(data=pb.y, model=model, residual=pb.y - model), **attrs)
