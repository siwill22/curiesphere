"""
CR (covariate regression, CONTEXT.md): VIS as a linear model in covariates plus a spatially correlated rest, from a
local fit (gmrf.py):  v = Bd theta + u, theta uniform on its box, u a GMRF; hyperparameters by maximum evidence.

  regression   the main design (default: class means + crust, age, undated, heat flow, LAB): coefficients +- s.d.
               (u keeps them honest), the posterior mean and s.d. of v (the result's map), the standard quantities
               (area mean, craton means, craton - rest); again with each alternative heat-flow product
  classes      class means with s and ell per class, against shared statistics (log evidence), with Laplace errors
  hypotheses   designs scored by their evidence (a) with u (hyperparameters fitted per design) and (b) without u (h
               integrated out): every subset of the covariate groups, the hand-written designs of a TOML file, and
               partitions of the cells (one mean each): an .npz of label arrays, or an FD-merge result (.nc)

Config keys used, besides those of problem.local and [covariates]:
  [method] design (columns), heat_flow_alternatives (covariate names), per_class (true/false),
           hypotheses = { auto = true, hand = path, labels = path }
"""
import itertools
import json
import os
import time
import tomllib

import numpy as np

from .. import core, covariates, gmrf, graph, parameterise, problem, results
from ..cache import Cache
from ..geometry import area_name


def fit_design(model, Bd, lo, hi, Q):
    fit = model.fit(Bd, (lo, hi))
    post = model.posterior(Bd, fit, Q)
    return fit, post


def summary(names, fit, post):
    return dict(names=names, theta=post['theta'].tolist(), sd=post['theta_sd'].tolist(), log_evidence=fit['ll'],
                h=fit['h'], s=fit['s'].tolist(), ell=fit['ell'].tolist(), quantities=post['functionals'])


def score(model, name, Bd, lo, hi):
    """a hypothesis: log evidence with u (fitted) and without u"""
    fit = model.fit(Bd, (lo, hi))
    plain = core.regression_evidence(model.Gw @ Bd, model.yw, lo, hi)
    return dict(name=name, K=Bd.shape[1], ll=fit['ll'], ll_plain=plain, h=fit['h'], s=float(fit['s'][0]),
                ell=float(fit['ell'][0]))


def hypotheses(model, cov, area, spec, base=None):
    """dict name -> (Bd, lo, hi) of the hypotheses in spec"""
    out = {}
    if spec.get('auto', True):
        G = parameterise.GROUPS
        for r in range(len(G) + 1):
            for gs in itertools.combinations(G, r):
                out['auto: ' + ('+'.join(gs) or 'mean only')] = parameterise.design(parameterise.columns(gs), cov, area)[:3]
    if spec.get('hand'):
        with open(os.path.expanduser(spec['hand']), 'rb') as f:
            for name, cols in tomllib.load(f).get('hypotheses', {}).items():
                out['hand: ' + name] = parameterise.design(cols, cov, area)[:3]
    if spec.get('labels'):
        path = os.path.expanduser(spec['labels'])
        if path.endswith('.nc'):                                    # an FD-merge result: one partition per setting
            ds = results.load(path)
            rows = json.loads(ds.attrs['settings'])
            labels = {f'merging lam x{r["lam_rel"]:.2g} gamma {r["gamma_rel"]:g} (K {r["K"]})': ds[f'domain_{i}'].values
                      for i, r in enumerate(rows)}
        else:
            z = np.load(path)
            labels = {name: z[name] for name in z.files}
        for name, lab in labels.items():
            B = parameterise.indicator(lab)
            out[name] = (B, np.full(B.shape[1], parameterise.LEVEL[0]), np.full(B.shape[1], parameterise.LEVEL[1]))
    return out


def run(cfg, cache=None, log=print):
    cache = cache or Cache()
    t0 = time.time()
    say = lambda msg: log(f'[{(time.time() - t0)/60:5.1f} min] {msg}')
    pb = problem.local(cfg, cache, say)
    cells, me = pb.cells, cfg['method']
    Gw, yw, area = pb.Gw, pb.yw, cells.area_km2
    cov = covariates.load(cells, cfg['covariates'], cache, area_name(cfg['target']['area']))
    edges, _ = graph.cell_edges(cells)
    Lw = graph.laplacian(edges, 1/graph.edge_lengths_km(edges, cells.clon, cells.clat)**2, len(cells))
    model = gmrf.GMRF(Gw, yw, area, Lw)
    Q = parameterise.quantities(cov, area)
    attrs = dict(**pb.info)

    cols = me.get('design', parameterise.columns(parameterise.GROUPS))
    variants = {'main': cols}
    for alt in me.get('heat_flow_alternatives', []):
        variants[alt] = [alt if c == 'heat_flow' else c for c in cols]
    regs, main = {}, None
    for vname, vc in variants.items():
        Bd, lo, hi, names = parameterise.design(vc, cov, area)
        fit, post = fit_design(model, Bd, lo, hi, Q)
        regs[vname] = summary(names, fit, post)
        main = main or (fit, post)
        say(f'regression ({vname}): ' + ', '.join(f'{n} {t:+.2f}+-{s:.2f}' for n, t, s in
                                                 zip(names, post['theta'], post['theta_sd'])) +
            f'; noise x {fit["h"]:.2f}, rest s.d. {fit["s"][0]:.2f} SI.km, length {fit["ell"][0]:.0f} km')
    attrs['regressions'] = regs
    fit, post = main
    attrs.update(log_evidence=fit['ll'], noise_scale_h=fit['h'], prior_sd_s=float(fit['s'][0]),
                 length_km=float(fit['ell'][0]))

    if me.get('per_class'):
        Bc, lo, hi, names = parameterise.design(['class'], cov, area)
        shared = model.fit(Bc, (lo, hi))
        mc = gmrf.GMRF(Gw, yw, area, Lw, cov['cls'])
        per = mc.fit_per_class(Bc, (lo, hi), start=dict(ell=shared['ell'], h=shared['h'], s=shared['s']))
        lsd = mc.hyper_sd(Bc, per, (lo, hi))
        p = mc.posterior(Bc, per)
        attrs['classes'] = dict(names=list(covariates.CLASS_NAMES), log_evidence_vs_shared=per['ll'] - shared['ll'],
                                h=per['h'], s=per['s'].tolist(), ell=per['ell'].tolist(), log_sd=lsd.tolist(),
                                mean=p['theta'].tolist(), mean_sd=p['theta_sd'].tolist(), converged=per['converged'])
        say(f'classes: log evidence {per["ll"] - shared["ll"]:+.1f} for separate statistics; ' +
            ', '.join(f'{n} {m:.2f}+-{sd:.2f} (s.d. {s:.2f}, length {e:.0f} km)' for n, m, sd, s, e in
                      zip(covariates.CLASS_NAMES, p['theta'], p['theta_sd'], per['s'], per['ell'])))

    if me.get('hypotheses'):
        hyps = hypotheses(model, cov, area, me['hypotheses'])
        res = [score(model, name, *h) for name, h in hyps.items()]
        res.sort(key=lambda r: -r['ll'])
        attrs['hypotheses'] = res
        best = res[0]['ll']
        say(f'{len(res)} hypotheses; best: ' + '; '.join(f'{r["name"]} ({r["ll"] - best:+.1f})' for r in res[:5]))

    cell_vars = dict(vis=post['v'], vis_sd=post['v_sd'])
    if pb.truth is not None:
        cell_vars['truth'] = pb.truth
    model_br = pb.G @ post['v']
    return results.make(cfg, cells, cell_vars, points=(pb.plon, pb.plat),
                        point_vars=dict(data=pb.y, model=model_br, residual=pb.y - model_br), **attrs)
