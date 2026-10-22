"""
First-order inversions with one linear-Gaussian model (domains/gmrf.py): v = B theta + u, u a GMRF
(standard deviation s, correlation length ell), noise h, all fitted by maximising the evidence.

  (5) GP map      B = intercept: posterior mean and sd of the VIS; the MAP with v >= 0; and the L2 and
                  TV maps (fixed penalties, length ell of the GP fit, reference the H&M mean, beta at the
                  misfit N h^2 of the GP fit)
  (1) regression  B = class means + standardised crust, age (+ undated), heat flow, LAB: coefficients
                  +- sd (u keeps them honest); again with each alternative heat-flow product
  (3) statistics  B = class means; s and ell per class against shared, with Laplace errors
  (4) hypotheses  all subsets of the covariate groups, the merging models and regions/<r>/hypotheses.toml,
                  each scored (a) with u (hyperparameters fitted per hypothesis) and (b) without u, h
                  integrated out (the scale of the domain samplers)
Cases
  cov_truth   synthetic: v = class means + crust and heat-flow terms + a GMRF with s and ell per class
  hast        realistic_synthetic.py truths and data (noise x1): one value per Hasterok province
  hast_var    the same with smooth variation within provinces
  hast_ms     make_multiscale_truth.py: hast_var plus ~50 km detail and small bodies (if made for the region)
  real        LCS-1 (or the band's data model)
Part gp2: the GP with two scales (broad + detail, gmrf.Model.fit2), its mean and sd and the broad and detail
parts of its mean.
Bands: env BAND=<name> runs on a variant of the problem (build_bands.py); output in its directory.
Output: linear_models.npz (updated in place: keys of parts not rerun are kept) and the printed log.

Run from notebooks/ (conda env pygmt17), lightly:
    N_WORKERS=3 nice -n 15 env PYTHONPATH=..:$PYTHONPATH python run_linear_models.py regions/australia/config.toml
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '1')
import sys
import time
import itertools
import tomllib
import multiprocessing

import numpy as np

from domains import bounds, covariates, gmrf, graph, linear, problem
from domains import design as dz
from domains.design import GROUPS, LEVEL, columns, indicator

N_WORKERS = int(os.environ.get('N_WORKERS', 3))
CASES = os.environ.get('CASES', 'cov_truth,hast,hast_var,real').split(',')
PARTS = os.environ.get('PARTS', 'gp,regression,classes,hypotheses').split(',')
SEED = 20261006
MERGE_MODE = os.environ.get('MERGE_MODE', 'fine')        # which run_merge.py mode's partitions are hypotheses

pb = problem.load(sys.argv[1])
Gw, area, edges = pb['Gw'], pb['area'], pb['edges']
N, n = Gw.shape
cov = dict(np.load(f'{pb["shared"]}/covariates.npz'))
Lw = graph.laplacian(edges, 1/graph.edge_lengths_km(edges, pb['clon'], pb['clat'])**2, n)
D = graph.difference_operator(edges, pb['clon'], pb['clat'])
GtG = Gw.T @ Gw
hm_mean = float(np.average(cov['hm_vis'], weights=area))
craton = np.isin(cov['group'], covariates.CRATON_GROUPS)
FUNCTIONALS = {'area mean': bounds.mean_functional(area, np.ones(n, bool)),
               **{g.split()[0][0] + 'AC mean': bounds.mean_functional(area, cov['group'] == g)
                  for g in covariates.CRATON_GROUPS},
               'craton - rest': bounds.mean_functional(area, craton) - bounds.mean_functional(area, ~craton)}
with open(f'{pb["reg"]}/hypotheses.toml', 'rb') as f:
    HAND = tomllib.load(f).get('hypotheses', {})


def design(cols):
    return dz.design(cols, cov, area)


# ------------------------------------------------------------------ cases
rs = dict(np.load(f'{pb["shared"]}/realistic_synthetic.npz', allow_pickle=True))   # eager (forked workers)
B_TRUE_COLS = ['class', 'crust', 'heat_flow']
THETA_TRUE = np.array([0.9, 0.6, 0.5, 0.12, -0.10])
S_TRUE, ELL_TRUE = np.array([0.25, 0.15, 0.10]), np.array([250., 150., 100.])


def make_cases():
    rng = np.random.default_rng(SEED)
    out = {}
    mc = gmrf.Model(Gw, np.zeros(N), area, Lw, cov['cls'], n_cache=1)
    u = np.linalg.cholesky(mc.prior_cov(S_TRUE, ELL_TRUE) + 1e-10*np.eye(n)) @ rng.standard_normal(n)
    v = design(B_TRUE_COLS)[0] @ THETA_TRUE + u
    out['cov_truth'] = dict(truth=v, dw=Gw @ v + pb['noise_w'][90])
    for j, name in enumerate(('hast', 'hast_var')):          # as run_realistic_synthetic.py, with this band's G
        out[name] = dict(truth=rs[f'truth_{name}'], dw=Gw @ rs[f'truth_{name}'] + pb['noise_w'][j])
        if not pb['band']:
            assert np.allclose(out[name]['dw'], rs[f'data_{name}_n1'], rtol=0, atol=1e-9)
    if os.path.exists(f'{pb["shared"]}/multiscale_truth.npz'):
        ms = np.load(f'{pb["shared"]}/multiscale_truth.npz')['truth']
        out['hast_ms'] = dict(truth=ms, dw=Gw @ ms + pb['noise_w'][2])   # noise draw as run_realistic_synthetic.py
    out['real'] = dict(truth=None, dw=pb['dw_real'])
    return out


def merging_models(case):
    if case == 'real':
        m = np.load(f'{pb["shared"]}/merge_points_sweep.npz', allow_pickle=True)
        return {f'merging lam x{s[0]:.2g} gamma {s[1]:g} (K {k})': lab
                for s, lab, k in zip(m['settings'], m[f'{MERGE_MODE}_label'], m[f'{MERGE_MODE}_K'])}
    if case in ('hast', 'hast_var'):
        return {f'merging lam x{lr} (K {int(rs[f"{case}_n1_l{lr}_K"])})': rs[f'{case}_n1_l{lr}_label']
                for lr in ('1.00', '0.33')}
    return {}


def score(name, B, lo, hi, m1, dw):
    """a hypothesis: evidence with u (fitted) and without u (h integrated out)"""
    fit = m1.fit(B, lo, hi, dw=dw)
    plain = linear.log_evidence(Gw @ B, dw, lo, hi)[0]
    return dict(name=name, K=B.shape[1], ll=fit['ll'], ll_plain=plain, h=fit['h'], s=fit['s'][0],
                ell=fit['ell'][0], theta=fit['theta'])


def run_case(case):
    t0 = time.time()
    dw = CASE_DATA[case]['dw']
    m1 = gmrf.Model(Gw, dw, area, Lw)
    out, log = {}, []
    say = lambda s: log.append(f'[{case}] {s}')

    # (5) GP map, MAP, L2, TV
    B0, lo0, hi0, _ = design(['intercept'])
    gp = m1.fit(B0, lo0, hi0)
    post = m1.posterior(B0, gp['h'], gp['s'], gp['ell'], functionals=FUNCTIONALS)
    fun = post['functionals']
    out.update(gp_h=gp['h'], gp_s=gp['s'][0], gp_ell=gp['ell'][0], gp_ll=gp['ll'], gp_v=post['v'], gp_sd=post['v_sd'],
               gp_fun_names=np.array(list(fun)), gp_fun=np.array(list(fun.values())))
    say(f'GP: noise h {gp["h"]:.2f}, VIS sd {gp["s"][0]:.2f} SI.km, correlation length {gp["ell"][0]:.0f} km; '
        f'{np.mean(post["v"] < 0):.1%} of cells below 0')
    truth = CASE_DATA[case]['truth']
    say('  GP posterior: ' + ', '.join(f'{k} {m:.2f}+-{sd:.2f}' + (f' (truth {FUNCTIONALS[k] @ truth:.2f})' if truth is not None else '')
                                       for k, (m, sd) in fun.items()))
    if 'gp' in PARTS:
        gp_map = gmrf.map_estimate(Gw, dw, m1.prior_rows(gp['s'], gp['ell']), B0 @ post['theta'], h=gp['h'], GtG=GtG)
        target = N*gp['h']**2
        ell = gp['ell'][0]
        b2, l2, c2 = gmrf.discrepancy(lambda b: gmrf.tikhonov(Gw, dw, area, D, ell, b, hm_mean, GtG=GtG),
                                      Gw, dw, target)
        b1, tv, c1 = gmrf.discrepancy(lambda b: gmrf.tikhonov(Gw, dw, area, D, ell, b, hm_mean, p=1, GtG=GtG),
                                      Gw, dw, target, n_bisect=12)
        out.update(gp_map=gp_map, l2_v=l2, l2_beta=b2, l2_chi2=c2, tv_v=tv, tv_beta=b1, tv_chi2=c1, chi2_target=target)
        say(f'L2 beta {b2:.3g} (chi2 {c2:.0f}), TV beta {b1:.3g} (chi2 {c1:.0f}); target {target:.0f}')
        if truth is not None:
            for k, v in (('GP mean', post['v']), ('GP MAP v>=0', gp_map), ('L2', l2), ('TV', tv)):
                say(f'  {k:12s} RMS error {np.sqrt(np.average((v - truth)**2, weights=area)):.3f} SI.km, '
                    f'correlation with truth {np.corrcoef(v, truth)[0, 1]:.2f}')

    if 'gp2' in PARTS:
        f2 = m1.fit2(B0, lo0, hi0, gp)
        p2 = m1.posterior2(B0, f2, functionals=FUNCTIONALS)
        out.update(gp2_h=f2['h'], gp2_s1=f2['s1'], gp2_ell1=f2['ell1'], gp2_s2=f2['s2'], gp2_ell2=f2['ell2'],
                   gp2_ll=f2['ll'], gp2_grid=f2['grid'], gp2_v=p2['v'], gp2_sd=p2['v_sd'], gp2_v_broad=p2['v_broad'],
                   gp2_v_detail=p2['v_detail'], gp2_fun=np.array(list(p2['functionals'].values())))
        say(f'GP, two scales: noise h {f2["h"]:.2f}; broad: sd {f2["s1"]:.2f} SI.km, length {f2["ell1"]:.0f} km; '
            f'detail: sd {f2["s2"]:.2f} SI.km, length {f2["ell2"]:.0f} km; log evidence {f2["ll"] - gp["ll"]:+.1f} vs one scale')
        say('  posterior: ' + ', '.join(f'{k} {m:.2f}+-{sd:.2f}' + (f' (truth {FUNCTIONALS[k] @ truth:.2f})' if truth is not None else '')
                                        for k, (m, sd) in p2['functionals'].items()))
        if truth is not None:
            say(f'  GP 2 scales  RMS error {np.sqrt(np.average((p2["v"] - truth)**2, weights=area)):.3f} SI.km, '
                f'correlation with truth {np.corrcoef(p2["v"], truth)[0, 1]:.2f}')

    # (1) regression
    if 'regression' in PARTS:
        variants = {'main': columns(GROUPS)}
        for alt in [k for k in cov if k.startswith('heat_flow_')]:
            variants[alt] = [alt if c == 'heat_flow' else c for c in columns(GROUPS)]
        for vname, cols in variants.items():
            B, lo, hi, names = design(cols)
            fit = m1.fit(B, lo, hi)
            p = m1.posterior(B, fit['h'], fit['s'], fit['ell'])
            out.update({f'reg_{vname}_names': np.array(names), f'reg_{vname}_theta': p['theta'],
                        f'reg_{vname}_sd': p['theta_sd'], f'reg_{vname}_ll': fit['ll'],
                        f'reg_{vname}_hyper': np.r_[fit['h'], fit['s'], fit['ell']]})
            say(f'regression ({vname}): ' + ', '.join(f'{nm} {t:+.2f}+-{s:.2f}' for nm, t, s in
                                                     zip(names, p['theta'], p['theta_sd'])))

    # (3) statistics by class
    if 'classes' in PARTS:
        Bc, loc, hic, names = design(['class'])
        shared = m1.fit(Bc, loc, hic)
        mc = gmrf.Model(Gw, dw, area, Lw, cov['cls'], n_cache=2)
        per = mc.fit(Bc, loc, hic, per_class=True, start=shared)
        sd = mc.hyper_sd(Bc, loc, hic, per)
        p = mc.posterior(Bc, per['h'], per['s'], per['ell'])
        out.update(cls_shared_ll=shared['ll'], cls_ll=per['ll'], cls_h=per['h'], cls_s=per['s'], cls_ell=per['ell'],
                   cls_log_sd=sd, cls_mean=p['theta'], cls_mean_sd=p['theta_sd'], cls_converged=per['converged'])
        say(f'classes: log evidence {per["ll"] - shared["ll"]:+.1f} for separate statistics (vs shared)')
        for k, cname in enumerate(covariates.CLASS_NAMES):
            say(f'  {cname:22s} mean {p["theta"][k]:.2f}+-{p["theta_sd"][k]:.2f}, sd {per["s"][k]:.2f} '
                f'(x/ {np.exp(sd[1+k]):.2f}), length {per["ell"][k]:.0f} km (x/ {np.exp(sd[4+k]):.2f})')

    # (4) hypotheses
    if 'hypotheses' in PARTS:
        hyps = {}
        for r in range(len(GROUPS) + 1):
            for gs in itertools.combinations(GROUPS, r):
                hyps['auto: ' + ('+'.join(gs) or 'mean only')] = design(columns(gs))[:3]
        for name, lab in merging_models(case).items():
            B = indicator(lab)
            hyps[name] = (B, np.full(B.shape[1], LEVEL[0]), np.full(B.shape[1], LEVEL[1]))
        for name, cols in HAND.items():
            hyps['hand: ' + name] = design(cols)[:3]
        res = [score(nm, B, lo, hi, m1, dw) for nm, (B, lo, hi) in hyps.items()]
        order = np.argsort([-r['ll'] for r in res])
        out.update(hyp_names=np.array([r['name'] for r in res]), hyp_ll=np.array([r['ll'] for r in res]),
                   hyp_ll_plain=np.array([r['ll_plain'] for r in res]), hyp_K=np.array([r['K'] for r in res]),
                   hyp_hyper=np.array([[r['h'], r['s'], r['ell']] for r in res]))
        say('hypotheses, best first (log evidence relative to the best; with u | without u):')
        best, best_plain = res[order[0]]['ll'], max(r['ll_plain'] for r in res)
        for i in order:
            r = res[i]
            say(f'  {r["ll"] - best:8.1f} | {r["ll_plain"] - best_plain:9.1f}  K {r["K"]:2d}  {r["name"]}')
    say(f'done in {time.time() - t0:.0f} s')
    return case, out, log


CASE_DATA = make_cases()

if __name__ == '__main__':
    print(f'{n} cells, {N} dimensions; H&M mean {hm_mean:.3f} SI.km; cases {CASES}; parts {PARTS}', flush=True)
    with multiprocessing.get_context('fork').Pool(min(N_WORKERS, len(CASES))) as pool:
        path = f'{pb["cache"]}/linear_models.npz'
        save = dict(np.load(path)) if os.path.exists(path) else {}       # parts run earlier are kept
        all_cases = list(dict.fromkeys([str(c) for c in save.get('cases', [])] + CASES))
        save.update(cases=np.array(all_cases), hm_mean=hm_mean, theta_true=THETA_TRUE, s_true=S_TRUE,
                    ell_true=ELL_TRUE, b_true_cols=np.array(B_TRUE_COLS))
        for case, out, log in pool.imap_unordered(run_case, CASES):
            print('\n'.join(log), flush=True)
            save.update({f'{case}_{k}': v for k, v in out.items()})
            if CASE_DATA[case]['truth'] is not None:
                save[f'{case}_truth'] = CASE_DATA[case]['truth']
            save[f'{case}_dw'] = CASE_DATA[case]['dw']
            np.savez(path, **save)
