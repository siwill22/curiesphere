"""
Stage 2 acceptance (docs/inversion-toolbox-spec.md): PD, CR and bounds for Australia against the first inversions
(notebooks/inversion/real_LCS1.csv and columns.npz; notebooks/regions/australia/cache: covariates.npz,
linear_models.npz, null_regression.npz, bounds.npz), and the mode agreement of explore against final.

  PD       estimator: bounded least squares on the first inversion's own polygon-exact G and data reproduces its 39
           values. End to end: domains follow cell edges (ADR 0001), so the values are compared in units of the first
           inversion's noise s.d.; the noise s.d. from draws against its own.
  CR       covariates against the first build's; at its hyperparameters, on its whitened data, the regression
           coefficients and evidence; end to end, the coefficients, class statistics and hypothesis ranking; the null
           test (are the coefficient s.d. honest?) repeated with the toolbox: spreads against the first build's.
  bounds   on the first build's whitened data with its misfit limit and roughness, the extremal values; end to end
           reported.
  modes    explore against final for each method, reported for review (ADR 0006).

Run from the repository root (conda env curiesphere):
    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 15 python acceptance/stage2.py
"""
import importlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from remit.inversion import cache, config, core, covariates, geometry, gmrf, graph, parameterise, problem, results

OLD = 'notebooks/regions/australia/cache'
ok = True


def check(label, value, expected, tol):
    global ok
    good = abs(value - expected) <= tol
    ok &= good
    print(f'  {label:58s} {value:11.5g}  expected {expected:9.4g} +- {tol:g}  {"ok" if good else "FAIL"}', flush=True)


def result(name):
    path = f'inversions/results/{name}.nc'
    if not os.path.exists(path):
        cfg = config.load(f'inversions/{name}.toml')
        results.save(importlib.import_module(f'remit.inversion.methods.{cfg["method"]["name"]}').run(cfg), path)
    return results.load(path)


C = cache.Cache()
oc, od, E = np.load(f'{OLD}/cells.npz'), np.load(f'{OLD}/data.npz'), np.load(f'{OLD}/errors.npz')
lm, cov_old = np.load(f'{OLD}/linear_models.npz'), np.load(f'{OLD}/covariates.npz')
P_old = np.linalg.svd(E['T'].T, full_matrices=False)[0]
sigma = float(np.sqrt(np.mean((od['d_lcs'] - od['d_mf7'])**2)))
G0 = np.load(f'{OLD}/G.npz')['G']
Gw0, dw0 = P_old.T @ G0/sigma, P_old.T @ (od['d_lcs'] - od['b'])/sigma
area0, edges0 = oc['area_km2'], oc['edges']
Lw0 = graph.laplacian(edges0, 1/graph.edge_lengths_km(edges0, oc['clon'], oc['clat'])**2, len(area0))
D0 = graph.difference_operator(edges0, oc['clon'], oc['clat'])

# ------------------------------------------------------------------ PD
print('PD')
old = pd.read_csv('notebooks/inversion/real_LCS1.csv').set_index('param')
z = np.load('notebooks/inversion/columns.npz')
s_pd = float(np.sqrt(np.mean((z['d_lcs'] - z['d_mf7'])**2)))
v = core.box_qp((z['G']/s_pd).T @ (z['G']/s_pd), (z['G']/s_pd).T @ ((z['d_lcs'] - z['b'])/s_pd), 0., np.inf, tol=1e-12)
check('estimator on the first inversion\'s G: max |v - its v| (SI.km)', np.abs(v - old.v.values).max(), 0, 1e-5)
ds = result('pd_aus_lcs1_16-133')
t = json.loads(ds.attrs['domains'])
dv = np.array([(t[p]['v'] - old.loc[p, 'v'])/old.loc[p, 'noise_std'] for p in t])
ratio = np.array([t[p]['sd_draws']/old.loc[p, 'noise_std'] for p in t])
print(f'  end to end ({len(t)} domains on cell edges): |v - first| / its noise s.d.: median {np.median(np.abs(dv)):.2f}, '
      f'max {np.abs(dv).max():.2f}; noise s.d. ratio median {np.median(ratio):.2f}, range {ratio.min():.2f}-{ratio.max():.2f}')
check('end to end: max |v - first| / first noise s.d.', np.abs(dv).max(), 0, 2.)
check('end to end: median noise s.d. ratio', np.median(ratio), 1, 0.15)
ex = result('pd_aus_lcs1_16-133_explore')
te = json.loads(ex.attrs['domains'])
de = np.array([(te[p]['v'] - t[p]['v'])/t[p]['sd_draws'] for p in t if p in te])
print(f'  modes (explore: the same cells, 30 noise draws): |explore - final| / final noise s.d.: median '
      f'{np.median(np.abs(de)):.2f}, max {np.abs(de).max():.2f}; noise s.d. ratio explore/final median '
      f'{np.median([te[p]["sd_draws"]/t[p]["sd_draws"] for p in t]):.2f}')

# ------------------------------------------------------------------ CR
print('CR')
cfg = config.load('inversions/cr_aus_lcs1_16-133.toml')
mask = geometry.target_mask(cfg['target']['area'], C)
cells = geometry.make_cells(mask, 64)
cov = covariates.load(cells, cfg['covariates'], C, geometry.area_name(cfg['target']['area']))
for k in ('heat_flow', 'heat_flow_seisheat', 'heat_flow_heatr', 'age', 'undated', 'cls'):
    check(f'covariate {k}: max difference', np.nanmax(np.abs(cov[k] - cov_old[k])), 0, 1e-9)
check('covariate group: cells differing', np.sum(cov['group'] != cov_old['group']), 0, 0)
check('covariate crust: max difference (km)', np.abs(cov['crust'] - cov_old['crust']).max(), 0, 0.5)
dl = np.abs(cov['lab'] - cov_old['lab'])
print(f'  covariate lab: {np.sum(dl > 0.01)} cells differ (max {dl.max():.1f} km, at {cells.clat[np.argmax(dl)]:.1f} deg: the '
      f'northern edge of the AuSREM grid, interpolated next to missing values)')
check('covariate lab: cells differing by more than 0.01 km', np.sum(dl > 0.01), 0, 25)
m0 = gmrf.GMRF(Gw0, dw0, area0, Lw0)
Bd, lo, hi, names = parameterise.design(parameterise.columns(parameterise.GROUPS), cov, area0)
h, s, ell = lm['real_reg_main_hyper']
lr, _ = m0.model(Bd, ell)
sol = lr.solve(dw0, h**2/s**2)
sd = h*np.sqrt(np.diag(np.linalg.inv(sol['P'])))
check('at the first build\'s hyperparameters: max |theta - its| / s.d.',
      np.max(np.abs(sol['theta'] - lm['real_reg_main_theta'])/lm['real_reg_main_sd']), 0, 0.03)
check('  ... max relative difference of the s.d.', np.max(np.abs(sd/lm['real_reg_main_sd'] - 1)), 0, 0.01)
check('  ... log evidence - its', m0.evidence(Bd, h, s, ell, (lo, hi)) - float(lm['real_reg_main_ll']), 0, 0.05)

ds = result('cr_aus_lcs1_16-133')
regs = json.loads(ds.attrs['regressions'])
for vname in ('main', 'heat_flow_seisheat', 'heat_flow_heatr'):
    r = regs[vname]
    d = (np.array(r['theta']) - lm[f'real_reg_{vname}_theta'])/lm[f'real_reg_{vname}_sd']
    check(f'end to end, regression {vname}: max |theta - first| / first s.d.', np.abs(d).max(), 0, 0.1)
cl = json.loads(ds.attrs['classes'])
print(f'  classes: log evidence {cl["log_evidence_vs_shared"]:+.1f} for separate statistics (first build '
      f'{float(lm["real_cls_ll"] - lm["real_cls_shared_ll"]):+.1f})')
for k, n in enumerate(cl['names']):
    print(f'    {n:22s} mean {cl["mean"][k]:.2f}+-{cl["mean_sd"][k]:.2f} ({lm["real_cls_mean"][k]:.2f}+-{lm["real_cls_mean_sd"][k]:.2f}), '
          f's.d. {cl["s"][k]:.2f} ({lm["real_cls_s"][k]:.2f}), length {cl["ell"][k]:.0f} km ({lm["real_cls_ell"][k]:.0f})')
check('classes: log evidence gain - first build\'s', cl['log_evidence_vs_shared'] - float(lm['real_cls_ll'] - lm['real_cls_shared_ll']), 0.5, 1.5)
check('classes: max |mean - first| / first s.d.', np.max(np.abs(np.array(cl['mean']) - lm['real_cls_mean'])/lm['real_cls_mean_sd']), 0, 0.5)
hy = {r['name']: r for r in json.loads(ds.attrs['hypotheses'])}
old_h = dict(zip([str(n) for n in lm['real_hyp_names']], lm['real_hyp_ll']))
common = [n for n in hy if n in old_h and np.isfinite(old_h[n]) and np.isfinite(hy[n]['ll'])]
ref = 'auto: mean only'
d = np.array([(hy[n]['ll'] - hy[ref]['ll']) - (old_h[n] - old_h[ref]) for n in common])
print(f'  hypotheses: {len(common)} in common; log evidence relative to "mean only", toolbox - first: median '
      f'{np.median(d):+.2f}, range {d.min():+.2f} to {d.max():+.2f}')
check('hypotheses: max |difference of relative log evidence|', np.abs(d).max(), 0, 1.)
best_t = max(common, key=lambda n: hy[n]['ll'])
best_o = max(common, key=lambda n: old_h[n])
print(f'  best covariate hypothesis: toolbox "{best_t}", first build "{best_o}"')

print('  null test (truths without covariate effects; 30 replicates, toolbox noise draws)')
nr = np.load(f'{OLD}/null_regression.npz')
pb = problem.local(cfg, C, lambda m: None)
mt = gmrf.GMRF(pb.Gw, pb.yw, area0, Lw0)
prov = np.unique(cov['key'], return_inverse=True)[1]
th = []
t0 = time.time()
for r in range(30):
    rng = np.random.default_rng(20261007 + r)
    truth = np.clip(np.exp(float(nr['log_mu']) + float(nr['log_sd'])*rng.standard_normal(prov.max() + 1)), 0.01, 4.)[prov]
    m = gmrf.GMRF(pb.Gw, pb.Gw @ truth + pb.W @ pb.draw(rng, model_error=False), area0, Lw0)
    m._parts = mt._parts                      # the factors depend on the cells only
    f = m.fit(Bd, (lo, hi))
    p = m.posterior(Bd, f)
    th.append((p['theta'], p['theta_sd']))
th, sds = np.array([a for a, _ in th]), np.array([b for _, b in th])
print(f'    ({time.time() - t0:.0f} s)')
for i in nr['slopes']:
    sp, sp_old = th[:, i].std(ddof=1), nr['theta'][:, i].std(ddof=1)
    print(f'    {names[i]:10s} spread {sp:.3f} (first {sp_old:.3f}), mean reported s.d. {sds[:, i].mean():.3f} '
          f'({nr["sd"][:, i].mean():.3f}), |slope| > 2 s.d. {np.mean(np.abs(th[:, i]) > 2*sds[:, i]):.0%} '
          f'({np.mean(np.abs(nr["theta"][:, i]) > 2*nr["sd"][:, i]):.0%})')
    check(f'null spread of {names[i]} / first build\'s', sp/sp_old, 1, 0.5)
ex = result('cr_aus_lcs1_16-133_explore')
re_ = json.loads(ex.attrs['regressions'])['main']
d = (np.array(re_['theta']) - np.array(regs['main']['theta']))/np.array(regs['main']['sd'])
print(f'  modes: |explore - final| / final s.d. of the main coefficients: max {np.abs(d).max():.2f}')

# ------------------------------------------------------------------ bounds
print('bounds')
bo = np.load(f'{OLD}/bounds.npz')
Q = parameterise.quantities(cov, area0)
for q in ('NAC mean', 'craton - rest'):
    for tag, k in (('pos', None), ('r1', 1.)):
        target = float(bo[f'real_{q}_{tag}_target'])
        rough = None if k is None else k*float(np.linalg.norm(D0 @ lm['real_l2_v']))
        o = core.extremes(Gw0, dw0, Q[q], target, 0., 4., D=D0, rough_max=rough)
        for e in ('min', 'max'):
            check(f'on the first build\'s data: {q}, {tag}, {e}', o[e]['value'], float(bo[f'real_{q}_{tag}_{e}']), 2e-3)
ds = result('bounds_aus_lcs1_16-133')
tb = json.loads(ds.attrs['bounds_table'])
key = {'positivity': 'pos', 'roughness 4 x L2': 'r4', 'roughness 2 x L2': 'r2', 'roughness 1 x L2': 'r1'}
for q, rows in tb.items():
    print(f'  {q:14s} ' + '; '.join(f'{tag}: {a:.2f} to {b:.2f} (first {float(bo[f"real_{q}_{key[tag]}_min"]):.2f} to '
                                     f'{float(bo[f"real_{q}_{key[tag]}_max"]):.2f})' for tag, (a, b) in rows.items()))
ex = json.loads(result('bounds_aus_lcs1_16-133_explore').attrs['bounds_table'])
print('  modes (explore): ' + '; '.join(f'{q} r1 {ex[q]["roughness 1 x L2"][0]:.2f} to {ex[q]["roughness 1 x L2"][1]:.2f}'
                                        for q in ex))
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
