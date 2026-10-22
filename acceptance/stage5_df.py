"""
Stage 5 synthetic tests of DF (ADR 0007), on DF-AUS's problem (Australia's polygon domains, LCS-1 noise, 16-133):
  1 own model   truths drawn from DF itself (known means, strengths, beta 2.5, ell 500 km), 20 noise draws: the
                domain averages of VIS and the means must fall within their error bars as often as the error bars say
                (|z| < 2 in ~95%), and beta, ell and the strengths must come back near the truth; error bars at the
                best-fit hyperparameters (plug-in) and averaged over sampled hyperparameters (one chain of STEPS)
  2 realistic   the truth of provinces with ~200 km variation (truth_hast_var of the first build's
                realistic_synthetic.npz; not fractal): what DF recovers when its assumptions are wrong
Run from the repo root (env curiesphere): python acceptance/stage5_df.py [n_draws] [chain steps]
"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from remit.inversion import config, problem
from remit.inversion.cache import Cache
from remit.inversion.methods import df

N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
TRUE = dict(beta=2.5, ell=500.)
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
N_POST = 60
REALISTIC = 'notebooks/regions/australia/cache/realistic_synthetic.npz'

cfg = config.load('inversions/df_aus_lcs1_16-133.toml')
pb = problem.local(cfg, Cache(), print)
cells = pb.cells
cell_km2 = 4*np.pi*df.R_KM**2/(12*cells.nside**2)
lays, _ = df.layouts(cells, cfg, Cache(), print)
label, names = lays['polygons']
K = label.max() + 1
area = cells.area_km2


def model(yw):
    return df.DomainFractal(pb.Gw, yw, label, cells.clon, cells.clat, cell_km2, 0., 4.)


base = model(pb.yw)
rng = np.random.default_rng(20261009)
theta_true = rng.uniform(0.5, 2.0, K)
s_true = np.exp(rng.uniform(np.log(0.1), np.log(0.6), K))
_, Ls = base.factors(TRUE['beta'], TRUE['ell'])
W = np.asarray(pb.W.todense()) if hasattr(pb.W, 'todense') else pb.W
avg_w = base.S*area[:, None]
avg_w /= avg_w.sum(0)


def one(i):
    r = np.random.default_rng([20261009, i])
    v = base.S @ theta_true
    for b, L in zip(base.blocks, Ls):
        v[b] += s_true[label[b][0]]*(L @ r.standard_normal(len(b)))
    yw = pb.Gw @ v + W @ pb.draw(r, model_error=False)
    m = model(yw)
    f = m.fit()
    p = m.posterior(f, area)
    bnds = m.bounds((0.01, 10.), (2., 3.), (50., 5000.), False)
    X, _, acc = m.chain(f, bnds, STEPS, [20261009, i, 1], df.proposal_cov(m.hessian(f), bnds), start_sd=0)
    pick = X[STEPS//3:][np.linspace(0, STEPS - STEPS//3 - 1, N_POST).astype(int)]
    q = df.mixture(m, [m.fit_at(x, False) for x in pick], area)
    true_avg = avg_w.T @ v
    return dict(z_avg=(p['avg'] - true_avg)/p['avg_sd'], z_theta=(p['theta'] - theta_true)/p['theta_sd'],
                z_avg_s=(q['avg'] - true_avg)/q['avg_sd'], z_theta_s=(q['theta'] - theta_true)/q['theta_sd'],
                acc=acc, beta=f['beta'], ell=f['ell'], log_s=np.log(f['s']), converged=f['converged'])


if __name__ == '__main__':
    t0 = time.time()
    with mp.get_context('fork').Pool(3) as pool:
        out = pool.map(one, range(N_DRAWS))
    za, zt = np.concatenate([o['z_avg'] for o in out]), np.concatenate([o['z_theta'] for o in out])
    ls = np.array([o['log_s'] for o in out])
    big = np.bincount(label, area) > np.median(np.bincount(label, area))
    print(f'\n1 own model: {N_DRAWS} draws, {(time.time() - t0)/60:.1f} min, converged {sum(o["converged"] for o in out)}')
    zas, zts = np.concatenate([o['z_avg_s'] for o in out]), np.concatenate([o['z_theta_s'] for o in out])
    print(f'  domain averages of VIS: |z| < 2 in {np.mean(abs(za) < 2):.1%} (sampled {np.mean(abs(zas) < 2):.1%}), '
          f'z s.d. {za.std():.2f} (sampled {zas.std():.2f})')
    print(f'  process means theta:    |z| < 2 in {np.mean(abs(zt) < 2):.1%} (sampled {np.mean(abs(zts) < 2):.1%}), '
          f'z s.d. {zt.std():.2f} (sampled {zts.std():.2f}); acceptance {np.mean([o["acc"] for o in out]):.2f}')
    print(f'  beta {np.mean([o["beta"] for o in out]):.2f} +- {np.std([o["beta"] for o in out]):.2f} (true {TRUE["beta"]}), '
          f'ell {np.exp(np.mean(np.log([o["ell"] for o in out]))):.0f} km x/ {np.exp(np.std(np.log([o["ell"] for o in out]))):.2f} (true {TRUE["ell"]:.0f})')
    err = ls - np.log(s_true)
    print(f'  log strength error, larger half of the domains: median {np.median(err[:, big]):+.2f}, '
          f'RMS {np.sqrt(np.mean(err[:, big]**2)):.2f}; smaller half: median {np.median(err[:, ~big]):+.2f}, '
          f'RMS {np.sqrt(np.mean(err[:, ~big]**2)):.2f}')

    z = np.load(REALISTIC)
    truth = z['truth_hast_var']
    assert len(truth) == len(cells)
    yw = pb.Gw @ truth + W @ pb.draw(np.random.default_rng(20261010), model_error=False)
    m = model(yw)
    f = m.fit()
    p = m.posterior(f, area)
    za = (p['avg'] - avg_w.T @ truth)/p['avg_sd']
    print(f'\n2 realistic truth (hast_var), one noise draw: beta {f["beta"]:.2f}, ell {f["ell"]:.0f} km, '
          f'strengths at the lower bound {int(f["at_bound"][:K].sum())}/{K}')
    print(f'  domain averages: |z| < 2 in {np.mean(abs(za) < 2):.0%}, z s.d. {za.std():.2f}, '
          f'RMS error {np.sqrt(np.mean((p["avg"] - avg_w.T @ truth)**2)):.3f} SI km; '
          f'cells: RMS error {np.sqrt(np.mean((p["v"] - truth)**2)):.3f}, truth s.d. {truth.std():.3f}')
