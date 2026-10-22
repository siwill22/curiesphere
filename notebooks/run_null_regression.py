"""
Null test of the covariate regression (run_linear_models.py, part 1): are its error bars honest?

Truths with no covariate effect: every cell takes the VIS of its Hasterok province, with province values
drawn afresh from the lognormal of realistic_synthetic.py (median and spread of its 'hast' truth), as
in that script. Data: Gw truth + a random-phase LCS-1 - MF7 noise realisation (whitened; realisations
not used elsewhere). Each replicate is fitted with the main regression (class means + crust, age,
undated, heat flow, LAB; GMRF rest, hyperparameters by maximum evidence).
For each covariate slope (true value 0): the spread across replicates (the real uncertainty), the mean
reported sd, and how often |slope| > 2 reported sd (5% if the error bars were honest). The real-data
slopes are then compared with the null spread.
Output: null_regression.npz in the region cache.

Run from notebooks/ (conda env pygmt17), lightly:
    N_WORKERS=3 nice -n 15 env PYTHONPATH=..:$PYTHONPATH python run_null_regression.py regions/australia/config.toml
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '1')
import sys
import time
import multiprocessing

import numpy as np

from domains import gmrf, graph, problem
from domains.design import GROUPS, columns, design

N_WORKERS = int(os.environ.get('N_WORKERS', 3))
N_REP = int(os.environ.get('N_REP', 30))
NOISE_IDX = [j for j in range(55, 100) if j != 90]           # realisations not used by other runs
SEED = 20261007

pb = problem.load(sys.argv[1])
Gw, area, edges = pb['Gw'], pb['area'], pb['edges']
N, n = Gw.shape
cov = dict(np.load(f'{pb["shared"]}/covariates.npz'))
rs = dict(np.load(f'{pb["shared"]}/realistic_synthetic.npz', allow_pickle=True))
lm = dict(np.load(f'{pb["cache"]}/linear_models.npz'))
Lw = graph.laplacian(edges, 1/graph.edge_lengths_km(edges, pb['clon'], pb['clat'])**2, n)
prov = rs['prov']
log_mu = float(rs['hm_lmu'])
log_sd = float(np.sqrt(np.mean((np.log(rs['v_prov']) - log_mu)**2)))     # spread of the 'hast' province values
B, lo, hi, names = design(columns(GROUPS), cov, area)
slopes = [i for i, nm in enumerate(names) if not nm.startswith('mean')]


def job(r):
    t = time.time()
    rng = np.random.default_rng(SEED + r)
    truth = np.clip(np.exp(log_mu + log_sd*rng.standard_normal(prov.max() + 1)), 0.01, 4.)[prov]
    dw = Gw @ truth + pb['noise_w'][NOISE_IDX[r % len(NOISE_IDX)]]
    m = gmrf.Model(Gw, dw, area, Lw)
    fit = m.fit(B, lo, hi)
    p = m.posterior(B, fit['h'], fit['s'], fit['ell'])
    return r, p['theta'], p['theta_sd'], np.r_[fit['h'], fit['s'], fit['ell']], time.time() - t


if __name__ == '__main__':
    print(f'{N_REP} null replicates; province log-VIS mean {log_mu:.2f}, sd {log_sd:.2f}', flush=True)
    th, sd, hy = np.zeros((N_REP, len(names))), np.zeros((N_REP, len(names))), np.zeros((N_REP, 3))
    with multiprocessing.get_context('fork').Pool(N_WORKERS) as pool:
        for r, t_, s_, h_, sec in pool.imap_unordered(job, range(N_REP)):
            th[r], sd[r], hy[r] = t_, s_, h_
            print(f'  replicate {r:2d}: ' + ', '.join(f'{names[i]} {t_[i]:+.2f}' for i in slopes) + f' ({sec:.0f} s)', flush=True)
    print('slope         null spread   mean reported sd   |slope| > 2 sd   real slope (z vs null spread)')
    for i in slopes:
        spread = th[:, i].std(ddof=1)
        real = lm.get('real_reg_main_theta')
        z = f'{real[i]:+.2f} ({real[i]/spread:+.1f})' if real is not None else '-'
        print(f'{names[i]:12s}  {spread:10.3f}   {sd[:, i].mean():16.3f}   {np.mean(np.abs(th[:, i]) > 2*sd[:, i]):14.0%}   {z}')
    np.savez(f'{pb["cache"]}/null_regression.npz', names=np.array(names), slopes=np.array(slopes), theta=th, sd=sd,
             hyper=hy, log_mu=log_mu, log_sd=log_sd)
