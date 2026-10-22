"""
Sensitivity of the L2 and TV maps to their two choices, for the viewer (export_viewer_inversion.py):
the smoothing length ell (200 km and the GP's fitted length) and the penalty weight beta (0.3, 1 and 3
x the beta at which the misfit equals N h^2, h the GP noise level). Reference VIS: the H&M mean.
Cases and GP fits from linear_models.npz (run_linear_models.py must have run).
Output: map_sweep.npz in the region cache.

Run from notebooks/ (conda env pygmt17), lightly:
    N_WORKERS=3 nice -n 15 env PYTHONPATH=..:$PYTHONPATH python run_map_sweep.py regions/australia/config.toml
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '1')
import sys
import time
import multiprocessing

import numpy as np

from domains import gmrf, graph, problem

N_WORKERS = int(os.environ.get('N_WORKERS', 3))
BETA_REL = [0.3, 1., 3.]
ELLS = ['200', 'gp']

pb = problem.load(sys.argv[1])
Gw, area = pb['Gw'], pb['area']
N = Gw.shape[0]
lm = dict(np.load(f'{pb["cache"]}/linear_models.npz'))
cov = dict(np.load(f'{pb["shared"]}/covariates.npz'))
hm_mean = float(np.average(cov['hm_vis'], weights=area))
D = graph.difference_operator(pb['edges'], pb['clon'], pb['clat'])
GtG = Gw.T @ Gw
CASES = [str(c) for c in lm['cases']]


def job(args):
    case, ell_key, p = args
    t = time.time()
    dw = lm[f'{case}_dw']
    ell = float(lm[f'{case}_gp_ell']) if ell_key == 'gp' else float(ell_key)
    target = N*float(lm[f'{case}_gp_h'])**2
    solve = lambda b: gmrf.tikhonov(Gw, dw, area, D, ell, b, hm_mean, p=p, GtG=GtG)
    beta0, _, _ = gmrf.discrepancy(solve, Gw, dw, target, n_bisect=16 if p == 2 else 12)
    out = {}
    for r in BETA_REL:
        v = solve(beta0*r)
        out[r] = (v, float(np.sum((Gw @ v - dw)**2)))
    return case, ell_key, ell, p, beta0, out, time.time() - t


if __name__ == '__main__':
    jobs = [(c, e, p) for p in (2, 1) for c in CASES for e in ELLS]
    save = dict(cases=np.array(CASES), beta_rel=np.array(BETA_REL), ells=np.array(ELLS))
    with multiprocessing.get_context('fork').Pool(N_WORKERS) as pool:
        for case, ek, ell, p, beta0, out, sec in pool.imap_unordered(job, jobs):
            m = 'l2' if p == 2 else 'tv'
            for r, (v, chi2) in out.items():
                key = f'{case}_{m}_ell{ek}_b{r:g}'
                save.update({f'{key}_v': v, f'{key}_chi2': chi2, f'{key}_beta': beta0*r, f'{key}_ell': ell})
            print(f'[{case}] {m} ell {ell:.0f} km: beta at target {beta0:.3g}; chi2 at x0.3/1/3: '
                  + ', '.join(f'{out[r][1]:.0f}' for r in BETA_REL) + f' ({sec:.0f} s)', flush=True)
            np.savez(f'{pb["cache"]}/map_sweep.npz', **save)
