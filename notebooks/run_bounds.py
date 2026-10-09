"""
Extremal bounds (domains/bounds.py): the smallest and largest area-weighted mean VIS of the North, West
and South Australian Cratons, and of the craton-minus-rest contrast, over all models with
0 <= v <= 4 SI.km that fit the data to chi2 <= h^2 chi2_95(N), with h the noise level of the GP fit
(run_linear_models.py, which must have run first). With positivity alone the interval is what the
data force; it is repeated with a roughness limit |D v| <= k x the roughness of the case's L2 map
(k in ROUGH, env, default 4,2,1), to show how much smoothness it takes to pin the means down.
Cases as in run_linear_models.py (synthetic truths are checked for bracketing).
Output: bounds.npz in the region cache.

Run from notebooks/ (conda env pygmt17), lightly:
    N_WORKERS=3 nice -n 15 env PYTHONPATH=..:$PYTHONPATH python run_bounds.py regions/australia/config.toml
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '1')
import sys
import time
import multiprocessing

import numpy as np
from scipy.stats import chi2 as chi2_dist

from domains import bounds, covariates, graph, linear, problem

N_WORKERS = int(os.environ.get('N_WORKERS', 3))
ROUGH = [float(x) for x in os.environ.get('ROUGH', '4,2,1').split(',')]

pb = problem.load(sys.argv[1])
Gw, area = pb['Gw'], pb['area']
N = Gw.shape[0]
cov = dict(np.load(f'{pb["shared"]}/covariates.npz'))
lm = dict(np.load(f'{pb["cache"]}/linear_models.npz'))           # eager: forked workers must not share a zip handle
CASES = [str(c) for c in lm['cases']]
craton = np.isin(cov['group'], covariates.CRATON_GROUPS)
D = graph.difference_operator(pb['edges'], pb['clon'], pb['clat'])
QUANT = {g.split()[0][0] + 'AC mean': bounds.mean_functional(area, cov['group'] == g) for g in covariates.CRATON_GROUPS}
QUANT['craton - rest'] = bounds.mean_functional(area, craton) - bounds.mean_functional(area, ~craton)


def job(args):
    case, q, k = args
    t = time.time()
    dw = lm[f'{case}_dw']
    target = float(lm[f'{case}_gp_h'])**2*chi2_dist.ppf(0.95, N)
    rough = None if k is None else k*float(np.linalg.norm(D @ lm[f'{case}_l2_v']))
    o = bounds.extremes(Gw, dw, QUANT[q], target, linear.V_LO, linear.V_HI, D=D, rough_max=rough)
    return case, q, k, target, o, time.time() - t


if __name__ == '__main__':
    jobs = [(c, q, k) for c in CASES for q in QUANT for k in [None] + ROUGH]
    save = dict(cases=np.array(CASES), quantities=np.array(list(QUANT)), rough=np.array(ROUGH))
    with multiprocessing.get_context('fork').Pool(N_WORKERS) as pool:
        for case, q, k, target, o, sec in pool.imap_unordered(job, jobs):
            tag = 'positivity only' if k is None else f'roughness <= {k:g} x L2'
            line = f'[{case}] {q:14s} {tag:20s} {o["min"]["value"]:6.2f} to {o["max"]["value"]:5.2f} SI.km'
            truth = f'{case}_truth'
            if truth in lm:
                tv = float(QUANT[q] @ lm[truth])
                inside = o['min']['value'] - 1e-6 <= tv <= o['max']['value'] + 1e-6
                line += f'; truth {tv:.2f} ({"inside" if inside else "OUTSIDE"})'
            print(line + f' ({sec:.0f} s)', flush=True)
            key = f'{case}_{q}_{"pos" if k is None else f"r{k:g}"}'
            save.update({f'{key}_min': o['min']['value'], f'{key}_max': o['max']['value'], f'{key}_vmin': o['min']['v'],
                         f'{key}_vmax': o['max']['v'], f'{key}_target': target})
            np.savez(f'{pb["cache"]}/bounds.npz', **save)
