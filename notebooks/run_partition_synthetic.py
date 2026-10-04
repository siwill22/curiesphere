"""
Checks of the cell-partition sampler (domains/partition.py) on the synthetic truths of
run_lvoronoi_synthetic.py (same data, same seed):
  vor6   6 Voronoi domains (two pairs with almost equal values), noise h = 1 and 2
  real7  the A2 merging model on the real data (lambda_95, gamma 0.01 lambda_95), noise h = 1
Merging penalties lam = LAM_REL lambda_95 (default 1/3) and gamma = GAMMAS x lambda_95; the sampler's prior
exp(-beta L - kappa U) is their equivalent at the noise level of the merging fit (partition.merging_prior);
the reported score is the merging score chi2 + lam U + gamma L. Chain 0 starts from the merging result
(merge.run) on the same data; chain 1 from a single unit. Both chains are parallel-tempered (TEMPS).
Also: the exact test of the moves on all 203 partitions of a 2 x 3 grid.

Run from notebooks/ (conda env pygmt17):
    PYTHONPATH=..:$PYTHONPATH python run_partition_synthetic.py regions/australia/config.toml
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '1')
import sys
import time
import zlib
import multiprocessing

import numpy as np

from domains import voronoi, merge, partition, outputs

REG = os.path.dirname(os.path.abspath(sys.argv[1]))
C = f'{REG}/cache'
TEMPS = list(np.geomspace(1., 50., 8))
N_SWEEPS, THIN, SWEEP = int(os.environ.get('N_SWEEPS', 2000)), 5, 50
LAM_REL = float(os.environ.get('LAM_REL', 1/3))
GAMMAS = [float(x) for x in os.environ.get('GAMMAS', '0.001,0.003,0.01').split(',')]
TAG = os.environ.get('TAG', '')

E = np.load(f'{C}/errors.npz')
z = np.load(f'{C}/data.npz')
G = np.load(f'{C}/G.npz')['G']
cells = np.load(f'{C}/cells.npz')
sigma = float(np.sqrt(np.mean((z['d_lcs'] - z['d_mf7'])**2)))
T = np.linalg.svd(E['T'].T, full_matrices=False)[0].T/sigma
Gw = T @ G
edges = cells['edges']
lam95 = float(np.load(f'{C}/merge_points_sweep.npz')['fine_lam95'])
PV = voronoi.setup(Gw, cells['clon'], cells['clat'], None)

# truths and data exactly as in run_lvoronoi_synthetic.py
rng = np.random.default_rng(20261004)
truths = {}
seeds6 = rng.choice(PV['n'], 6, replace=False)
truths['vor6'] = (voronoi.assign(PV, seeds6), 0.6*np.exp(0.5*rng.standard_normal(6)))
m = np.load(f'{C}/merge_points_sweep.npz', allow_pickle=True)
i = [k for k, s in enumerate(m['settings']) if np.isclose(s[0], 1.) and np.isclose(s[1], 0.01)][0]
lab7 = np.unique(m['fine_label'][i], return_inverse=True)[1]
truths['real7'] = (lab7, np.array([m['fine_v'][i][lab7 == k][0] for k in range(lab7.max()+1)]))
CASES = [('vor6', 1.), ('vor6', 2.), ('real7', 1.)]
data = {}
for name, h in CASES:
    lab, v = truths[name]
    data[(name, h)] = Gw @ v[lab] + h*rng.standard_normal(Gw.shape[0])


def job(args):
    name, h, g, chain = args
    r = np.random.default_rng(1000*chain + zlib.crc32(f'{name}{h}{g}'.encode()) % 997)
    t = time.time()
    dw = data[(name, h)]
    lam, gamma = LAM_REL*lam95, g*lam95
    mr = merge.run(merge.setup(Gw, edges), dw, lam, gamma, [0.03, 0.1, 0.3])
    beta, kappa, h2 = partition.merging_prior(lam, gamma, mr['chi2'], Gw.shape[0], mr['K'])
    P = partition.setup(Gw, dw, edges, beta=beta, kappa=kappa, u_max=60)
    info = dict(merge_label=mr['label'], merge_K=mr['K'], merge_score=mr['chi2'] + lam*mr['K'] + gamma*mr['L'],
                merge_seconds=time.time() - t, beta=beta, kappa=kappa, h2=h2)
    lab0 = mr['label'] if chain == 0 else np.zeros(P['n'], int)
    out, acc = partition.sample(P, lab0, N_SWEEPS, TEMPS, r, thin=THIN, sweep=SWEEP)
    U = np.array([o['U'] for o in out]); L = np.array([o['L'] for o in out]); chi2 = np.array([o['chi2'] for o in out])
    return dict(name=name, h=h, g=g, chain=chain, U=U, L=L, chi2=chi2, score=chi2 + lam*U + gamma*L,
                label=np.array([o['label'] for o in out]), v=[o['v'] for o in out], acc=acc,
                seconds=time.time() - t, **info)


if __name__ == '__main__':
    t = time.time()
    grid_edges = np.array([(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)])
    ex, fr = partition.partition_prior_test(grid_edges, 6, lambda U, L: -0.7*L - 0.5*U, 300000,
                                            np.random.default_rng(1))
    print(f'exact test, 203 partitions of a 2 x 3 grid, target exp(-0.7 L - 0.5 U): total variation '
          f'{0.5*np.abs(fr - ex).sum():.4f} ({time.time()-t:.0f} s)', flush=True)
    jobs = [(n, h, g, c) for n, h in CASES for g in GAMMAS for c in range(2)]
    with multiprocessing.get_context('fork').Pool(int(os.environ.get('N_WORKERS', 12))) as pool:
        res = pool.map(job, jobs)
    save = dict(edges=edges, gammas=GAMMAS, lam_rel=LAM_REL, lam95=lam95,
                **{f'truth_{k}_label': truths[k][0] for k in truths}, **{f'truth_{k}_v': truths[k][1] for k in truths})
    for name, h in CASES:
        lab_t, v_t = truths[name]
        cut_t = lab_t[edges[:, 0]] != lab_t[edges[:, 1]]
        chi2_t = float(np.sum((Gw @ v_t[lab_t] - data[(name, h)])**2))
        for g in GAMMAS:
            score_t = chi2_t + LAM_REL*lam95*v_t.size + g*lam95*cut_t.sum()
            for c in range(2):
                r = [x for x in res if (x['name'], x['h'], x['g'], x['chain']) == (name, h, g, c)][0]
                burn = len(r['U'])//3
                labs = r['label'][burn:]
                prob = outputs.boundary_probability(labs, edges)
                vmean = np.mean([np.asarray(v)[l] for v, l in zip(r['v'][burn:], labs)], axis=0)
                aris = [outputs.ari(lab_t, l) for l in labs[::5]]
                start = (f'merging K {r["merge_K"]}, score {r["merge_score"]:.0f}; beta {r["beta"]:.2f}/edge, kappa '
                         f'{r["kappa"]:.0f}/unit; ' + ('merging start' if c == 0 else 'single-unit start'))
                print(f'{name} h {h:g}, gamma {g:g}, chain {c} ({start}): units mode {np.bincount(r["U"][burn:]).argmax()} '
                      f'(5-95% {np.percentile(r["U"][burn:], 5):.0f}-{np.percentile(r["U"][burn:], 95):.0f}; true {v_t.size}); '
                      f'L median {np.median(r["L"][burn:]):.0f} (truth {cut_t.sum()}); score best {r["score"].min():.0f}, '
                      f'median {np.median(r["score"][burn:]):.0f} (truth {score_t:.0f}); ARI median {np.median(aris):.2f}; '
                      f'true edges p>0.5 {np.mean(prob[cut_t] > 0.5):.2f}, others {np.mean(prob[~cut_t] > 0.5):.3f}; '
                      f'|VIS error| {np.mean(np.abs(vmean - v_t[lab_t])):.3f}; acceptance '
                      + ', '.join(f'{k} {a:.2f}' for k, a in r['acc'].items()) + f'; {r["seconds"]/60:.1f} min', flush=True)
                key = f'{name}_h{h:g}_g{g:g}_c{c}'
                for k_ in ('U', 'L', 'chi2', 'score'):
                    save[f'{key}_{k_}'] = r[k_]
                save[f'{key}_prob'], save[f'{key}_vmean'], save[f'{key}_label_last'] = prob, vmean, r['label'][-1]
                save[f'{key}_score_truth'] = score_t
                save[f'{key}_beta'], save[f'{key}_kappa'] = r['beta'], r['kappa']
                if c == 0:
                    save[f'{key}_merge_label'], save[f'{key}_merge_score'] = r['merge_label'], r['merge_score']
    np.savez(f'{C}/partition_synthetic{TAG}.npz', **save)
