"""
Step 4 checks of the Voronoi sampler and annealing (docs/domain-inversion-framework.md), Australia:
  prior     data switched off (constant likelihood): P(K) must be uniform on 1..K_MAX_PRIOR
  vor6      truth = 6 Voronoi domains (random seeds), values lognormal around 0.6 SI.km
  real7     truth = the A2 merging model on the real data (lambda x1, gamma 0.01 lambda_95): 7
            domains that are not Voronoi regions
Synthetic data = Gw v_true + h_true * standard normal in the compressed point-weighted space (A2;
units of sigma), i.e. exactly the likelihood's assumptions: this tests the sampler, not the noise model.
Each case runs in two independent chains (different random starts); h is integrated out.
Annealing (method b) for vor6, K = 2..10.

Run from notebooks/ (conda env pygmt17):
    PYTHONPATH=..:$PYTHONPATH python run_voronoi_synthetic.py regions/australia/config.toml
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '1')
import sys
import time
import multiprocessing

import numpy as np

from domains import voronoi, outputs

REG = os.path.dirname(os.path.abspath(sys.argv[1]))
C = f'{REG}/cache'
TEMPS = list(np.geomspace(1., 200., 10))
N_SWEEPS, THIN, SWEEP = 2500, 5, 20
K_MAX, K_MAX_PRIOR = 40, 10

E = np.load(f'{C}/errors.npz')
z = np.load(f'{C}/data.npz')
G = np.load(f'{C}/G.npz')['G']
cells = np.load(f'{C}/cells.npz')
sigma = float(np.sqrt(np.mean((z['d_lcs'] - z['d_mf7'])**2)))
T = np.linalg.svd(E['T'].T, full_matrices=False)[0].T/sigma
Gw = T @ G
P0 = voronoi.setup(Gw, cells['clon'], cells['clat'], T @ (z['d_lcs'] - z['b']))
edges = cells['edges']

rng = np.random.default_rng(20261004)
truths = {}
seeds6 = rng.choice(P0['n'], 6, replace=False)
truths['vor6'] = (voronoi.assign(P0, seeds6), 0.6*np.exp(0.5*rng.standard_normal(6)))
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
    kind, name, h, chain = args
    r = np.random.default_rng(1000*chain + (hash((kind, name, h)) % 997))
    t = time.time()
    if kind == 'prior':
        orig = voronoi.evaluate
        voronoi.evaluate = lambda P, s, h=None: (0., 0., np.zeros(len(s)), voronoi.assign(P, s))
        out, acc = voronoi.sample(P0, 4000, [1.], r, k_max=K_MAX_PRIOR, k_start=3, thin=2)
        voronoi.evaluate = orig
    elif kind == 'sample':
        P = dict(P0, dw=data[(name, h)])
        out, acc = voronoi.sample(P, N_SWEEPS, TEMPS, r, k_max=K_MAX, k_start=3, thin=THIN, sweep=SWEEP)
    else:                                                       # anneal, K = chain
        P = dict(P0, dw=data[(name, h)])
        best = voronoi.anneal(P, chain, 20000, r, restarts=2)
        return dict(kind=kind, name=name, h=h, K=chain, chi2=best[0], label=best[2], seconds=time.time()-t)
    return dict(kind=kind, name=name, h=h, chain=chain, K=np.array([o['K'] for o in out]),
                label=np.array([o['label'] for o in out]), v=[o['v'] for o in out],
                chi2=np.array([o['chi2'] for o in out]), acc=acc, seconds=time.time()-t)


if __name__ == '__main__':
    jobs = [('prior', 'prior', 0., c) for c in range(2)]
    jobs += [('sample', n, h, c) for n, h in CASES for c in range(2)]
    jobs += [('anneal', 'vor6', 1., K) for K in range(2, 11)]
    with multiprocessing.get_context('fork').Pool(int(os.environ.get('N_WORKERS', 10))) as pool:
        res = pool.map(job, jobs)
    save = dict(truth_vor6_label=truths['vor6'][0], truth_vor6_v=truths['vor6'][1],
                truth_real7_label=truths['real7'][0], truth_real7_v=truths['real7'][1], edges=edges,
                k_max=K_MAX, k_max_prior=K_MAX_PRIOR)
    for r in res:
        if r['kind'] == 'prior':
            Ks = r['K'][len(r['K'])//4:]
            hist = np.bincount(Ks, minlength=K_MAX_PRIOR+1)[1:]/len(Ks)
            print(f'prior chain {r["chain"]}: P(K=1..{K_MAX_PRIOR}) = {np.round(hist, 3)} (uniform: {1/K_MAX_PRIOR:.3f}); '
                  f'acceptance {r["acc"]}', flush=True)
            save[f'prior_{r["chain"]}_K'] = r['K']
    for name, h in CASES:
        lab_t, v_t = truths[name]
        cut_t = lab_t[edges[:, 0]] != lab_t[edges[:, 1]]
        chi2_t = float(np.sum((Gw @ v_t[lab_t] - data[(name, h)])**2))
        for c in range(2):
            r = [x for x in res if x['kind'] == 'sample' and x['name'] == name and x['h'] == h and x['chain'] == c][0]
            burn = len(r['K'])//3
            Ks, labs = r['K'][burn:], r['label'][burn:]
            prob = outputs.boundary_probability(labs, edges)
            aris = [outputs.ari(lab_t, l) for l in labs[::5]]
            h_est = np.sqrt(np.median(r['chi2'][burn:]/(Gw.shape[0] - Ks)))
            vmean = np.mean([np.asarray(v)[l] for v, l in zip(r['v'][burn:], labs)], axis=0)
            print(f'{name}, h {h:g}, chain {c}: K mode {np.bincount(Ks).argmax()} (true {v_t.size}), '
                  f'5-95% {np.percentile(Ks, 5):.0f}-{np.percentile(Ks, 95):.0f}; h estimate {h_est:.2f}; '
                  f'ARI median {np.median(aris):.2f}; true boundaries with p > 0.5: {np.mean(prob[cut_t] > 0.5):.2f}, '
                  f'other edges with p > 0.5: {np.mean(prob[~cut_t] > 0.5):.3f}; mean |VIS error| '
                  f'{np.mean(np.abs(vmean - v_t[lab_t])):.3f} SI.km; best chi2 {r["chi2"].min():.0f} (truth {chi2_t:.0f}); acceptance '
                  + ', '.join(f'{k} {a:.2f}' for k, a in r['acc'].items()) + f'; {r["seconds"]/60:.1f} min', flush=True)
            key = f'{name}_h{h:g}_c{c}'
            save[f'{key}_K'] = r['K']; save[f'{key}_prob'] = prob; save[f'{key}_vmean'] = vmean
            save[f'{key}_chi2'] = r['chi2']; save[f'{key}_label_last'] = r['label'][-1]
    lab_t, v_t = truths['vor6']
    chi2_true = float(np.sum((Gw @ v_t[lab_t] - data[('vor6', 1.)])**2))
    ann = sorted([x for x in res if x['kind'] == 'anneal'], key=lambda x: x['K'])
    print(f'annealing, vor6 h 1 (true K 6, chi2 of the truth {chi2_true:.0f}): ' +
          ', '.join(f'K {a["K"]}: chi2 {a["chi2"]:.0f} (ARI {outputs.ari(lab_t, a["label"]):.2f})' for a in ann))
    save['anneal_K'] = [a['K'] for a in ann]; save['anneal_chi2'] = [a['chi2'] for a in ann]
    save['anneal_label'] = np.array([a['label'] for a in ann]); save['anneal_chi2_true'] = chi2_true
    np.savez(f'{C}/voronoi_synthetic.npz', **save)
