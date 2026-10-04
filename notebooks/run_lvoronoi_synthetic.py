"""
Step 4 checks of the LABELLED Voronoi sampler (domains/lvoronoi.py), same truths and data as
run_voronoi_synthetic.py (plain Voronoi); the number of units U is the number of domains.
Original description of the checks (plain sampler):
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

from domains import voronoi, lvoronoi, outputs

REG = os.path.dirname(os.path.abspath(sys.argv[1]))
C = f'{REG}/cache'
TEMPS = list(np.geomspace(1., 200., 10))
N_SWEEPS, THIN, SWEEP = 2500, 5, 20
M_MAX = 60

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
    name, h, chain = args
    r = np.random.default_rng(1000*chain + (hash((name, h)) % 997))
    t = time.time()
    P = dict(P0, dw=data[(name, h)])
    out, acc = lvoronoi.sample(P, N_SWEEPS, TEMPS, r, m_max=M_MAX, m_start=6, thin=THIN, sweep=SWEEP, edges=edges)
    return dict(name=name, h=h, chain=chain, U=np.array([o['U'] for o in out]), M=np.array([o['M'] for o in out]),
                pieces=np.array([o['pieces'] for o in out]), label=np.array([o['label'] for o in out]),
                v=[o['v'] for o in out], chi2=np.array([o['chi2'] for o in out]), acc=acc, seconds=time.time()-t)


if __name__ == '__main__':
    jobs = [(n, h, c) for n, h in CASES for c in range(2)]
    with multiprocessing.get_context('fork').Pool(int(os.environ.get('N_WORKERS', 10))) as pool:
        res = pool.map(job, jobs)
    save = dict(truth_vor6_label=truths['vor6'][0], truth_vor6_v=truths['vor6'][1],
                truth_real7_label=truths['real7'][0], truth_real7_v=truths['real7'][1], edges=edges)
    for name, h in CASES:
        lab_t, v_t = truths[name]
        cut_t = lab_t[edges[:, 0]] != lab_t[edges[:, 1]]
        chi2_t = float(np.sum((Gw @ v_t[lab_t] - data[(name, h)])**2))
        for c in range(2):
            r = [x for x in res if x['name'] == name and x['h'] == h and x['chain'] == c][0]
            burn = len(r['U'])//3
            Us, Ms, pcs, labs = r['U'][burn:], r['M'][burn:], r['pieces'][burn:], r['label'][burn:]
            prob = outputs.boundary_probability(labs, edges)
            aris = [outputs.ari(lab_t, l) for l in labs[::5]]
            h_est = np.sqrt(np.median(r['chi2'][burn:]/(Gw.shape[0] - Us)))
            vmean = np.mean([np.asarray(v)[l] for v, l in zip(r['v'][burn:], labs)], axis=0)
            print(f'{name}, h {h:g}, chain {c}: units mode {np.bincount(Us).argmax()} (true {v_t.size}), '
                  f'5-95% {np.percentile(Us, 5):.0f}-{np.percentile(Us, 95):.0f}; seeds median {np.median(Ms):.0f}; '
                  f'connected pieces median {np.median(pcs):.0f}; h estimate {h_est:.2f}; ARI median {np.median(aris):.2f}; '
                  f'true boundaries p > 0.5: {np.mean(prob[cut_t] > 0.5):.2f}, other edges p > 0.5: '
                  f'{np.mean(prob[~cut_t] > 0.5):.3f}; |VIS error| {np.mean(np.abs(vmean - v_t[lab_t])):.3f}; '
                  f'best chi2 {r["chi2"].min():.0f} (truth {chi2_t:.0f}); acceptance '
                  + ', '.join(f'{k} {a:.2f}' for k, a in r['acc'].items()) + f'; {r["seconds"]/60:.1f} min', flush=True)
            key = f'{name}_h{h:g}_c{c}'
            for k_ in ('U', 'M', 'pieces', 'chi2'):
                save[f'{key}_{k_}'] = r[k_]
            save[f'{key}_prob'] = prob; save[f'{key}_vmean'] = vmean; save[f'{key}_label_last'] = r['label'][-1]
    np.savez(f'{C}/lvoronoi_synthetic.npz', **save)
