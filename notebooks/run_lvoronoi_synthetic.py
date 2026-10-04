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

Boundary-length penalty: env GAMMA = gamma relative to lambda_95 of the A2 merging (default 0); the
output is lvoronoi_synthetic{TAG}.npz (env TAG, default '').
Search (env GUIDED=1, default): unit births guided by the residual (lvoronoi.setup_births); chain 0
starts from the merging result on the same synthetic data (lambda_95/3, the same gamma; an
over-split start, since merges are easy moves and births hard), chain 1 from a random start.
A prior test (likelihood switched off, guided births on) checks P(M) and P(U | M) are uniform.

Run from notebooks/ (conda env pygmt17):
    PYTHONPATH=..:$PYTHONPATH python run_voronoi_synthetic.py regions/australia/config.toml
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '1')
import sys
import time
import zlib
import multiprocessing

import numpy as np

from domains import voronoi, lvoronoi, merge, outputs

REG = os.path.dirname(os.path.abspath(sys.argv[1]))
C = f'{REG}/cache'
TEMPS = list(np.geomspace(1., 200., 10))
N_SWEEPS, THIN, SWEEP = 2500, 5, 20
M_MAX = 60
GAMMA_REL = float(os.environ.get('GAMMA', 0.))
TAG = os.environ.get('TAG', '')
GUIDED = os.environ.get('GUIDED', '1') == '1'
M_MAX_PRIOR = 8

E = np.load(f'{C}/errors.npz')
z = np.load(f'{C}/data.npz')
G = np.load(f'{C}/G.npz')['G']
cells = np.load(f'{C}/cells.npz')
sigma = float(np.sqrt(np.mean((z['d_lcs'] - z['d_mf7'])**2)))
T = np.linalg.svd(E['T'].T, full_matrices=False)[0].T/sigma
Gw = T @ G
P0 = voronoi.setup(Gw, cells['clon'], cells['clat'], T @ (z['d_lcs'] - z['b']))
edges = cells['edges']
lam95 = float(np.load(f'{C}/merge_points_sweep.npz')['fine_lam95'])
P0['edges'], P0['gamma'] = edges, GAMMA_REL*lam95
if GUIDED:
    lvoronoi.setup_births(P0)
print(f'gamma {GAMMA_REL:g} lambda_95 = {P0["gamma"]:.2f} per edge; guided {GUIDED}', flush=True)

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
    r = np.random.default_rng(1000*chain + zlib.crc32(f'{name}{h}'.encode()) % 997)
    t = time.time()
    if name == 'prior':
        P = dict(P0, dw=data[('vor6', 1.)])
        flat = lambda P, s, u, h=None: (0., 0., np.zeros(u.max() + 1), u[voronoi.assign(P, s)])
        out, acc = lvoronoi.sample(P, 6000, [1.], r, m_max=M_MAX_PRIOR, m_start=3, thin=2, evaluate_fn=flat)
        return dict(name=name, chain=chain, M=np.array([o['M'] for o in out]), U=np.array([o['U'] for o in out]),
                    acc=acc)
    P = dict(P0, dw=data[(name, h)])
    start, info = None, {}
    if chain == 0 and GUIDED:
        mr = merge.run(merge.setup(Gw, edges), P['dw'], lam95/3, P0['gamma'], [0.03, 0.1, 0.3])
        start = lvoronoi.seeds_from_partition(P, mr['label'], max_seeds=M_MAX - 10)
        ll, chi2_s, _, lab_s = lvoronoi.evaluate(P, *start)
        info = dict(merge_K=mr['K'], merge_chi2=mr['chi2'], merge_L=mr['L'], start_M=len(start[0]),
                    start_chi2=chi2_s, start_L=lvoronoi.boundary_length(edges, lab_s), start_label=lab_s,
                    merge_label=mr['label'])
    out, acc = lvoronoi.sample(P, N_SWEEPS, TEMPS, r, m_max=M_MAX, m_start=6, thin=THIN, sweep=SWEEP, edges=edges,
                               start=start)
    return dict(name=name, h=h, chain=chain, U=np.array([o['U'] for o in out]), M=np.array([o['M'] for o in out]),
                pieces=np.array([o['pieces'] for o in out]), L=np.array([o['L'] for o in out]),
                label=np.array([o['label'] for o in out]), v=[o['v'] for o in out],
                chi2=np.array([o['chi2'] for o in out]), acc=acc, seconds=time.time()-t, **info)


if __name__ == '__main__':
    jobs = [(n, h, c) for n, h in CASES for c in range(2)] + [('prior', 0., c) for c in range(2)]
    with multiprocessing.get_context('fork').Pool(int(os.environ.get('N_WORKERS', 10))) as pool:
        res = pool.map(job, jobs)
    save = dict(truth_vor6_label=truths['vor6'][0], truth_vor6_v=truths['vor6'][1],
                truth_real7_label=truths['real7'][0], truth_real7_v=truths['real7'][1], edges=edges, gamma=P0['gamma'], gamma_rel=GAMMA_REL)
    for r in [x for x in res if x['name'] == 'prior']:
        Ms, Us = r['M'][len(r['M'])//4:], r['U'][len(r['U'])//4:]
        pM = np.bincount(Ms, minlength=M_MAX_PRIOR + 1)[1:]/len(Ms)
        pU = [np.round(np.bincount(Us[Ms == m], minlength=m + 1)[1:]/max((Ms == m).sum(), 1), 2) for m in (2, 4, 8)]
        print(f'prior chain {r["chain"]}: P(M = 1..{M_MAX_PRIOR}) = {np.round(pM, 3)} (uniform {1/M_MAX_PRIOR:.3f}); '
              f'P(U | M = 2, 4, 8) = {pU}; acceptance ' + ', '.join(f'{k} {a:.2f}' for k, a in r['acc'].items()), flush=True)
        save[f'prior_{r["chain"]}_M'], save[f'prior_{r["chain"]}_U'] = r['M'], r['U']
    for name, h in CASES:
        lab_t, v_t = truths[name]
        cut_t = lab_t[edges[:, 0]] != lab_t[edges[:, 1]]
        chi2_t = float(np.sum((Gw @ v_t[lab_t] - data[(name, h)])**2))
        for c in range(2):
            r = [x for x in res if x['name'] == name and x['h'] == h and x['chain'] == c][0]
            if 'merge_K' in r:
                print(f'{name}, h {h:g}, chain {c} start: merging K {r["merge_K"]}, chi2 {r["merge_chi2"]:.0f}, '
                      f'L {r["merge_L"]}; as Voronoi: {r["start_M"]} seeds, chi2 {r["start_chi2"]:.0f}, L {r["start_L"]}',
                      flush=True)
                for k_ in ('merge_K', 'merge_chi2', 'merge_L', 'start_M', 'start_chi2', 'start_L', 'start_label', 'merge_label'):
                    save[f'{name}_h{h:g}_c{c}_{k_}'] = r[k_]
            pen = r['chi2'] + P0['gamma']*r['L']
            burn = len(r['U'])//3
            Us, Ms, pcs, labs = r['U'][burn:], r['M'][burn:], r['pieces'][burn:], r['label'][burn:]
            Ls = r['L'][burn:]
            prob = outputs.boundary_probability(labs, edges)
            aris = [outputs.ari(lab_t, l) for l in labs[::5]]
            h_est = np.sqrt(np.median(r['chi2'][burn:]/(Gw.shape[0] - Us)))
            vmean = np.mean([np.asarray(v)[l] for v, l in zip(r['v'][burn:], labs)], axis=0)
            print(f'{name}, h {h:g}, chain {c}: units mode {np.bincount(Us).argmax()} (true {v_t.size}), '
                  f'5-95% {np.percentile(Us, 5):.0f}-{np.percentile(Us, 95):.0f}; seeds median {np.median(Ms):.0f}; '
                  f'connected pieces median {np.median(pcs):.0f}; L median {np.median(Ls):.0f} (truth {int(cut_t.sum())}); h estimate {h_est:.2f}; ARI median {np.median(aris):.2f}; '
                  f'true boundaries p > 0.5: {np.mean(prob[cut_t] > 0.5):.2f}, other edges p > 0.5: '
                  f'{np.mean(prob[~cut_t] > 0.5):.3f}; |VIS error| {np.mean(np.abs(vmean - v_t[lab_t])):.3f}; '
                  f'best chi2 {r["chi2"].min():.0f} (truth {chi2_t:.0f}); best chi2 + gamma L {pen.min():.0f} '
                  f'(truth {chi2_t + P0["gamma"]*cut_t.sum():.0f}); acceptance '
                  + ', '.join(f'{k} {a:.2f}' for k, a in r['acc'].items()) + f'; {r["seconds"]/60:.1f} min', flush=True)
            key = f'{name}_h{h:g}_c{c}'
            for k_ in ('U', 'M', 'pieces', 'L', 'chi2'):
                save[f'{key}_{k_}'] = r[k_]
            save[f'{key}_prob'] = prob; save[f'{key}_vmean'] = vmean; save[f'{key}_label_last'] = r['label'][-1]
    np.savez(f'{C}/lvoronoi_synthetic{TAG}.npz', **save)
