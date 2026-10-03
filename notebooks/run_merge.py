"""
Method (a), merging, on whitened data (docs/domain-inversion-framework.md, step 3), in two modes:
  'refine'  path and choice of K on the coarse (NSIDE/2, ~200 km) basis, clean-up on the fine basis
  'fine'    everything on the fine (~100 km) basis
Steps, for each mode:
  1. lambda_95 from N_NULL null data sets (uniform V0 + a whitened error realisation, i.e. a
     standard normal vector), gamma = 0, as in merge_domains.py
  2. check: N_NULL fresh null data sets through the full method should give K = 1 in ~95%
  3. real data, settings lambda = lam_rel * lambda_95 and gamma = g_rel * lambda_95 (gamma is per
     fine edge, ~100 km); g_rel 0.01 is about the old gamma 10 (10/1313 = 0.008)
  4. parametric bootstrap for BOOT_SETTINGS: N_BOOT data sets = the setting's fitted model + a
     whitened error realisation; boundary probability per fine edge
Errors: the region's error model (error_model.py) with its reference amplitudes.

Run from notebooks/ (conda env pygmt17):
    PYTHONPATH=..:$PYTHONPATH python run_merge.py regions/australia/config.toml
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '1')
import sys
import time
import tomllib
import multiprocessing

import numpy as np

from domains import merge, outputs

cfg_path = sys.argv[1]
with open(cfg_path, 'rb') as f:
    cfg = tomllib.load(f)
REG = os.path.dirname(os.path.abspath(cfg_path))
CACHE = f'{REG}/cache'
MUS = [0.03, 0.1, 0.3]
N_NULL, N_BOOT = 20, 30
V0 = 0.6
SETTINGS = [(lr, g) for lr in (1., 1/3) for g in (0., 0.01, 0.03, 0.1)]
BOOT_SETTINGS = [(1., 0.), (1., 0.01)]
N_WORKERS = int(os.environ.get('N_WORKERS', 8))
SEED = 20261003

# whitened problem
E = np.load(f'{CACHE}/errors.npz')
D = float(E['sn2']) + float(E['s2'])*E['lam']
T = E['T']/np.sqrt(D)[:, None]
G = np.load(f'{CACHE}/G.npz')['G']
z = np.load(f'{CACHE}/data.npz')
cells = np.load(f'{CACHE}/cells.npz')
Gw = T @ G
dw_real = T @ (z['d_lcs'] - z['b'])
parent, edges = cells['parent'], cells['edges']
P = {'fine': merge.setup(Gw, edges), 'coarse': merge.coarse_problem(Gw, edges, parent)}
print(f'{Gw.shape[1]} fine cells, {P["coarse"]["n"]} coarse cells, {Gw.shape[0]} whitened dimensions; '
      f'error amplitudes sigma_n {np.sqrt(E["sn2"]):.1f} nT, s {np.sqrt(E["s2"]):.1f} nT', flush=True)


def run(mode, dw, lam, gamma):
    if mode == 'fine':
        return merge.run(P['fine'], dw, lam, gamma, MUS)
    return merge.run(P['coarse'], dw, lam, gamma, MUS, refine=(P['fine'], parent))


def null_paths(args):
    mode, dw = args
    Q = P['fine'] if mode == 'fine' else P['coarse']
    out = {}
    for mu in MUS:
        Ks, chi2, L, _ = merge.path_scores(Q, merge.path(Q, dw, mu, 0.), dw, 0., 0.)
        out[mu] = (Ks, chi2, L)
    return out


def task(args):
    mode, dw, lam, gamma = args
    t = time.time()
    r = run(mode, dw, lam, gamma)
    r['seconds'] = time.time() - t
    return r


if __name__ == '__main__':
    rng = np.random.default_rng(SEED)
    null_dw = [Gw @ (V0*np.ones(Gw.shape[1])) + rng.standard_normal(Gw.shape[0]) for _ in range(2*N_NULL)]
    out = {}
    pool = multiprocessing.get_context('fork').Pool(N_WORKERS)
    for mode in ('refine', 'fine'):
        t = time.time()
        curves = pool.map(null_paths, [(mode, dw) for dw in null_dw[:N_NULL]])
        lam95 = merge.calibrate_lambda(curves)
        print(f'[{mode}] lambda_95 = {lam95:.1f} ({time.time()-t:.0f} s)', flush=True)
        t = time.time()
        chk = pool.map(task, [(mode, dw, lam95, 0.) for dw in null_dw[N_NULL:]])
        Kc = np.array([r['K'] for r in chk])
        print(f'[{mode}] fresh null data: K = 1 in {np.mean(Kc == 1):.0%} (K: {sorted(Kc)}) ({time.time()-t:.0f} s)', flush=True)
        t = time.time()
        real = pool.map(task, [(mode, dw_real, lr*lam95, g*lam95) for lr, g in SETTINGS])
        print(f'[{mode}] real data, {len(SETTINGS)} settings ({time.time()-t:.0f} s):', flush=True)
        for (lr, g), r in zip(SETTINGS, real):
            print(f'    lambda x{lr:.2f}, gamma {g:.2f} lambda_95: K {r["K"]:2d}, chi2 {r["chi2"]:.1f}, '
                  f'L {r["L"]:.0f}, mu {r["mu"]}, {r["seconds"]:.0f} s', flush=True)
        boot = {}
        for lr, g in BOOT_SETTINGS:
            r0 = real[SETTINGS.index((lr, g))]
            pred = Gw @ r0['v']
            t = time.time()
            bs = pool.map(task, [(mode, pred + rng.standard_normal(len(pred)), lr*lam95, g*lam95) for _ in range(N_BOOT)])
            labs = np.array([b['label'] for b in bs])
            boot[(lr, g)] = dict(label=labs, K=np.array([b['K'] for b in bs]),
                                 prob=outputs.boundary_probability(labs, edges))
            print(f'[{mode}] bootstrap lambda x{lr:.2f}, gamma {g:.2f}: K median {np.median(boot[(lr, g)]["K"]):.0f} '
                  f'(range {boot[(lr, g)]["K"].min()}-{boot[(lr, g)]["K"].max()}) ({time.time()-t:.0f} s)', flush=True)
        out[mode] = dict(lam95=lam95, null_K=Kc, real=real, boot=boot, null_curves=curves)
    pool.close()

    save = dict(settings=np.array(SETTINGS), boot_settings=np.array(BOOT_SETTINGS), mus=MUS, edges=edges)
    for mode, o in out.items():
        save[f'{mode}_lam95'] = o['lam95']
        save[f'{mode}_null_K'] = o['null_K']
        for key in ('label', 'v', 'K', 'chi2', 'L', 'mu', 'seconds'):
            save[f'{mode}_{key}'] = np.array([r[key] for r in o['real']])
        # trade-off: chi2 against K on the path of the chosen restart, per setting
        save[f'{mode}_path_K'] = np.array([r['curves'][r['mu']][0] for r in o['real']], dtype=object)
        save[f'{mode}_path_chi2'] = np.array([r['curves'][r['mu']][1] for r in o['real']], dtype=object)
        for i, s in enumerate(BOOT_SETTINGS):
            save[f'{mode}_boot{i}_prob'] = o['boot'][s]['prob']
            save[f'{mode}_boot{i}_K'] = o['boot'][s]['K']
            save[f'{mode}_boot{i}_label'] = o['boot'][s]['label']
    np.savez(f'{CACHE}/merge.npz', **save)

    # comparison of the two modes
    print('\nrefine against fine, real data:')
    for i, (lr, g) in enumerate(SETTINGS):
        a, b = out['refine']['real'][i], out['fine']['real'][i]
        print(f'    lambda x{lr:.2f}, gamma {g:.2f}: K {a["K"]} / {b["K"]}, chi2 {a["chi2"]:.1f} / {b["chi2"]:.1f}, '
              f'ARI {outputs.ari(a["label"], b["label"]):.2f}, time {a["seconds"]:.0f} / {b["seconds"]:.0f} s')
    for i, s in enumerate(BOOT_SETTINGS):
        pa, pb = out['refine']['boot'][s]['prob'], out['fine']['boot'][s]['prob']
        print(f'    bootstrap {s}: boundary probability, correlation {np.corrcoef(pa, pb)[0, 1]:.2f}, '
              f'mean |difference| {np.mean(np.abs(pa - pb)):.3f}')
