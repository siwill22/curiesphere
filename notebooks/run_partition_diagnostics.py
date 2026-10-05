"""
Why does the cell-partition sampler never change the number of units? Three diagnostics on the synthetic
cases of run_partition_synthetic.py where beta >= 1.2 per edge (same data, lam = lambda_95/3):
  1. profile   log pi of the best partition found at each U: the merging path's partition at K = U,
               polished by the sampler with U fixed (T = 1). If neighbouring U are far below the best,
               a flat U trace is the posterior, not a mixing failure.
  2. proposals every T = 1 proposal from the merging start, with its parts: cluster size, q, dU, dL,
               d log pi and the proposal term; shows what blocks new units and merges.
  3. tie test  kappa reset so that the best partitions at U0 and U0 + 1 have equal log pi (from 1);
               two parallel-tempered chains, started at each. If the moves can change U, both chains
               visit both values; if neither crosses, it is a mixing failure.
Output: partition_diagnostics.npz in the region cache.

Run from notebooks/ (conda env pygmt17), lightly:
    N_WORKERS=3 nice -n 15 env PYTHONPATH=..:$PYTHONPATH python run_partition_diagnostics.py regions/australia/config.toml
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '1')
import sys
import time
import zlib
import multiprocessing

import numpy as np

import run_partition_synthetic as R
from domains import merge, partition

CASES = [('vor6', 1., 0.003), ('vor6', 1., 0.01), ('vor6', 2., 0.01), ('real7', 1., 0.01)]
DK = 3                                   # profile over U = K0 - DK .. K0 + DK
POLISH_SWEEPS, LOG_SWEEPS, TIE_SWEEPS = 60, 300, 1200
N_WORKERS = int(os.environ.get('N_WORKERS', 3))


def setup_case(name, h, g):
    dw = R.data[(name, h)]
    lam, gamma = R.LAM_REL*R.lam95, g*R.lam95
    Pm = merge.setup(R.Gw, R.edges)
    mr = merge.run(Pm, dw, lam, gamma, [0.03, 0.1, 0.3])
    beta, kappa, h2 = partition.merging_prior(lam, gamma, mr['chi2'], R.Gw.shape[0], mr['K'])
    return dw, lam, gamma, Pm, mr, beta, kappa


def profile_and_log(args):
    name, h, g = args
    t = time.time()
    rng = np.random.default_rng(zlib.crc32(f'{name}{h}{g}'.encode()))
    dw, lam, gamma, Pm, mr, beta, kappa = setup_case(name, h, g)
    P = partition.setup(R.Gw, dw, R.edges, beta=beta, kappa=kappa, u_max=60)
    saved = merge.path(Pm, dw, mr['mu'], gamma)
    prof = {}
    for K in range(max(1, mr['K'] - DK), mr['K'] + DK + 1):
        st0 = partition.state(P, saved[K])
        out, _ = partition.sample(P, saved[K], POLISH_SWEEPS, [1.], rng, thin=1, fix_u=True)
        best = max(out, key=lambda o: o['ll']) if out else None
        if best is None or not np.isfinite(best['ll']):
            continue
        prof[K] = dict(ll_raw=st0['ll'], ll=best['ll'], chi2=best['chi2'], L=best['L'], label=best['label'])
    log = []
    partition.sample(P, mr['label'], LOG_SWEEPS, [1.], rng, thin=10**9, record=False, log=log)
    return dict(name=name, h=h, g=g, beta=beta, kappa=kappa, merge_K=mr['K'], prof=prof,
                log=np.array([(k == 'new', k == 'merge', n0, q, dU, dL, dll, ex) for k, n0, q, dU, dL, dll, ex in log]),
                log_kind=np.array([x[0] for x in log]), seconds=time.time() - t)


def tie_chain(args):
    name, h, g, kappa_tie, lab_start, chain = args
    t = time.time()
    rng = np.random.default_rng(100 + chain + zlib.crc32(f'{name}{h}{g}'.encode()) % 997)
    dw, lam, gamma, Pm, mr, beta, kappa = setup_case(name, h, g)
    P = partition.setup(R.Gw, dw, R.edges, beta=beta, kappa=kappa_tie, u_max=60)
    out, acc = partition.sample(P, lab_start, TIE_SWEEPS, R.TEMPS, rng, thin=5)
    return dict(name=name, h=h, g=g, chain=chain, U=np.array([o['U'] for o in out]),
                ll=np.array([o['ll'] for o in out]), acc=acc, seconds=time.time() - t)


if __name__ == '__main__':
    save = {}
    with multiprocessing.get_context('fork').Pool(N_WORKERS) as pool:
        res = pool.map(profile_and_log, CASES)
        ties = []
        for r in res:
            key = f'{r["name"]}_h{r["h"]:g}_g{r["g"]:g}'
            Ks = sorted(r['prof'])
            ll = np.array([r['prof'][K]['ll'] for K in Ks])
            print(f'\n{key}: beta {r["beta"]:.2f}/edge, kappa {r["kappa"]:.0f}/unit, merging K {r["merge_K"]} '
                  f'({r["seconds"]/60:.1f} min)', flush=True)
            for K, l_ in zip(Ks, ll):
                p = r['prof'][K]
                print(f'  U {K:2d}: log pi best {l_:9.1f} (raw path {p["ll_raw"]:9.1f}), relative {l_ - ll.max():8.1f}; '
                      f'chi2 {p["chi2"]:.0f}, L {p["L"]}', flush=True)
            lg, kinds = r['log'], r['log_kind']
            for kind in ('flip', 'move', 'new', 'merge'):
                m = kinds == kind
                if not m.any():
                    continue
                a = lg[m, 6] + lg[m, 7]                                  # log acceptance ratio
                best = np.argmax(a)
                print(f'  {kind:5s}: {m.sum():6d} proposals; log alpha median {np.median(a):8.1f}, best {a[best]:8.1f} '
                      f'(|V0| {lg[m][best, 2]:.0f}, q {lg[m][best, 3]:.2f}, dL {lg[m][best, 5]:.0f}, '
                      f'dlogpi {lg[m][best, 6]:.1f}, proposal term {lg[m][best, 7]:.1f}); accepted share '
                      f'{np.mean(np.log(np.random.default_rng(0).random(m.sum())) < a):.3f}', flush=True)
            save[f'{key}_profile_U'] = np.array(Ks); save[f'{key}_profile_ll'] = ll
            save[f'{key}_profile_ll_raw'] = np.array([r['prof'][K]['ll_raw'] for K in Ks])
            save[f'{key}_log'] = lg; save[f'{key}_log_kind'] = kinds
            save[f'{key}_beta'], save[f'{key}_kappa'] = r['beta'], r['kappa']
            # tie test on (U0, U0 + 1), U0 = the best U of the profile
            U0 = Ks[int(np.argmax(ll))]
            if U0 + 1 in r['prof']:
                d = r['prof'][U0 + 1]['ll'] - r['prof'][U0]['ll']           # < 0: U0 + 1 is worse
                kappa_tie = r['kappa'] + d                                  # lowers the cost of a unit by -d
                save[f'{key}_tie_U0'], save[f'{key}_tie_kappa'] = U0, kappa_tie
                print(f'  tie test: U {U0} vs {U0 + 1}, kappa {r["kappa"]:.1f} -> {kappa_tie:.1f}', flush=True)
                for c, K in enumerate((U0, U0 + 1)):
                    ties.append((r['name'], r['h'], r['g'], kappa_tie, r['prof'][K]['label'], c))
        tres = pool.map(tie_chain, ties)
    print('\ntie tests (posterior share of each U, second two thirds of the chain):', flush=True)
    for t in tres:
        key = f'{t["name"]}_h{t["h"]:g}_g{t["g"]:g}'
        U = t['U'][len(t['U'])//3:]
        share = {int(u): round(float(np.mean(U == u)), 2) for u in np.unique(U)}
        print(f'  {key}, start U {save[f"{key}_tie_U0"] + t["chain"]}: U visited {share}; changes of U along the '
              f'chain {int(np.sum(np.diff(t["U"]) != 0))}; acceptance ' +
              ', '.join(f'{k} {a:.3f}' for k, a in t['acc'].items()) + f'; {t["seconds"]/60:.1f} min', flush=True)
        save[f'{key}_tie_c{t["chain"]}_U'], save[f'{key}_tie_c{t["chain"]}_ll'] = t['U'], t['ll']
    np.savez(f'{R.C}/partition_diagnostics.npz', **save)
