"""
Self-consistency test: can the free-domain method recover its own real-data models?

Each truth is a real-data model of merge_compact.py (partition and values), and is inverted
with the setting that produced it. Two kinds of synthetic data:
  'noise'    truth + one noise realisation (LCS-1 - MF7 level), N_REAL realisations
  'misfit'   truth + a noise realisation scaled to the RMS of that model's actual LCS-1 residual
             (~2 sigma): the real fit leaves that much unexplained, so this is the realistic test
Noise realisations 40-49 are used, distinct from those used in earlier runs.

Run from notebooks/ (conda env pygmt17), after merge_compact.py:
    PYTHONPATH=..:$PYTHONPATH python merge_selfconsistency.py
"""
import multiprocessing
import time

import numpy as np

import merge_domains as md
import merge_compact as mc

OUT = md.OUT
TRUTHS = [((1., 0.), 0.), ((1., 10.), 0.45), ((1/3, 10.), 0.45)]    # (lambda_rel, gamma), v floor
N_REAL, FIRST_NOISE = 10, 40
cmp_ = dict(np.load(f'{OUT}/merge_compact{md.TAG}.npz'))     # into memory: workers must not share the open file
dw_real = (md.d_lcs - md.b)/md.sigma


def truth(t):
    (lr, g), vmin = TRUTHS[t]
    i = int(np.where(np.isclose(cmp_['lam_rel'], lr) & (cmp_['gamma'] == g))[0][0])
    lab = cmp_['label'][i]
    v = cmp_[f'v_{vmin}'][i]
    return lab, v


def run(args):
    t, kind, j = args
    (lam_rel, gamma), vmin = TRUTHS[t]
    lam = lam_rel*mc.LAM95
    lab_t, v_t = truth(t)
    y = md.Gw @ v_t                                            # truth, in sigma units
    if kind == 'noise':
        dw = y + md.noise[FIRST_NOISE + j]/md.sigma
    else:
        scale = np.sqrt(np.mean((dw_real - y)**2))/np.sqrt(np.mean((md.noise/md.sigma)**2))
        dw = y + scale*md.noise[FIRST_NOISE + j]/md.sigma
    best = None
    for mu in md.MUS:
        saved = mc.path(dw, mu, gamma)
        Ks = sorted(saved)
        score = [md.misfit(saved[K], dw) + lam*K + gamma*mc.cut_length(saved[K]) for K in Ks]
        lab = mc.clean_up(saved[Ks[int(np.argmin(score))]], dw, gamma)
        s = md.misfit(lab, dw) + lam*len(np.unique(lab)) + gamma*mc.cut_length(lab)
        if best is None or s < best[0]:
            best = (s, lab)
    lab = best[1]
    v, chi = mc.values(lab, dw, vmin)
    a = md.area_km2
    doms = np.unique(lab_t)
    rec = np.array([np.average(v[lab_t == k], weights=a[lab_t == k]) for k in doms])
    tru = np.array([v_t[lab_t == k][0] for k in doms])
    s_true = md.misfit(lab_t, dw) + lam*len(doms) + gamma*mc.cut_length(lab_t)
    return dict(label=lab, v=v, K=len(np.unique(lab)), ari=md.ari(lab_t, lab), chi=chi,
                rel_err=np.sqrt(np.mean(((rec - tru)/tru)**2)), score_found=best[0], score_true=s_true)


if __name__ == '__main__':
    jobs = [(t, kind, j) for t in range(len(TRUTHS)) for kind in ('noise', 'misfit') for j in range(N_REAL)]
    t0 = time.time()
    with multiprocessing.get_context('fork').Pool(md.N_WORKERS) as pool:
        res = pool.map(run, jobs)
    print(f'{len(jobs)} runs: {time.time()-t0:.0f} s')
    shape = (len(TRUTHS), 2, N_REAL)
    get = lambda k: np.array([r[k] for r in res]).reshape(shape + np.shape(res[0][k]))
    np.savez(f'{OUT}/merge_selfconsistency{md.TAG}.npz', label=get('label'), v=get('v'), K=get('K'), ari=get('ari'),
             chi=get('chi'), rel_err=get('rel_err'), score_found=get('score_found'), score_true=get('score_true'),
             true_label=np.array([truth(t)[0] for t in range(len(TRUTHS))]),
             true_v=np.array([truth(t)[1] for t in range(len(TRUTHS))]),
             truths=np.array([[lr, g, vm] for (lr, g), vm in TRUTHS]), kinds=['noise', 'misfit'])
    for t, ((lr, g), vm) in enumerate(TRUTHS):
        K_t = len(np.unique(truth(t)[0]))
        for k, kind in enumerate(('noise', 'misfit')):
            K, A, E = get('K')[t, k], get('ari')[t, k], get('rel_err')[t, k]
            worse = np.mean(get('score_true')[t, k] > get('score_found')[t, k])
            print(f'truth lambda x{lr:.2f} gamma {g:.0f} (K {K_t}), {kind:8s}: K {np.median(K):.0f} '
                  f'({K.min()}-{K.max()}), ARI median {np.median(A):.2f} ({A.min():.2f}-{A.max():.2f}), '
                  f'value error median {np.median(E):.2f}; truth scores worse than found in {worse*100:.0f}%')
