"""
Synthetic-block test of the compact free-domain settings (merge_compact.py).

Same truths and noise as merge_domains.py (inversion/merge_synthetic.npz: 3 random 8-block
Voronoi partitions, block values lognormal around 0.6 SI.km, noise realisation = case index),
noise levels 0, 0.25, 0.5, 1 x the LCS-1 - MF7 estimate. Values are refitted with v >= 0 (no
floor: some true block values are below the real-data floor).

Run from notebooks/ (conda env pygmt17), after merge_domains.py:
    PYTHONPATH=..:$PYTHONPATH python merge_compact_synthetic.py
"""
import multiprocessing
import time

import numpy as np

import merge_domains as md
import merge_compact as mc

OUT = md.OUT
SETTINGS = [(1., 0.), (1., 10.), (1/3, 10.), (1/3, 20.), (1/3, 30.)]
syn = np.load(f'{OUT}/merge_synthetic.npz')
TRUE_LABEL, TRUE_V, NOISE_LEVELS = syn['true_label'], syn['true_v'], list(syn['noise_levels'])


def run(args):
    (lam_rel, gamma), case, n = args
    lam = lam_rel*mc.LAM95
    tb, vb = TRUE_LABEL[case], TRUE_V[case]
    dw = (md.G @ vb[tb] + NOISE_LEVELS[n]*md.noise[case])/md.sigma
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
    v, chi = mc.values(lab, dw, 0.)
    chi_true = np.sqrt(md.misfit(tb, dw)/len(dw))
    # block-value error: recovered values averaged over each true block, against the true value
    area = md.area_km2
    rec = np.array([np.average(v[tb == k], weights=area[tb == k]) for k in range(len(vb))])
    return dict(label=lab, v=v, K=len(np.unique(lab)), L=mc.cut_length(lab), ari=md.ari(tb, lab),
                chi=chi, chi_true=chi_true, rel_err=np.sqrt(np.mean(((rec - vb)/vb)**2)))


if __name__ == '__main__':
    jobs = [(s, c, n) for s in SETTINGS for c in range(len(TRUE_V)) for n in range(len(NOISE_LEVELS))]
    t = time.time()
    with multiprocessing.get_context('fork').Pool(md.N_WORKERS) as pool:
        res = pool.map(run, jobs)
    print(f'{len(jobs)} runs: {time.time()-t:.0f} s')
    shape = (len(SETTINGS), len(TRUE_V), len(NOISE_LEVELS))
    get = lambda k: np.array([r[k] for r in res]).reshape(shape + np.shape(res[0][k]))
    np.savez(f'{OUT}/merge_compact_synthetic.npz', settings=np.array(SETTINGS), noise_levels=NOISE_LEVELS,
             label=get('label'), v=get('v'), K=get('K'), L=get('L'), ari=get('ari'), chi=get('chi'),
             chi_true=get('chi_true'), rel_err=get('rel_err'), true_label=TRUE_LABEL, true_v=TRUE_V,
             true_L=[mc.cut_length(tb) for tb in TRUE_LABEL])
    print('true boundary length (cut edges): ' + ', '.join(str(mc.cut_length(tb)) for tb in TRUE_LABEL))
    for i, (lr, g) in enumerate(SETTINGS):
        print(f'lambda x{lr:.2f}, gamma {g:.0f}:')
        for c in range(len(TRUE_V)):
            print(f'  case {c}: ' + '; '.join(
                f'x{f:g}: K {get("K")[i, c, n]}, ARI {get("ari")[i, c, n]:.2f}, L {get("L")[i, c, n]}, '
                f'err {get("rel_err")[i, c, n]:.2f}' for n, f in enumerate(NOISE_LEVELS)))
