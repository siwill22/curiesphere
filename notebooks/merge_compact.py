"""
Free domains with a boundary-length penalty (more compact, less branching) and a lower bound on VIS.

Builds on merge_domains.py (same cells, background, data, noise, merge path and clean-up), with
two changes:
  - objective chi2 + lambda K + gamma L, where L is the number of cell-graph edges that lie on a
    domain boundary (each ~100 km). In the merge path, the priority of merging two domains is
    their constraint cost minus gamma times their shared edges, so long shared boundaries (and
    the branches that make them) are merged first. The number of domains is chosen with
    chi2 + lambda K + gamma L, and the boundary clean-up accepts a cell move if it lowers
    chi2 + gamma L.
  - final values refitted with v >= V_MIN, where V_MIN is a floor taken from H&M's eastern
    (Phanerozoic) average.
Settings are a small grid of (lambda, gamma) on the real data; lambda is given relative to the
null-calibrated lambda_95 of merge_domains.py.

Run from notebooks/ (conda env pygmt17), after merge_domains.py:
    PYTHONPATH=..:$PYTHONPATH python merge_compact.py
"""
import multiprocessing
import time

import numpy as np
from scipy.optimize import lsq_linear

import merge_domains as md
from merge_domains import n_cell, cell_edges, Gw, GtG, Lap, scale, ridge, neighbours, misfit, indicator

OUT = md.OUT
LAM95 = float(np.load(f'{OUT}/merge_real.npz')['lam'])
LAM_REL = [1., 1/3]
GAMMAS = [0., 10., 30., 100.]
V_MINS = [0., 0.33, 0.45]          # SI.km: none, H&M eastern 10th percentile, H&M eastern mean
K_SAVE = 100


def cut_length(label):
    return int(np.sum(label[cell_edges[:, 0]] != label[cell_edges[:, 1]]))


def path(dw, mu, gamma):
    """merge path with priority delta - gamma * shared edges; returns {K: label}"""
    H = GtG + mu*scale*Lap + ridge*np.eye(n_cell)
    C = np.linalg.inv(H)
    g = Gw.T @ dw
    v = C @ g
    label = np.arange(n_cell)
    ea, eb = cell_edges[:, 0].copy(), cell_edges[:, 1].copy()
    w = np.ones(len(ea))                        # number of cell edges shared by the two domains
    saved = {}
    for step in range(n_cell - 1):
        K = n_cell - step
        if K <= K_SAVE:
            saved[K] = label.copy()
        den = C[ea, ea] + C[eb, eb] - 2*C[ea, eb]
        i = np.argmin((v[ea] - v[eb])**2/np.maximum(den, 1e-300) - gamma*w)
        a, bb = ea[i], eb[i]
        Cc = C[:, a] - C[:, bb]
        den_i = Cc[a] - Cc[bb]
        v -= Cc*((v[a] - v[bb])/den_i)
        C -= np.multiply.outer(Cc, Cc/den_i)
        label[label == bb] = a
        ea, eb = np.where(ea == bb, a, ea), np.where(eb == bb, a, eb)
        lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
        keep = lo != hi
        e, inv = np.unique(np.stack([lo[keep], hi[keep]], 1), axis=0, return_inverse=True)
        w = np.bincount(inv.ravel(), weights=w[keep])
        ea, eb = e[:, 0], e[:, 1]
        if (step + 1) % md.RESYNC == 0 and K - 1 > 1:
            A, idx = indicator(label)
            CR = np.linalg.inv(A.T @ H @ A)
            v, C = (CR @ (A.T @ g))[idx], CR[np.ix_(idx, idx)]
    saved[1] = label.copy()
    return saved


def clean_up(label, dw, gamma, max_pass=10):
    """as merge_domains.clean_up, accepting a move if chi2 + gamma L decreases"""
    label = label.copy()
    while True:
        regions = md.region_lists(label)
        small = [r for r, m in regions.items() if len(m) < md.MIN_CELLS]
        if not small:
            break
        r = small[0]
        nbrs = {label[n] for c in regions[r] for n in neighbours[c]} - {r}
        best = min(nbrs, key=lambda q: misfit(np.where(label == r, q, label), dw)
                   + gamma*cut_length(np.where(label == r, q, label)))
        label[label == r] = best
    reps = list(np.unique(label))
    col = {r: i for i, r in enumerate(reps)}
    GR = np.stack([Gw[:, label == r].sum(1) for r in reps], 1)

    def ss(GR):
        vR, *_ = np.linalg.lstsq(GR, dw, rcond=None)
        return np.sum((GR @ vR - dw)**2)
    current = ss(GR)
    for _ in range(max_pass):
        moved = 0
        regions = md.region_lists(label)
        for c in range(n_cell):
            r = label[c]
            targets = {label[n] for n in neighbours[c]} - {r}
            if not targets or len(regions[r]) <= md.MIN_CELLS:
                continue
            if not md.connected([m for m in regions[r] if m != c]):
                continue
            n_r = sum(label[n] == r for n in neighbours[c])
            for q in targets:
                dL = n_r - sum(label[n] == q for n in neighbours[c])     # change in cut edges
                GR[:, col[r]] -= Gw[:, c]
                GR[:, col[q]] += Gw[:, c]
                f = ss(GR)
                if f + gamma*dL < current - 1e-9:
                    label[c], current = q, f
                    regions[r].remove(c)
                    regions[q].append(c)
                    moved += 1
                    break
                GR[:, col[r]] += Gw[:, c]
                GR[:, col[q]] -= Gw[:, c]
        if moved == 0:
            break
    return label


def values(label, dw, v_min):
    A, idx = indicator(label)
    GR = Gw @ A
    vR = lsq_linear(GR, dw, bounds=(v_min, np.inf)).x
    return vR[idx], np.sqrt(np.mean((GR @ vR - dw)**2))


def run(args):
    lam_rel, gamma = args
    lam = lam_rel*LAM95
    dw = (md.d_lcs - md.b)/md.sigma
    best = None
    for mu in md.MUS:
        saved = path(dw, mu, gamma)
        Ks = sorted(saved)
        score = [misfit(saved[K], dw) + lam*K + gamma*cut_length(saved[K]) for K in Ks]
        lab = clean_up(saved[Ks[int(np.argmin(score))]], dw, gamma)
        s = misfit(lab, dw) + lam*len(np.unique(lab)) + gamma*cut_length(lab)
        if best is None or s < best[0]:
            best = (s, lab, mu)
    _, lab, mu = best
    out = dict(lam_rel=lam_rel, gamma=gamma, mu=mu, label=lab, K=len(np.unique(lab)), L=cut_length(lab))
    for v_min in V_MINS:
        out[f'v_{v_min}'], out[f'chi_{v_min}'] = values(lab, dw, v_min)
    return out


if __name__ == '__main__':
    t = time.time()
    settings = [(lr, g) for lr in LAM_REL for g in GAMMAS]
    with multiprocessing.get_context('fork').Pool(md.N_WORKERS) as pool:
        res = pool.map(run, settings)
    print(f'{len(settings)} settings: {time.time()-t:.0f} s')
    # the current model (merge_domains.py) with the lower bounds, for comparison
    cur = np.load(f'{OUT}/merge_real.npz')['label'][0]
    dw = (md.d_lcs - md.b)/md.sigma
    print(f'current 8-domain model: L = {cut_length(cur)}; misfit/sigma with v >= ' +
          ', '.join(f'{vm}: {values(cur, dw, vm)[1]:.3f}' for vm in V_MINS))
    for r in res:
        print(f'  lambda x{r["lam_rel"]:.2f}, gamma {r["gamma"]:5.0f}: K {r["K"]:2d}, L {r["L"]:4d}, mu {r["mu"]}; '
              'misfit/sigma ' + ', '.join(f'v>={vm}: {r[f"chi_{vm}"]:.3f}' for vm in V_MINS), flush=True)
    np.savez(f'{OUT}/merge_compact{md.TAG}.npz', lam_rel=[r['lam_rel'] for r in res], gamma=[r['gamma'] for r in res],
             K=[r['K'] for r in res], L=[r['L'] for r in res], label=np.array([r['label'] for r in res]),
             **{f'v_{vm}': np.array([r[f'v_{vm}'] for r in res]) for vm in V_MINS},
             **{f'chi_{vm}': np.array([r[f'chi_{vm}'] for r in res]) for vm in V_MINS},
             v_mins=V_MINS, lam95=LAM95,
             current_label=cur, **{f'current_v_{vm}': values(cur, dw, vm)[0] for vm in V_MINS},
             **{f'current_chi_{vm}': values(cur, dw, vm)[1] for vm in V_MINS})
