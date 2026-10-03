"""
Method (a): agglomerative merging of cells into domains (as notebooks/merge_domains.py and
merge_compact.py), on whitened data, with a coarse-then-refine option.

Score of a partition: chi2 + lam K + gamma L
  chi2  misfit of whitened data (unit variance per dimension) with one value per domain
  K     number of domains
  L     boundary length in fine-cell edges (~100 km each): on the coarse graph an edge carries the
        number of fine edges between the two coarse cells, so L means the same at both levels
Merge path: every cell starts as a domain; the two touching domains whose merger costs least are
merged, with priority (exact constraint cost for J_mu) - gamma * shared edges, J_mu = chi2 +
mu s v'Lv (s: mean diagonal of G'G, L: graph Laplacian) only guiding the order. Restarts over mu.
K is chosen on the path by the score; then boundary clean-up (cells move to a touching domain if
that lowers chi2 + gamma L and keeps both connected; domains below MIN_CELLS fine cells are
absorbed). Values are refitted within the VIS bounds.

A "problem" P is a dict made by setup() for one basis (fine or coarse); functions take P first.
"""
from collections import deque

import numpy as np
from scipy.optimize import lsq_linear

RESYNC = 100
K_SAVE = 100
MIN_CELLS = 3


def setup(Gw, edges, weights=None, size=None, ridge_rel=1e-6):
    """Gw: whitened columns (dims x cells); edges (n_edge x 2); weights: fine edges per edge;
    size: fine cells per cell"""
    n = Gw.shape[1]
    w = np.ones(len(edges)) if weights is None else np.asarray(weights, float)
    GtG = Gw.T @ Gw
    Lap = np.zeros((n, n))
    for (i, j), wij in zip(edges, w):
        Lap[i, j] -= wij; Lap[j, i] -= wij; Lap[i, i] += wij; Lap[j, j] += wij
    nbr = [dict() for _ in range(n)]
    for (i, j), wij in zip(edges, w):
        nbr[i][j] = wij
        nbr[j][i] = wij
    return dict(Gw=Gw, GtG=GtG, Lap=Lap, scale=np.mean(np.diag(GtG)), ridge=ridge_rel*np.mean(np.diag(GtG)),
                edges=np.asarray(edges), w=w, nbr=nbr, n=n, size=np.ones(n) if size is None else np.asarray(size))


def coarse_problem(Gw_fine, edges_fine, parent):
    """the coarse basis: columns summed over children, edges weighted by the fine edges they hold"""
    n_c = parent.max() + 1
    A = np.zeros((len(parent), n_c))
    A[np.arange(len(parent)), parent] = 1.
    pe = np.sort(parent[edges_fine], axis=1)
    pe = pe[pe[:, 0] != pe[:, 1]]
    e, w = np.unique(pe, axis=0, return_counts=True)
    return setup(Gw_fine @ A, e, w, size=np.bincount(parent))


def indicator(label):
    reps, idx = np.unique(label, return_inverse=True)
    A = np.zeros((len(label), len(reps)))
    A[np.arange(len(label)), idx] = 1.
    return A, idx


def misfit(P, label, dw):
    GR = P['Gw'] @ indicator(label)[0]
    vR, *_ = np.linalg.lstsq(GR, dw, rcond=None)
    return float(np.sum((GR @ vR - dw)**2))


def cut_length(P, label):
    e = P['edges']
    return float(np.sum(P['w'][label[e[:, 0]] != label[e[:, 1]]]))


def path(P, dw, mu, gamma):
    """merge path; returns {K: label} for K <= K_SAVE"""
    n, edges = P['n'], P['edges']
    H = P['GtG'] + mu*P['scale']*P['Lap'] + P['ridge']*np.eye(n)
    C = np.linalg.inv(H)
    g = P['Gw'].T @ dw
    v = C @ g
    label = np.arange(n)
    ea, eb, w = edges[:, 0].copy(), edges[:, 1].copy(), P['w'].copy()
    saved = {}
    for step in range(n - 1):
        K = n - step
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
        if (step + 1) % RESYNC == 0 and K - 1 > 1:
            A, idx = indicator(label)
            CR = np.linalg.inv(A.T @ H @ A)
            v, C = (CR @ (A.T @ g))[idx], CR[np.ix_(idx, idx)]
    saved[1] = label.copy()
    return saved


def path_scores(P, saved, dw, lam, gamma):
    Ks = np.array(sorted(saved))
    chi2 = np.array([misfit(P, saved[K], dw) for K in Ks])
    L = np.array([cut_length(P, saved[K]) for K in Ks])
    return Ks, chi2, L, chi2 + lam*Ks + gamma*L


def _connected(P, members):
    members = set(members)
    if not members:
        return True
    start = next(iter(members))
    seen, q = {start}, deque([start])
    while q:
        c = q.popleft()
        for m in P['nbr'][c]:
            if m in members and m not in seen:
                seen.add(m)
                q.append(m)
    return len(seen) == len(members)


def clean_up(P, label, dw, gamma, max_pass=10):
    """absorb domains of fewer than MIN_CELLS fine cells; move boundary cells while chi2 + gamma L falls"""
    label = label.copy()
    Gw, nbr, size = P['Gw'], P['nbr'], P['size']
    while True:
        regions = {}
        for c, r in enumerate(label):
            regions.setdefault(r, []).append(c)
        small = [r for r, m in regions.items() if size[m].sum() < MIN_CELLS]
        if not small:
            break
        r = small[0]
        cand = {label[m] for c in regions[r] for m in nbr[c]} - {r}
        best = min(cand, key=lambda q: misfit(P, np.where(label == r, q, label), dw)
                   + gamma*cut_length(P, np.where(label == r, q, label)))
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
        regions = {}
        for c, r in enumerate(label):
            regions.setdefault(r, []).append(c)
        for c in range(P['n']):
            r = label[c]
            targets = {label[m] for m in nbr[c]} - {r}
            if not targets or size[regions[r]].sum() - size[c] < MIN_CELLS:
                continue
            if not _connected(P, [m for m in regions[r] if m != c]):
                continue
            w_r = sum(wm for m, wm in nbr[c].items() if label[m] == r)
            for q in targets:
                dL = w_r - sum(wm for m, wm in nbr[c].items() if label[m] == q)
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


def values(P, label, dw, bounds=(0., 4.)):
    """VIS per cell (refitted within bounds) and chi2"""
    A, idx = indicator(label)
    GR = P['Gw'] @ A
    vR = lsq_linear(GR, dw, bounds=bounds).x
    return vR[idx], float(np.sum((GR @ vR - dw)**2))


def run(P, dw, lam, gamma, mus, refine=None):
    """full method on one data vector. With refine = (P_fine, parent), the path and the choice of
    K are on P (coarse), then labels go to the fine cells and the clean-up is on P_fine.
    Returns the best partition over the mu restarts (lowest final score) and every restart's path."""
    best, curves = None, {}
    for mu in mus:
        saved = path(P, dw, mu, gamma)
        Ks, chi2, L, score = path_scores(P, saved, dw, lam, gamma)
        curves[mu] = (Ks, chi2, L)
        lab = saved[int(Ks[np.argmin(score)])]
        Q = P
        if refine is not None:
            Q, parent = refine
            lab = lab[parent]
        lab = clean_up(Q, lab, dw, gamma)
        K = len(np.unique(lab))
        s = misfit(Q, lab, dw) + lam*K + gamma*cut_length(Q, lab)
        if best is None or s < best[0]:
            best = (s, lab, mu)
    s, lab, mu = best
    Q = P if refine is None else refine[0]
    v, chi2 = values(Q, lab, dw)
    return dict(label=lab, v=v, K=len(np.unique(lab)), chi2=chi2, L=cut_length(Q, lab), score=s, mu=mu,
                curves=curves)


def calibrate_lambda(null_curves, quantile=0.95, lams=np.logspace(-1, 4, 600)):
    """smallest lambda for which a fraction `quantile` of null runs choose K = 1 on every restart's
    path (gamma = 0). null_curves: list over runs of {mu: (Ks, chi2, L)}"""
    ok = []
    for lam in lams:
        ok.append(np.mean([all(np.argmin(c[1] + lam*c[0]) == 0 for c in run_.values()) for run_ in null_curves]))
    ok = np.array(ok)
    return float(lams[np.argmax(ok >= quantile)])
