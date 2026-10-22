"""
FD-merge (free domains by agglomerative merging, CONTEXT.md), from a local fit.

Score of a partition: chi2 + lam K + gamma L
  chi2  misfit of the whitened data with one value per domain
  K     number of domains
  L     boundary length in fine-cell edges (~100 km each; on the coarse graph an edge carries the number of fine edges
        between its two coarse cells, so L means the same at both levels)
Merge path: every cell starts as a domain; the two touching domains whose merger costs least are merged, with priority
(the exact cost of the constraint v_a = v_b for J_mu) - gamma x shared edges, where J_mu = chi2 + mu s v'Lv (s: the mean
diagonal of Gw'Gw, L: the graph Laplacian) only guides the order. Restarts over mu. K is chosen on the path by the
score; then a boundary clean-up (cells move to a touching domain if that lowers chi2 + gamma L and keeps both connected;
domains below MIN_CELLS fine cells are absorbed). Values are refitted within the VIS bounds.

Calibration (the first build's 'fine' mode, run_merge.py): lambda_95 is the smallest lambda for which 95% of null data
sets (a uniform level plus an error-model draw) choose K = 1 on every restart's path (gamma = 0); a fresh set of null
data checks it. Real data are run at settings lam = lam_rel x lambda_95, gamma = gamma_rel x lambda_95 (gamma per fine
edge), and a parametric bootstrap (the fitted model plus error-model draws) gives the probability that each edge of
the cell graph is a domain boundary.
Modes: 'fine' (everything on the problem's cells) or 'refine' (path and K on cells coarsened once, labels then moved
to the fine cells for the clean-up).

Config keys used, besides those of problem.local:
  [method] mode, mus, lam_rels, gamma_rels, null_runs, null_level, bootstrap ([[lam_rel, gamma_rel], ...]),
           bootstrap_runs, bounds, seed, lambda_95 ('calibrate', or a number to skip the calibration);
  [compute] workers
"""
import multiprocessing
import time
from collections import deque

import numpy as np

from .. import core, geometry, graph, problem, results
from ..cache import Cache

RESYNC = 100
K_SAVE = 100
MIN_CELLS = 3


def setup(Gw, edges, weights=None, size=None, ridge_rel=1e-6):
    """the merging problem of one basis: Gw (data x cells), edges, fine edges per edge, fine cells per cell"""
    n = Gw.shape[1]
    w = np.ones(len(edges)) if weights is None else np.asarray(weights, float)
    GtG = Gw.T @ Gw
    nbr = [dict() for _ in range(n)]
    for (i, j), wij in zip(edges, w):
        nbr[i][j] = wij
        nbr[j][i] = wij
    return dict(Gw=Gw, GtG=GtG, Lap=graph.laplacian(np.asarray(edges), w, n), scale=np.mean(np.diag(GtG)),
                ridge=ridge_rel*np.mean(np.diag(GtG)), edges=np.asarray(edges), w=w, nbr=nbr, n=n,
                size=np.ones(n) if size is None else np.asarray(size))


def coarse_problem(Gw_fine, edges_fine, parent):
    """the coarse basis: columns summed over children, edges weighted by the fine edges they hold"""
    A = np.zeros((len(parent), parent.max() + 1))
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
    """VIS per cell (refitted within the bounds) and chi2"""
    A, idx = indicator(label)
    GR = P['Gw'] @ A
    vR = core.box_qp(GR.T @ GR, GR.T @ dw, bounds[0], bounds[1], tol=1e-12)
    return vR[idx], float(np.sum((GR @ vR - dw)**2))


def merge(P, dw, lam, gamma, mus, refine=None, bounds=(0., 4.)):
    """the full method on one data vector. With refine = (P_fine, parent), the path and the choice of K are on P
    (coarse), then the labels go to the fine cells and the clean-up is on P_fine. Returns the best partition over
    the mu restarts (lowest final score) and every restart's path."""
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
    v, chi2 = values(Q, lab, dw, bounds)
    return dict(label=lab, v=v, K=len(np.unique(lab)), chi2=chi2, L=cut_length(Q, lab), score=s, mu=mu, curves=curves)


def calibrate_lambda(null_curves, quantile=0.95, lams=np.logspace(-1, 4, 600)):
    """the smallest lambda for which a fraction `quantile` of null runs choose K = 1 on every restart's path
    (gamma = 0); null_curves: a list over runs of {mu: (Ks, chi2, L)}"""
    ok = np.array([np.mean([all(np.argmin(c[1] + lam*c[0]) == 0 for c in run_.values()) for run_ in null_curves])
                   for lam in lams])
    return float(lams[np.argmax(ok >= quantile)])


def boundary_probability(labels, edges):
    """fraction of the partitions (n_runs x n_cells) that cut each edge"""
    labels = np.atleast_2d(labels)
    return np.mean(labels[:, edges[:, 0]] != labels[:, edges[:, 1]], axis=0)


def ari(x, y):
    """adjusted Rand index of two partitions of the same cells (1 = identical, 0 = chance)"""
    from scipy.special import comb
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    ct = np.zeros((xi.max() + 1, yi.max() + 1))
    np.add.at(ct, (xi, yi), 1)
    s = comb(ct, 2).sum()
    sa, sb = comb(ct.sum(1), 2).sum(), comb(ct.sum(0), 2).sum()
    expected = sa*sb/comb(len(x), 2)
    return (s - expected)/(0.5*(sa + sb) - expected)


# ------------------------------------------------------------------ the calibrated method on one problem
_job = {}


def _null_paths(dw):
    Q, mus = _job['path_problem'], _job['mus']
    return {mu: merge_path_scores(Q, dw, mu) for mu in mus}


def merge_path_scores(Q, dw, mu):
    Ks, chi2, L, _ = path_scores(Q, path(Q, dw, mu, 0.), dw, 0., 0.)
    return Ks, chi2, L


def _task(args):
    dw, lam, gamma = args
    t = time.time()
    r = merge(_job['path_problem'], dw, lam, gamma, _job['mus'], _job['refine'], _job['bounds'])
    r['seconds'] = time.time() - t
    return r


def calibrated(Gw, dw, edges, errors, me, parent=None, workers=3, say=print):
    """lambda_95, the null check, every setting on the data, and the bootstraps. errors: a list of whitened error
    realisations (null runs, then fresh null runs, then bootstrap runs)"""
    P = setup(Gw, edges)
    refine = None
    if me.get('mode', 'fine') == 'refine':
        refine = (P, parent)
        P = coarse_problem(Gw, edges, parent)
    _job.update(path_problem=P, mus=me.get('mus', [0.03, 0.1, 0.3]), refine=refine,
                bounds=tuple(me.get('bounds', [0., 4.])))
    n_null, n_boot = int(me.get('null_runs', 20)), int(me.get('bootstrap_runs', 30))
    v0 = me.get('null_level', 0.6)
    settings = [(lr, g) for lr in me.get('lam_rels', [1., 1/3]) for g in me.get('gamma_rels', [0., 0.001, 0.002, 0.005, 0.01])]
    with multiprocessing.get_context('fork').Pool(workers) as pool:
        if isinstance(me.get('lambda_95'), (int, float)):              # given: no calibration
            lam95, curves, null_K = float(me['lambda_95']), [], np.array([], int)
            say(f'lambda_95 = {lam95:.1f} (given)')
        else:
            null_dw = [Gw @ np.full(Gw.shape[1], v0) + errors[j] for j in range(2*n_null)]
            t = time.time()
            curves = pool.map(_null_paths, null_dw[:n_null])
            lam95 = calibrate_lambda(curves)
            say(f'lambda_95 = {lam95:.1f} ({time.time() - t:.0f} s)')
            t = time.time()
            chk = pool.map(_task, [(d, lam95, 0.) for d in null_dw[n_null:]])
            null_K = np.array([r['K'] for r in chk])
            say(f'fresh null data: K = 1 in {np.mean(null_K == 1):.0%} (K: {sorted(null_K)}) ({time.time() - t:.0f} s)')
        t = time.time()
        real = pool.map(_task, [(dw, lr*lam95, g*lam95) for lr, g in settings])
        say(f'data, {len(settings)} settings ({time.time() - t:.0f} s):')
        for (lr, g), r in zip(settings, real):
            say(f'    lambda x{lr:.2f}, gamma {g:.3f} lambda_95: K {r["K"]:2d}, chi2 {r["chi2"]:.1f}, L {r["L"]:.0f}, '
                f'mu {r["mu"]}')
        boot = []
        for lr, g in me.get('bootstrap', [[1., 0.002]]):
            k = [i for i, s in enumerate(settings) if np.isclose(s[0], lr) and np.isclose(s[1], g)][0]
            pred = Gw @ real[k]['v']
            t = time.time()
            bs = pool.map(_task, [(pred + errors[2*n_null + j], lr*lam95, g*lam95) for j in range(n_boot)])
            labs = np.array([b['label'] for b in bs])
            Ks = np.array([b['K'] for b in bs])
            boot.append(dict(setting=[lr, g], K=Ks, label=labs, prob=boundary_probability(labs, edges)))
            say(f'bootstrap lambda x{lr:.2f}, gamma {g:.3f}: K median {np.median(Ks):.0f} (range {Ks.min()}-{Ks.max()}) '
                f'({time.time() - t:.0f} s)')
    return dict(lam95=lam95, null_K=null_K, settings=settings, real=real, boot=boot, null_curves=curves)


def run(cfg, cache=None, log=print):
    cache = cache or Cache()
    t0 = time.time()
    say = lambda msg: log(f'[{(time.time() - t0)/60:5.1f} min] {msg}')
    pb = problem.local(cfg, cache, say)
    cells, me = pb.cells, cfg['method']
    edges, _ = graph.cell_edges(cells)
    parent = geometry.coarsen(cells)[1] if me.get('mode', 'fine') == 'refine' else None
    rng = np.random.default_rng(me.get('seed', 0))
    n_err = 2*int(me.get('null_runs', 20)) + int(me.get('bootstrap_runs', 30))
    errors = [pb.W @ pb.draw(rng, model_error=False) for _ in range(n_err)]
    out = calibrated(pb.Gw, pb.yw, edges, errors, me, parent, int(cfg['compute'].get('workers', 3)), say)
    main = out['real'][0]
    cell_vars = dict(vis=main['v'], domain=main['label'].astype(float))
    for i, r in enumerate(out['real']):
        cell_vars[f'domain_{i}'] = r['label'].astype(float)
    if pb.truth is not None:
        cell_vars['truth'] = pb.truth
    table = [dict(lam_rel=lr, gamma_rel=g, K=r['K'], chi2=r['chi2'], L=r['L'], mu=r['mu'])
             for (lr, g), r in zip(out['settings'], out['real'])]
    edge_vars = {f'boundary_probability_{i}': b['prob'] for i, b in enumerate(out['boot'])}
    attrs = dict(lambda_95=out['lam95'], null_K=out['null_K'].tolist(), settings=table,
                 bootstrap=[dict(setting=b['setting'], K=b['K'].tolist()) for b in out['boot']], **pb.info)
    model = pb.G @ main['v']
    return results.make(cfg, cells, cell_vars, points=(pb.plon, pb.plat),
                        point_vars=dict(data=pb.y, model=model, residual=pb.y - model), edges=edges,
                        edge_vars=edge_vars, **attrs)
