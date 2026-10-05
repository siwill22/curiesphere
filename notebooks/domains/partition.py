"""
Method (c), cell-partition sampler: MCMC over partitions of the cells into units, with the merging
score as the posterior (the Bayesian counterpart of merging, on the same representation).

Target
  Units are sets of cells (not necessarily connected: a unit is a value class); each has one VIS
  value, uniform on [V_LO, V_HI], integrated out analytically; the noise scale h (Jeffreys) too.
  Prior on partitions (unlabelled, U <= U_MAX): p ~ exp(-beta L - kappa U), L = cell-graph edges
  between different units; this is proper (a finite set) and does not depend on h. Then
      log pi = -U log(V_HI - V_LO) - 1/2 log|GR'GR| + log Gamma((N-U)/2) - (N-U)/2 log(pi chi2) - beta L - kappa U
  A partition whose least-squares values leave [V_LO, V_HI] gets zero mass.
  Relation to merging (score chi2 + lam U + gamma L): beta = gamma/(2 h^2), kappa = lam/(2 h^2) at the
  noise level h of the fit (merging_prior). Putting the penalties inside the chi2 term instead (scaled
  by the integrated h) does not work here: a longer boundary then mostly inflates h, and the
  number of fragmented partitions (their entropy) wins; the sampler drifts to L ~ 2000.
Moves: Swendsen-Wang cuts (Barbu & Zhu 2005, IEEE PAMI 27, 1239)
  1. q is drawn from Q_SET (each q is a separate kernel in detailed balance on its own).
  2. Each edge inside a unit is switched on with probability q; a cell v is drawn uniformly and V0 is
     its connected component of on-edges (q = 0: V0 = {v}, a boundary flip; q near 1: most of the
     unit, a merge of the whole unit with another).
  3. V0 gets a new label l': another unit, weight 1 if adjacent to V0, W_FAR if not, or a new unit,
     weight W_NEW.
  Acceptance: pi(B)/pi(A) x (1-q)^(c_new - c_old) x q_label(l | B)/q_label(l' | A), where c_old/c_new
  are the numbers of edges between V0 and the rest of its old/new unit (they must be off for V0 to be
  a component). The probability of the internal edges of V0 and |V0|/n are the same both ways.
Check: partition_prior_test enumerates all partitions of a small graph and compares visit
  frequencies with the exact target.
"""
import itertools

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import gammaln

V_LO, V_HI = 0., 4.
Q_SET = (0., 0.3, 0.6, 0.9, 0.97)
W_FAR, W_NEW = 0.2, 1.


def setup(Gw, dw, edges, beta=0., kappa=0., u_max=60):
    return dict(Gw=Gw, dw=dw, N=Gw.shape[0], n=Gw.shape[1], e0=edges[:, 0], e1=edges[:, 1], edges=edges,
                beta=beta, kappa=kappa, u_max=u_max)


def merging_prior(lam, gamma, chi2, N, K):
    """beta and kappa equivalent to the merging penalties at the noise level of a fit (chi2, K units)"""
    h2 = chi2/(N - K)
    return gamma/(2*h2), lam/(2*h2), h2


def score(P, GR, L):
    """log target, chi2 and values of a partition given its unit columns GR (N x U) and length L"""
    U = GR.shape[1]
    M = GR.T @ GR
    try:
        cf = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return -np.inf, np.inf, None
    v = np.linalg.solve(M, GR.T @ P['dw'])
    if (v < V_LO).any() or (v > V_HI).any():
        return -np.inf, np.inf, v
    chi2 = float(np.sum((GR @ v - P['dw'])**2))
    N = P['N']
    ll = (-U*np.log(V_HI - V_LO) - np.sum(np.log(np.diag(cf))) + gammaln((N - U)/2)
          - (N - U)/2*np.log(np.pi*chi2) - P['beta']*L - P['kappa']*U)
    return float(ll), chi2, v


def state(P, lab, score_fn=score):
    lab = np.unique(lab, return_inverse=True)[1]
    U = lab.max() + 1
    GR = np.stack([P['Gw'][:, lab == k].sum(1) for k in range(U)], 1)
    L = int(np.sum(lab[P['e0']] != lab[P['e1']]))
    ll, chi2, v = score_fn(P, GR, L)
    return dict(lab=lab, GR=GR, L=L, ll=ll, chi2=chi2, v=v)


def _label_weight(u, adj, new):
    return W_NEW if new else (1. if u in adj else W_FAR)


def _label_norm(units, own, adj):
    """sum of the label-proposal weights over the candidates (other units and a new one)"""
    return sum(1. if u in adj else W_FAR for u in units if u != own) + W_NEW


def propose(P, st, rng):
    """one Swendsen-Wang-cut proposal: (kind, new state without score, log proposal ratio) or None"""
    lab, n, e0, e1 = st['lab'], P['n'], P['e0'], P['e1']
    U = st['GR'].shape[1]
    q = Q_SET[rng.integers(len(Q_SET))]
    v0 = rng.integers(n)
    l = lab[v0]
    if q > 0:
        inside = (lab[e0] == l) & (lab[e1] == l)
        on = inside & (rng.random(len(e0)) < q)
        g = csr_matrix((np.ones(on.sum()), (e0[on], e1[on])), shape=(n, n))
        comp = connected_components(g, directed=False)[1]
        in0 = comp == comp[v0]
    else:
        in0 = np.zeros(n, bool)
        in0[v0] = True
    cut = in0[e0] != in0[e1]
    other = np.where(in0[e0[cut]], lab[e1[cut]], lab[e0[cut]])          # unit across each cut edge
    adj = set(other.tolist()) - {l}
    whole = in0.sum() == np.sum(lab == l)
    # forward label
    units = range(U)
    w = np.array([0. if u == l else _label_weight(u, adj, False) for u in units] + [W_NEW])
    k = rng.choice(U + 1, p=w/w.sum())
    new = k == U
    if new and (whole or U >= P['u_max']):
        return None
    log_qf = np.log(w[k]/w.sum())
    c_old = int(np.sum(other == l))
    c_new = 0 if new else int(np.sum(other == k))
    # new partition
    gsum = P['Gw'][:, in0].sum(1)
    lab_b, GR_b = lab.copy(), st['GR'].copy()
    GR_b[:, l] -= gsum
    if new:
        lab_b[in0] = U
        GR_b = np.hstack([GR_b, gsum[:, None]])
    else:
        lab_b[in0] = k
        GR_b[:, k] += gsum
    own_b = U if new else k
    units_b = list(range(U)) + ([U] if new else [])
    if whole:                                                            # unit l disappears
        units_b.remove(l)
    adj_b = set(other.tolist()) - {own_b}
    log_qr = np.log(_label_weight(l, adj_b, whole)/_label_norm(units_b, own_b, adj_b))
    if whole:
        GR_b = np.delete(GR_b, l, axis=1)
        lab_b = np.where(lab_b > l, lab_b - 1, lab_b)
    L_b = st['L'] + c_old - c_new
    kind = 'merge' if whole else ('new' if new else ('flip' if q == 0 else 'move'))
    log_cut = (c_new - c_old)*np.log1p(-q) if q > 0 else 0.
    return kind, dict(lab=lab_b, GR=GR_b, L=L_b, n0=int(in0.sum()), q=q), log_cut + log_qr - log_qf


def sample(P, lab0, n_sweeps, temps, rng, thin=5, sweep=50, score_fn=score, record=True, fix_u=False, log=None):
    """parallel-tempered sampler from partition lab0; returns the T = 1 samples and acceptance rates.
    fix_u: proposals that change the number of units are rejected (the target restricted to U = U0).
    log: a list that receives, for every T = 1 proposal, (kind, |V0|, q, dU, dL, dll, extra)."""
    chains = [state(P, lab0, score_fn) for _ in temps]
    kinds = ('flip', 'move', 'new', 'merge')
    acc = {k: [0, 0] for k in kinds + ('swap',)}
    out = []
    for it in range(n_sweeps):
        for ci, T in enumerate(temps):
            for _ in range(sweep):
                pr = propose(P, chains[ci], rng)
                if pr is None:
                    continue
                kind, nb, extra = pr
                dU = nb['GR'].shape[1] - chains[ci]['GR'].shape[1]
                if fix_u and dU:
                    continue
                ll, chi2, v = score_fn(P, nb['GR'], nb['L'])
                if ci == 0:
                    acc[kind][1] += 1
                    if log is not None:
                        log.append((kind, nb['n0'], nb['q'], dU, nb['L'] - chains[ci]['L'], ll - chains[ci]['ll'], extra))
                if np.isfinite(ll) and np.log(rng.random()) < (ll - chains[ci]['ll'])/T + extra:
                    chains[ci] = dict(nb, ll=ll, chi2=chi2, v=v)
                    if ci == 0:
                        acc[kind][0] += 1
        for ci in range(len(temps) - 1):
            a = (chains[ci+1]['ll'] - chains[ci]['ll'])*(1/temps[ci] - 1/temps[ci+1])
            acc['swap'][1] += 1
            if np.log(rng.random()) < a:
                chains[ci], chains[ci+1] = chains[ci+1], chains[ci]
                acc['swap'][0] += 1
        if it % thin == 0 and record:
            c = chains[0]
            out.append(dict(U=c['GR'].shape[1], L=c['L'], chi2=c['chi2'], ll=c['ll'], label=c['lab'].copy(),
                            v=None if c['v'] is None else np.asarray(c['v']).copy()))
    return out, {k: a/max(t, 1) for k, (a, t) in acc.items()}


def set_partitions(n):
    """all partitions of range(n) as canonical label arrays (restricted growth strings)"""
    def rec(prefix, m):
        if len(prefix) == n:
            yield np.array(prefix)
            return
        for k in range(m + 1):
            yield from rec(prefix + [k], max(m, k + 1))
    yield from rec([0], 1)


def partition_prior_test(edges, n, toy_ll, n_steps, rng, u_max=None):
    """exact check of the moves: sample the toy target exp(toy_ll(U, L)) over all partitions of a small
    graph and return (exact probabilities, visit frequencies) per partition"""
    parts = list(set_partitions(n))

    def key(lab):                                   # labels in order of first appearance
        first = {}
        return tuple(first.setdefault(x, len(first)) for x in lab.tolist())
    e0, e1 = edges[:, 0], edges[:, 1]
    u_max = u_max or n
    logp = np.array([toy_ll(p.max() + 1, int(np.sum(p[e0] != p[e1]))) if p.max() + 1 <= u_max else -np.inf
                     for p in parts])
    exact = np.exp(logp - logp.max())
    exact /= exact.sum()
    P = setup(np.zeros((1, n)), np.zeros(1), edges, u_max=u_max)
    fn = lambda P, GR, L: (toy_ll(GR.shape[1], L), 0., None)
    st = state(P, np.zeros(n, int), fn)
    index = {key(p): i for i, p in enumerate(parts)}
    counts = np.zeros(len(parts))
    for _ in range(n_steps):
        pr = propose(P, st, rng)
        if pr is not None:
            kind, nb, extra = pr
            ll = fn(P, nb['GR'], nb['L'])[0]
            if np.log(rng.random()) < ll - st['ll'] + extra:
                st = dict(nb, ll=ll)
        counts[index[key(st['lab'])]] += 1
    return exact, counts/counts.sum()
