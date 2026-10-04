"""
Method (c), labelled Voronoi: transdimensional sampling of units made of Voronoi pieces.

Model
  M seeds on cell centres; each seed carries a unit label (U units, every unit has >= 1 seed).
  A cell belongs to its nearest seed (great circle) and through it to that seed's unit; each unit
  has one VIS value. Units can be non-convex, and can also be disconnected (reported separately).
Likelihood: as voronoi.evaluate, with the units in place of the Voronoi regions (VIS values,
  uniform on [V_LO, V_HI], and the noise scale h, Jeffreys, integrated out analytically).
Boundary-length penalty (P['gamma'], default 0; the analogue of gamma in merging): L = number of
  cell-graph edges (P['edges']) between different units (edges between seeds of the same unit cost
  nothing). The joint prior of geometry and h carries a factor exp(-gamma L / (2 h^2)), so that
  -2 log posterior = (chi2 + gamma L)/h^2 + ..., the same trade-off of misfit against length as the
  merging score chi2 + gamma L. With h integrated out, chi2 is replaced by chi2 + gamma L in the
  marginal likelihood; with h fixed, the log posterior gains -gamma L/(2 h^2). (The factor is not
  normalised over geometries: it is part of an improper joint prior, as the Jeffreys prior is.)
Prior
  p(M, U, seed set, grouping) = p(M) p(seed set | M) p(U | M) p(grouping | M, U)
    M uniform on 1..M_MAX; seed set uniform: 1/C(n, M); U uniform on 1..M: 1/M;
    grouping of the M seeds into U (unlabelled) non-empty units uniform: 1/S(M, U) (Stirling
    numbers of the second kind).
Moves (each step picks one of the 8 kinds with probability 1/8; impossible proposals are rejected)
  seed_birth  new seed on a free cell (uniform), joining a unit (uniform)            M+1
  seed_death  remove a seed whose unit keeps >= 1 seed (uniform over such seeds)     M-1
  unit_birth  new seed on a free cell forming a new unit                             M+1, U+1
  unit_death  remove a single-seed unit (uniform over them)                          M-1, U-1
  move        a seed moves (adjacent cell, within R_LOCAL, or anywhere; as voronoi.propose)
  relabel     a seed from a unit with >= 2 seeds joins another unit (uniform)
  merge       two units (uniform over pairs) become one                              U-1
  split       a unit with >= 2 seeds (uniform) is split into two non-empty groups,
              uniformly over the 2^(m-1) - 1 unordered splits of its m seeds         U+1
The log prior x proposal ratios ("extra") are derived in the comments of propose(); the prior test
(likelihood switched off) checks them: P(M) uniform on 1..M_MAX and P(U | M) uniform on 1..M.
"""
import numpy as np

from . import voronoi
from .voronoi import V_LO, V_HI, MOVE_MIX

KINDS = ('seed_birth', 'seed_death', 'unit_birth', 'unit_death', 'move', 'relabel', 'merge', 'split')


def log_stirling2(m_max):
    """logS[m, k] = log S(m, k), Stirling numbers of the second kind (-inf where zero)"""
    L = np.full((m_max + 2, m_max + 2), -np.inf)
    L[0, 0] = 0.
    for m in range(1, m_max + 2):
        for k in range(1, m + 1):
            L[m, k] = np.logaddexp(np.log(k) + L[m-1, k], L[m-1, k-1])
    return L


def boundary_length(edges, lab):
    """number of cell-graph edges between different units"""
    return int(np.sum(lab[edges[:, 0]] != lab[edges[:, 1]]))


def evaluate(P, seeds, units, h=None):
    """log marginal likelihood (with the boundary-length penalty), chi2, unit values, and the unit
    of every cell"""
    U = units.max() + 1
    lab = units[voronoi.assign(P, seeds)]
    GR = np.stack([P['Gw'][:, lab == k].sum(1) for k in range(U)], 1)
    M = GR.T @ GR
    try:
        cf = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return -np.inf, np.inf, None, lab
    v = np.linalg.solve(M, GR.T @ P['dw'])
    if (v < V_LO).any() or (v > V_HI).any():
        return -np.inf, np.inf, v, lab
    chi2 = float(np.sum((GR @ v - P['dw'])**2))
    gamma = P.get('gamma', 0.)
    pen = gamma*boundary_length(P['edges'], lab) if gamma else 0.
    N = P['N']
    base = -U*np.log(V_HI - V_LO) - np.sum(np.log(np.diag(cf)))
    if h is None:
        from scipy.special import gammaln
        ll = base + gammaln((N - U)/2) - (N - U)/2*np.log(np.pi*(chi2 + pen))
    else:
        ll = base - (N - U)/2*np.log(2*np.pi*h**2) - (chi2 + pen)/(2*h**2)
    return float(ll), chi2, v, lab


def _compact(units):
    """relabel units to 0..U-1 in order of first appearance"""
    _, inv = np.unique(units, return_inverse=True)
    return inv


def propose(P, seeds, units, rng, m_max, logS):
    """one proposal: (kind, new seeds, new units, extra) with extra = log(prior ratio x q_rev/q_fwd);
    extra = None means the proposal is impossible here (rejected)"""
    n, M = P['n'], len(seeds)
    U = units.max() + 1
    c = np.bincount(units, minlength=U)
    kind = KINDS[rng.integers(len(KINDS))]

    if kind == 'seed_birth':
        # prior ratio (M+1)/(n-M) * M/(M+1) * S(M,U)/S(M+1,U); q_fwd 1/((n-M) U); q_rev 1/D'
        if M >= m_max:
            return kind, seeds, units, None
        u = rng.integers(U)
        new_s = np.append(seeds, rng.choice(np.setdiff1d(np.arange(n), seeds)))
        new_u = np.append(units, u)
        D_new = np.sum(np.bincount(new_u)[new_u] >= 2)
        extra = np.log(M) + logS[M, U] - logS[M+1, U] + np.log(U) - np.log(D_new)
        return kind, new_s, new_u, extra

    if kind == 'seed_death':
        # reverse of seed_birth from the smaller state (M-1): extra = -[log(M-1) + S(M-1,U) - S(M,U) + log U - log D]
        deletable = np.where(c[units] >= 2)[0]
        if len(deletable) == 0:
            return kind, seeds, units, None
        j = rng.choice(deletable)
        new_s, new_u = np.delete(seeds, j), np.delete(units, j)
        extra = -(np.log(M - 1) + logS[M-1, U] - logS[M, U] + np.log(U) - np.log(len(deletable)))
        return kind, new_s, new_u, extra

    if kind == 'unit_birth':
        # prior ratio (M+1)/(n-M) * M/(M+1) * S(M,U)/S(M+1,U+1); q_fwd 1/(n-M); q_rev 1/S1'
        if M >= m_max:
            return kind, seeds, units, None
        new_s = np.append(seeds, rng.choice(np.setdiff1d(np.arange(n), seeds)))
        new_u = np.append(units, U)
        S1_new = np.sum(np.bincount(new_u) == 1)
        extra = np.log(M) + logS[M, U] - logS[M+1, U+1] - np.log(S1_new)
        return kind, new_s, new_u, extra

    if kind == 'unit_death':
        # reverse of unit_birth from (M-1, U-1): extra = -[log(M-1) + S(M-1,U-1) - S(M,U) - log S1]
        singles = np.where(c == 1)[0]
        if len(singles) == 0 or U < 2:
            return kind, seeds, units, None
        u = rng.choice(singles)
        j = np.where(units == u)[0][0]
        new_s, new_u = np.delete(seeds, j), _compact(np.delete(units, j))
        extra = -(np.log(M - 1) + logS[M-1, U-1] - logS[M, U] - np.log(len(singles)))
        return kind, new_s, new_u, extra

    if kind == 'move':
        # positions only; the grouping and the prior are unchanged
        _, new_s, corr = voronoi.propose(P, seeds, rng, m_max, moves=('move',))
        return kind, new_s, units, corr

    if kind == 'relabel':
        # a seed from a unit with >= 2 seeds joins one of the U-1 others; q_fwd 1/(E (U-1)),
        # q_rev 1/(E' (U-1)); prior unchanged (same M, U)
        elig = np.where(c[units] >= 2)[0]
        if len(elig) == 0 or U < 2:
            return kind, seeds, units, None
        j = rng.choice(elig)
        others = np.delete(np.arange(U), units[j])
        new_u = units.copy()
        new_u[j] = rng.choice(others)
        E_new = np.sum(np.bincount(new_u, minlength=U)[new_u] >= 2)
        return kind, seeds, new_u, np.log(len(elig)) - np.log(E_new)

    if kind == 'merge':
        # prior ratio S(M,U)/S(M,U-1); q_fwd 1/(U(U-1)/2); q_rev 1/(W' (2^(m-1) - 1))
        if U < 2:
            return kind, seeds, units, None
        a, b = rng.choice(U, 2, replace=False)
        new_u = _compact(np.where(units == b, a, units))
        m = c[a] + c[b]
        W_new = np.sum(np.bincount(new_u) >= 2)
        extra = (logS[M, U] - logS[M, U-1] + np.log(U*(U-1)/2) - np.log(W_new)
                 - np.log(2.**(m-1) - 1))
        return kind, seeds, new_u, extra

    if kind == 'split':
        # reverse of merge from (U+1): extra = -[S(M,U+1) - S(M,U) + log((U+1)U/2) - log W - log(2^(m-1) - 1)]
        multi = np.where(c >= 2)[0]
        if len(multi) == 0:
            return kind, seeds, units, None
        u = rng.choice(multi)
        members = np.where(units == u)[0]
        m = len(members)
        while True:                                   # uniform over unordered non-empty splits
            side = np.r_[0, rng.integers(0, 2, m - 1)]
            if side.any():
                break
        new_u = units.copy()
        new_u[members[side == 1]] = U
        extra = -(logS[M, U+1] - logS[M, U] + np.log((U+1)*U/2) - np.log(len(multi)) - np.log(2.**(m-1) - 1))
        return kind, seeds, new_u, extra


def components(lab, nbr_edges):
    """number of connected pieces of all units together (cell graph edges within a unit)"""
    n = len(lab)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    same = lab[nbr_edges[:, 0]] == lab[nbr_edges[:, 1]]
    for i, j in nbr_edges[same]:
        parent[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def sample(P, n_sweeps, temps, rng, m_max=60, m_start=6, h=None, thin=5, sweep=20, evaluate_fn=None, edges=None):
    """parallel-tempered sampler; returns T = 1 samples (dicts) and acceptance rates per move kind"""
    ev = evaluate if evaluate_fn is None else evaluate_fn
    logS = log_stirling2(m_max + 1)
    n = P['n']
    state = []
    for _ in temps:
        while True:
            s = rng.choice(n, m_start, replace=False)
            u = _compact(rng.integers(0, max(1, m_start//2), m_start))
            ll, chi2, v, lab = ev(P, s, u, h)
            if np.isfinite(ll):
                break
        state.append([s, u, ll, chi2, v, lab])
    acc = {k: [0, 0] for k in KINDS + ('swap',)}
    out = []
    for it in range(n_sweeps):
        for ci, T in enumerate(temps):
            for _ in range(sweep):
                s, u, ll0 = state[ci][0], state[ci][1], state[ci][2]
                kind, ns, nu, extra = propose(P, s, u, rng, m_max, logS)
                if ci == 0:
                    acc[kind][1] += 1
                if extra is None:
                    continue
                ll, chi2, v, lab = ev(P, ns, nu, h)
                if np.isfinite(ll) and np.log(rng.random()) < (ll - ll0)/T + extra:
                    state[ci] = [ns, nu, ll, chi2, v, lab]
                    if ci == 0:
                        acc[kind][0] += 1
        for ci in range(len(temps) - 1):
            a = (state[ci+1][2] - state[ci][2])*(1/temps[ci] - 1/temps[ci+1])
            acc['swap'][1] += 1
            if np.log(rng.random()) < a:
                state[ci], state[ci+1] = state[ci+1], state[ci]
                acc['swap'][0] += 1
        if it % thin == 0:
            s, u, ll, chi2, v, lab = state[0]
            rec = dict(M=len(s), U=int(u.max()) + 1, seeds=s.copy(), units=u.copy(), label=lab.copy(),
                       v=None if v is None else v.copy(), chi2=chi2, ll=ll)
            if edges is not None:
                rec['pieces'] = components(lab, edges)
                rec['L'] = boundary_length(edges, lab)
            out.append(rec)
    return out, {k: a/max(t, 1) for k, (a, t) in acc.items()}
