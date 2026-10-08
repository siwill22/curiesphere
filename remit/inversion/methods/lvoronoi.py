"""
FD-lvor (free domains from labelled Voronoi pieces, CONTEXT.md), from a local fit: transdimensional sampling of
units made of Voronoi pieces.

Model
  M seeds on cell centres; each seed carries a unit label (U units, every unit has >= 1 seed). A cell belongs to its
  nearest seed (great circle) and through it to that seed's unit; each unit has one VIS value. Units can be
  non-convex, and also disconnected (reported separately).
Likelihood: as FD-vor's, with the units in place of the Voronoi regions (values uniform on the VIS bounds and the
  noise scale h, Jeffreys, integrated out analytically).
Boundary-length penalty (P['gamma'], default 0; the analogue of gamma in merging): L = the number of cell-graph edges
  between different units. The joint prior of geometry and h carries exp(-gamma L/(2 h^2)), so -2 log posterior =
  (chi2 + gamma L)/h^2 + ..., the same trade-off as the merging score chi2 + gamma L; with h integrated out, chi2 is
  replaced by chi2 + gamma L in the marginal likelihood.
Prior: p(M, U, seed set, grouping) = p(M) p(seed set | M) p(U | M) p(grouping | M, U): M uniform on 1..M_MAX; seed set
  uniform, 1/C(n, M); U uniform on 1..M, 1/M; the grouping of the M seeds into U (unlabelled) non-empty units uniform,
  1/S(M, U) (Stirling numbers of the second kind).
Moves (each step one of 8 kinds, probability 1/8; impossible proposals are rejected): seed_birth, seed_death,
  unit_birth, unit_death, move (as FD-vor), relabel, merge, split. The log prior x proposal ratios are derived in the
  comments of _propose; the prior test (likelihood switched off) checks them: P(M) uniform on 1..M_MAX and P(U | M)
  uniform on 1..M.
Residual-guided unit births (after setup_births): the cell of a new unit is drawn from a mixture of uniform over free
  cells (BIRTH_MIX) and proportional to the misfit reduction of giving the 400 km disc around the cell a value of its
  own; q depends only on the state, so detailed balance holds with q in the Hastings ratio of unit_birth and of its
  reverse.
Starting point: seeds_from_partition turns a cell partition (e.g. an FD-merge result) into seeds and units.

Config keys used, besides those of problem.local:
  [method] temps ([T_min, T_max, n]), sweeps, sweep, thin, m_max, m_start, chains, seed, burn, bounds, gamma_rel
  (the boundary penalty relative to lambda_95 of an FD-merge calibration, given as lambda_95), guided, start
  ('random' or an FD-merge result .nc and the index of its setting);  [compute] workers
"""
import time

import numpy as np

from .. import core, graph, problem, results, sampling
from ..cache import Cache
from . import voronoi
from .voronoi import V_LO, V_HI

KINDS = ('seed_birth', 'seed_death', 'unit_birth', 'unit_death', 'move', 'relabel', 'merge', 'split')
BIRTH_MIX = 0.5                                     # uniform share of the unit-birth proposal


def setup_births(P):
    """columns of the 400 km disc (P['near']['local']) around each cell, for guided unit births"""
    D = np.stack([P['Gw'][:, nb].sum(1) for nb in P['near']['local']], 1)
    P['Gdisc'], P['gdisc2'] = D, np.sum(D**2, 0)
    return P


def birth_q(P, seeds, v, lab):
    """probability of each cell as the seed of a new unit (0 on occupied cells)"""
    free = np.ones(P['n'], bool)
    free[seeds] = False
    uniform = np.where(free, 1./free.sum(), 0.)
    if 'Gdisc' not in P or v is None:
        return uniform
    r = P['dw'] - P['Gw'] @ np.asarray(v)[lab]
    w = np.where(free, (P['Gdisc'].T @ r)**2/P['gdisc2'], 0.)
    if not w.sum() > 0:
        return uniform
    return BIRTH_MIX*uniform + (1 - BIRTH_MIX)*w/w.sum()


def seeds_from_partition(P, label, tol=0.02, max_seeds=None):
    """seeds and units whose Voronoi units reproduce a cell partition except for a fraction tol of
    the cells. One seed per unit at its most interior cell (farthest, great circle, from any cell of
    another unit); then, repeatedly, (1) the seed (any free cell, with that cell's unit) that
    corrects the most cells is added, and (2) every seed in turn moves to the adjacent cell
    (within R_STEP_KM) that most reduces the number of wrong cells, until no move helps.
    Stops at tol, at max_seeds, or when no added seed helps."""
    lab = np.unique(label, return_inverse=True)[1]
    dots = P['xyz'] @ P['xyz'].T
    same = lab[:, None] == lab[None, :]
    depth = -np.where(~same, dots, -1.).max(1)                              # larger = more interior
    seeds = [int(idx[np.argmax(depth[idx])]) for idx in (np.where(lab == k)[0] for k in range(lab.max() + 1))]
    n_wrong = lambda s: int(np.sum(lab[s][np.argmax(dots[:, s], 1)] != lab))
    while True:
        # (2) local moves
        improved = True
        while improved:
            improved = False
            for i in range(len(seeds)):
                cur = n_wrong(seeds)
                for c in P['near']['step'][seeds[i]]:
                    if c in seeds:
                        continue
                    trial = seeds[:i] + [int(c)] + seeds[i+1:]
                    if lab[c] == lab[seeds[i]] and n_wrong(trial) < cur:
                        seeds, cur, improved = trial, n_wrong(trial), True
        near = dots[:, seeds].max(1)
        wrong = lab[seeds][np.argmax(dots[:, seeds], 1)] != lab
        if wrong.sum() <= tol*len(lab) or len(seeds) >= (max_seeds or P['n']):
            break
        # (1) best added seed
        take = dots > near[:, None]                                         # take[i, j]: seed at j takes cell i
        gain = np.sum(take & wrong[:, None] & same, 0) - np.sum(take & ~wrong[:, None] & ~same, 0)
        gain[seeds] = -1
        if gain.max() <= 0:
            break
        seeds.append(int(np.argmax(gain)))
    seeds = np.array(seeds)
    return seeds, _compact(lab[seeds])


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
    gamma = P.get('gamma', 0.)
    pen = gamma*boundary_length(P['edges'], lab) if gamma else 0.
    ll, chi2, v = core.box_evidence(GR, P['dw'], *P.get('bounds', (V_LO, V_HI)), h, pen)
    return ll, chi2, v, lab


def _compact(units):
    """relabel units to 0..U-1 in order of first appearance"""
    _, inv = np.unique(units, return_inverse=True)
    return inv


def propose(P, seeds, units, rng, m_max, logS, qfun=None):
    """one proposal: (kind, new seeds, new units, extra, post) with extra = log(prior ratio x
    q_rev/q_fwd); extra = None means the proposal is impossible here (rejected). With guided births
    (qfun() = birth_q of the current state), unit_birth uses q, and unit_death returns post = the
    removed cell: its extra still lacks log q_new(cell) + log(n - M + 1), added by the caller once
    the smaller state has been evaluated."""
    kind, new_s, new_u, extra = _propose(P, seeds, units, rng, m_max, logS, qfun)
    post = None
    if kind == 'unit_death' and extra is not None and qfun is not None:
        post = int(np.setdiff1d(seeds, new_s)[0])
    return kind, new_s, new_u, extra, post


def _propose(P, seeds, units, rng, m_max, logS, qfun):
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
        # guided: q_fwd q(c) instead of 1/(n-M), i.e. extra gains -log q(c) - log(n-M)
        if M >= m_max:
            return kind, seeds, units, None
        if qfun is None:
            c, corr = rng.choice(np.setdiff1d(np.arange(n), seeds)), 0.
        else:
            q = qfun()
            c = rng.choice(n, p=q)
            corr = -np.log(q[c]) - np.log(n - M)
        new_s = np.append(seeds, c)
        new_u = np.append(units, U)
        S1_new = np.sum(np.bincount(new_u) == 1)
        extra = np.log(M) + logS[M, U] - logS[M+1, U+1] - np.log(S1_new) + corr
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


def sample(P, n_sweeps, temps, rng, m_max=60, m_start=6, h=None, thin=5, sweep=20, evaluate_fn=None, edges=None,
           start=None):
    """parallel-tempered sampler; returns T = 1 samples (dicts) and acceptance rates per move kind.
    start = (seeds, units): every chain starts there (else random starts with m_start seeds).
    Unit births are residual-guided if setup_births(P) has been called."""
    ev = evaluate if evaluate_fn is None else evaluate_fn
    logS = log_stirling2(m_max + 1)
    n = P['n']
    guided = 'Gdisc' in P
    state = []
    for _ in temps:
        ll = -np.inf
        if start is not None:
            s, u = np.array(start[0]), _compact(np.array(start[1]))
            ll, chi2, v, lab = ev(P, s, u, h)
        while not np.isfinite(ll):
            s = rng.choice(n, m_start, replace=False)
            u = _compact(rng.integers(0, max(1, m_start//2), m_start))
            ll, chi2, v, lab = ev(P, s, u, h)
        state.append([s, u, ll, chi2, v, lab])
    acc = {k: [0, 0] for k in KINDS + ('swap',)}
    out = []
    for it in range(n_sweeps):
        for ci, T in enumerate(temps):
            for _ in range(sweep):
                s, u, ll0 = state[ci][0], state[ci][1], state[ci][2]
                qfun = (lambda st=state[ci]: birth_q(P, st[0], st[4], st[5])) if guided else None
                kind, ns, nu, extra, post = propose(P, s, u, rng, m_max, logS, qfun)
                if ci == 0:
                    acc[kind][1] += 1
                if extra is None:
                    continue
                ll, chi2, v, lab = ev(P, ns, nu, h)
                if post is not None:                    # reverse of a guided birth into the old state
                    if not np.isfinite(ll):
                        continue
                    extra += np.log(birth_q(P, ns, v, lab)[post]) + np.log(n - len(s) + 1)
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


def run(cfg, cache=None, log=print):
    cache = cache or Cache()
    t0 = time.time()
    say = lambda msg: log(f'[{(time.time() - t0)/60:5.1f} min] {msg}')
    pb = problem.local(cfg, cache, say)
    cells, me = pb.cells, cfg['method']
    edges, _ = graph.cell_edges(cells)
    P = voronoi.setup(pb.Gw, cells.clon, cells.clat, pb.yw, me.get('bounds', [V_LO, V_HI]))
    lam95 = me.get('lambda_95', 0.)
    if lam95 == 'start':                                       # lambda_95 of the FD-merge result started from
        lam95 = results.load(me['start'][0]).attrs['lambda_95']
    P.update(edges=edges, gamma=float(me.get('gamma_rel', 0.))*float(lam95))
    if me.get('guided', True):
        setup_births(P)
    start = None
    if isinstance(me.get('start'), list):                     # [FD-merge result, setting index]
        ds = results.load(me['start'][0])
        start = seeds_from_partition(P, ds[f'domain_{int(me["start"][1])}'].values.astype(int))
    temps = voronoi.temperatures(me.get('temps', [1., 200., 10]))
    seed = int(me.get('seed', 0))

    def chain(c):
        t = time.time()
        out, acc = sample(P, int(me.get('sweeps', 2500)), temps, np.random.default_rng(seed + c),
                          m_max=int(me.get('m_max', 60)), m_start=int(me.get('m_start', 6)), thin=int(me.get('thin', 5)),
                          sweep=int(me.get('sweep', 20)), edges=edges, start=start if c == 0 else None)
        return dict(K=np.array([o['U'] for o in out]), label=np.array([o['label'] for o in out]),
                    v=[o['v'] for o in out], acc=acc, seconds=time.time() - t)
    res = sampling.chains(chain, int(me.get('chains', 2)), int(cfg['compute'].get('workers', 3)))
    burn = me.get('burn', 1/3)
    sums = [sampling.summary(r['label'], r['v'], r['K'], edges, burn) for r in res]
    for c, (r, s) in enumerate(zip(res, sums)):
        say(f'chain {c}: units mode {s["K_mode"]}; acceptance ' + ', '.join(f'{k} {a:.2f}' for k, a in r['acc'].items())
            + f'; {r["seconds"]/60:.1f} min')
    cut = lambda r: int(len(r['K'])*burn)
    allc = sampling.summary(np.concatenate([r['label'][cut(r):] for r in res]), sum([r['v'][cut(r):] for r in res], []),
                            np.concatenate([r['K'][cut(r):] for r in res]), edges, 0.)
    cell_vars = dict(vis=allc['vis'], vis_sd=allc['vis_sd'])
    if pb.truth is not None:
        cell_vars['truth'] = pb.truth
    edge_vars = dict(boundary_probability=allc['boundary_probability'],
                     **{f'boundary_probability_{c}': s['boundary_probability'] for c, s in enumerate(sums)})
    attrs = dict(K_hist=allc['K_hist'], K_mode=allc['K_mode'], chains=[dict(K_hist=s['K_hist'], K_mode=s['K_mode'],
                 acceptance=r['acc']) for r, s in zip(res, sums)], **pb.info)
    model = pb.G @ allc['vis']
    return results.make(cfg, cells, cell_vars, points=(pb.plon, pb.plat),
                        point_vars=dict(data=pb.y, model=model, residual=pb.y - model), edges=edges,
                        edge_vars=edge_vars, **attrs)
