"""
FD-vor (free domains as Voronoi regions, CONTEXT.md), from a local fit: found by annealing or sampled by
transdimensional MCMC.

Model
  K seeds, each on the centre of one cell (a set of K distinct cells); every cell belongs to its nearest seed (great
  circle: the largest dot product of unit vectors), so each domain is a spherical Voronoi region made of whole cells.
Likelihood: yw ~ N(Gw A v, h^2 I), A the cell-to-domain indicator; the values v (uniform on [lo, hi]) and h (Jeffreys)
  are integrated out analytically (core.box_evidence); a layout whose least-squares values leave the box gets zero mass.
Priors: K uniform on 1..k_max; given K, the seed set uniform over all sets of K cells.
Moves (one per step, equally likely; values never proposed)
  birth  a new seed on a free cell drawn uniformly            acceptance ratio = likelihood ratio
  death  a seed drawn uniformly is removed                    acceptance ratio = likelihood ratio
  move   a seed moves to a free cell within R_STEP km (an adjacent cell; probability 1/2), within R_LOCAL km (3/10)
         or anywhere (1/5); each local kernel's acceptance includes the ratio of the neighbourhood sizes
  (with the uniform priors on K and on seed sets the birth/death prior and proposal ratios cancel)
Parallel tempering: chains at temperatures T_i see likelihood^(1/T_i); neighbours swap after each sweep; samples
  come from the T = 1 chain.
Annealing: moves only, at fixed K, temperature lowered geometrically on chi2.

Config keys used, besides those of problem.local:
  [method] temps ([T_min, T_max, n]), sweeps, sweep, thin, k_max, k_start, chains, seed, burn, bounds;
  [compute] workers
"""
import time

import numpy as np

from .. import core, graph, problem, results, sampling
from ..cache import Cache

R_LOCAL_KM = 400.
R_STEP_KM = 150.
MOVE_MIX = ((0.5, 'step'), (0.3, 'local'), (0.2, 'global'))
R_KM = 6371.
V_LO, V_HI = 0., 4.


def setup(Gw, clon, clat, dw, bounds=(V_LO, V_HI)):
    xyz = graph.unit(clon, clat)
    dots = xyz @ xyz.T
    near = {'local': [np.where(r >= np.cos(R_LOCAL_KM/R_KM))[0] for r in dots],
            'step': [np.where(r >= np.cos(R_STEP_KM/R_KM))[0] for r in dots]}
    return dict(Gw=Gw, xyz=xyz, near=near, n=Gw.shape[1], N=Gw.shape[0], dw=dw, bounds=tuple(bounds))


def assign(P, seeds):
    """domain index (0..K-1) of every cell: its nearest seed"""
    return np.argmax(P['xyz'] @ P['xyz'][seeds].T, axis=1)


def evaluate(P, seeds, h=None):
    """log marginal likelihood (h fixed, or integrated out if None), chi2, least-squares values, labels"""
    lab = assign(P, seeds)
    GR = np.stack([P['Gw'][:, lab == k].sum(1) for k in range(len(seeds))], 1)
    ll, chi2, v = core.box_evidence(GR, P['dw'], *P.get('bounds', (V_LO, V_HI)), h)
    return ll, chi2, v, lab


def propose(P, seeds, rng, k_max, moves=('birth', 'death', 'move')):
    """one proposal: kind, new seed array and log proposal-ratio correction (0 except for local moves)"""
    K, n = len(seeds), P['n']
    kind = moves[rng.integers(len(moves))]
    if kind == 'birth' and K < k_max:
        free = np.setdiff1d(np.arange(n), seeds)
        return kind, np.append(seeds, rng.choice(free)), 0.
    if kind == 'death' and K > 1:
        return kind, np.delete(seeds, rng.integers(K)), 0.
    if kind == 'move' or kind in ('birth', 'death'):
        i = rng.integers(K)
        new = seeds.copy()
        u, acc_p = rng.random(), 0.
        for prob, kernel in MOVE_MIX:
            acc_p += prob
            if u < acc_p:
                break
        if kernel == 'global':
            new[i] = rng.choice(np.setdiff1d(np.arange(n), seeds))
            return 'move', new, 0.
        near = P['near'][kernel]
        cand = np.setdiff1d(near[seeds[i]], seeds)
        if len(cand) == 0:
            return 'move', seeds.copy(), 0.
        new[i] = rng.choice(cand)
        # q(old -> new) = 1/|free cells near old|, q(new -> old) = 1/|free cells near new|
        back = np.setdiff1d(near[new[i]], new)
        return 'move', new, float(np.log(len(cand)) - np.log(len(back)))
    return kind, seeds.copy(), 0.


def sample(P, n_sweeps, temps, rng, k_max=40, k_start=5, h=None, thin=10, sweep=20, evaluate_fn=None):
    """parallel-tempered transdimensional sampler; the T = 1 samples (every `thin` sweeps) as dicts with K, seeds,
    labels, values, chi2 and log-likelihood; and acceptance rates. evaluate_fn replaces evaluate (prior tests)."""
    ev = evaluate if evaluate_fn is None else evaluate_fn
    n = P['n']
    state = []
    for _ in temps:
        s = rng.choice(n, k_start, replace=False)
        ll, chi2, v, lab = ev(P, s, h)
        while not np.isfinite(ll):
            s = rng.choice(n, k_start, replace=False)
            ll, chi2, v, lab = ev(P, s, h)
        state.append([s, ll, chi2, v, lab])
    acc = {k: [0, 0] for k in ('birth', 'death', 'move', 'swap')}
    out = []
    for it in range(n_sweeps):
        for c, T in enumerate(temps):
            for _ in range(sweep):
                kind, new, corr = propose(P, state[c][0], rng, k_max)
                ll, chi2, v, lab = ev(P, new, h)
                if c == 0:
                    acc[kind][1] += 1
                if np.isfinite(ll) and np.log(rng.random()) < (ll - state[c][1])/T + corr:
                    state[c] = [new, ll, chi2, v, lab]
                    if c == 0:
                        acc[kind][0] += 1
        for c in range(len(temps) - 1):
            a = (state[c+1][1] - state[c][1])*(1/temps[c] - 1/temps[c+1])
            acc['swap'][1] += 1
            if np.log(rng.random()) < a:
                state[c], state[c+1] = state[c+1], state[c]
                acc['swap'][0] += 1
        if it % thin == 0:
            s, ll, chi2, v, lab = state[0]
            out.append(dict(K=len(s), seeds=s.copy(), label=lab.copy(), v=v.copy(), chi2=chi2, ll=ll))
    return out, {k: a/max(t, 1) for k, (a, t) in acc.items()}


def anneal(P, K, n_steps, rng, t0=None, t1=0.1, restarts=1):
    """minimise chi2 at fixed K by simulated annealing with move proposals only. t0 defaults to 1% of the chi2 of a
    random start; returns (chi2, seeds, labels, values) of the best"""
    best = None
    for _ in range(restarts):
        seeds = rng.choice(P['n'], K, replace=False)
        _, chi2, v, lab = evaluate(P, seeds, h=1.)
        while not np.isfinite(chi2):
            seeds = rng.choice(P['n'], K, replace=False)
            _, chi2, v, lab = evaluate(P, seeds, h=1.)
        T0 = 0.01*chi2 if t0 is None else t0
        cur = (chi2, seeds, lab, v)
        for i in range(n_steps):
            T = T0*(t1/T0)**(i/max(n_steps - 1, 1)) if T0 > t1 else t1
            _, new, corr = propose(P, cur[1], rng, K, moves=('move',))
            _, c2, v, lab = evaluate(P, new, h=1.)
            if np.isfinite(c2) and (c2 < cur[0] or rng.random() < np.exp(-(c2 - cur[0])/T)):
                cur = (c2, new, lab, v)
                if best is None or c2 < best[0]:
                    best = cur
        if best is None or cur[0] < best[0]:
            best = cur
    return best


def temperatures(spec):
    lo, hi, n = spec
    return list(np.geomspace(float(lo), float(hi), int(n)))


def run(cfg, cache=None, log=print):
    cache = cache or Cache()
    t0 = time.time()
    say = lambda msg: log(f'[{(time.time() - t0)/60:5.1f} min] {msg}')
    pb = problem.local(cfg, cache, say)
    cells, me = pb.cells, cfg['method']
    edges, _ = graph.cell_edges(cells)
    P = setup(pb.Gw, cells.clon, cells.clat, pb.yw, me.get('bounds', [V_LO, V_HI]))
    temps = temperatures(me.get('temps', [1., 200., 10]))
    seed = int(me.get('seed', 0))

    def chain(c):
        t = time.time()
        out, acc = sample(P, int(me.get('sweeps', 2500)), temps, np.random.default_rng(seed + c),
                          k_max=int(me.get('k_max', 40)), k_start=int(me.get('k_start', 3)),
                          thin=int(me.get('thin', 5)), sweep=int(me.get('sweep', 20)))
        return dict(K=np.array([o['K'] for o in out]), label=np.array([o['label'] for o in out]),
                    v=[o['v'] for o in out], chi2=np.array([o['chi2'] for o in out]), acc=acc, seconds=time.time() - t)
    res = sampling.chains(chain, int(me.get('chains', 2)), int(cfg['compute'].get('workers', 3)))
    sums = [sampling.summary(r['label'], r['v'], r['K'], edges, me.get('burn', 1/3)) for r in res]
    for c, (r, s) in enumerate(zip(res, sums)):
        say(f'chain {c}: K mode {s["K_mode"]}; acceptance ' + ', '.join(f'{k} {a:.2f}' for k, a in r['acc'].items()) +
            f'; {r["seconds"]/60:.1f} min')
    allc = sampling.summary(np.concatenate([r['label'][int(len(r['K'])*me.get('burn', 1/3)):] for r in res]),
                            sum([r['v'][int(len(r['K'])*me.get('burn', 1/3)):] for r in res], []),
                            np.concatenate([r['K'][int(len(r['K'])*me.get('burn', 1/3)):] for r in res]), edges, 0.)
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
