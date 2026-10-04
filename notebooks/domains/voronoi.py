"""
Methods (b) and (c): Voronoi domains, found by annealing (b) or sampled by transdimensional MCMC (c).

Model
  K seeds, each on the centre of one fine cell (the seed positions are a set of K distinct cells);
  every cell belongs to its nearest seed (great-circle distance, i.e. largest dot product of unit
  vectors), so each domain is a spherical Voronoi region made of whole cells.
Data and likelihood
  dw = T (d - b): data in the compressed, weighted space (A2: point weighting, sigma = 1 unit);
  dw ~ N(Gw A v, h^2 I), with A the cell-to-domain indicator.
VIS values v are integrated out analytically: with a uniform prior on [V_LO, V_HI]^K and the
  Gaussian integral taken over all of R^K (a layout whose least-squares values fall outside the
  bounds is rejected, i.e. given zero prior mass),
      log p(dw | layout, h) = -K log(V_HI - V_LO) - (N-K)/2 log(2 pi h^2) - 1/2 log|GR'GR| - chi2/(2 h^2)
  with GR = Gw A and chi2 the least-squares misfit. With h unknown (Jeffreys prior p(h) ~ 1/h) h is
  integrated out too: log p = -K log(V_HI - V_LO) - 1/2 log|GR'GR| + log Gamma((N-K)/2)
  - (N-K)/2 log(pi chi2) (constants in K dropped consistently).
Priors
  K uniform on 1..K_MAX; given K, the seed set uniform over all sets of K cells.
Moves (each step one of them, with equal probability; values never proposed)
  birth  a new seed on a free cell drawn uniformly         acceptance ratio = likelihood ratio
  death  a seed drawn uniformly is removed                 acceptance ratio = likelihood ratio
  move   a seed moves to a free cell within R_STEP km (an adjacent cell; probability 1/2), within
         R_LOCAL km (3/10) or anywhere (1/5); each local kernel's acceptance includes the ratio
         of the neighbourhood sizes (each kernel satisfies detailed balance on its own)
  (with the uniform priors on K and on seed sets, the birth/death prior and proposal ratios cancel:
  prior (K+1)/(n-K) x proposal (n-K)/(K+1) = 1)
Parallel tempering: chains at temperatures T_i see likelihood^(1/T_i); neighbours swap after each
  sweep with the usual Metropolis rule. Samples are taken from the T = 1 chain.
Annealing (method b): moves only, at fixed K, temperature lowered geometrically on chi2.
"""
import numpy as np
from scipy.special import gammaln

V_LO, V_HI = 0., 4.
R_LOCAL_KM = 400.
R_STEP_KM = 150.
MOVE_MIX = ((0.5, 'step'), (0.3, 'local'), (0.2, 'global'))
R_KM = 6371.


def setup(Gw, clon, clat, dw):
    xyz = np.stack([np.cos(np.radians(clat))*np.cos(np.radians(clon)),
                    np.cos(np.radians(clat))*np.sin(np.radians(clon)), np.sin(np.radians(clat))], -1)
    dots = xyz @ xyz.T
    near = {'local': [np.where(r >= np.cos(R_LOCAL_KM/R_KM))[0] for r in dots],
            'step': [np.where(r >= np.cos(R_STEP_KM/R_KM))[0] for r in dots]}
    return dict(Gw=Gw, xyz=xyz, near=near, n=Gw.shape[1], N=Gw.shape[0], dw=dw)


def assign(P, seeds):
    """domain index (0..K-1) of every cell: its nearest seed"""
    return np.argmax(P['xyz'] @ P['xyz'][seeds].T, axis=1)


def evaluate(P, seeds, h=None):
    """log marginal likelihood (h fixed, or h integrated out if None), chi2 and LSQ values"""
    lab = assign(P, seeds)
    K = len(seeds)
    GR = np.stack([P['Gw'][:, lab == k].sum(1) for k in range(K)], 1)
    M = GR.T @ GR
    try:
        cf = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return -np.inf, np.inf, None, lab
    v = np.linalg.solve(M, GR.T @ P['dw'])
    if (v < V_LO).any() or (v > V_HI).any():
        return -np.inf, np.inf, v, lab
    chi2 = float(np.sum((GR @ v - P['dw'])**2))
    logdet = 2*np.sum(np.log(np.diag(cf)))
    N = P['N']
    base = -K*np.log(V_HI - V_LO) - 0.5*logdet
    if h is None:
        ll = base + gammaln((N - K)/2) - (N - K)/2*np.log(np.pi*chi2)
    else:
        ll = base - (N - K)/2*np.log(2*np.pi*h**2) - chi2/(2*h**2)
    return float(ll), chi2, v, lab


def propose(P, seeds, rng, k_max, moves=('birth', 'death', 'move')):
    """one proposal: new seed array and log proposal-ratio correction (0 except for local moves)"""
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
        # q(old -> new) = 1/|free cells near old|, q(new -> old) = 1/|free cells near new| (the old
        # position is free after the move and, distances being symmetric, near the new one)
        back = np.setdiff1d(near[new[i]], new)
        return 'move', new, float(np.log(len(cand)) - np.log(len(back)))
    return kind, seeds.copy(), 0.


def sample(P, n_sweeps, temps, rng, k_max=40, k_start=5, h=None, thin=10, sweep=20):
    """parallel-tempered transdimensional sampler; returns the T = 1 samples (every `thin` sweeps)
    as dicts with K, seeds, labels, values, chi2, log-likelihood; and acceptance rates"""
    n = P['n']
    state = []
    for _ in temps:
        s = rng.choice(n, k_start, replace=False)
        ll, chi2, v, lab = evaluate(P, s, h)
        while not np.isfinite(ll):
            s = rng.choice(n, k_start, replace=False)
            ll, chi2, v, lab = evaluate(P, s, h)
        state.append([s, ll, chi2, v, lab])
    acc = {k: [0, 0] for k in ('birth', 'death', 'move', 'swap')}
    out = []
    for it in range(n_sweeps):
        for c, T in enumerate(temps):
            for _ in range(sweep):
                kind, new, corr = propose(P, state[c][0], rng, k_max)
                ll, chi2, v, lab = evaluate(P, new, h)
                if c == 0:
                    acc[kind][1] += 1
                if np.isfinite(ll) and np.log(rng.random()) < (ll - state[c][1])/T + corr:
                    state[c] = [new, ll, chi2, v, lab]
                    if c == 0:
                        acc[kind][0] += 1
        for c in range(len(temps) - 1):                         # swaps between neighbouring temperatures
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
    """method (b): minimise chi2 at fixed K by simulated annealing with move proposals only.
    t0 defaults to 1% of the chi2 of a random start; returns (chi2, seeds, labels, values) of the best"""
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
