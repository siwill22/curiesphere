"""
What the domain samplers (FD-vor, FD-lvor, FD-part) share: running chains in parallel, and summarising their samples
as maps and boundary probabilities.

  chains     independent chains, each with its own seed (seed + chain number), in a pool of worker processes
  summary    after discarding a burn-in fraction: the posterior mean and s.d. of the VIS of each cell (over samples of
             the domain values), the histogram of the number of domains, and the probability that each edge of the
             cell graph separates two domains
"""
import multiprocessing

import numpy as np

_job = {}


def chains(fn, n_chains, workers=3):
    """[fn(0), fn(1), ...] in a pool (fork: fn may be a closure over large arrays)"""
    _job['fn'] = fn
    if workers <= 1 or n_chains == 1:
        return [fn(c) for c in range(n_chains)]
    with multiprocessing.get_context('fork').Pool(min(workers, n_chains)) as pool:
        return pool.map(_call, range(n_chains))


def _call(c):
    return _job['fn'](c)


def boundary_probability(labels, edges):
    labels = np.atleast_2d(labels)
    return np.mean(labels[:, edges[:, 0]] != labels[:, edges[:, 1]], axis=0)


def summary(labels, values, K, edges, burn=1/3):
    """labels (samples x cells), values (list of per-domain value arrays), K (samples): posterior summaries"""
    k0 = int(len(K)*burn)
    labs, vals, Ks = np.asarray(labels[k0:]), values[k0:], np.asarray(K[k0:])
    maps = np.array([np.asarray(v)[l] for v, l in zip(vals, labs)])
    return dict(vis=maps.mean(0), vis_sd=maps.std(0), K_hist=np.bincount(Ks).tolist(), K_mode=int(np.bincount(Ks).argmax()),
                boundary_probability=boundary_probability(labs, edges), n_samples=len(Ks))
