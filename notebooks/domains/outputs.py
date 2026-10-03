"""
Common outputs of the three methods, in the same form so they can be compared directly.

  cell-graph edges        pairs of cells that touch along a 0.1 deg grid edge, plus bridges that
                          join islands to the nearest cell (great circle) of another piece
  boundary probability    for each edge, the fraction of models (bootstrap reruns, starts or
                          posterior samples, optionally weighted) in which it separates two domains
  trade-off               misfit against K for each method and setting, with P(K) for the sampler
"""
import numpy as np
from scipy.spatial import cKDTree

from .basis import unit


def cell_edges(cell, clon, clat):
    """edges of the cell graph (n_edge x 2, i < j) and the bridges added to connect it"""
    n = cell.max() + 1
    pairs = []
    for a1, a2 in [(cell[:, :-1], cell[:, 1:]), (cell[:-1, :], cell[1:, :])]:
        m = (a1 >= 0) & (a2 >= 0) & (a1 != a2)
        pairs.append(np.stack([np.minimum(a1[m], a2[m]), np.maximum(a1[m], a2[m])], 1))
    # the 0 / 360 deg seam of the grid
    a1, a2 = cell[:, -1], cell[:, 0]
    m = (a1 >= 0) & (a2 >= 0) & (a1 != a2)
    pairs.append(np.stack([np.minimum(a1[m], a2[m]), np.maximum(a1[m], a2[m])], 1))
    edges = np.unique(np.vstack(pairs), axis=0)

    def components(e):
        parent = np.arange(n)
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, j in e:
            parent[find(i)] = find(j)
        return np.array([find(i) for i in range(n)])

    xyz = unit(clon, clat)
    bridges = []
    comp = components(edges)
    while len(np.unique(comp)) > 1:
        small = min(np.unique(comp), key=lambda c: (comp == c).sum())
        inside, outside = np.where(comp == small)[0], np.where(comp != small)[0]
        dist, j = cKDTree(xyz[outside]).query(xyz[inside])
        i = np.argmin(dist)
        bridges.append((min(inside[i], outside[j[i]]), max(inside[i], outside[j[i]])))
        comp = components(np.vstack([edges, bridges]))
    bridges = np.array(bridges, dtype=edges.dtype).reshape(-1, 2)
    return np.unique(np.vstack([edges, bridges]), axis=0), bridges


def boundary_probability(labels, edges, weights=None):
    """fraction (weighted) of the partitions in labels (n_model x n_cell) that cut each edge"""
    labels = np.atleast_2d(labels)
    cut = labels[:, edges[:, 0]] != labels[:, edges[:, 1]]
    return np.average(cut, axis=0, weights=weights)


def tradeoff_record(method, setting, K, misfit, **extra):
    """one row of a region's trade-off table: misfit (chi2 of whitened data) against K"""
    return dict(method=method, setting=setting, K=np.asarray(K), misfit=np.asarray(misfit), **extra)


def ari(x, y):
    """adjusted Rand index of two partitions of the same cells (1 = identical, 0 = chance)"""
    from scipy.special import comb
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    ct = np.zeros((xi.max()+1, yi.max()+1))
    np.add.at(ct, (xi, yi), 1)
    s = comb(ct, 2).sum()
    sa, sb = comb(ct.sum(1), 2).sum(), comb(ct.sum(0), 2).sum()
    expected = sa*sb/comb(len(x), 2)
    return (s - expected)/(0.5*(sa + sb) - expected)
