"""
The cell graph: which cells touch, and the operators on it used by smoothness priors and penalties.

  edges        pairs of cells that touch along a 0.1 deg grid edge (across the 0/360 deg seam too), plus bridges
               joining each island to the nearest cell (great circle between centres) of another piece, so the
               graph is connected
  difference   D (edges x cells): (D v)_e = (v_i - v_j)/d_ij, d_ij the great-circle distance between the two
               centres (km), so D is a gradient on the sphere and D'D the Laplacian with weights 1/d^2
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

R_KM = 6371.0


def unit(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], -1)


def _components(edges, n):
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in edges:
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


def cell_edges(cells):
    """edges (n_edges x 2, i < j) of the cells' graph, and the bridges among them"""
    cell, n = cells.cell, len(cells)
    pairs = []
    # neighbours along rows and columns of the grid, and across the seam: 359.9 deg against 0 (masks never hold
    # the 360 deg column, which repeats 0)
    for a1, a2 in [(cell[:, :-1], cell[:, 1:]), (cell[:-1, :], cell[1:, :]), (cell[:, -2], cell[:, 0])]:
        m = (a1 >= 0) & (a2 >= 0) & (a1 != a2)
        pairs.append(np.stack([np.minimum(a1[m], a2[m]), np.maximum(a1[m], a2[m])], -1).reshape(-1, 2))
    edges = np.unique(np.vstack(pairs), axis=0)
    xyz = unit(cells.clon, cells.clat)
    bridges = []
    comp = _components(edges, n)
    while len(np.unique(comp)) > 1:
        small = min(np.unique(comp), key=lambda c: (comp == c).sum())
        inside, outside = np.flatnonzero(comp == small), np.flatnonzero(comp != small)
        dist, j = cKDTree(xyz[outside]).query(xyz[inside])
        i = np.argmin(dist)
        bridges.append((min(inside[i], outside[j[i]]), max(inside[i], outside[j[i]])))
        comp = _components(np.vstack([edges, bridges]), n)
    bridges = np.array(bridges, dtype=edges.dtype).reshape(-1, 2)
    return np.unique(np.vstack([edges, bridges]), axis=0), bridges


def edge_lengths_km(edges, clon, clat):
    u = unit(clon, clat)
    return R_KM*np.arccos(np.clip(np.sum(u[edges[:, 0]]*u[edges[:, 1]], 1), -1., 1.))


def laplacian(edges, w, n):
    """dense graph Laplacian with edge weights w"""
    i, j = edges[:, 0], edges[:, 1]
    L = np.zeros((n, n))
    np.add.at(L, (i, j), -w)
    np.add.at(L, (j, i), -w)
    np.add.at(L, (i, i), w)
    np.add.at(L, (j, j), w)
    return L


def difference_operator(edges, clon, clat):
    d = edge_lengths_km(edges, clon, clat)
    rows = np.repeat(np.arange(len(edges)), 2)
    vals = np.stack([1/d, -1/d], 1).ravel()
    return csr_matrix(coo_matrix((vals, (rows, edges.ravel())), shape=(len(edges), len(clon))))
