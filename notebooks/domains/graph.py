"""
The cell graph: Laplacian and difference operator on the edges between touching cells.

Distances are great-circle distances between cell centres (km), so the difference operator is a
gradient on the sphere: row e of D is (v_i - v_j)/d_ij, and D'D is the Laplacian with weights 1/d_ij^2.
"""
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

R_KM = 6371.0


def laplacian(edges, w=None, n=None):
    """dense graph Laplacian with edge weights w (default 1)"""
    edges = np.asarray(edges)
    n = int(edges.max()) + 1 if n is None else n
    w = np.ones(len(edges)) if w is None else np.asarray(w, float)
    i, j = edges[:, 0], edges[:, 1]
    L = np.zeros((n, n))
    np.add.at(L, (i, j), -w)
    np.add.at(L, (j, i), -w)
    np.add.at(L, (i, i), w)
    np.add.at(L, (j, j), w)
    return L


def edge_lengths_km(edges, clon, clat):
    """great-circle distance between the centres of the two cells of each edge"""
    from .basis import unit
    u = unit(clon, clat)
    dot = np.sum(u[edges[:, 0]]*u[edges[:, 1]], 1)
    return R_KM*np.arccos(np.clip(dot, -1., 1.))


def difference_operator(edges, clon, clat):
    """sparse D (n_edges x n_cells), (D v)_e = (v_i - v_j)/d_ij in 1/km"""
    edges = np.asarray(edges)
    d = edge_lengths_km(edges, clon, clat)
    m = len(edges)
    rows = np.repeat(np.arange(m), 2)
    cols = edges.ravel()
    vals = np.stack([1/d, -1/d], 1).ravel()
    return csr_matrix(coo_matrix((vals, (rows, cols)), shape=(m, len(clon))))
