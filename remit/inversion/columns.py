"""
Cell columns: the Gauss coefficients (packed g, h by k = l(l+1)/2 + m, referenced to R0, nT per SI.km) of 1 SI.km
of VIS spread uniformly through a layer in one cell, magnetised along a direction model.

  node sum    the finite sum of the global transform (Gubbins et al. 2011 eq. 32-34, Driscoll-Healy weights)
              restricted to the cell's own grid nodes, vectorised as one matrix product per (cell, grid row)
              (docs/benchmark-stage0-speed.md). For row i and the cell's nodes j in it, with
                  X_j = (mr, mt, mp)_j rad_j(l),   T_j = (cos, sin)(m phi_j),   Q = sum_j X_j T_j',
                  g_lm += w_i (l p_lm Q_rc + dp_lm Q_tc) - wsin_i m p_lm Q_ps
                  h_lm += w_i (l p_lm Q_rs + dp_lm Q_ts) + wsin_i m p_lm Q_pc
              where rad_j(l) = sum_q wq_q/2 (r_q/R0)^(l+1) over Gauss-Legendre radii between the layer's top and
              base, so the radial factor is exact. Backends: 'numpy' (float64), 'mps' (torch on the Apple GPU,
              float32, ADR 0005) and 'torch-cpu' (float64, for checks).
  transform   one global remit transform per cell (depth.slab_vis_coeffs): the reference, slow.

ColumnStore keeps the columns of all cells of a mask in the cache (ADR 0004), in blocks of cells, keyed by
everything that determines them, including the backend's precision.
"""
import os
import time

import numpy as np
import pyshtools

from remit import depth
from remit.vhtools import _dh_weights
from . import geometry

N_GL = 8                                  # Gauss-Legendre radii per layer: exact to rounding at lmax 185
VERSION = 2                               # of the column algorithm; part of the cache key, so a fix never reuses old columns


def packed_lm(lmax):
    """degree and order of each packed (l, m) index (k = l(l+1)/2 + m, as pyshtools.legendre)"""
    return (np.concatenate([np.full(l + 1, l) for l in range(lmax + 1)]),
            np.concatenate([np.arange(l + 1) for l in range(lmax + 1)]))


class Setup:
    """what the node sums need, computed once: magnetisation per SI.km, layer radii, quadrature, Legendre rows"""

    def __init__(self, layer, direction, lmax, mask=None, g=None):
        g = g or geometry.grid()
        self.lmax = lmax
        self.L, self.M = packed_lm(lmax)
        self.comps = geometry.direction_components(direction, g)
        self.r_top, self.r_base = geometry.layer_radii(layer, mask, g)
        self.nth = len(g.lat) - 1                     # DH rows: north pole to one row short of the south pole
        self.w, self.wsin = _dh_weights(np.arange(self.nth)*np.pi/self.nth)
        self.dphi = 2*np.pi/(2*self.nth)
        self.x, self.wq = np.polynomial.legendre.leggauss(N_GL)
        self.K = 1e9*pyshtools.constants.mu0.value/(4*np.pi*depth.R0)
        self._legendre = {}

    def legendre(self, rows):
        """Schmidt P_lm and dP_lm/dtheta (packed) at grid rows: (len(rows), 2, n_lm)"""
        for i in rows:
            if i not in self._legendre:
                th = i*np.pi/self.nth
                p, dp = pyshtools.legendre.PlmSchmidt_d1(self.lmax, np.cos(th))
                self._legendre[i] = np.stack([p, -dp*np.sin(th)])
        return np.stack([self._legendre[i] for i in rows])


def _nodes(cell, idx):
    """grid nodes of the cells idx, sorted by (position in idx, row), and the (cell, row) groups"""
    pos = np.full(cell.max() + 1, -1)
    pos[idx] = np.arange(len(idx))
    rows, cols = np.nonzero(np.isin(cell, idx))
    if (rows == cell.shape[0] - 1).any() or (cols == cell.shape[1] - 1).any():
        raise ValueError('cells touch the south-pole row or the 360 deg column')
    cp = pos[cell[rows, cols]]
    o = np.lexsort((rows, cp))
    rows, cols, cp = rows[o], cols[o], cp[o]
    key = cp.astype(np.int64)*cell.shape[0] + rows
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    stop = np.r_[start[1:], len(rows)]
    return rows, cols, start, stop, cp[start], rows[start]


def node_sum(cell, idx, S, backend='numpy'):
    """columns (2, len(idx), n_lm) of the cells idx of the cell grid, by node sum"""
    rows, cols, start, stop, gcell, grow = _nodes(cell, np.asarray(idx))
    urows = np.unique(grow)
    P = S.legendre(urows)
    prow = np.searchsorted(urows, grow)                   # each group's row in P
    args = (rows, cols, start, stop, gcell, prow, P, S, len(idx))
    if backend == 'numpy':
        return _node_sum_numpy(*args)
    if backend in ('mps', 'torch-cpu'):
        return _node_sum_torch(*args, device='mps' if backend == 'mps' else 'cpu')
    raise ValueError(f'columns backend {backend!r}')


def _node_inputs(xp, rows, cols, S, dtype):
    """per node: X (n, 3(l+1)) magnetisation x radial factor, T (n, 2(l+1)) cos and sin of m phi"""
    l = xp.arange(S.lmax + 1, dtype=dtype) + 1.
    rt = xp.asarray(S.r_top[rows, cols], dtype=dtype)/depth.R0
    rb = xp.asarray(S.r_base[rows, cols], dtype=dtype)/depth.R0
    rad = 0.
    for xq, wq in zip(S.x, S.wq):
        rad = rad + wq/2*xp.exp(l[None, :]*xp.log(rt + (rb - rt)*(xq + 1)/2)[:, None])
    X = xp.concatenate([xp.asarray(c[rows, cols], dtype=dtype)[:, None]*rad for c in S.comps], 1)
    # m phi reduced exactly in integers first ((col m) mod 2 nth): float32 trig of arguments up to ~1000 rad
    # would lose ~1e-4 of absolute accuracy
    ph = (np.asarray(cols, np.int64)[:, None]*np.arange(S.lmax + 1)[None, :]) % (2*S.nth)
    ph = xp.asarray(ph, dtype=dtype)*S.dphi
    return X, xp.concatenate([xp.cos(ph), xp.sin(ph)], 1)


def _node_sum_numpy(rows, cols, start, stop, gcell, prow, P, S, n_cells):
    X, T = _node_inputs(np, rows, cols, S, np.float64)
    n, L, M = S.lmax + 1, S.L, S.M
    out = np.zeros((2, n_cells, len(L)))
    for s, e, c, k, i in zip(start, stop, gcell, prow, rows[start]):
        Q = X[s:e].T @ T[s:e]
        pk = lambda a, b: Q[a*n + L, b*n + M]
        p, dp = P[k]
        out[0, c] += S.w[i]*(L*p*pk(0, 0) + dp*pk(1, 0)) - S.wsin[i]*M*p*pk(2, 1)
        out[1, c] += S.w[i]*(L*p*pk(0, 1) + dp*pk(1, 1)) + S.wsin[i]*M*p*pk(2, 0)
    out *= S.K*S.dphi
    out[:, :, L == 0] = 0.
    return out


def _node_sum_torch(rows, cols, start, stop, gcell, prow, P, S, n_cells, device, chunk=512):
    import torch
    dtype = torch.float32 if device == 'mps' else torch.float64

    class TorchNP:                                           # the numpy calls _node_inputs makes, on the device
        @staticmethod
        def asarray(a, dtype=None):
            return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)
        arange = staticmethod(lambda n, dtype=None: torch.arange(n, dtype=dtype, device=device))
        exp, log, cos, sin = torch.exp, torch.log, torch.cos, torch.sin
        concatenate = staticmethod(lambda xs, axis: torch.cat(xs, axis))

    X, T = _node_inputs(TorchNP, rows, cols, S, dtype)
    t = lambda a, dt=dtype: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
    Pt, wt, wst = t(P), t(S.w), t(S.wsin)
    Li, Mi = t(S.L, torch.long), t(S.M, torch.long)
    Lf, Mf = t(S.L), t(S.M)
    n = S.lmax + 1
    out = torch.zeros((2, n_cells, len(S.L)), dtype=dtype, device=device)
    size = stop - start
    width = np.maximum(8, 2**np.ceil(np.log2(size)).astype(int))
    grows = rows[start]
    for wd in np.unique(width):
        gs = np.flatnonzero(width == wd)
        for k in range(0, len(gs), chunk):
            b = gs[k:k + chunk]
            idx = start[b][:, None] + np.arange(wd)[None, :]
            valid = idx < stop[b][:, None]
            idx_t = t(np.where(valid, idx, 0), torch.long)
            mask = t(valid)[..., None]
            Q = torch.bmm((X[idx_t]*mask).transpose(1, 2), T[idx_t]*mask)       # (G, 3n, 2n)
            pk = lambda a, c: Q[:, a*n + Li, c*n + Mi]
            pr = t(prow[b], torch.long)
            p, dp = Pt[pr, 0], Pt[pr, 1]
            i = t(grows[b], torch.long)
            wi, wsi = wt[i][:, None], wst[i][:, None]
            c = t(gcell[b], torch.long)
            out[0].index_add_(0, c, wi*(Lf*p*pk(0, 0) + dp*pk(1, 0)) - wsi*Mf*p*pk(2, 1))
            out[1].index_add_(0, c, wi*(Lf*p*pk(0, 1) + dp*pk(1, 1)) + wsi*Mf*p*pk(2, 0))
    out *= S.K*S.dphi
    out[:, :, Li == 0] = 0.
    if device == 'mps':
        torch.mps.synchronize()
    return out.cpu().double().numpy()


def transform_columns(cell, idx, S, g=None):
    """columns of the cells idx by one global transform each (the reference; about 10 s per cell)"""
    g = g or geometry.grid()
    to_gmm = depth.induced_to_gmm(g.lon, g.lat)
    out = np.zeros((2, len(idx), len(S.L)))
    for k, i in enumerate(idx):
        c = depth.slab_vis_coeffs(np.where(cell == i, 1., 0.), S.r_top, S.r_base, to_gmm, lmax=S.lmax).coeffs
        out[:, k] = c[:, S.L, S.M]
    return out


def resolve_backend(backend):
    """the backend actually available: 'mps' falls back to 'numpy' without torch or the Apple GPU"""
    if backend != 'mps':
        return backend
    try:
        import torch
        if torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    print('columns: the GPU (MPS) is not available, using numpy float64')
    return 'numpy'


class ColumnStore:
    """columns of all cells of a mask, in the cache, built in blocks of cells (resumable)"""

    def __init__(self, cells, mask_name, layer, direction, lmax, backend, cache, block=200):
        self.cells, self.lmax, self.block = cells, lmax, block
        self.backend = resolve_backend(backend)
        self.spec = dict(grid=geometry.AGE_GRID, sediment=os.path.basename(depth.SEDIMENT_GRID),
                         relief='earth_relief_06m', mask=mask_name, nside=cells.nside, n_cells=len(cells),
                         layer=layer, direction=direction, lmax=lmax, n_gl=N_GL, block=block, version=VERSION,
                         precision='float32' if self.backend == 'mps' else 'float64')
        self.layer, self.direction = layer, direction
        self.dir = cache.entry('cell_columns', self.spec)
        self.blocks = [np.arange(k, min(k + block, len(cells))) for k in range(0, len(cells), block)]

    def _file(self, b):
        return os.path.join(self.dir, f'{b:05d}.npy')

    def build(self, log=print):
        todo = [b for b in range(len(self.blocks)) if not os.path.exists(self._file(b))]
        if not todo:
            return self
        log(f'cell columns: {len(todo)} of {len(self.blocks)} blocks to compute ({self.backend}) in {self.dir}')
        S = Setup(self.layer, self.direction, self.lmax, self.cells.cell >= 0)
        t0 = time.time()
        for n, b in enumerate(todo, 1):
            cols = node_sum(self.cells.cell, self.blocks[b], S, self.backend)
            from .cache import save_atomic
            save_atomic(self._file(b), lambda p: np.save(p, cols))
            if n % 10 == 0 or n == len(todo):
                log(f'  {n}/{len(todo)} blocks, {(time.time() - t0)/60:.1f} min')
        return self

    def get(self, idx, lmax=None):
        """columns (2, len(idx), n_lm) of cell indices idx, to degree lmax (default the store's)"""
        idx = np.asarray(idx)
        nk = (lmax + 1)*(lmax + 2)//2 if lmax is not None else None
        out = None
        for b in np.unique(idx//self.block):
            sel = np.flatnonzero(idx//self.block == b)
            a = np.load(self._file(b), mmap_mode='r')
            if out is None:
                out = np.zeros((2, len(idx), nk or a.shape[2]))
            out[:, sel] = a[:, idx[sel] % self.block, :out.shape[2]]
        return out
