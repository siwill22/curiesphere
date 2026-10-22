"""Stage 0a: node-sum cell columns, vectorised as one matrix product per (cell, grid row).
For a row i and the nodes j of a cell in it, with X_j = (mr, mt, mp)_j rad_j(l) and T_j = (cos, sin)(m phi_j):
    Q = sum_j X_j T_j'  (one product), then  g_lm += w_i (L p Q_rc + dp Q_tc) - wsin_i M p Q_ps,
                                              h_lm += w_i (L p Q_rs + dp Q_ts) + wsin_i M p Q_pc
the same finite sum as basis.direct_columns. Run in the benchmark venv (numpy/scipy on Accelerate, torch MPS)."""
import os, sys, time
import numpy as np
import torch

D = os.environ.get('STAGE0_DIR', os.path.expanduser('~/.cache/curiesphere/stage0'))
torch.set_num_threads(1)
z = np.load(f'{D}/inputs.npz')
cell, test, ref = z['cell'], z['test'], z['ref']
L, M = z['L'], z['M']
LMAX = int(L.max()); NL = len(L); l = np.arange(LMAX + 1)
R0, K, dphi = float(z['R0']), float(z['K']), float(z['dphi'])
w, wsin, x, wq = z['w'], z['wsin'], z['x'], z['wq']


def node_arrays(cells_sel):
    """nodes of the selected cells sorted by (cell position, row): per-node X (n, 3(l)), trig (n, 2(l)), groups"""
    pos = np.full(cell.max() + 1, -1); pos[cells_sel] = np.arange(len(cells_sel))
    rows, cols = np.nonzero(np.isin(cell, cells_sel))
    cp = pos[cell[rows, cols]]
    o = np.lexsort((rows, cp)); rows, cols, cp = rows[o], cols[o], cp[o]
    rt, rb = z['r_top'][rows, cols], z['r_base'][rows, cols]
    rad = np.zeros((len(rows), LMAX + 1))
    for xq, wq_ in zip(x, wq):
        rad += wq_/2*((rt + (rb - rt)*(xq + 1)/2)/R0)[:, None]**(l + 1.)[None, :]
    X = np.concatenate([z[c][rows, cols][:, None]*rad for c in ('mr', 'mt', 'mp')], 1)
    ph = cols[:, None]*dphi*l[None, :]
    T = np.concatenate([np.cos(ph), np.sin(ph)], 1)
    key = cp*10000 + rows
    start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    stop = np.r_[start[1:], len(rows)]
    return X, T, start, stop, cp[start], rows[start]


def combine(Q, p, dp, wi, wsi):
    n = LMAX + 1
    pick = lambda a, b: Q[..., a*n + L, b*n + M] if Q.ndim == 2 else Q[:, a*n + L, b*n + M]
    g = wi*(L*p*pick(0, 0) + dp*pick(1, 0)) - wsi*M*p*pick(2, 1)
    h = wi*(L*p*pick(0, 1) + dp*pick(1, 1)) + wsi*M*p*pick(2, 0)
    return g, h


def numpy_columns(cells_sel, P):
    X, T, start, stop, gcell, grow = node_arrays(cells_sel)
    out = np.zeros((2, len(cells_sel), NL))
    for s, e, c, i in zip(start, stop, gcell, grow):
        g, h = combine(X[s:e].T @ T[s:e], P[i, 0], P[i, 1], w[i], wsin[i])
        out[0, c] += g; out[1, c] += h
    out *= K*dphi; out[:, :, L == 0] = 0
    return out


def torch_columns(cells_sel, P, device, dtype, chunk=512):
    X, T, start, stop, gcell, grow = node_arrays(cells_sel)
    dev = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
    Xt, Tt, Pt = dev(X), dev(T), dev(P)
    Lt, Mt = torch.as_tensor(L, device=device), torch.as_tensor(M, device=device)
    Lf, Mf = dev(L), dev(M)
    wt, wst = dev(w), dev(wsin)
    out = torch.zeros((2, len(cells_sel), NL), dtype=dtype, device=device)
    n = LMAX + 1
    size = stop - start
    width = np.maximum(8, 2**np.ceil(np.log2(size)).astype(int))
    for wd in np.unique(width):
        gs = np.flatnonzero(width == wd)
        for k in range(0, len(gs), chunk):
            b = gs[k:k + chunk]
            idx = start[b][:, None] + np.arange(wd)[None, :]
            valid = idx < stop[b][:, None]
            idx = torch.as_tensor(np.where(valid, idx, 0), device=device)
            mask = dev(valid)[..., None]
            Xb, Tb = Xt[idx]*mask, Tt[idx]*mask                       # (G, wd, 3n), (G, wd, 2n)
            Q = torch.bmm(Xb.transpose(1, 2), Tb)                       # (G, 3n, 2n)
            pk = lambda a, c: Q[:, a*n + Lt, c*n + Mt]
            i = torch.as_tensor(grow[b], device=device)
            p, dp = Pt[i, 0], Pt[i, 1]
            wi, wsi = wt[i][:, None], wst[i][:, None]
            g = wi*(Lf*p*pk(0, 0) + dp*pk(1, 0)) - wsi*Mf*p*pk(2, 1)
            h = wi*(Lf*p*pk(0, 1) + dp*pk(1, 1)) + wsi*Mf*p*pk(2, 0)
            c = torch.as_tensor(gcell[b], device=device)
            out[0].index_add_(0, c, g); out[1].index_add_(0, c, h)
    out *= K*dphi; out[:, :, Lt == 0] = 0
    if device == 'mps':
        torch.mps.synchronize()
    return out.cpu().double().numpy()


def err(a):
    return np.abs(a - ref).max()/np.abs(ref).max()


P64 = np.load(f'{D}/P64.npy')
P32 = np.load(f'{D}/P.npy')
nodes, n_all = int(z['nodes_test']), int(z['n_nodes_all'])
print(f'reference (basis.direct_columns, pygmt17 numpy/OpenBLAS): {float(z["t_ref"]):.1f} s for {nodes} nodes '
      f'-> all {n_all} nodes {float(z["t_ref"])/nodes*n_all/60:.1f} min')
for name, fn in [('numpy f64 (Accelerate, 1 thread)', lambda: numpy_columns(test, P64)),
                 ('torch cpu f64 (1 thread)', lambda: torch_columns(test, P64, 'cpu', torch.float64)),
                 ('torch mps f32', lambda: torch_columns(test, P32, 'mps', torch.float32))]:
    fn() if 'mps' in name else None                                     # warm-up for the GPU
    t = time.time(); out = fn(); dt = time.time() - t
    print(f'{name:34s} {dt:6.2f} s, max rel diff {err(out):.1e} -> all nodes {dt/nodes*n_all/60:.1f} min')
