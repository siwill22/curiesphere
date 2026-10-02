"""
Domains that form from the data: merge cells of uniform VIS into as few domains as the data need.

Basis: the cell columns of cell_basis.py (HEALPix NSIDE 64 cells, ~100 km, slab from top of
basement to 30 km). Background and data points as in invert_domains.py: LCS-1 Br, degrees 16-133.
No polygon boundaries are used inside the continent.

Method (version 2, 2026-10-02; version 1 merged from the unregularised cell fit, whose values
are mostly noise, and its early merges were close to arbitrary)
  1. Merge path. Every cell starts as its own domain. The path is built for the objective
         J_mu = |G v - d|^2 / sigma^2 + mu * s * v' L v      (L: graph Laplacian of the cells,
                                                             s: mean diagonal of G'G/sigma^2)
     Merging two touching domains imposes one equality constraint, whose cost for J_mu is exact
     and closed-form, (v_a - v_b)^2 / (C_aa + C_bb - 2 C_ab) with C the inverse Hessian; v and C
     are updated by a rank-1 formula, with an exact refit in domain space every RESYNC steps.
     The smoothness term only guides the order of merging; the partitions on the path are
     scored by the plain data misfit chi2(K) = min_v |G v - d|^2 / sigma^2.
  2. Restarts: the path is built for each mu in MUS.
  3. Number of domains: minimise chi2(K) + lambda K. lambda is calibrated on null data (one
     uniform domain + correlated noise) with the same pipeline: the smallest lambda that returns
     K = 1 in 95% of realisations. This includes the gain that flexible boundaries get from
     fitting noise, at least for small K.
  4. Clean-up at the chosen K: domains smaller than MIN_CELLS are merged into their cheapest
     neighbour; boundary cells move to a touching domain if that lowers chi2 and leaves the
     domain connected. The restart with the lowest chi2 + lambda K is kept; values are refitted
     with v >= 0.

Synthetic test: SYN_CASES random Voronoi partitions (seeds on the sphere, great-circle nearest
seed) into SYN_K blocks, block values lognormal around 0.6 SI.km, noise scaled by NOISE_LEVELS.
Real data: LCS-1, and LCS-1 plus each of N_REAL noise realisations, to map how often each
boundary appears.

Outputs (inversion/): merge_null.npz, merge_synthetic.npz, merge_real.npz.

Run from notebooks/ (conda env pygmt17), after invert_domains.py and cell_basis.py:
    PYTHONPATH=..:$PYTHONPATH python merge_domains.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import multiprocessing
import time
from collections import deque

import numpy as np
import pandas as pd
import pygmt
import pyshtools
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy_healpix import healpy as hp
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree
from scipy.special import comb

REGION = os.environ.get('REGION', 'australia')     # another region: set up by region_setup.py
OUT = 'inversion' if REGION == 'australia' else f'inversion_{REGION}'
RIDGE = 1e-6                  # relative to the mean diagonal of G'G
RESYNC = 100
MIN_CELLS = 3
MUS = [0.03, 0.1, 0.3]
N_NULL = 20
SYN_K, NOISE_LEVELS = 8, [0., 0.25, 0.5, 1.]
SYN_CASES = int(os.environ.get('SYN_CASES', 3))       # 0 skips the synthetic blocks
N_REAL = int(os.environ.get('N_REAL', 20))
K_SAVE = 100
V0 = 0.6
SEED = 20261002
N_WORKERS = int(os.environ.get('N_WORKERS', 8))

cells = np.load(f'{OUT}/cells.npz')
cell, cell_ids, area_km2, nside = cells['cell'], cells['cell_ids'], cells['area_km2'], int(cells['nside'])
n_cell = len(cell_ids)
cc = np.load(f'{OUT}/cell_columns.npz')
G = cc['G133']
z = np.load(f'{OUT}/columns.npz')
assert np.array_equal(z['plon'], cc['plon'])
b, d_lcs, d_mf7 = z['b'], z['d_lcs'], z['d_mf7']
sigma = np.sqrt(np.mean((d_lcs - d_mf7)**2))
# data target: LCS-1 (default) or, with TARGET=mean3, the mean of LCS-1, MF7 and MLI_SHA_2D
# (noise_models.py) with its estimated error. The variable keeps the name d_lcs.
TARGET = os.environ.get('TARGET', 'lcs1')
TAG = '' if TARGET == 'lcs1' else f'_{TARGET}'
if TARGET == 'mean3':
    tm = np.load(f'{OUT}/target_mean3.npz')
    assert np.array_equal(tm['plon'], cc['plon'])
    d_lcs, sigma = tm['d'], float(tm['sigma'])
# SIGMA (nT) overrides the noise level, e.g. to change the target with everything else equal
if 'SIGMA' in os.environ:
    sigma = float(os.environ['SIGMA'])
    TAG += f'_sigma{sigma:g}'
noise = np.load(f'{OUT}/noise.npz')['noise']
noise *= sigma/np.sqrt(np.mean(noise**2))
Gw = G/sigma
GtG = Gw.T @ Gw
ridge = RIDGE*np.mean(np.diag(GtG))
print(f'{REGION}: {n_cell} cells, {G.shape[0]} points, target {TARGET}, sigma {sigma:.2f} nT')

# cell adjacency from the 0.1 deg grid: cells touching along a grid edge
pairs = []
for a1, a2 in [(cell[:, :-1], cell[:, 1:]), (cell[:-1, :], cell[1:, :])]:
    m = (a1 >= 0) & (a2 >= 0) & (a1 != a2)
    pairs.append(np.stack([np.minimum(a1[m], a2[m]), np.maximum(a1[m], a2[m])], 1))
cell_edges = np.unique(np.vstack(pairs), axis=0)
clon, clat = hp.pix2ang(nside, cell_ids, lonlat=True)
clon, clat = np.asarray(clon, float), np.asarray(clat, float)


def unit(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], -1)


def components(edges):
    parent = np.arange(n_cell)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in edges:
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n_cell)])


# islands and offshore plateaus: bridge each separate piece to the nearest cell (great circle)
# of another piece, until the extent is one connected graph
xyz_c = unit(clon, clat)
bridges = []
comp = components(cell_edges)
while len(np.unique(comp)) > 1:
    small = min(np.unique(comp), key=lambda c: (comp == c).sum())
    inside, outside = np.where(comp == small)[0], np.where(comp != small)[0]
    dist, j = cKDTree(xyz_c[outside]).query(xyz_c[inside])
    i = np.argmin(dist)
    bridges.append((min(inside[i], outside[j[i]]), max(inside[i], outside[j[i]])))
    comp = components(np.vstack([cell_edges, bridges]))
print(f'{len(bridges)} bridges join the separate pieces of the extent')
cell_edges = np.unique(np.vstack([cell_edges, np.array(bridges, dtype=cell_edges.dtype).reshape(-1, 2)]), axis=0)
neighbours = [[] for _ in range(n_cell)]
for i, j in cell_edges:
    neighbours[i].append(j)
    neighbours[j].append(i)


# ============================================================ merge path
Lap = np.zeros((n_cell, n_cell))
for i, j in cell_edges:
    Lap[i, j] -= 1; Lap[j, i] -= 1; Lap[i, i] += 1; Lap[j, j] += 1
scale = np.mean(np.diag(GtG))


def indicator(label):
    reps, idx = np.unique(label, return_inverse=True)
    A = np.zeros((n_cell, len(reps)))
    A[np.arange(n_cell), idx] = 1.
    return A, idx


def path(dw, mu):
    """merge path for J_mu; returns {K: label} for K <= K_SAVE"""
    H = GtG + mu*scale*Lap + ridge*np.eye(n_cell)
    C = np.linalg.inv(H)
    g = Gw.T @ dw
    v = C @ g
    label = np.arange(n_cell)
    ea, eb = cell_edges[:, 0].copy(), cell_edges[:, 1].copy()
    saved = {}
    for step in range(n_cell - 1):
        K = n_cell - step
        if K <= K_SAVE:
            saved[K] = label.copy()
        den = C[ea, ea] + C[eb, eb] - 2*C[ea, eb]
        i = np.argmin((v[ea] - v[eb])**2/np.maximum(den, 1e-300))
        a, bb = ea[i], eb[i]
        Cc = C[:, a] - C[:, bb]
        den_i = Cc[a] - Cc[bb]
        v -= Cc*((v[a] - v[bb])/den_i)
        C -= np.multiply.outer(Cc, Cc/den_i)
        label[label == bb] = a
        ea, eb = np.where(ea == bb, a, ea), np.where(eb == bb, a, eb)
        lo, hi = np.minimum(ea, eb), np.maximum(ea, eb)
        keep = lo != hi
        e = np.unique(np.stack([lo[keep], hi[keep]], 1), axis=0)
        ea, eb = e[:, 0], e[:, 1]
        if (step + 1) % RESYNC == 0 and K - 1 > 1:
            A, idx = indicator(label)
            CR = np.linalg.inv(A.T @ H @ A)
            v, C = (CR @ (A.T @ g))[idx], CR[np.ix_(idx, idx)]
    saved[1] = label.copy()
    return saved


def chi2_path(saved, dw):
    return {K: misfit(lab, dw) for K, lab in saved.items()}


def choose_k(chi2, lam):
    Ks = np.array(sorted(chi2))
    return int(Ks[np.argmin(np.array([chi2[k] for k in Ks]) + lam*Ks)])


# ============================================================ clean-up at a chosen K
def region_lists(label):
    out = {}
    for c, r in enumerate(label):
        out.setdefault(r, []).append(c)
    return out


def connected(members):
    members = set(members)
    if not members:
        return True
    start = next(iter(members))
    seen, q = {start}, deque([start])
    while q:
        c = q.popleft()
        for n in neighbours[c]:
            if n in members and n not in seen:
                seen.add(n)
                q.append(n)
    return len(seen) == len(members)


def misfit(label, dw):
    reps, idx = np.unique(label, return_inverse=True)
    A = np.zeros((n_cell, len(reps)))
    A[np.arange(n_cell), idx] = 1.
    GR = Gw @ A
    vR, *_ = np.linalg.lstsq(GR, dw, rcond=None)
    return np.sum((GR @ vR - dw)**2)


def clean_up(label, dw, max_pass=10):
    label = label.copy()
    # merge small domains into their cheapest neighbour
    while True:
        regions = region_lists(label)
        small = [r for r, m in regions.items() if len(m) < MIN_CELLS]
        if not small:
            break
        r = small[0]
        nbrs = {label[n] for c in regions[r] for n in neighbours[c]} - {r}
        best = min(nbrs, key=lambda q: misfit(np.where(label == r, q, label), dw))
        label[label == r] = best
    # move boundary cells; the domain columns GR are updated in place for each trial
    reps = list(np.unique(label))
    col = {r: i for i, r in enumerate(reps)}
    GR = np.stack([Gw[:, label == r].sum(1) for r in reps], 1)

    def ss(GR):
        vR, *_ = np.linalg.lstsq(GR, dw, rcond=None)
        return np.sum((GR @ vR - dw)**2)
    current = ss(GR)
    for _ in range(max_pass):
        moved = 0
        regions = region_lists(label)
        for c in range(n_cell):
            r = label[c]
            targets = {label[n] for n in neighbours[c]} - {r}
            if not targets or len(regions[r]) <= MIN_CELLS:
                continue
            if not connected([m for m in regions[r] if m != c]):
                continue
            for q in targets:
                GR[:, col[r]] -= Gw[:, c]
                GR[:, col[q]] += Gw[:, c]
                f = ss(GR)
                if f < current - 1e-9:
                    label[c], current = q, f
                    regions[r].remove(c)
                    regions[q].append(c)
                    moved += 1
                    break
                GR[:, col[r]] += Gw[:, c]
                GR[:, col[q]] -= Gw[:, c]
        if moved == 0:
            break
    return label


def final_values(label, dw):
    reps, idx = np.unique(label, return_inverse=True)
    A = np.zeros((n_cell, len(reps)))
    A[np.arange(n_cell), idx] = 1.
    GR = Gw @ A
    vR = lsq_linear(GR, dw, bounds=(0., np.inf)).x
    return vR[idx], np.sqrt(np.mean((GR @ vR - dw)**2))


def ari(x, y):
    """adjusted Rand index of two partitions of the cells"""
    _, xi = np.unique(x, return_inverse=True)
    _, yi = np.unique(y, return_inverse=True)
    ct = np.zeros((xi.max()+1, yi.max()+1))
    np.add.at(ct, (xi, yi), 1)
    s = comb(ct, 2).sum()
    sa, sb = comb(ct.sum(1), 2).sum(), comb(ct.sum(0), 2).sum()
    expected = sa*sb/comb(len(x), 2)
    return (s - expected)/(0.5*(sa + sb) - expected)


# ============================================================ one full run
def run(dw, lam):
    """restarts over MUS; returns the best cleaned partition, its values, chi2 and K, and the chi2 paths"""
    best, paths = None, {}
    for mu in MUS:
        saved = path(dw, mu)
        chi2 = chi2_path(saved, dw)
        paths[mu] = chi2
        if lam is None:
            continue
        K0 = choose_k(chi2, lam)
        lab = clean_up(saved[K0], dw)
        K = len(np.unique(lab))
        score = misfit(lab, dw) + lam*K
        if best is None or score < best[0]:
            best = (score, lab, mu)
    if lam is None:
        return paths
    _, lab, mu = best
    v, chi = final_values(lab, dw)
    return dict(label=lab, v=v, misfit_over_sigma=chi, K=len(np.unique(lab)), mu=mu, paths=paths)


def null_job(j):
    return run((G @ (V0*np.ones(n_cell)) + noise[j])/sigma, None)


def task(args):
    kind, payload, lam = args
    return run(payload, lam)


if __name__ == '__main__':
    pool = multiprocessing.get_context('fork').Pool(N_WORKERS)

    # 1. calibrate lambda on null data, same pipeline
    t = time.time()
    cache = f'{OUT}/merge_null.npz'
    if os.path.exists(cache):
        chi_null = np.load(cache)['chi2']                       # [run, mu, K]
    else:
        res = pool.map(null_job, range(N_NULL))
        chi_null = np.array([[[r[mu][K] for K in range(1, K_SAVE+1)] for mu in MUS] for r in res])
        np.savez(cache, chi2=chi_null, mus=MUS)
    lams = np.logspace(0, 4, 500)
    Ks = np.arange(1, K_SAVE+1)
    # a null run returns K = 1 if, for every restart, chi2(K) + lambda K is smallest at K = 1
    ok = np.array([np.mean([all(np.argmin(chi_null[j, m] + lam*Ks) == 0 for m in range(len(MUS)))
                            for j in range(N_NULL)]) for lam in lams])
    LAM = lams[np.argmax(ok >= 0.95)]
    print(f'null calibration: lambda_95 = {LAM:.1f} ({time.time()-t:.0f} s)', flush=True)

    # 2. synthetic blocks at several noise levels
    if SYN_CASES:
        rng = np.random.default_rng(SEED)
        xyz = unit(clon, clat)
        truths, jobs = [], []
        for case in range(SYN_CASES):
            seeds = rng.choice(n_cell, SYN_K, replace=False)
            _, tb = cKDTree(xyz[seeds]).query(xyz)
            vb = V0*np.exp(0.5*rng.standard_normal(SYN_K))
            truths.append((tb, vb))
            for f in NOISE_LEVELS:
                jobs.append(('syn', (G @ vb[tb] + f*noise[case])/sigma, LAM))
        t = time.time()
        syn = pool.map(task, jobs)
        print(f'synthetic runs: {time.time()-t:.0f} s', flush=True)
        lab_syn = np.array([r['label'] for r in syn]).reshape(SYN_CASES, len(NOISE_LEVELS), n_cell)
        v_syn = np.array([r['v'] for r in syn]).reshape(SYN_CASES, len(NOISE_LEVELS), n_cell)
        K_syn = np.array([r['K'] for r in syn]).reshape(SYN_CASES, len(NOISE_LEVELS))
        chi_syn = np.array([r['misfit_over_sigma'] for r in syn]).reshape(SYN_CASES, len(NOISE_LEVELS))
        chi_true = np.array([[np.sqrt(misfit(tb, jobs[c*len(NOISE_LEVELS)+n][1])/len(b))
                              for n in range(len(NOISE_LEVELS))] for c, (tb, vb) in enumerate(truths)])
        ari_syn = np.array([[ari(truths[c][0], lab_syn[c, n]) for n in range(len(NOISE_LEVELS))] for c in range(SYN_CASES)])
        np.savez(f'{OUT}/merge_synthetic.npz', true_label=np.array([t_[0] for t_ in truths]),
                 true_v=np.array([t_[1] for t_ in truths]), label=lab_syn, v=v_syn, K=K_syn,
                 misfit_over_sigma=chi_syn, misfit_true_over_sigma=chi_true, ari=ari_syn,
                 noise_levels=NOISE_LEVELS, lam=LAM)
        for c, (tb, vb) in enumerate(truths):
            print(f'  case {c} (block contrast std/mean {vb.std()/vb.mean():.2f}): ' + '; '.join(
                f'noise x{f}: K {K_syn[c, n]}, ARI {ari_syn[c, n]:.2f}, misfit {chi_syn[c, n]:.3f} (truth {chi_true[c, n]:.3f})'
                for n, f in enumerate(NOISE_LEVELS)), flush=True)

    # 3. real data, and real data plus noise realisations
    t = time.time()
    dw_real = (d_lcs - b)/sigma
    jobs = [('real', dw_real, LAM)] + [('real', dw_real + noise[j]/sigma, LAM) for j in range(N_REAL)]
    real = pool.map(task, jobs)
    print(f'real-data runs: {time.time()-t:.0f} s', flush=True)
    np.savez(f'{OUT}/merge_real.npz', label=np.array([r['label'] for r in real]),
             v=np.array([r['v'] for r in real]), K=np.array([r['K'] for r in real]),
             misfit_over_sigma=np.array([r['misfit_over_sigma'] for r in real]),
             mu=np.array([r['mu'] for r in real]), lam=LAM,
             chi2_path=np.array([[real[0]['paths'][mu][K] for K in range(1, K_SAVE+1)] for mu in MUS]), mus=MUS)
    r0 = real[0]
    print(f'  LCS-1: K = {r0["K"]}, misfit/sigma {r0["misfit_over_sigma"]:.3f}, mu {r0["mu"]}')
    Kp = [r['K'] for r in real[1:]]
    if Kp:
        print(f'  LCS-1 + noise: K median {np.median(Kp):.0f} (range {min(Kp)}-{max(Kp)})')
    pool.close()
