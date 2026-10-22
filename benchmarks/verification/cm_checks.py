"""
Verification of CM (remit.inversion.methods.cm: the GMRF map), Australia local fit, as df_checks.py for DF.

CM-AUS weights the data with the 'subspace' weighting (errors.weights): the data are projected onto the leading
eigenvectors of the error model and divided by the noise RMS, so the weighted errors are uncorrelated but not of unit
variance; CM treats them as white with one fitted scale h (as the first Australian build).

C1  the algebra: the log evidence, posterior mean and s.d. of every cell at fixed h, s and ell against an independent
    solver (the bordered system of df_checks.reference, with the correlation matrix from a direct dense inverse);
    the evidence with h at its best; the fitted ell against a fine profile of the evidence; the weighted errors
    (correlation and spread of variances); a uniform truth without noise, which CM must return exactly
C2  calibration: truths drawn from CM's own model (v0 + s u, u with correlation length ell); data errors as CM assumes
    them (white, h = 1: settings fixed at the truth and fitted) and from the error model (noise and model error, as on
    real data: settings fitted); the errors divided by the reported s.d.
C3  shrinking checkerboards (df_checks.SIZES, 1.5 / 0.5 SI km): without noise, with the field model's noise (the same
    draw as DF's), and with noise and model error from the error model (which gives h as in the real CM-AUS fit)

Run from the repository root (each a few minutes to ~20 min with 3 workers):
    PYTHONPATH=benchmarks/verification:benchmarks:notebooks:$PYTHONPATH OMP_NUM_THREADS=1 \
        nice -n 15 conda run -n curiesphere python benchmarks/verification/cm_checks.py c1|c2 [n]|c3
Outputs: benchmarks/verification/results/cm_*.json, cm_*.npz
"""
import json
import multiprocessing as mp
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from df_checks import RESULTS, SIZES, WORKERS, log, squares
from remit.inversion import config, gmrf, graph, problem
from remit.inversion.cache import Cache

CONFIG = 'inversions/cm_aus_lcs1_16-133.toml'       # CM-AUS: subspace weighting, noise and model error (33.4 nT)
TRUE = dict(v0=0.65, s=0.45, ell=1400., h=1.)       # as the real CM-AUS fit (laybg: v0 0.65, s 0.47, ell 1406 km)


def setup():
    cfg = config.load(CONFIG)
    pb = problem.local(cfg, Cache(), log)
    cells = pb.cells
    edges, _ = graph.cell_edges(cells)
    Lw = graph.laplacian(edges, 1/graph.edge_lengths_km(edges, cells.clon, cells.clat)**2, len(cells))
    return SimpleNamespace(cfg=cfg, pb=pb, Lw=Lw, area=cells.area_km2, one=np.ones((len(cells), 1)),
                           W=np.asarray(pb.W))


def at(P, yw, h, s, ell):
    """CM's posterior and log evidence at fixed settings"""
    gm = gmrf.GMRF(P.pb.Gw, yw, P.area, P.Lw)
    lr, L = gm.model(P.one, ell)
    post = gm.posterior(P.one, dict(ridge=lr, L=L, alpha2=h**2/s**2, h=h))
    return gm, post, gm.evidence(P.one, h, s, ell)


def fitted(P, yw):
    gm = gmrf.GMRF(P.pb.Gw, yw, P.area, P.Lw)
    f = gm.fit(P.one)
    return f, gm.posterior(P.one, f)


def correlation(P, ell):
    """the correlation matrix of CM's prior, directly: the inverse of diag(a/ell^2) + L scaled to unit diagonal"""
    a = P.area/np.mean(P.area)
    Qi = np.linalg.inv(np.diag(a/ell**2) + P.Lw)
    d = 1/np.sqrt(np.diag(Qi))
    return Qi*d[:, None]*d[None, :]


def reference(G, y, Sigma, X, h):
    """log evidence, posterior mean and s.d. of y = G v + e, v = X theta + u, u ~ N(0, Sigma), e ~ N(0, h^2 I),
    theta flat: the bordered system [[K, U], [U', 0]] by LU (as df_checks.reference)"""
    N, p = len(y), X.shape[1]
    U = G @ X
    K = G @ Sigma @ G.T + h**2*np.eye(N)
    B = np.block([[K, U], [U.T, np.zeros((p, p))]])
    lu = lu_factor(B)
    logdet = np.linalg.slogdet(B)[1]
    sol = lu_solve(lu, np.r_[y, np.zeros(p)])
    a, theta = sol[:N], sol[N:]
    ll = -0.5*logdet - 0.5*y @ a - 0.5*(N - p)*np.log(2*np.pi)
    c = np.vstack([G @ Sigma, X.T])
    cov = Sigma - c.T @ lu_solve(lu, c)
    return dict(ll=float(ll), v=X @ theta + Sigma @ G.T @ a, v_sd=np.sqrt(np.maximum(np.diag(cov), 0)))


def draw_truth(P, rng, L=None):
    L = gmrf.GMRF(P.pb.Gw, P.pb.yw, P.area, P.Lw).factor(TRUE['ell']) if L is None else L
    return TRUE['v0'] + TRUE['s']*(L @ rng.standard_normal(len(P.area)))


def errors(P, rng):
    """weighted data errors from the error model (noise and model error)"""
    return P.W @ P.pb.draw(rng, model_error=True)


# ---------------------------------------------------------------- C1: the algebra

def c1():
    P = setup()
    pb = P.pb
    rng = np.random.default_rng(20261020)
    v_true = draw_truth(P, rng)
    yw = pb.Gw @ v_true + rng.standard_normal(len(pb.yw))          # errors as CM assumes them (h = 1)
    out, maps = {}, {}
    for h, s, ell in ((1., 0.45, 1400.), (2., 0.3, 300.)):
        gm, post, ll = at(P, yw, h, s, ell)
        R = correlation(P, ell)
        L = gm.factor(ell)
        ref = reference(pb.Gw, yw, s**2*R, P.one, h)
        key = f'h {h:g}, s {s:g}, ell {ell:g} km'
        if not maps:
            maps = dict(v_cm=post['v'], v_ref=ref['v'], sd_cm=post['v_sd'], sd_ref=ref['v_sd'])
        out[key] = dict(correlation_max_diff=float(np.abs(L @ L.T - R).max()), ll=ll, ll_ref=ref['ll'],
                        ll_diff=abs(ll - ref['ll']), v_max_diff=float(np.abs(post['v'] - ref['v']).max()),
                        sd_max_rel_diff=float(np.abs(post['v_sd']/ref['v_sd'] - 1).max()))
        log(f'{key}: correlation {out[key]["correlation_max_diff"]:.1e}, log evidence {ll:.4f} vs {ref["ll"]:.4f}, '
            f'map {out[key]["v_max_diff"]:.1e} SI km, s.d. {out[key]["sd_max_rel_diff"]:.1e} (relative)')

    # h at its best: the evidence of the fit equals the evidence at its own h and s
    f, post = fitted(P, yw)
    ell, h, s = float(f['ell'][0]), f['h'], float(f['s'][0])
    ref = reference(pb.Gw, yw, s**2*correlation(P, ell), P.one, h)
    out['fit'] = dict(h=h, s=s, ell_km=ell, ll=f['ll'], ll_ref=ref['ll'], ll_diff=abs(f['ll'] - ref['ll']),
                      v_max_diff=float(np.abs(post['v'] - ref['v']).max()))
    log(f'fit: h {h:.3f}, s {s:.3f}, ell {ell:.0f} km; log evidence {f["ll"]:.4f} vs {ref["ll"]:.4f} at those settings')

    # the fitted ell against a fine profile (h and s at their best at each ell)
    gm = gmrf.GMRF(pb.Gw, yw, P.area, P.Lw)
    grid = np.geomspace(40., 3000., 41)
    prof = np.array([gm.best(P.one, e)[0] for e in grid])
    out['profile'] = dict(ell_km=grid.tolist(), ll=prof.tolist(), best_grid_ell=float(grid[np.argmax(prof)]),
                          best_grid_ll=float(prof.max()), fit_minus_best_grid=float(f['ll'] - prof.max()))
    log(f'profile: best of 41 lengths {grid[np.argmax(prof)]:.0f} km, log evidence {prof.max():.4f}; the fit is '
        f'{f["ll"] - prof.max():+.4f} from it')

    # the weighted errors: uncorrelated (the subspace is the error model's eigenvectors), variances unequal
    E = np.array([errors(P, np.random.default_rng(20261100 + k)) for k in range(2000)])
    C = np.corrcoef(E.T)
    var = E.var(0)
    off = C[~np.eye(len(C), dtype=bool)]
    out['weighting'] = dict(variance_mean=float(var.mean()), variance_range=[float(var.min()), float(var.max())],
                            correlation_rms=float(np.sqrt(np.mean(off**2))), chance_rms=float(1/np.sqrt(2000)),
                            sqrt_variance_mean=float(np.sqrt(var.mean())))
    log(f'weighted errors: variance {var.mean():.1f} ({var.min():.2f}-{var.max():.1f}; sqrt of the mean '
        f'{np.sqrt(var.mean()):.2f}), correlation RMS {out["weighting"]["correlation_rms"]:.3f} (chance '
        f'{1/np.sqrt(2000):.3f})')

    # a uniform truth without noise: the map must be the truth
    v_u = np.full(len(P.area), 0.75)
    f, post = fitted(P, pb.Gw @ v_u)
    out['uniform'] = dict(max_abs_error=float(np.abs(post['v'] - v_u).max()), v0=float(post['theta'][0]),
                          s=float(f['s'][0]), ell_km=float(f['ell'][0]), h=f['h'])
    log(f'uniform truth, no noise: max |error| {out["uniform"]["max_abs_error"]:.1e} SI km (s {f["s"][0]:.1e})')

    with open(os.path.join(RESULTS, 'cm_c1.json'), 'w') as fo:
        json.dump(out, fo, indent=1)
    np.savez(os.path.join(RESULTS, 'cm_c1.npz'), clon=pb.cells.clon, clat=pb.cells.clat, v_true=v_true,
             prof_ell=grid, prof_ll=prof, **maps)
    log('C1 done')


# ---------------------------------------------------------------- C2: calibration

P2 = None


def _worker():
    """each worker builds the problem itself ('spawn'): forked workers hung in their first LAPACK call here"""
    global P2
    P2 = setup()
    P2.L = None


def pool():
    return mp.get_context('spawn').Pool(WORKERS, initializer=_worker)


def c2_draw(i):
    P = P2
    if P.L is None:                                   # per worker: a large LAPACK call before the fork can hang it
        P.L = gmrf.GMRF(P.pb.Gw, P.pb.yw, P.area, P.Lw).factor(TRUE['ell'])
    rng = np.random.default_rng(20261021 + i)
    v_true = draw_truth(P, rng, P.L)
    yw = P.pb.Gw @ v_true + rng.standard_normal(len(P.pb.yw))         # errors as CM assumes them (h = 1)
    _, fixed, _ = at(P, yw, TRUE['h'], TRUE['s'], TRUE['ell'])
    f, post = fitted(P, yw)
    yr = P.pb.Gw @ v_true + errors(P, rng)                             # errors from the error model, as real data
    fr, real = fitted(P, yr)
    out = dict(z_fixed=(fixed['v'] - v_true)/fixed['v_sd'], z_fit=(post['v'] - v_true)/post['v_sd'],
               z_real=(real['v'] - v_true)/real['v_sd'], h=f['h'], s=float(f['s'][0]), ell=float(f['ell'][0]),
               h_real=fr['h'], s_real=float(fr['s'][0]), ell_real=float(fr['ell'][0]))
    if i < 3:
        out.update(v_true=v_true, v_fixed=fixed['v'], sd_fixed=fixed['v_sd'], v_fit=post['v'], sd_fit=post['v_sd'],
                   v_real=real['v'], sd_real=real['v_sd'])
    log(f'  truth {i}: fitted h {f["h"]:.2f}, s {f["s"][0]:.2f}, ell {f["ell"][0]:.0f} km (error model: h '
        f'{fr["h"]:.2f}, ell {fr["ell"][0]:.0f} km); within 1 s.d.: fixed {np.mean(np.abs(out["z_fixed"]) < 1):.2f}, '
        f'fitted {np.mean(np.abs(out["z_fit"]) < 1):.2f}, error model {np.mean(np.abs(out["z_real"]) < 1):.2f}')
    return out


def c2(n):
    global P2
    P2 = setup()
    log(f'C2: {n} truths from CM at {TRUE}, {WORKERS} workers')
    with pool() as pl:
        res = pl.map(c2_draw, range(n), chunksize=1)
    arrays = {k: np.array([r[k] for r in res]) for k in ('z_fixed', 'z_fit', 'z_real', 'h', 's', 'ell', 'h_real',
                                                          's_real', 'ell_real')}
    for k in ('v_true', 'v_fixed', 'sd_fixed', 'v_fit', 'sd_fit', 'v_real', 'sd_real'):
        arrays[k] = np.array([r[k] for r in res if k in r])
    np.savez(os.path.join(RESULTS, 'cm_c2.npz'), clon=P2.pb.cells.clon, clat=P2.pb.cells.clat, **arrays)
    for k in ('z_fixed', 'z_fit', 'z_real'):
        z = arrays[k]
        log(f'  {k}: s.d. of z {z.std():.3f}, within 1 s.d. {np.mean(np.abs(z) < 1):.3f}, within 2 s.d. '
            f'{np.mean(np.abs(z) < 2):.3f} (expected 1, 0.683, 0.954)')
    for k in ('h', 's', 'ell', 'h_real', 's_real', 'ell_real'):
        log(f'  fitted {k}: median {np.median(arrays[k]):.3g}, middle 68% {np.percentile(arrays[k], 16):.3g}-'
            f'{np.percentile(arrays[k], 84):.3g} (true {TRUE[k.split("_")[0]]:g})')
    log('C2 done')


# ---------------------------------------------------------------- C3: shrinking checkerboards

NOISE = ('none', 'noise', 'noise and model error')  # none; the field model's noise (as DF); the full error model


def c3_job(job):
    size, noise = job
    P = P2
    lon, lat = P.pb.cells.clon, P.pb.cells.clat
    _, high = squares(lon, lat, size)
    v_true = np.where(high, 1.5, 0.5)
    # the noise as DF's shrinking test (its point-space draw, seed 20261018), whitened by CM's error model
    e = (0. if noise == 'none' else P.W @ P.pb.draw(np.random.default_rng(20261018), model_error=noise != 'noise'))
    t0 = time.time()
    f, post = fitted(P, P.pb.Gw @ v_true + e)
    v, sd = post['v'], post['v_sd']
    err = v - v_true
    r = dict(size=size, noise=noise, contrast=float(v[high].mean() - v[~high].mean()),
             rms_error=float(np.sqrt(np.mean(err**2))), within_1sd=float(np.mean(np.abs(err/sd) < 1)),
             h=f['h'], s=float(f['s'][0]), ell_km=float(f['ell'][0]), ll=f['ll'], v=v, sd=sd)
    log(f'  {size:g} deg, noise {noise}: contrast {r["contrast"]:.3f}, RMS error {r["rms_error"]:.3f}, within 1 s.d. '
        f'{r["within_1sd"]:.2f}, ell {r["ell_km"]:.0f} km, h {f["h"]:.2f} ({time.time() - t0:.0f} s)')
    return r


def c3():
    global P2
    P2 = setup()
    jobs = [(s, n) for s in SIZES for n in NOISE]
    log(f'C3: shrinking checkerboards, {len(jobs)} fits')
    with pool() as pl:
        res = pl.map(c3_job, jobs, chunksize=1)
    maps = {f'{r["size"]:g}_{r["noise"]}': np.c_[r.pop('v'), r.pop('sd')] for r in res}
    with open(os.path.join(RESULTS, 'cm_shrink.json'), 'w') as fo:
        json.dump(res, fo, indent=1)
    np.savez(os.path.join(RESULTS, 'cm_shrink.npz'), clon=P2.pb.cells.clon, clat=P2.pb.cells.clat, **maps)
    log('C3 done')


if __name__ == '__main__':
    if sys.argv[1] == 'c1':
        c1()
    elif sys.argv[1] == 'c2':
        c2(int(sys.argv[2]) if len(sys.argv) > 2 else 45)
    elif sys.argv[1] == 'c3':
        c3()
