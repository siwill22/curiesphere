"""
Verification of DF (remit.inversion.methods.df, ADR 0007), Australia local fit: the pass/fail tests.

V1  the algebra, on the real Australian problem (1406 cells, 39 polygon domains, 849 whitened data, ring error)
    a  the correlation of cell averages: df's tabulated and interpolated R_k against a direct Legendre sum, with the
       cap window from numerical quadrature (not df's closed form), at degree 2000 and 4000
    b  log evidence, posterior mean and s.d. of every cell at fixed hyperparameters, against an independent solver:
       the bordered (universal-kriging) system [[K, U], [U', 0]] solved by LU, where df uses Cholesky factors and a
       Schur complement; for Gaussian means and for box means
    c  the gradient of the log evidence against central differences of the independent evidence
    d  the optimiser: fits from three starting points reach the same maximum
    e  the whitening: W times draws of the error model has unit covariance
V2  calibration: truths drawn from DF's own model (domain means from N(v0, tau^2), a fractal pattern inside each
    domain, noise from the error model), inverted with the hyperparameters fixed at the truth and fitted; the errors
    divided by the reported s.d. should have spread 1, and 68% / 95% of cells should fall within 1 / 2 s.d.

Run from the repository root (V1 a few minutes; V2 about 50 min with 3 workers):
    PYTHONPATH=benchmarks/verification:benchmarks:notebooks:$PYTHONPATH OMP_NUM_THREADS=1 \
        nice -n 15 conda run -n curiesphere python benchmarks/verification/df_checks.py v1|v1maps|v2 [n]|checker|shrink|mismatch|matching
Outputs: benchmarks/verification/results/df_v1.json, df_v2.npz
"""
import json
import multiprocessing as mp
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from remit.inversion import config, errors, parameterise, problem
from remit.inversion.cache import Cache
from remit.inversion.graph import unit
from remit.inversion.methods import df

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
CONFIG = 'inversions/df_aus_syn_rf_gm.toml'          # DF-AUS, Gaussian means, ring error; synthetic data
TRUE = dict(v0=0.75, tau=0.3, s=0.3, beta=2.25, ell=1000., ring=0.1)   # V2 truth (ring at its lower bound: none)
WORKERS = int(os.environ.get('VERIFY_WORKERS', '3'))


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def setup():
    cfg = config.load(CONFIG)
    cache = Cache()
    pb = problem.local(cfg, cache, log)
    lays, _ = df.layouts(pb.cells, cfg, cache, log, pb.Gw, pb.yw)
    me = cfg['method']
    xyz_c, xyz_p = unit(pb.cells.clon, pb.cells.clat), unit(pb.plon, pb.plat)
    dist_km = np.arccos(np.clip((xyz_p @ xyz_c.T).max(1), -1., 1.))*df.R_KM
    ring = dist_km > me['ring_error']['beyond_km']
    Wd = np.asarray(pb.W)
    Cr = errors.point_shape(pb.spectra['model'], pb.band, pb.plon[ring], pb.plat[ring])
    return SimpleNamespace(cfg=cfg, pb=pb, label=lays['polygons'][0], names=lays['polygons'][1], layouts=lays,
                           extra=[Wd[:, ring] @ Cr @ Wd[:, ring].T], ring=ring,
                           cell_km2=4*np.pi*df.R_KM**2/(12*pb.cells.nside**2), lo=me['bounds'][0], hi=me['bounds'][1],
                           opts=dict(s_range=tuple(me['s_range']), beta_range=tuple(me['beta_range']),
                                     ell_range=tuple(me['ell_range'])),
                           tau_range=tuple(me.get('tau_range', [0.01, 4.])))


def model(P, yw, means='gaussian'):
    pb = P.pb
    return df.DomainFractal(pb.Gw, yw, P.label, pb.cells.clon, pb.cells.clat, P.cell_km2, P.lo, P.hi, P.extra, means,
                            P.tau_range)


def x_of(m, s, beta, ell, sig):
    """df's hyperparameter vector: log s_k, beta, log ell, log sigma_j (noise scale h fixed at 1)"""
    return np.r_[np.log(np.broadcast_to(s, (m.K,))), beta, np.log(ell), np.log(sig)]


# ---------------------------------------------------------------- the independent solver (V1b)

def reference(m, x, R=None):
    """log evidence, posterior mean, s.d. and theta-free summaries of DF's model by the bordered system.
    R: the correlation blocks (default: df's own, L L', so that only the algebra is compared)"""
    s, beta, ell, h, sig = m.unpack(x, False)
    if R is None:
        R = [L @ L.T for L in m.factors(beta, ell)[1]]
    n, N = len(m.label), len(m.yw)
    Sigma = np.zeros((n, n))
    for k, b in enumerate(m.blocks):
        Sigma[np.ix_(b, b)] = s[k]**2*R[k]
    extra = list(m.extra)
    if m.gaussian:                                   # theta_k = v0 + delta_k: the offsets join the random field
        tau = sig[-1]
        Sigma += tau**2*(m.S @ m.S.T)
        X = np.ones((n, 1))
        extra, sig = extra[:-1], sig[:-1]            # tau^2 Udom Udom' is already in Gw Sigma Gw'
    else:
        X = m.S
    G = m.Gw
    U = G @ X
    Kd = G @ Sigma @ G.T + h**2*np.eye(N)
    for sj, E in zip(sig, extra):
        Kd += sj**2*E
    p = X.shape[1]
    B = np.block([[Kd, U], [U.T, np.zeros((p, p))]])
    lu = lu_factor(B)
    sign, logdet = np.linalg.slogdet(B)              # log|K| + log|U'K^-1 U|, sign (-1)^p
    sol = lu_solve(lu, np.r_[m.yw, np.zeros(p)])
    a, theta = sol[:N], sol[N:]
    lo, hi = np.broadcast_to(np.asarray(m.lo, float), (p,)), np.broadcast_to(np.asarray(m.hi, float), (p,))
    ll = -0.5*logdet - 0.5*m.yw @ a - 0.5*(N - p)*np.log(2*np.pi) - np.sum(np.log(hi - lo))
    c = np.vstack([G @ Sigma, X.T])                  # (N + p) x n
    Z = lu_solve(lu, c)
    mean = X @ theta + Sigma @ G.T @ a
    cov = Sigma - c.T @ Z
    return dict(ll=float(ll), v=mean, v_sd=np.sqrt(np.maximum(np.diag(cov), 0)), sign=float(sign))


# ---------------------------------------------------------------- the correlation (V1a)

def cap_window_quadrature(area_km2, lmax, n=6000):
    """W_l = mean of P_l over a spherical cap, by Gauss-Legendre quadrature on [x0, 1]"""
    x0 = 1 - area_km2/(2*np.pi*df.R_KM**2)
    t, w = np.polynomial.legendre.leggauss(n)
    x = x0 + (1 - x0)*(t + 1)/2
    w = w/2                                          # weights of the mean over [x0, 1]
    out = np.empty(lmax + 1)
    p0, p1 = np.ones_like(x), x.copy()
    out[0], out[1] = 1., w @ p1
    for l in range(1, lmax):
        p0, p1 = p1, ((2*l + 1)*x*p1 - l*p0)/(l + 1)
        out[l + 1] = w @ p1
    return out


def correlation_direct(beta, ell, psi, window):
    """sum_l (2l+1) S_l W_l^2 P_l(cos psi) / sum_l (2l+1) S_l W_l^2, by the Legendre recurrence at each psi"""
    lmax = len(window) - 1
    l = np.arange(lmax + 1)
    c = (2*l + 1)*(l*(l + 1) + (df.R_KM/ell)**2)**(-beta/2)*window**2
    x = np.cos(psi)
    p0, p1 = np.ones_like(x), x.copy()
    acc = c[0]*p0 + c[1]*p1
    for k in range(1, lmax):
        p0, p1 = p1, ((2*k + 1)*x*p1 - k*p0)/(k + 1)
        acc += c[k + 1]*p1
    return acc/c.sum()


# ---------------------------------------------------------------- V1

def v1():
    P = setup()
    out = {}
    m = model(P, P.pb.yw)
    beta, ell = 2.25, 1000.

    # a: the correlation blocks
    log('V1a correlation of cell averages')
    _, Ls = m.factors(beta, ell)
    R_df = [L @ L.T - df.JITTER*np.eye(len(L)) for L in Ls]
    for lmax in (2000, 4000):
        win = cap_window_quadrature(P.cell_km2, lmax)
        err = max(np.abs(correlation_direct(beta, ell, p, win) - r).max() for p, r in zip(m.psi, R_df))
        out[f'V1a correlation, max abs difference (direct to degree {lmax})'] = float(err)
        log(f'  direct sum to degree {lmax}: max |R_df - R_direct| {err:.1e}')
    win_df = df.cap_window(P.cell_km2)
    win_q = cap_window_quadrature(P.cell_km2, df.LMAX)
    out['V1a cap window: closed form vs quadrature, max abs difference'] = float(np.abs(win_df - win_q).max())

    # b: evidence and posterior at fixed hyperparameters (production-like values)
    for means in ('gaussian', 'box'):
        mm = model(P, P.pb.yw, means)
        sig = [62.] + ([0.3] if mm.gaussian else [])
        x = x_of(mm, 0.3, beta, ell, sig)
        t = time.time()
        post = mm.posterior(mm.fit_at(x, False), P.pb.cells.area_km2)
        ref = reference(mm, x)
        log(f'V1b {means} means ({time.time() - t:.0f} s)')
        res = {'log evidence: df': post['ll'], 'log evidence: reference': ref['ll'],
               'log evidence: difference': post['ll'] - ref['ll'],
               'posterior mean: max abs difference (SI km)': float(np.abs(post['v'] - ref['v']).max()),
               'posterior mean: max relative to the s.d.': float(np.abs(post['v'] - ref['v']).max()/ref['v_sd'].min()),
               'posterior s.d.: max relative difference': float(np.abs(post['v_sd']/ref['v_sd'] - 1).max())}
        # end to end, with the directly summed correlation in place of df's table
        R_dir = [correlation_direct(beta, ell, p, win_q) + df.JITTER*np.eye(len(p)) for p in mm.psi]
        ref2 = reference(mm, x, R_dir)
        res['log evidence: df vs reference with the direct correlation'] = post['ll'] - ref2['ll']
        res['posterior mean: max abs difference with the direct correlation'] = float(np.abs(post['v'] - ref2['v']).max())
        for k, v in res.items():
            log(f'  {k}: {v:.3e}')
        out[f'V1b {means} means'] = res

    # c: gradient
    log('V1c gradient')
    sig = [62., 0.3]
    x = x_of(m, np.exp(np.random.default_rng(1).normal(np.log(0.3), 0.3, m.K)), beta, ell, sig)
    _, g = m.gradient(x)
    gr = np.zeros_like(x)
    for j in range(len(x)):
        e = np.zeros_like(x)
        e[j] = 1e-4
        gr[j] = (reference(m, x + e)['ll'] - reference(m, x - e)['ll'])/2e-4
    rel = np.abs(g - gr)/np.maximum(np.abs(gr), 1e-3*np.abs(gr).max())
    out['V1c gradient: max relative difference'] = float(rel.max())
    out['V1c gradient: max relative difference, strengths and errors (analytic)'] = float(
        np.delete(rel, [m.K, m.K + 1]).max())
    out['V1c gradient: beta and log ell (central differences in df)'] = [float(rel[m.K]), float(rel[m.K + 1])]
    log(f'  max relative difference {rel.max():.1e} (analytic parts {np.delete(rel, [m.K, m.K + 1]).max():.1e})')

    # d: the optimiser from three starts
    log('V1d fits from three starting points')
    fits = []
    rng = np.random.default_rng(2)
    bnds = m.bounds(*P.opts.values(), False)
    for k in range(3):
        x0 = None if k == 0 else np.array([rng.uniform(lo + 0.1*(hi - lo), hi - 0.1*(hi - lo)) for lo, hi in bnds])
        f = m.fit(x0=x0, **P.opts)
        fits.append(f)
        log(f'  start {k}: log evidence {f["ll"]:.4f}, beta {f["beta"]:.3f}, ell {f["ell"]:.0f} km, '
            f'{f["n_eval"]} evaluations, converged {f["converged"]}')
    lls = [f['ll'] for f in fits]
    out['V1d log evidence of three fits'] = lls
    out['V1d spread of the log evidence'] = float(np.ptp(lls))
    out['V1d beta, ell of three fits'] = [[f['beta'], f['ell']] for f in fits]
    best = fits[int(np.argmax(lls))]
    _, g = m.gradient(best['x'])
    free = ~best['at_bound']
    out['V1d gradient at the best fit, free parameters (max abs)'] = float(np.abs(g[free]).max())

    # e: whitening
    log('V1e whitening of the error model')
    rng = np.random.default_rng(3)
    Wd = np.asarray(P.pb.W)
    E = np.array([Wd @ P.pb.draw(rng, model_error=False) for _ in range(2000)])
    C = np.cov(E.T)
    out['V1e whitened draws: mean variance'] = float(np.diag(C).mean())
    out['V1e whitened draws: RMS off-diagonal covariance'] = float(np.sqrt(np.mean(C[~np.eye(len(C), dtype=bool)]**2)))
    out['V1e expected sampling error (1/sqrt(n))'] = float(1/np.sqrt(len(E)))
    log(f'  mean variance {np.diag(C).mean():.3f}, RMS off-diagonal {out["V1e whitened draws: RMS off-diagonal covariance"]:.3f}')

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'df_v1.json'), 'w') as f:
        json.dump(out, f, indent=1)
    log('V1 done')


def v1_maps():
    """the maps of V1b (Gaussian means): DF's map and s.d. and the independent solver's, for the report"""
    P = setup()
    m = model(P, P.pb.yw)
    x = x_of(m, 0.3, 2.25, 1000., [62., 0.3])
    post = m.posterior(m.fit_at(x, False), P.pb.cells.area_km2)
    ref = reference(m, x)
    np.savez(os.path.join(RESULTS, 'df_v1.npz'), clon=P.pb.cells.clon, clat=P.pb.cells.clat, v_df=post['v'],
             sd_df=post['v_sd'], v_ref=ref['v'], sd_ref=ref['v_sd'], truth=P.pb.truth, label=P.label)
    log('V1 maps saved')


# ---------------------------------------------------------------- V2

P2 = None


def draw_truth(m, rng):
    """v from DF's own model at TRUE: v0 + delta_domain + s L_k z inside each domain"""
    _, Ls = m.factors(TRUE['beta'], TRUE['ell'])
    v = TRUE['v0'] + rng.normal(0., TRUE['tau'], m.K)[m.label]
    for b, L in zip(m.blocks, Ls):
        v[b] += TRUE['s']*(L @ rng.standard_normal(len(b)))
    return v


def one_draw(i):
    P = P2
    rng = np.random.default_rng(20261016 + i)
    m0 = model(P, P.pb.yw)
    v_true = draw_truth(m0, rng)
    yw = P.pb.Gw @ v_true + np.asarray(P.pb.W) @ P.pb.draw(rng, model_error=False)
    m = model(P, yw)
    area = P.pb.cells.area_km2
    c = m.S*area[:, None]
    c /= c.sum(0)
    avg_true = c.T @ v_true
    x_true = x_of(m, TRUE['s'], TRUE['beta'], TRUE['ell'], [TRUE['ring'], TRUE['tau']])
    fixed = m.posterior(m.fit_at(x_true, False), area)
    f = m.fit(**P.opts)
    fitted = m.posterior(f, area)
    sd = m.hyper_sd(f)
    out = dict(z_fixed=(fixed['v'] - v_true)/fixed['v_sd'], z_fit=(fitted['v'] - v_true)/fitted['v_sd'],
               za_fixed=(fixed['avg'] - avg_true)/fixed['avg_sd'], za_fit=(fitted['avg'] - avg_true)/fitted['avg_sd'],
               x_fit=f['x'], x_sd=sd, x_true=x_true, at_bound=f['at_bound'], ll=f['ll'])
    if i < 3:
        out.update(v_true=v_true, v_fit=fitted['v'], v_sd=fitted['v_sd'], v_fixed=fixed['v'], v_sd_fixed=fixed['v_sd'])
    log(f'  draw {i}: beta {f["beta"]:.2f}, ell {f["ell"]:.0f} km, tau {f["sig"][-1]:.2f}, '
        f'cells within 1 s.d.: fixed {np.mean(np.abs(out["z_fixed"]) < 1):.2f}, fitted {np.mean(np.abs(out["z_fit"]) < 1):.2f}')
    return out


def v2(n_draws):
    global P2
    P2 = setup()
    log(f'V2: {n_draws} truths drawn from DF at {TRUE}, {WORKERS} workers')
    with mp.get_context('fork').Pool(WORKERS) as pool:
        res = pool.map(one_draw, range(n_draws), chunksize=1)
    keys = ['z_fixed', 'z_fit', 'za_fixed', 'za_fit', 'x_fit', 'x_sd', 'x_true', 'at_bound', 'll']
    arrays = {k: np.array([r[k] for r in res]) for k in keys}
    for k in ('v_true', 'v_fit', 'v_sd', 'v_fixed', 'v_sd_fixed'):
        arrays[k] = np.array([r[k] for r in res if k in r])
    arrays.update(label=P2.label, clon=P2.pb.cells.clon, clat=P2.pb.cells.clat, ring=P2.ring,
                  plon=P2.pb.plon, plat=P2.pb.plat)
    np.savez(os.path.join(RESULTS, 'df_v2.npz'), **arrays)
    for k in ('z_fixed', 'z_fit', 'za_fixed', 'za_fit'):
        z = arrays[k]
        log(f'  {k}: s.d. of z {z.std():.3f}, within 1 s.d. {np.mean(np.abs(z) < 1):.3f}, within 2 s.d. '
            f'{np.mean(np.abs(z) < 2):.3f} (expected 1, 0.683, 0.954)')
    log('V2 done')


# ---------------------------------------------------------------- checkerboards (exact recovery)

def tiles(clon, clat, nside):
    """HEALPix tile of each cell (nested numbering) and the checkerboard sign of the tile (x + y parity in its face)"""
    from astropy_healpix import healpy as hp
    t = hp.ang2pix(nside, clon, clat, lonlat=True, nest=True)
    p = t % (nside*nside)
    x = sum(((p >> (2*b)) & 1) << b for b in range(8))
    y = sum(((p >> (2*b + 1)) & 1) << b for b in range(8))
    return t, (x + y) % 2


def checker():
    """DF with one domain per checkerboard tile and the tiles' VIS as the truth (1.5 / 0.5 SI km). With box means and
    noise-free data the domain means are estimated exactly (generalised least squares of an exactly fitting model), so
    the map must equal the truth to rounding; Gaussian means shrink the tiles toward their shared level."""
    P = setup()
    pb = P.pb
    rows = []
    maps = {}
    rng = np.random.default_rng(20261017)
    noise = np.asarray(pb.W) @ pb.draw(rng, model_error=False)
    for nside in (4, 8, 16):
        t, parity = tiles(pb.cells.clon, pb.cells.clat, nside)
        label = np.unique(t, return_inverse=True)[1]
        v_true = np.where(parity == 0, 1.5, 0.5)
        for noisy in (False, True):
            yw = pb.Gw @ v_true + (noise if noisy else 0.)
            for means in ('box', 'gaussian'):
                m = df.DomainFractal(pb.Gw, yw, label, pb.cells.clon, pb.cells.clat, P.cell_km2, P.lo, P.hi, P.extra,
                                     means, P.tau_range)
                t0 = time.time()
                f = m.fit(**P.opts)
                post = m.posterior(f, pb.cells.area_km2)
                err = post['v'] - v_true
                r = dict(tile_nside=nside, tiles=int(m.K), noise=noisy, means=means, ll=f['ll'],
                         max_abs_error=float(np.abs(err).max()), rms_error=float(np.sqrt(np.mean(err**2))),
                         max_error_over_sd=float(np.abs(err/post['v_sd']).max()),
                         contrast_recovered=float((post['v'][parity == 0].mean() - post['v'][parity == 1].mean())),
                         beta=f['beta'], ell_km=f['ell'], s_median=float(np.median(f['s'])),
                         tau=float(f['sig'][m.tau]) if m.gaussian else None, n_eval=f['n_eval'])
                rows.append(r)
                maps[f'{nside}_{int(noisy)}_{means}'] = np.c_[post['v'], post['v_sd']]
                maps[f'{nside}_truth'] = v_true
                log(f'tiles NSIDE {nside} ({m.K} tiles), noise {noisy}, {means} means: max |error| '
                    f'{r["max_abs_error"]:.2e} SI km, contrast {r["contrast_recovered"]:.3f} (true 1.000), '
                    f'max |error|/s.d. {r["max_error_over_sd"]:.2f} ({time.time() - t0:.0f} s)')
    with open(os.path.join(RESULTS, 'df_checker.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    np.savez(os.path.join(RESULTS, 'df_checker.npz'), clon=pb.cells.clon, clat=pb.cells.clat, **maps)
    log('checkerboards done')


# ---------------------------------------------------------------- shrinking checkerboards (resolution)

SIZES = [8., 6., 5., 4., 3., 2.5, 2., 1.5]           # lon/lat squares (degrees), about 900 to 170 km


def shrink_job(job):
    """one fit: (size, layout 'tiles' or 'one', means, noisy)"""
    size, layout, means, noisy = job
    P = P2
    pb = P.pb
    lon, lat = pb.cells.clon, pb.cells.clat
    i, j = np.floor(lat/size).astype(int), np.floor(np.mod(lon, 360)/size).astype(int)
    high = (i + j) % 2 == 0
    v_true = np.where(high, 1.5, 0.5)
    label = np.unique(i*10000 + j, return_inverse=True)[1] if layout == 'tiles' else np.zeros(len(lon), int)
    yw = pb.Gw @ v_true + (P.noise if noisy else 0.)
    t0 = time.time()
    try:
        m = df.DomainFractal(pb.Gw, yw, label, lon, lat, P.cell_km2, P.lo, P.hi, P.extra, means, P.tau_range)
        f = m.fit(**P.opts)
        post = m.posterior(f, pb.cells.area_km2)
    except Exception as e:                           # e.g. more tile means than the data can fix
        log(f'  {job}: failed ({type(e).__name__}: {e})')
        return dict(size=size, layout=layout, means=means, noise=noisy, tiles=int(label.max()) + 1, failed=str(e))
    v, sd = post['v'], post['v_sd']
    err = v - v_true
    r = dict(size=size, layout=layout, means=means, noise=noisy, tiles=int(label.max()) + 1,
             contrast=float(v[high].mean() - v[~high].mean()), rms_error=float(np.sqrt(np.mean(err**2))),
             max_abs_error=float(np.abs(err).max()), within_1sd=float(np.mean(np.abs(err/sd) < 1)),
             beta=f['beta'], ell_km=f['ell'], ll=f['ll'], seconds=time.time() - t0, v=v, sd=sd)
    log(f'  {size:g} deg, {layout} ({r["tiles"]} domains), {means}, noise {noisy}: contrast {r["contrast"]:.3f}, '
        f'RMS error {r["rms_error"]:.3f}, within 1 s.d. {r["within_1sd"]:.2f} ({r["seconds"]:.0f} s)')
    return r


def shrink():
    global P2
    P2 = setup()
    P2.noise = np.asarray(P2.pb.W) @ P2.pb.draw(np.random.default_rng(20261018), model_error=False)
    jobs = [(s, lay, means, noisy) for s in SIZES for noisy in (False, True)
            for lay, means in (('tiles', 'box'), ('tiles', 'gaussian'), ('one', 'box'))]
    jobs.sort(key=lambda j: (j[1] != 'tiles', j[0]))      # the many-domain fits first
    log(f'shrinking checkerboards: {len(jobs)} fits, {WORKERS} workers')
    with mp.get_context('fork').Pool(WORKERS) as pool:
        res = pool.map(shrink_job, jobs, chunksize=1)
    maps = {}
    for r in res:
        if 'v' in r:
            key = f'{r["size"]:g}_{r["layout"]}_{r["means"]}_{int(r["noise"])}'
            maps[key] = np.c_[r.pop('v'), r.pop('sd')]
    with open(os.path.join(RESULTS, 'df_shrink.json'), 'w') as f:
        json.dump(res, f, indent=1)
    np.savez(os.path.join(RESULTS, 'df_shrink.npz'), clon=P2.pb.cells.clon, clat=P2.pb.cells.clat, **maps)
    log('shrinking done')


# ---------------------------------------------------------------- checkerboards with the wrong domains

def squares(lon, lat, size):
    i, j = np.floor(lat/size).astype(int), np.floor(np.mod(lon, 360)/size).astype(int)
    return np.unique(i*10000 + j, return_inverse=True)[1], (i + j) % 2 == 0


def mismatch_job(job):
    size, layout, noisy = job
    P = P2
    pb = P.pb
    lon, lat = pb.cells.clon, pb.cells.clat
    tile, high = squares(lon, lat, size)
    v_true = np.where(high, 1.5, 0.5)
    label = tile if layout == 'the squares' else P.layouts[layout][0]
    yw = pb.Gw @ v_true + (P.noise if noisy else 0.)
    m = df.DomainFractal(pb.Gw, yw, label, lon, lat, P.cell_km2, P.lo, P.hi, P.extra, 'gaussian', P.tau_range)
    f = m.fit(**P.opts)
    post = m.posterior(f, pb.cells.area_km2)
    v, sd = post['v'], post['v_sd']
    err = v - v_true
    r = dict(size=size, layout=layout, noise=noisy, domains=int(m.K), ll=f['ll'],
             contrast=float(v[high].mean() - v[~high].mean()), rms_error=float(np.sqrt(np.mean(err**2))),
             within_1sd=float(np.mean(np.abs(err/sd) < 1)), beta=f['beta'], ell_km=f['ell'],
             tau=float(f['sig'][m.tau]) if m.gaussian else None, v=v, sd=sd)
    log(f'  {size:g} deg, {layout} ({m.K} domains), noise {noisy}: log evidence {f["ll"]:.1f}, contrast '
        f'{r["contrast"]:.3f}, RMS error {r["rms_error"]:.3f}, within 1 s.d. {r["within_1sd"]:.2f}')
    return r


def mismatch():
    """8 and 4 deg checkerboards inverted with the layouts of real runs: the polygons, random tiles as many as the
    polygons (the geology-blind control), and one domain; the squares themselves as the reference"""
    global P2
    P2 = setup()
    P2.noise = np.asarray(P2.pb.W) @ P2.pb.draw(np.random.default_rng(20261018), model_error=False)
    layouts = ['the squares', 'polygons'] + [k for k in P2.layouts if k.startswith('random tiles')] + ['one domain']
    jobs = [(s, lay, noisy) for s in (8., 4.) for noisy in (False, True) for lay in layouts]
    log(f'checkerboards with the wrong domains: {len(jobs)} fits, layouts {layouts}')
    with mp.get_context('fork').Pool(WORKERS) as pool:
        res = pool.map(mismatch_job, jobs, chunksize=1)
    maps = {f'{r["size"]:g}_{r["layout"]}_{int(r["noise"])}': np.c_[r.pop('v'), r.pop('sd')] for r in res}
    labels = {f'label_{k}': P2.layouts[k][0] for k in layouts[1:]}
    with open(os.path.join(RESULTS, 'df_mismatch.json'), 'w') as f:
        json.dump(res, f, indent=1)
    np.savez(os.path.join(RESULTS, 'df_mismatch.npz'), clon=P2.pb.cells.clon, clat=P2.pb.cells.clat, **maps, **labels)
    log('mismatch done')


# ---------------------------------------------------------------- the matching check (polygons right)

MATCH_CASES = [(0.0, True), (0.1, True), (0.3, True), (0.6, True), (0.3, False)]
# (s.d. of the polygon levels, pattern restarts at polygon edges); the pattern has s 0.3. (0, True): the polygons show
# only as breaks in the pattern; (0.3, False): only as steps in the level, on one pattern across Australia
MATCH_DRAWS = 3
MATCH_RANDOM = 3                                     # random tilings, as many tiles as polygons


def matching_job(job):
    (tau, restart), draw, layout = job
    P = P2
    pb = P.pb
    v_true = P.truths[(tau, restart, draw)]
    label = P.match_layouts[layout]
    yw = pb.Gw @ v_true + P.noise[draw]
    m = df.DomainFractal(pb.Gw, yw, label, pb.cells.clon, pb.cells.clat, P.cell_km2, P.lo, P.hi, P.extra, 'gaussian',
                         P.tau_range)
    f = m.fit(**P.opts)
    post = m.posterior(f, pb.cells.area_km2)
    v, sd = post['v'], post['v_sd']
    err = v - v_true
    c = P.S*pb.cells.area_km2[:, None]                # area means of the true polygons, true and recovered
    c /= c.sum(0)
    lev_true, lev = c.T @ v_true, c.T @ v
    r = dict(tau=tau, restart=restart, draw=draw, layout=layout, domains=int(m.K), ll=f['ll'], rms_error=float(np.sqrt(np.mean(err**2))),
             within_1sd=float(np.mean(np.abs(err/sd) < 1)), within_2sd=float(np.mean(np.abs(err/sd) < 2)),
             level_rms_error=float(np.sqrt(np.mean((lev - lev_true)**2))),
             level_slope=float(np.polyfit(lev_true - lev_true.mean(), lev - lev.mean(), 1)[0]),
             beta=f['beta'], ell_km=f['ell'], s_median=float(np.median(f['s'])),
             tau_fit=float(f['sig'][m.tau]) if m.gaussian else None, v=v, sd=sd)
    log(f'  tau {tau}, restart {restart}, draw {draw}, {layout}: log evidence {f["ll"]:.1f}, RMS error {r["rms_error"]:.3f}, '
        f'polygon levels slope {r["level_slope"]:.2f}, within 1 s.d. {r["within_1sd"]:.2f}')
    return r


def matching():
    """truths built with the polygons (a level per polygon from N(0.75, tau^2) plus DF's fractal pattern inside, s 0.3,
    beta 2.25, ell 1000 km, restarting at polygon edges or one pattern across Australia), noisy data, inverted with the polygons, random tilings of as many tiles, and one domain.
    Does the evidence pick the polygons when they are right, and by how much?"""
    global P2
    P2 = setup()
    pb = P2.pb
    m0 = model(P2, pb.yw)
    P2.S = m0.S
    P2.noise = [np.asarray(pb.W) @ pb.draw(np.random.default_rng(20261019 + d), model_error=False)
                for d in range(MATCH_DRAWS)]
    _, Ls = m0.factors(TRUE['beta'], TRUE['ell'])
    m1 = df.DomainFractal(pb.Gw, pb.yw, np.zeros(len(P2.label), int), pb.cells.clon, pb.cells.clat, P2.cell_km2,
                          P2.lo, P2.hi, P2.extra, 'box', P2.tau_range)
    L1 = m1.factors(TRUE['beta'], TRUE['ell'])[1][0]
    P2.truths = {}
    for d in range(MATCH_DRAWS):                      # the same pattern and level offsets for each case
        rng = np.random.default_rng(20261020 + d)
        z, pattern = rng.standard_normal(m0.K), np.zeros(len(P2.label))
        for b, L in zip(m0.blocks, Ls):
            pattern[b] = TRUE['s']*(L @ rng.standard_normal(len(b)))
        whole = TRUE['s']*(L1 @ rng.standard_normal(len(P2.label)))
        for tau, restart in MATCH_CASES:
            P2.truths[(tau, restart, d)] = TRUE['v0'] + tau*z[P2.label] + (pattern if restart else whole)
    K = m0.K
    P2.match_layouts = {'polygons': P2.label, 'one domain': np.zeros(len(P2.label), int)}
    for i in range(MATCH_RANDOM):
        lab = parameterise.voronoi(pb.cells, K, parameterise.VORONOI_SEED + 1000 + i)
        P2.match_layouts[f'random tiles {i + 1}'] = np.unique(lab, return_inverse=True)[1]
    jobs = [(c, d, lay) for c in MATCH_CASES for d in range(MATCH_DRAWS) for lay in P2.match_layouts]
    log(f'matching check: {len(jobs)} fits, layouts {list(P2.match_layouts)}')
    with mp.get_context('fork').Pool(WORKERS) as pool:
        res = pool.map(matching_job, jobs, chunksize=1)
    maps = {f'{r["tau"]:g}_{int(r["restart"])}_{r["layout"]}': np.c_[r.pop('v'), r.pop('sd')] for r in res
            if r['draw'] == 0}
    for r in res:
        r.pop('v', None), r.pop('sd', None)
    truths = {f'truth_{t:g}_{int(rs)}': P2.truths[(t, rs, 0)] for t, rs in MATCH_CASES}
    labels = {f'label_{k}': v for k, v in P2.match_layouts.items()}
    with open(os.path.join(RESULTS, 'df_matching.json'), 'w') as f:
        json.dump(res, f, indent=1)
    np.savez(os.path.join(RESULTS, 'df_matching.npz'), clon=pb.cells.clon, clat=pb.cells.clat, **maps, **truths,
             **labels)
    log('matching done')


if __name__ == '__main__':
    if sys.argv[1] == 'matching':
        matching()
    elif sys.argv[1] == 'mismatch':
        mismatch()
    elif sys.argv[1] == 'v1maps':
        v1_maps()
    elif sys.argv[1] == 'v1':
        v1()
    elif sys.argv[1] == 'checker':
        checker()
    elif sys.argv[1] == 'shrink':
        shrink()
    else:
        v2(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
