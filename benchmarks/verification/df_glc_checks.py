"""
Verification of DF-GLC (remit.inversion.methods.df_global: DF's model in the global fit, ADR 0010), on the real
global problem (22591 cells, 17700 coefficients, the ocean-model error at 16 nT), as df_checks.py for DF.

DF-GLC: v = v0 + delta_d + p, delta ~ N(0, tau^2) per domain, p a fractal pattern inside each domain (shared s, beta,
ell), y = u v0 + G v + e after whitening. For one layout and shape (beta, ell) df_global.Shape decomposes
A = G blockdiag(R_d) G' once (~21 min); the evidence in (s, tau) and the posterior then cost little, so the
decompositions are cached on disk here and reused for every truth.

D1  correctness at one shape (groups layout, beta 2, ell 700 km): the log evidence, the posterior mean and s.d. of
    every cell against an independent solver (the bordered system with G Sigma built block by block, no
    eigendecomposition); the fast posterior used in D2 against Shape.posterior
D2  calibration: 20 truths from DF-GLC's own model (groups layout, v0 0.7, tau 0.1, s 0.4, beta 2, ell 700 km), white
    errors after whitening; the errors over the reported s.d., with the settings at the truth, fitted at the truth's
    shape, and fitted with the shape searched over the real run's grid (beta 2, 2.5, 3 x ell 350, 700, 1400 km)
D3  checkerboards of equal-area tiles (as gcm_checks g3), one domain (one pattern without breaks), the shape searched
    over the same grid; without noise and with noise

Run from the repository root (D1 ~45 min, D2 ~3.5 h, D3 ~3.5 h; decompositions cached, so a stopped run resumes):
    PYTHONPATH=benchmarks/verification:benchmarks:notebooks:$PYTHONPATH VECLIB_MAXIMUM_THREADS=3 \
        nice -n 15 conda run -n curiesphere python benchmarks/verification/df_glc_checks.py d1|d2|d3
Outputs: benchmarks/verification/results/df_glc_*.json, df_glc_*.npz
"""
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular

from df_checks import HERE, RESULTS, log, tiles
from remit.inversion import config, graph, parameterise
from remit.inversion.cache import Cache
from remit.inversion.methods import df, df_global as dg, gcm

CONFIG = 'inversions/df_glc_lcs1_16-133.toml'
TRUE = dict(v0=0.7, tau=0.1, s=0.4, beta=2.0, ell=700.)     # as the real groups fit (v0 0.69, s 0.42, beta 2, 700 km)
BETAS, ELLS = (2.0, 2.5, 3.0), (350., 700., 1400.)
CACHE = os.path.join(HERE, 'cache', 'df_glc')
MEMO, PROJ = {}, {}                                         # decomposed shapes and their projections, in memory


def setup():
    """the problem as df_global.run builds it: G (whitened coefficients of each cell's VIS), u, the ocean-model
    whitening, the layouts"""
    cfg = config.load(CONFIG)
    cache = Cache()
    pb = gcm.global_problem(cfg, cache, log)
    G = pb.B
    G *= np.sqrt(pb.a)[None, :]
    eb = cfg['errors']['background']
    C_oc, rms1, _ = gcm._ocean_covariance(cfg, pb.band, pb.w, pb.F, cache, log)
    rms = float(np.atleast_1d(eb['rms_nT'])[0])
    Lc = cholesky(np.eye(len(pb.y)) + (rms/rms1)**2*C_oc, lower=True)
    del C_oc
    G = solve_triangular(Lc, G, lower=True, overwrite_b=True)
    u = solve_triangular(Lc, pb.u, lower=True)
    lays, _ = parameterise.layouts(pb.cells, cfg['parameterisation']['tables'], cache, cfg['target']['area'],
                                   ['one', 'groups'])
    lo, hi = cfg['method'].get('bounds', [0., 4.])
    return SimpleNamespace(cfg=cfg, pb=pb, G=G, u=u, Lc=Lc, ll_whiten=-float(np.sum(np.log(np.diag(Lc)))),
                           lays=lays, lo=lo, hi=hi, xyz=graph.unit(pb.cells.clon, pb.cells.clat),
                           window=df.cap_window(4*np.pi*df.R_KM**2/(12*pb.cells.nside**2)))


def shape(P, layout, beta, ell, y):
    """df_global.Shape for this layout and shape, its decomposition read from the cache when there"""
    if (layout, beta, ell) in MEMO:
        return with_data(MEMO[layout, beta, ell], y)
    label = P.lays[layout][0]
    f = os.path.join(CACHE, f'{layout}_{beta:g}_{ell:g}.npz')
    if os.path.exists(f):
        z = np.load(f)
        sh = dg.Shape.__new__(dg.Shape)                 # Shape without its decomposition, then the cached one
        sh.G, sh.u, sh.y, sh.label = P.G, P.u, y, np.asarray(label)
        sh.beta, sh.ell, sh.lo, sh.hi, sh.ll_whiten = beta, ell, P.lo, P.hi, P.ll_whiten
        sh.K = int(sh.label.max()) + 1
        sh.blocks = [np.flatnonzero(sh.label == k) for k in range(sh.K)]
        sh.grid, sh.tab, sh.xyz = z['grid'], z['tab'], P.xyz
        sh.lam, sh.V = z['lam'], z['V']
        sh.ut = sh.V.T @ P.u
        sh.Ut = sh.V.T @ (P.G @ parameterise.indicator(sh.label)) if sh.K > 1 else None
    else:
        t = time.time()
        sh = dg.Shape(P.G, P.u, y, label, P.xyz, P.window, beta, ell, P.lo, P.hi, P.ll_whiten)
        os.makedirs(CACHE, exist_ok=True)
        np.savez(f + '.tmp.npz', lam=sh.lam, V=sh.V, grid=sh.grid, tab=sh.tab)
        os.replace(f + '.tmp.npz', f)
        log(f'  decomposed {layout}, beta {beta:g}, ell {ell:g} km ({(time.time() - t)/60:.1f} min)')
    MEMO[layout, beta, ell] = sh
    return with_data(sh, y)


def with_data(sh, y):
    sh.y, sh.yt = y, sh.V.T @ y
    return sh


def fast_posterior(sh, f, VM, VGS):
    """the posterior mean and s.d. of every cell as Shape.posterior, with V'(G R) and V'(G S) computed once per
    shape (they do not depend on the data or on s and tau)"""
    s, tau, h = f['s'], f['tau'] or 0., f['h']
    Ci, _ = sh._parts(s, tau if sh.K > 1 else 1., h)
    Cu = Ci(sh.ut)
    Au = float(sh.ut @ Cu)
    v0 = float(sh.ut @ Ci(sh.yt))/Au
    qt = Ci(sh.yt - sh.ut*v0)
    Gq = sh.G.T @ (sh.V @ qt)
    n = len(sh.label)
    pat = np.empty(n)
    for b in sh.blocks:
        pat[b] = sh.R(b) @ Gq[b]
    S = parameterise.indicator(sh.label) if sh.K > 1 else None
    v = v0 + s**2*pat + (S @ (tau**2*(S.T @ Gq)) if S is not None else 0.)
    prior = s**2 + (tau**2 if S is not None else 0.)
    red, bt = np.empty(n), np.empty(n)
    for k in range(0, n, 2000):
        Yk = s**2*VM[:, k:k + 2000] + (tau**2*(VGS @ np.asarray(S[k:k + 2000]).T) if S is not None else 0.)
        CY = Ci(Yk)
        red[k:k + 2000] = np.sum(Yk*CY, 0)
        bt[k:k + 2000] = 1. - CY.T @ sh.ut
    return dict(v=v, sd=np.sqrt(np.maximum(prior - red + bt**2/Au, 0.)), v0=v0)


def projections(sh, keep=2):
    """V'(G R) (data x cells) and V'(G S) (data x domains), the last `keep` shapes' kept in memory"""
    key = (int(sh.K), sh.beta, sh.ell)
    if key not in PROJ:
        while len(PROJ) >= keep:
            PROJ.pop(next(iter(PROJ)))
        PROJ[key] = (sh.V.T @ sh.pattern_columns(),
                     sh.V.T @ (sh.G @ parameterise.indicator(sh.label)) if sh.K > 1 else None)
    return PROJ[key]


def draw_truth(P, layout, rng, chol):
    label = P.lays[layout][0]
    K = label.max() + 1
    v = TRUE['v0'] + TRUE['tau']*rng.standard_normal(K)[label]
    for b, L in chol:
        v[b] += TRUE['s']*(L @ rng.standard_normal(len(b)))
    return v


# ---------------------------------------------------------------- D1

def d1():
    P = setup()
    rng = np.random.default_rng(20261027)
    sh0 = shape(P, 'groups', TRUE['beta'], TRUE['ell'], np.zeros(len(P.u)))
    chol = [(b, cholesky(sh0.R(b) + 1e-10*np.eye(len(b)), lower=True)) for b in sh0.blocks]
    truth = draw_truth(P, 'groups', rng, chol)
    y = P.G @ truth + rng.standard_normal(len(P.u))
    sh = with_data(sh0, y)
    f = dict(s=TRUE['s'], tau=TRUE['tau'], h=1.)
    ll, v0 = sh.evidence(f['s'], f['tau'], f['h'])
    post = sh.posterior(f, None, np.ones(len(y)), P.pb.cells.area_km2)
    fast = fast_posterior(sh, f, *projections(sh))
    PROJ.clear()

    log('reference: G Sigma block by block')
    S = parameterise.indicator(sh.label)
    GS = np.empty_like(P.G)
    for b in sh.blocks:
        GS[:, b] = f['s']**2*(P.G[:, b] @ sh.R(b))
    GS += f['tau']**2*((P.G @ S) @ S.T)
    log('reference: K = G Sigma G\' + I and its Cholesky factor')
    K = GS @ P.G.T
    K = 0.5*(K + K.T)
    K[np.diag_indices_from(K)] += f['h']**2
    cf = cho_factor(K, lower=True, overwrite_a=True)
    Ky, Ku = cho_solve(cf, y), cho_solve(cf, P.u)
    su = float(P.u @ Ku)
    v0r = float(P.u @ Ky)/su
    a = Ky - Ku*v0r
    N = len(y)
    ll_ref = (-float(np.sum(np.log(np.diag(cf[0])))) - 0.5*np.log(su) - 0.5*float((y - P.u*v0r) @ a)
              - 0.5*(N - 1)*np.log(2*np.pi) - np.log(P.hi - P.lo) + P.ll_whiten)
    v_ref = v0r + GS.T @ a
    log('reference: the s.d. of every cell')
    Z = solve_triangular(cf[0], GS, lower=True)                      # L^-1 (G Sigma)
    red = np.einsum('ij,ij->j', Z, Z)
    del Z
    bt = 1. - GS.T @ Ku
    sd_ref = np.sqrt(np.maximum(f['s']**2 + f['tau']**2 - red + bt**2/su, 0.))
    out = dict(ll=ll, ll_ref=ll_ref, ll_diff=abs(ll - ll_ref), v0=v0, v0_ref=v0r,
               map_max_diff=float(np.abs(post['v'] - v_ref).max()),
               sd_max_rel_diff=float(np.abs(post['sd']/sd_ref - 1).max()),
               fast_map_max_diff=float(np.abs(fast['v'] - post['v']).max()),
               fast_sd_max_rel_diff=float(np.abs(fast['sd']/post['sd'] - 1).max()))
    log(f'log evidence {ll:.4f} vs {ll_ref:.4f}; map {out["map_max_diff"]:.1e} SI km; s.d. {out["sd_max_rel_diff"]:.1e} '
        f'(relative); the fast posterior against Shape.posterior: map {out["fast_map_max_diff"]:.1e}, s.d. '
        f'{out["fast_sd_max_rel_diff"]:.1e}')
    with open(os.path.join(RESULTS, 'df_glc_d1.json'), 'w') as fo:
        json.dump(out, fo, indent=1)
    np.savez(os.path.join(RESULTS, 'df_glc_d1.npz'), clon=P.pb.cells.clon, clat=P.pb.cells.clat, truth=truth,
             v=post['v'], v_ref=v_ref, sd=post['sd'], sd_ref=sd_ref, label=sh.label)
    log('D1 done')


# ---------------------------------------------------------------- D2

def best_shape(P, layout, y):
    """the real run's search, with every shape of the grid evaluated (decompositions from the cache)"""
    fits = {}
    for beta in BETAS:
        for ell in ELLS:
            fits[beta, ell] = shape(P, layout, beta, ell, y).fit()
    return max(fits, key=lambda k: fits[k]['ll']), fits


def d2(n=20):
    P = setup()
    for beta in BETAS:                                     # all decompositions first (cached; ~3 h the first time)
        for ell in ELLS:
            shape(P, 'groups', beta, ell, np.zeros(len(P.u)))
    sh_t = shape(P, 'groups', TRUE['beta'], TRUE['ell'], np.zeros(len(P.u)))
    chol = [(b, cholesky(sh_t.R(b) + 1e-10*np.eye(len(b)), lower=True)) for b in sh_t.blocks]
    z = dict(fixed=[], fit_shape=[], fit=[])
    rows, keep = [], {}
    for i in range(n):
        rng = np.random.default_rng(20261028 + i)
        truth = draw_truth(P, 'groups', rng, chol)
        y = P.G @ truth + rng.standard_normal(len(P.u))
        key, fits = best_shape(P, 'groups', y)
        sh = shape(P, 'groups', TRUE['beta'], TRUE['ell'], y)
        fx = fast_posterior(sh, dict(s=TRUE['s'], tau=TRUE['tau'], h=1.), *projections(sh))
        f_t = fits[TRUE['beta'], TRUE['ell']]
        ft = fast_posterior(sh, f_t, *projections(sh))
        shb = shape(P, 'groups', key[0], key[1], y)
        fb = fast_posterior(shb, fits[key], *projections(shb))
        for k, p in (('fixed', fx), ('fit_shape', ft), ('fit', fb)):
            z[k].append((p['v'] - truth)/p['sd'])
        rows.append(dict(best_beta=key[0], best_ell=key[1], s=fits[key]['s'], tau=fits[key]['tau'],
                         s_at_truth_shape=f_t['s'], tau_at_truth_shape=f_t['tau']))
        if i == 0:
            keep = dict(truth=truth, v=fb['v'], sd=fb['sd'], label=P.lays['groups'][0])
        log(f'  truth {i}: best shape beta {key[0]:g}, ell {key[1]:g} km, s {fits[key]["s"]:.3f}, tau '
            f'{fits[key]["tau"]:.3f}; within 1 s.d.: fixed {np.mean(np.abs(z["fixed"][-1]) < 1):.3f}, fitted at the '
            f'true shape {np.mean(np.abs(z["fit_shape"][-1]) < 1):.3f}, shape searched {np.mean(np.abs(z["fit"][-1]) < 1):.3f}')
    arrays = {f'z_{k}': np.array(v) for k, v in z.items()}
    summary = {k: dict(sd=float(v.std()), within_1sd=float(np.mean(np.abs(v) < 1)),
                       within_2sd=float(np.mean(np.abs(v) < 2))) for k, v in arrays.items()}
    summary['fits'] = rows
    for k, v in summary.items():
        if k != 'fits':
            log(f'  {k}: {v}')
    with open(os.path.join(RESULTS, 'df_glc_d2.json'), 'w') as fo:
        json.dump(summary, fo, indent=1)
    np.savez(os.path.join(RESULTS, 'df_glc_d2.npz'), clon=P.pb.cells.clon, clat=P.pb.cells.clat, **arrays, **keep)
    log('D2 done')


# ---------------------------------------------------------------- D3

def d3():
    P = setup()
    for beta in BETAS:
        for ell in ELLS:
            shape(P, 'one', beta, ell, np.zeros(len(P.u)))
    rng = np.random.default_rng(20261026)                   # as gcm_checks g3: the same noise draw
    e = rng.standard_normal(len(P.u))
    lon, lat = P.pb.cells.clon, P.pb.cells.clat
    rows, maps = [], {}
    for nside in (4, 8, 16, 32):
        high = tiles(lon, lat, nside)[1] == 0
        truth = np.where(high, 1.5, 0.5)
        y0 = P.G @ truth
        for noise in ('none', 'noise'):
            y = y0 + (e if noise == 'noise' else 0.)
            key, fits = best_shape(P, 'one', y)
            sh = shape(P, 'one', key[0], key[1], y)
            post = fast_posterior(sh, fits[key], *projections(sh))
            err = post['v'] - truth
            r = dict(nside=nside, size_km=float(np.sqrt(4*np.pi*6371.**2/(12*nside**2))), noise=noise,
                     contrast=float(post['v'][high].mean() - post['v'][~high].mean()),
                     rms_error=float(np.sqrt(np.mean(err**2))), within_1sd=float(np.mean(np.abs(err/post['sd']) < 1)),
                     beta=key[0], ell_km=key[1], s=fits[key]['s'], ll=fits[key]['ll'])
            rows.append(r)
            maps[f'{nside}_{noise}'] = np.c_[post['v'], post['sd']]
            log(f'  tiles NSIDE {nside} ({r["size_km"]:.0f} km), {noise}: contrast {r["contrast"]:.3f}, RMS error '
                f'{r["rms_error"]:.3f}, within 1 s.d. {r["within_1sd"]:.2f}, shape beta {key[0]:g}, ell {key[1]:g} km')
    with open(os.path.join(RESULTS, 'df_glc_shrink.json'), 'w') as fo:
        json.dump(rows, fo, indent=1)
    np.savez(os.path.join(RESULTS, 'df_glc_shrink.npz'), clon=lon, clat=lat, **maps)
    log('D3 done')


if __name__ == '__main__':
    {'d1': d1, 'd2': lambda: d2(int(sys.argv[2]) if len(sys.argv) > 2 else 20), 'd3': d3}[sys.argv[1]]()
