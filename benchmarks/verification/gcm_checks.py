"""
Verification of GCM (remit.inversion.methods.gcm: the global cell map, one level), the global fit of all continental
crust to the Gauss coefficients of degrees 16-133 (22591 cells, 17700 data), as df_checks.py for DF.

GCM: m = v0 + x/sqrt(a) per cell, x ~ N(0, tau^2 I) (area-scaled, no correlation between cells), y = u v0 + B x + e,
e ~ N(0, sigma^2 I) after the per-degree noise weighting; v0 flat (profiled), alpha^2 = sigma^2/tau^2 by the
evidence, sigma fitted. Final mode solves by one eigendecomposition (cached).

G1  correctness: the map, the s.d. (given v0, as GCM reports it, and with v0's uncertainty) and the log evidence at
    fixed alpha^2 against an independent solver (a data-space Cholesky factor of I + BB'/alpha^2); a uniform truth
    without noise, which GCM must return exactly
G2  calibration: truths from GCM's own model (v0 0.75, tau 0.5 SI km) with white noise (sigma 1); the errors over the
    reported s.d., with alpha^2 and sigma at the truth and fitted
G3  checkerboards of equal-area HEALPix tiles (NSIDE 4, 8, 16, 32: about 1630, 815, 407 and 204 km across; 1.5 / 0.5
    SI km), without noise, with the noise of LCS-1 (sigma 1), and with that noise and a wrong ocean model (the true
    oceans with HM05 remanence, the background with GK07, as gcm_glc_syn_u1.0_oceanerr)

Run from the repository root (one process, ~10-20 GB of memory; G1 ~15 min, G2 and G3 a few min each):
    PYTHONPATH=benchmarks/verification:benchmarks:notebooks:$PYTHONPATH VECLIB_MAXIMUM_THREADS=3 \
        nice -n 15 conda run -n curiesphere python benchmarks/verification/gcm_checks.py g1|g2|g3
Outputs: benchmarks/verification/results/gcm_*.json, gcm_*.npz
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from df_checks import RESULTS, log, tiles
from remit.inversion import config, core, data
from remit.inversion.cache import Cache
from remit.inversion.methods import gcm

CONFIG = 'inversions/gcm_glc_lcs1_16-133.toml'
TRUE = dict(v0=0.75, tau=0.5, sigma=1.)             # as the real GCM-GLC fit (v0 0.751, prior s.d. 0.51 SI km)


def setup():
    cfg = config.load(CONFIG)
    cache = Cache()
    pb = gcm.global_problem(cfg, cache, log)
    comp, d = cfg['compute'], cfg['data']
    spec = dict(columns=pb.store.spec, coarsen=comp.get('coarsen', 0), band=list(d['band']),
                filter=d.get('filter', 'lcs1'), weights=[pb.model, pb.noise['reference']],
                products=comp['products_backend'], version=gcm.EIG_VERSION)
    d_eig = cache.entry('eig', spec)
    eig = (np.load(f'{d_eig}/lam.npy'), np.load(f'{d_eig}/E.npy')) if os.path.exists(f'{d_eig}/E.npy') else None
    log('eigendecomposition from the cache' if eig else 'eigendecomposition (not cached: ~15 min)')
    pb.solver = core.LevelRidge(pb.B, pb.u, 'eigh', comp['products_backend'], eig)
    pb.cfg, pb.cache = cfg, cache
    return pb


def ocean_error(pb, true='HM05'):
    """the weighted data of a wrong ocean model: the oceans' field with remanence model `true` minus that of the
    background GCM subtracts (GK07), as gcm.global_problem's ocean_error"""
    cfg = pb.cfg
    bg, geo, tg = cfg['fit']['background'], cfg['geometry'], cfg['target']
    C = [data.background(pb.cache, pb.mask, tg['area'], geo['lmax'], bg['vis_model'], rem, bg['near_ridge'])
         for rem in (true, bg['remanence'])]
    return pb.w*pb.F[pb.band.l]*pb.band.rows(C[0] - C[1])


def gcm_map(pb, y, alpha2=None, sigma=None):
    """GCM as run(): alpha^2 by the evidence (sigma fitted) unless given; the map, its s.d. (given v0, as GCM reports
    it) and with v0's uncertainty"""
    s, sa = pb.solver, np.sqrt(pb.a)
    if alpha2 is None:
        alpha2 = s.by_evidence(y, True)
    sol = s.solve(y, alpha2)
    sigma = s.sigma(sol, True) if sigma is None else sigma
    m = sol['v0'] + sol['x']/sa
    sd = s.posterior_sd_x(alpha2, sigma)/sa
    sd_full = s.cell_sd(alpha2, sigma, sa, np.ones((len(sa), 1)))
    return dict(m=m, sd=sd, sd_full=sd_full, v0=sol['v0'], alpha2=alpha2, sigma=sigma,
                ll=s.log_evidence(y, alpha2, False))


# ---------------------------------------------------------------- G1

def reference(pb, y, alpha2):
    """the same model by a data-space Cholesky factor of K = I + BB'/alpha^2 (sigma 1): v0 by generalised least
    squares, x = B'K^-1 (y - u v0)/alpha^2; the s.d. of x given v0 and with v0 integrated out; the log evidence with
    v0 at its estimate, -1/2 (log det K + r'K^-1 r) (GCM's 'profile' evidence with sigma fixed)"""
    B, u, sa = pb.B, pb.u, np.sqrt(pb.a)
    log('reference: K = I + BB\'/alpha^2')
    K = B @ B.T/alpha2
    K[np.diag_indices_from(K)] += 1.
    cf = cho_factor(K, lower=True, overwrite_a=True)
    Ky, Ku = cho_solve(cf, y), cho_solve(cf, u)
    s = float(u @ Ku)
    v0 = float(u @ Ky)/s
    r = y - u*v0
    Kr = Ky - Ku*v0
    x = B.T @ Kr/alpha2
    ll = -float(np.sum(np.log(np.diag(cf[0])))) - 0.5*float(r @ Kr)
    log('reference: diag(B\'K^-1 B)')
    Z = solve_triangular(cf[0], B, lower=True)                     # L^-1 B
    dBKB = np.einsum('ij,ij->j', Z, Z)
    del Z
    var_given = 1/alpha2 - dBKB/alpha2**2                          # x given v0
    BKu = B.T @ Ku/alpha2                                          # Sigma B'K^-1 u
    var_full = var_given + (sa - BKu)**2/s                         # x_total = sqrt(a) v0 + x, v0 integrated out
    return dict(m=v0 + x/sa, sd=np.sqrt(var_given)/sa, sd_full=np.sqrt(var_full)/sa, v0=v0, ll=ll)


def g1():
    pb = setup()
    rng = np.random.default_rng(20261024)
    sa = np.sqrt(pb.a)
    x_true = TRUE['tau']*rng.standard_normal(len(sa))
    truth = TRUE['v0'] + x_true/sa
    y = pb.u*TRUE['v0'] + pb.B @ x_true + rng.standard_normal(len(pb.y))
    alpha2 = (TRUE['sigma']/TRUE['tau'])**2
    g = gcm_map(pb, y, alpha2, 1.)
    ref = reference(pb, y, alpha2)
    out = dict(v0=g['v0'], v0_ref=ref['v0'], map_max_diff=float(np.abs(g['m'] - ref['m']).max()),
               sd_max_rel_diff=float(np.abs(g['sd']/ref['sd'] - 1).max()),
               sd_full_max_rel_diff=float(np.abs(g['sd_full']/ref['sd_full'] - 1).max()),
               sd_v0_share_max=float(np.max(ref['sd_full']/ref['sd'] - 1)))
    # the evidence: GCM's profile evidence differs from the reference by the constant it leaves out (none here), so
    # compare at two damping values as well
    g2 = gcm_map(pb, y, 2*alpha2, 1.)
    ref2 = reference(pb, y, 2*alpha2)
    out.update(ll=g['ll'], ll_ref=ref['ll'], ll_diff=abs(g['ll'] - ref['ll']),
               ll_change=g2['ll'] - g['ll'], ll_change_ref=ref2['ll'] - ref['ll'])
    log(f'map {out["map_max_diff"]:.1e} SI km, v0 {g["v0"]:.6f} vs {ref["v0"]:.6f}, s.d. given v0 '
        f'{out["sd_max_rel_diff"]:.1e}, s.d. with v0 {out["sd_full_max_rel_diff"]:.1e} (relative); v0 adds at most '
        f'{100*out["sd_v0_share_max"]:.2f}% to a cell s.d.; log evidence {g["ll"]:.4f} vs {ref["ll"]:.4f}, change '
        f'with alpha^2 x 2: {out["ll_change"]:.4f} vs {out["ll_change_ref"]:.4f}')
    u_true = np.full(len(sa), 0.75)
    gu = gcm_map(pb, pb.u*0.75)
    out['uniform'] = dict(max_abs_error=float(np.abs(gu['m'] - u_true).max()), v0=gu['v0'], alpha2=gu['alpha2'])
    log(f'uniform truth 0.75, no noise: max |error| {out["uniform"]["max_abs_error"]:.1e} SI km '
        f'(alpha^2 {gu["alpha2"]:.3g})')
    with open(os.path.join(RESULTS, 'gcm_g1.json'), 'w') as f:
        json.dump(out, f, indent=1)
    np.savez(os.path.join(RESULTS, 'gcm_g1.npz'), clon=pb.cells.clon, clat=pb.cells.clat, truth=truth, m=g['m'],
             m_ref=ref['m'], sd=g['sd'], sd_ref=ref['sd'])
    log('G1 done')


# ---------------------------------------------------------------- G2

def g2(n=30):
    pb = setup()
    sa = np.sqrt(pb.a)
    z = dict(fixed=[], fit=[], fit_full=[])
    fits, keep = [], {}
    for i in range(n):
        rng = np.random.default_rng(20261025 + i)
        truth = TRUE['v0'] + TRUE['tau']*rng.standard_normal(len(sa))/sa
        y = pb.u*TRUE['v0'] + pb.B @ (sa*(truth - TRUE['v0'])) + TRUE['sigma']*rng.standard_normal(len(pb.y))
        gx = gcm_map(pb, y, (TRUE['sigma']/TRUE['tau'])**2, TRUE['sigma'])
        gf = gcm_map(pb, y)
        z['fixed'].append((gx['m'] - truth)/gx['sd'])
        z['fit'].append((gf['m'] - truth)/gf['sd'])
        z['fit_full'].append((gf['m'] - truth)/gf['sd_full'])
        fits.append((gf['sigma'], gf['sigma']/np.sqrt(gf['alpha2']), gf['v0']))
        if i == 0:
            keep = dict(truth=truth, m=gf['m'], sd=gf['sd'])
        log(f'  truth {i}: sigma {fits[-1][0]:.3f}, tau {fits[-1][1]:.3f}, v0 {fits[-1][2]:.3f}; within 1 s.d.: '
            f'fixed {np.mean(np.abs(z["fixed"][-1]) < 1):.3f}, fitted {np.mean(np.abs(z["fit"][-1]) < 1):.3f}')
    arrays = {f'z_{k}': np.array(v) for k, v in z.items()}
    summary = {k: dict(sd=float(v.std()), within_1sd=float(np.mean(np.abs(v) < 1)),
                       within_2sd=float(np.mean(np.abs(v) < 2))) for k, v in arrays.items()}
    f = np.array(fits)
    summary['fitted'] = dict(sigma=np.percentile(f[:, 0], [16, 50, 84]).tolist(),
                             tau=np.percentile(f[:, 1], [16, 50, 84]).tolist(),
                             v0=np.percentile(f[:, 2], [16, 50, 84]).tolist())
    for k, v in summary.items():
        log(f'  {k}: {v}')
    with open(os.path.join(RESULTS, 'gcm_g2.json'), 'w') as fo:
        json.dump(summary, fo, indent=1)
    np.savez(os.path.join(RESULTS, 'gcm_g2.npz'), clon=pb.cells.clon, clat=pb.cells.clat, **arrays, **keep)
    log('G2 done')


# ---------------------------------------------------------------- G3

def g3():
    pb = setup()
    sa = np.sqrt(pb.a)
    rng = np.random.default_rng(20261026)
    e = rng.standard_normal(len(pb.y))
    oe = ocean_error(pb)
    log(f'ocean-model error: RMS {np.sqrt(np.mean(oe**2)):.2f} noise s.d. per datum')
    rows, maps = [], {}
    for nside in (4, 8, 16, 32):
        _, parity = tiles(pb.cells.clon, pb.cells.clat, nside)
        high = parity == 0
        truth = np.where(high, 1.5, 0.5)
        y0 = pb.B @ (sa*truth)
        for noise in ('none', 'noise', 'noise and ocean error'):
            g = gcm_map(pb, y0 + {'none': 0., 'noise': e, 'noise and ocean error': e + oe}[noise])
            err = g['m'] - truth
            r = dict(nside=nside, size_km=float(np.sqrt(4*np.pi*6371.**2/(12*nside**2))), noise=noise,
                     contrast=float(g['m'][high].mean() - g['m'][~high].mean()),
                     rms_error=float(np.sqrt(np.mean(err**2))), within_1sd=float(np.mean(np.abs(err/g['sd']) < 1)),
                     sigma=g['sigma'], tau=g['sigma']/np.sqrt(g['alpha2']), v0=g['v0'])
            rows.append(r)
            maps[f'{nside}_{noise}'] = np.c_[g['m'], g['sd']]
            maps[f'{nside}_truth'] = truth
            log(f'  tiles NSIDE {nside} ({r["size_km"]:.0f} km), {noise}: contrast {r["contrast"]:.3f}, RMS error '
                f'{r["rms_error"]:.3f}, within 1 s.d. {r["within_1sd"]:.2f}, prior s.d. {r["tau"]:.2f}')
    with open(os.path.join(RESULTS, 'gcm_shrink.json'), 'w') as f:
        json.dump(rows, f, indent=1)
    np.savez(os.path.join(RESULTS, 'gcm_shrink.npz'), clon=pb.cells.clon, clat=pb.cells.clat, **maps)
    log('G3 done')


if __name__ == '__main__':
    {'g1': g1, 'g2': lambda: g2(int(sys.argv[2]) if len(sys.argv) > 2 else 30), 'g3': g3}[sys.argv[1]]()
