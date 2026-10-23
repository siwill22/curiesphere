"""
Verification of GCM with domain levels (GCM-L: remit.inversion.methods.gcm with [parameterisation] kind = 'layouts',
ADR 0008) in the global fit with the ocean-model error at 16 nT (ADR 0009), as the candidate-set runs use it; then
GCM-L against DF-GLC on the same truths: does each method's evidence pick the right layout, and can the domain means
each reports be read with their s.d.?

GCM-L: v = v0 + delta_d + x/sqrt(a) per cell, delta_d ~ N(0, sigma^2/beta) per domain, x ~ N(0, sigma^2/alpha^2)
(independent cells), v0 flat (profiled); y = G v + e after whitening (e white, sigma 1: the noise held at its
measured level). alpha^2 and beta by the evidence.

L1  correctness on the provinces layout: gcm._fit_layout against an independent solver (the data-space covariance
    K = G Sigma G' + I built from Sigma = diag(1/(alpha^2 a)) + S S'/beta, Cholesky, no eigendecomposition) at the
    fitted (alpha^2, beta): the log evidence (and its change between two settings), the map, the s.d. of every cell,
    the domain levels and their s.d., and the s.d. of each domain mean; a truth of one level per province without
    noise, which GCM-L must return exactly
L2  layouts: four truths (A GCM-L's own model on the provinces; B DF-GLC's model on the provinces, the pattern
    restarting at province edges; C one pattern across all continents plus a level per province; D one pattern, no
    levels: a null that ignores geology), each drawn several times with white noise, fitted with nine layouts (one
    level, groups, provinces, 3 random Voronoi tilings and 3 spun province sets of as many domains) by GCM-L and by
    DF-GLC at one shape (beta 2, ell 700 km, the shape of the truths and of the real groups and provinces fits);
    scored on the log evidence, the map error, and the province means with their s.d.

Run from the repository root (one process, ~40-60 GB of memory; L1 ~30 min; L2 ~2.6 min per GCM-L fit plus 15-20
min per new DF-GLC layout decomposition, cached; every fit is saved, so a stopped run resumes):
    PYTHONPATH=benchmarks/verification:benchmarks:notebooks:$PYTHONPATH VECLIB_MAXIMUM_THREADS=3 \
        nice -n 15 conda run -n curiesphere python benchmarks/verification/gcml_checks.py l1|l2 [first_draw n_draws]
Outputs: benchmarks/verification/results/gcml_*.json, gcml_*.npz, gcml_l2_fits.jsonl
"""
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular

import df_glc_checks as dgc
from df_checks import RESULTS, log
from remit.inversion import core, parameterise
from remit.inversion.cache import Cache
from remit.inversion.methods import gcm

TRUE_A = dict(v0=0.71, tau_d=0.25, tau_c=0.49)      # GCM-L provinces fit to LCS-1: v0 0.713, domains 0.247, cells 0.489
TRUE_F = dict(v0=0.7, tau=0.2, s=0.4, beta=2.0, ell=700.)
LAYOUTS = ['one', 'groups', 'provinces', 'voronoi_provinces:0', 'voronoi_provinces:1', 'voronoi_provinces:2',
           'rotated_provinces:0', 'rotated_provinces:1', 'rotated_provinces:2']
TRUTHS = {'A': 'levels per province, independent cells (GCM-L\'s model)',
          'B': 'levels per province, a pattern restarting at province edges (DF-GLC\'s model)',
          'C': 'one pattern across the continents, levels per province',
          'D': 'one pattern, no levels (null)'}
FITS = os.path.join(RESULTS, 'gcml_l2_fits.jsonl')


def setup():
    """DF-GLC's problem (whitened G with VIS columns, u, the ocean-model whitening at 16 nT) and the nine layouts;
    GCM-L's base solver on the same whitened problem, with the eigendecomposition of the candidate-set runs"""
    P = dgc.setup()
    cfg, pb = P.cfg, P.pb
    cache = Cache()
    P.lays, _ = parameterise.layouts(pb.cells, cfg['parameterisation']['tables'], cache, cfg['target']['area'],
                                     LAYOUTS)
    P.sa = np.sqrt(pb.a)
    P.Bw = P.G/P.sa[None, :]
    comp, d = cfg['compute'], cfg['data']
    spec = dict(columns=pb.store.spec, coarsen=comp.get('coarsen', 0), band=list(d['band']),
                filter=d.get('filter', 'lcs1'), weights=[pb.model, pb.noise['reference']],
                products=comp['products_backend'], version=gcm.EIG_VERSION)
    _, _, oc_spec = gcm._ocean_covariance(cfg, pb.band, pb.w, pb.F, cache, log)
    rms = float(np.atleast_1d(cfg['errors']['background']['rms_nT'])[0])
    d_eig = cache.entry('eig', dict(spec, background=oc_spec, rms_nT=rms))
    eig = (np.load(f'{d_eig}/lam.npy'), np.load(f'{d_eig}/E.npy')) if os.path.exists(f'{d_eig}/E.npy') else None
    log('GCM-L eigendecomposition from the cache' if eig else 'GCM-L eigendecomposition (not cached: ~15 min)')
    P.base = core.LinearRidge(P.Bw, P.u, 'eigh', eig=eig)
    P.base.Q                                                # cell-space eigenvectors, shared by every layout
    area = pb.cells.area_km2
    lab = P.lays['provinces'][0]
    P.prov = lab
    P.W = np.zeros((len(lab), lab.max() + 1))               # area weights of each province's mean (cells x provinces)
    P.W[np.arange(len(lab)), lab] = area
    P.W /= P.W.sum(0)
    return P


# ---------------------------------------------------------------- GCM-L

class GCML:
    """one layout of GCM-L, as gcm._fit_layout, with the solver kept for many data sets"""

    def __init__(self, P, layout):
        label, self.names = P.lays[layout]
        self.label, self.k = np.asarray(label), len(self.names)
        self.P = P
        if self.k > 1:
            self.D = np.eye(self.k)[self.label]
            Ud = P.Bw @ (P.sa[:, None]*self.D)
            self.solver = core.LinearRidge(P.Bw, np.hstack([P.u[:, None], Ud]), 'eigh',
                                           eig=(P.base.lam, P.base.E), prior=np.r_[False, np.ones(self.k, bool)])
            self.solver._Q = P.base._Q
        else:
            self.D, self.solver = np.zeros((len(self.label), 0)), P.base
        # the province means v_p = W'(D theta + x/sa): their sensitivity to theta and to x
        self.Dp = np.hstack([np.ones((P.W.shape[1], 1)), P.W.T @ self.D])
        self.Mp = P.W.T/P.sa[None, :]
        self.MBE = (self.Mp @ P.Bw.T) @ P.base.E

    def fit(self, y, alpha2_one, settings=None):
        s, sa = self.solver, self.P.sa
        if settings is not None:
            alpha2, beta = settings
        elif self.k > 1:
            alpha2, beta = s.by_evidence_beta(y, False, alpha2_one)
        else:
            alpha2, beta = alpha2_one, 0.
        sol = s.solve(y, alpha2, beta)
        theta = sol['theta']
        level = theta[0] + (self.D @ theta[1:] if self.k > 1 else 0.)
        m = level + sol['x']/sa
        Dc = np.hstack([np.ones((len(sa), 1)), self.D])
        sd = s.cell_sd(alpha2, 1., sa, Dc, beta)
        return dict(alpha2=float(alpha2), beta=float(beta), ll=float(s.log_evidence(y, alpha2, False, beta=beta))
                    + self.P.ll_whiten, v=m, sd=sd, pm=self.P.W.T @ m, psd=self.province_sd(alpha2, beta),
                    tau_cells=float(1/np.sqrt(alpha2)), tau_domains=float(1/np.sqrt(beta)) if self.k > 1 else 0.)

    def province_sd(self, alpha2, beta):
        """as LinearRidge.posterior_sd_map (sigma 1) for the province means, with M B'E kept"""
        s = self.solver
        den = s.lam + alpha2
        var = (np.sum(self.Mp**2, 1) - np.sum(self.MBE**2/den, 1))/alpha2
        S = self.Dp - self.MBE @ (s.c/den[:, None])
        Pm = s._precision(alpha2, beta)[0]
        return np.sqrt(np.maximum(var + np.sum(S*np.linalg.solve(Pm, S.T).T, 1), 0.))


def reference(P, y, label, alpha2, beta):
    """GCM-L by the data-space covariance (sigma 1): Sigma = diag(1/(alpha^2 a)) + S S'/beta, K = G Sigma G' + I, v0
    by generalised least squares; the profile log evidence -1/2 (log det K + r'K^-1 r), the map, the s.d. of every
    cell, of each domain level v0 + delta_d and of each province mean"""
    G, u, a = P.G, P.u, P.pb.a
    S = np.eye(label.max() + 1)[label]
    log('reference: G Sigma, K = G Sigma G\' + I')
    GS = G/(alpha2*a)[None, :]
    GSd = (G @ S)/beta                                      # G Sigma's domain part, per domain (N x k)
    GS += GSd @ S.T
    K = GS @ G.T
    K = 0.5*(K + K.T)
    K[np.diag_indices_from(K)] += 1.
    cf = cho_factor(K, lower=True, overwrite_a=True)
    Ky, Ku = cho_solve(cf, y), cho_solve(cf, u)
    su = float(u @ Ku)
    v0 = float(u @ Ky)/su
    q = Ky - Ku*v0
    r = y - u*v0
    ll = -float(np.sum(np.log(np.diag(cf[0])))) - 0.5*float(r @ q)
    v = v0 + GS.T @ q
    level = v0 + GSd.T @ q

    def sd(Y, prior):                                       # posterior s.d. of v0 + functionals with cov(data) Y
        Z = solve_triangular(cf[0], Y, lower=True)
        bt = 1. - Y.T @ Ku
        return np.sqrt(np.maximum(prior - np.einsum('ij,ij->j', Z, Z) + bt**2/su, 0.))
    log('reference: s.d. of every cell, domain level and province mean')
    sd_cell = sd(GS, 1/(alpha2*a) + 1/beta)
    sd_level = sd(GSd, np.full(GSd.shape[1], 1/beta))
    StW = S.T @ P.W
    GSW = G @ (P.W/(alpha2*a)[:, None]) + GSd @ StW
    sd_prov = sd(GSW, np.sum(P.W**2/(alpha2*a)[:, None], 0) + np.sum(StW**2, 0)/beta)
    return dict(ll=ll, v0=v0, v=v, sd=sd_cell, level=level, level_sd=sd_level, pm=P.W.T @ v, psd=sd_prov)


def l1():
    P = setup()
    pb = P.pb
    rng = np.random.default_rng(20261101)
    lab, names = P.lays['provinces']
    lab = np.asarray(lab)
    k = len(names)
    truth = TRUE_A['v0'] + TRUE_A['tau_d']*rng.standard_normal(k)[lab] + TRUE_A['tau_c']*rng.standard_normal(len(lab))/P.sa
    y = P.G @ truth + rng.standard_normal(len(P.u))
    alpha2_one = P.base.by_evidence(y, False)
    log('production: gcm._fit_layout')
    t = time.time()
    row, (m, sd, level, label, _) = gcm._fit_layout(P.base, P.Bw, P.u, y, P.pb.a, lab, names, pb.cells, pb.band,
                                                    pb.w, False, P.Lc, P.ll_whiten)
    secs = time.time() - t
    lv = np.array([row['domains'][n]['level'] for n in names])
    lsd = np.array([row['domains'][n]['level_sd'] for n in names])
    g = GCML(P, 'provinces')
    mine = g.fit(y, alpha2_one, (row['alpha2'], row['beta']))
    ref = reference(P, y, lab, row['alpha2'], row['beta'])
    ref2 = reference(P, y, lab, 2*row['alpha2'], 0.5*row['beta'])
    ll2 = g.solver.log_evidence(y, 2*row['alpha2'], False, beta=0.5*row['beta'])
    ll1 = row['log_evidence'] - P.ll_whiten
    out = dict(seconds_per_fit=secs, alpha2=row['alpha2'], beta=row['beta'], tau_cells=row['tau_cells'],
               tau_domains=row['tau_domains'], v0=row['v0'], v0_ref=ref['v0'],
               ll=ll1, ll_ref=ref['ll'], ll_diff=abs(ll1 - ref['ll']),
               ll_change=ll2 - ll1, ll_change_ref=ref2['ll'] - ref['ll'],
               map_max_diff=float(np.abs(m - ref['v']).max()),
               sd_max_rel_diff=float(np.abs(sd/ref['sd'] - 1).max()),
               level_max_diff=float(np.abs(lv - ref['level']).max()),
               level_sd_max_rel_diff=float(np.abs(lsd/ref['level_sd'] - 1).max()),
               province_mean_max_diff=float(np.abs(mine['pm'] - ref['pm']).max()),
               province_sd_max_rel_diff=float(np.abs(mine['psd']/ref['psd'] - 1).max()),
               wrapper_map_max_diff=float(np.abs(mine['v'] - m).max()),
               wrapper_ll_diff=abs(mine['ll'] - row['log_evidence']))
    log(f'GCM-L fit {secs/60:.1f} min: cells {row["tau_cells"]:.3f} (true {TRUE_A["tau_c"]}), domains '
        f'{row["tau_domains"]:.3f} (true {TRUE_A["tau_d"]}); against the reference: log evidence {ll1:.4f} vs '
        f'{ref["ll"]:.4f}, change {out["ll_change"]:.4f} vs {out["ll_change_ref"]:.4f}; map {out["map_max_diff"]:.1e} '
        f'SI km; cell s.d. {out["sd_max_rel_diff"]:.1e}; levels {out["level_max_diff"]:.1e}, their s.d. '
        f'{out["level_sd_max_rel_diff"]:.1e}; province means {out["province_mean_max_diff"]:.1e}, their s.d. '
        f'{out["province_sd_max_rel_diff"]:.1e}; the L2 wrapper against _fit_layout: map '
        f'{out["wrapper_map_max_diff"]:.1e}, log evidence {out["wrapper_ll_diff"]:.1e}')
    # exact case: one level per province, no noise
    if hasattr(P.base, '_alpha2_one'):                     # _fit_layout keeps the first data set's start
        del P.base._alpha2_one
    lv_true = TRUE_A['v0'] + TRUE_A['tau_d']*rng.standard_normal(k)
    t_ex = lv_true[lab]
    y_ex = P.G @ t_ex
    r_ex, (m_ex, *_) = gcm._fit_layout(P.base, P.Bw, P.u, y_ex, P.pb.a, lab, names, pb.cells, pb.band, pb.w, False,
                                       P.Lc, P.ll_whiten)
    out['exact'] = dict(max_abs_error=float(np.abs(m_ex - t_ex).max()), rms_error=float(np.sqrt(np.mean((m_ex - t_ex)**2))),
                        tau_cells=r_ex['tau_cells'], tau_domains=r_ex['tau_domains'])
    log(f'one level per province, no noise: max |error| {out["exact"]["max_abs_error"]:.1e} SI km, RMS '
        f'{out["exact"]["rms_error"]:.1e} (cells {r_ex["tau_cells"]:.2e}, domains {r_ex["tau_domains"]:.3f})')
    with open(os.path.join(RESULTS, 'gcml_l1.json'), 'w') as fo:
        json.dump(out, fo, indent=1)
    np.savez(os.path.join(RESULTS, 'gcml_l1.npz'), clon=pb.cells.clon, clat=pb.cells.clat, truth=truth, v=m,
             v_ref=ref['v'], sd=sd, sd_ref=ref['sd'], label=lab, exact_truth=t_ex, exact_v=m_ex)
    log('L1 done')


# ---------------------------------------------------------------- L2

def truths(P, first, n):
    """the four truths of each draw (seeded per truth and draw): {(truth, draw): VIS per cell}"""
    lab = P.prov
    kp = lab.max() + 1
    sh_p = dgc.shape(P, 'provinces', TRUE_F['beta'], TRUE_F['ell'], np.zeros(len(P.u)))
    chol_p = [(b, cholesky(sh_p.R(b) + 1e-10*np.eye(len(b)), lower=True)) for b in sh_p.blocks]
    sh_1 = dgc.shape(P, 'one', TRUE_F['beta'], TRUE_F['ell'], np.zeros(len(P.u)))
    log('one pattern across the continents: Cholesky of its correlation')
    R1 = sh_1.R(sh_1.blocks[0])
    for jitter in (1e-10, 1e-8, 1e-6):                     # the smallest that factorises (rounding in the table)
        try:
            L1 = cholesky(R1 + jitter*np.eye(len(lab)), lower=True)
            break
        except np.linalg.LinAlgError:
            log(f'  correlation not positive definite with jitter {jitter:g}')
    del R1
    out = {}
    for i in range(first, first + n):
        for t in TRUTHS:
            rng = np.random.default_rng(20261102 + 1000*i + ord(t))
            if t == 'A':
                v = (TRUE_A['v0'] + TRUE_A['tau_d']*rng.standard_normal(kp)[lab]
                     + TRUE_A['tau_c']*rng.standard_normal(len(lab))/P.sa)
            elif t == 'B':
                v = TRUE_F['v0'] + TRUE_F['tau']*rng.standard_normal(kp)[lab]
                for b, L in chol_p:
                    v[b] += TRUE_F['s']*(L @ rng.standard_normal(len(b)))
            else:
                v = TRUE_F['v0'] + TRUE_F['s']*(L1 @ rng.standard_normal(len(lab)))
                if t == 'C':
                    v += TRUE_F['tau']*rng.standard_normal(kp)[lab]
            out[t, i] = v
    del L1
    return out


def scores(P, truth, f):
    """the map error and the province means' errors in units of their s.d."""
    area = P.pb.cells.area_km2
    tp = P.W.T @ truth
    zc = (f['v'] - truth)/f['sd']
    zp = (f['pm'] - tp)/f['psd']
    return dict(rms_error=float(np.sqrt(np.sum(area*(f['v'] - truth)**2)/area.sum())),
                province_rms_error=float(np.sqrt(np.mean((f['pm'] - tp)**2))),
                cells_within_1sd=float(np.mean(np.abs(zc) < 1)), cells_within_2sd=float(np.mean(np.abs(zc) < 2)),
                provinces_within_1sd=float(np.mean(np.abs(zp) < 1)),
                provinces_within_2sd=float(np.mean(np.abs(zp) < 2)),
                province_z_sd=float(np.std(zp)), median_province_sd=float(np.median(f['psd'])))


def done_fits():
    if not os.path.exists(FITS):
        return set()
    return {(r['method'], r['layout'], r['truth'], r['draw']) for r in map(json.loads, open(FITS))}


class DFGLC:
    """DF-GLC at one shape for one layout, with the province means and their s.d."""

    def __init__(self, P, layout):
        self.P = P
        self.sh = dgc.shape(P, layout, TRUE_F['beta'], TRUE_F['ell'], np.zeros(len(P.u)))
        self.VM, self.VGS = dgc.projections(self.sh, keep=1)
        W, sh = P.W, self.sh
        self.VMW = self.VM @ W                                  # V'G R W
        self.RW = np.zeros_like(W)
        for b in sh.blocks:                                     # R W, block by block, for w'R w
            self.RW[b] = sh.R(b) @ W[b]
        self.wRw = np.sum(W*self.RW, 0)
        if sh.K > 1:
            S = np.eye(sh.K)[sh.label]
            self.StW = S.T @ W
            self.VGSW = self.VGS @ self.StW
        else:
            self.StW = self.VGSW = None

    def fit(self, y):
        sh = dgc.with_data(self.sh, y)
        f = sh.fit()
        post = dgc.fast_posterior(sh, f, self.VM, self.VGS)
        s, tau = f['s'], f['tau'] or 0.
        Ci, _ = sh._parts(s, tau if sh.K > 1 else 1., f['h'])
        Y = s**2*self.VMW + (tau**2*self.VGSW if self.VGSW is not None else 0.)
        prior = s**2*self.wRw + (tau**2*np.sum(self.StW**2, 0) if self.StW is not None else 0.)
        CY = Ci(Y)
        Au = float(sh.ut @ Ci(sh.ut))
        bt = 1. - CY.T @ sh.ut
        psd = np.sqrt(np.maximum(prior - np.sum(Y*CY, 0) + bt**2/Au, 0.))
        return dict(ll=float(f['ll']), s=f['s'], tau=f['tau'], v=post['v'], sd=post['sd'],
                    pm=self.P.W.T @ post['v'], psd=psd)


def l2(first=0, n=5):
    P = setup()
    T = truths(P, first, n)
    Y = {key: P.G @ v + np.random.default_rng(20261202 + 1000*key[1] + ord(key[0])).standard_normal(len(P.u))
         for key, v in T.items()}
    a1 = {key: P.base.by_evidence(y, False) for key, y in Y.items()}
    have = done_fits()
    maps = dict(np.load(os.path.join(RESULTS, 'gcml_l2_maps.npz'))) if os.path.exists(
        os.path.join(RESULTS, 'gcml_l2_maps.npz')) else {}
    for key, v in T.items():
        if key[1] == 0:
            maps[f'truth_{key[0]}'] = v
    maps.update(clon=P.pb.cells.clon, clat=P.pb.cells.clat, provinces=P.prov)
    for layout in LAYOUTS:
        for method, make in (('GCM-L', GCML), ('DF-GLC', DFGLC)):
            todo = [key for key in T if (method, layout, key[0], key[1]) not in have]
            if not todo:
                continue
            t0 = time.time()
            m = make(P, layout)
            log(f'{method} {layout}: set up in {(time.time() - t0)/60:.1f} min; {len(todo)} fits')
            for key in todo:
                t1 = time.time()
                f = m.fit(Y[key], a1[key]) if method == 'GCM-L' else m.fit(Y[key])
                row = dict(method=method, layout=layout, truth=key[0], draw=key[1], ll=f['ll'],
                           seconds=time.time() - t1, **scores(P, T[key], f),
                           **{k: f[k] for k in ('alpha2', 'beta', 'tau_cells', 'tau_domains', 's', 'tau') if k in f})
                with open(FITS, 'a') as fo:
                    fo.write(json.dumps(row) + '\n')
                if key[1] == 0:
                    maps[f'{method}_{layout}_{key[0]}'] = f['v']
                    maps[f'{method}_{layout}_{key[0]}_sd'] = f['sd']
                log(f'  {method} {layout}, truth {key[0]} draw {key[1]}: log evidence {f["ll"]:.1f}, map error '
                    f'{row["rms_error"]:.3f}, provinces within 1 s.d. {row["provinces_within_1sd"]:.2f} '
                    f'({row["seconds"]:.0f} s)')
            np.savez(os.path.join(RESULTS, 'gcml_l2_maps.npz'), **maps)
            del m
            dgc.MEMO.clear()
            dgc.PROJ.clear()
    log('L2 done')


def l1x():
    """the exact case again with the noise level fitted (sigma by the evidence): no shrinkage should remain"""
    P = setup()
    pb = P.pb
    rng = np.random.default_rng(20261101)
    lab, names = P.lays['provinces']
    lab = np.asarray(lab)
    t_ex = (TRUE_A['v0'] + TRUE_A['tau_d']*rng.standard_normal(len(names)))[lab]
    r, (m, *_) = gcm._fit_layout(P.base, P.Bw, P.u, P.G @ t_ex, pb.a, lab, names, pb.cells, pb.band, pb.w, True,
                                 P.Lc, P.ll_whiten)
    out = dict(max_abs_error=float(np.abs(m - t_ex).max()), rms_error=float(np.sqrt(np.mean((m - t_ex)**2))),
               sigma=r['sigma'], tau_cells=r['tau_cells'], tau_domains=r['tau_domains'])
    log(f'one level per province, no noise, noise level fitted: max |error| {out["max_abs_error"]:.1e} SI km, RMS '
        f'{out["rms_error"]:.1e} (noise x {r["sigma"]:.1e})')
    with open(os.path.join(RESULTS, 'gcml_l1x.json'), 'w') as fo:
        json.dump(out, fo, indent=1)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'l1'
    if what == 'l1':
        l1()
    elif what == 'l1x':
        l1x()
    elif what == 'l2':
        args = [int(a) for a in sys.argv[2:4]]
        l2(*args)
