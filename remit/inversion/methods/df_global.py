"""
DF-GLC (CONTEXT.md; ADRs 0007, 0008, 0010): DF's model in the global fit to the Gauss coefficients. VIS on all
continental crust = v0 + delta_d (the offset of the cell's domain, delta ~ N(0, tau^2)) + a Gaussian fractal pattern
inside each domain, independent between domains, of one strength s shared by all domains:

    y = u v0 + G (S delta + p) + e,    p ~ N(0, s^2 blockdiag(R_d)),    e ~ N(0, I + rho C_oc)

G: the whitened coefficients of each cell's VIS (the columns of GCM, not area-scaled); R_d: DF's correlation of cell
averages (df.correlation), restricted to the cells of domain d; C_oc: the ocean-model error of ADR 0009 at a fixed
amplitude (the data are whitened by its Cholesky factor once, whose Jacobian joins the evidence); v0 uniform on the box.
For one domain shape (beta, ell), A = G blockdiag(R_d) G' is decomposed once (A = V lam V'), after which the evidence
in (s, tau), and the noise scale h when fitted, costs O(N k^2) (Woodbury on the k domain offsets): so (s, tau) are
maximised exactly on each shape, and the shapes are searched on a small grid (beta first, then ell). The best shape
of each layout gives its evidence; the best layout gives the map (posterior mean, s.d., domain levels).

Config keys, besides GCM's ([target], [geometry], [data], [errors.noise], [fit.background], [compute] products):
  [parameterisation] kind = 'layouts', tables;  [errors.background] kind = 'ocean_cells', layer, rms_nT (one value)
  [method] name = 'df' with [fit] type = 'global'; layouts (names from parameterise.LAYOUTS + CONTROLS; 'one' is
           the whole area as one domain, one pattern without breaks); bounds ([lo, hi] of v0); beta_grid, ell_grid_km
           (the shape search: beta over beta_grid at the middle ell, then ell over ell_grid_km at the best beta, one
           step further out when the best is at an end); noise ('fixed' or 'fitted')
"""
import json
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, eigh, solve_triangular
from scipy.optimize import minimize

from .. import graph, parameterise, results
from ..cache import Cache
from . import df, gcm

N_TAB = 4001
S_RANGE, TAU_RANGE, H_RANGE = (1e-3, 10.), (1e-3, 4.), (0.2, 5.)    # SI km, SI km, noise scale


class Shape:
    """the decomposed problem of one layout at one (beta, ell): evidence in (s, tau, h), and the posterior"""

    def __init__(self, G, u, y, label, xyz, window, beta, ell, lo, hi, ll_whiten):
        self.G, self.u, self.y, self.label = G, u, y, np.asarray(label)
        self.beta, self.ell, self.lo, self.hi, self.ll_whiten = beta, ell, lo, hi, ll_whiten
        self.K = int(self.label.max()) + 1
        self.blocks = [np.flatnonzero(self.label == k) for k in range(self.K)]
        psi_max = max(np.arccos(np.clip((xyz[b] @ xyz[b].T).min(), -1, 1)) for b in self.blocks)
        self.grid = np.linspace(0., psi_max*(1 + 1e-9) + 1e-12, N_TAB)
        self.tab = df.correlation(beta, ell, self.grid, window)
        self.xyz = xyz
        M = self.pattern_columns()                      # G R (data x cells)
        A = M @ G.T
        del M
        self.lam, self.V = eigh(0.5*(A + A.T), overwrite_a=True, check_finite=False)
        self.lam = np.maximum(self.lam, 0.)
        self.yt, self.ut = self.V.T @ y, self.V.T @ u
        self.Ut = self.V.T @ (G @ parameterise.indicator(self.label)) if self.K > 1 else None

    def R(self, b):
        p = np.arccos(np.clip(self.xyz[b] @ self.xyz[b].T, -1., 1.))
        return np.interp(p, self.grid, self.tab)

    def pattern_columns(self):
        M = np.empty_like(self.G)
        for b in self.blocks:
            M[:, b] = self.G[:, b] @ self.R(b)
        return M

    def _parts(self, s, tau, h):
        """C^-1 applied in the eigenbasis (a function), and log det C"""
        dinv = 1/(h**2 + s**2*self.lam)
        ld = float(np.sum(np.log(h**2 + s**2*self.lam)))
        if self.Ut is None:
            return (lambda z: dinv[:, None]*z if z.ndim == 2 else dinv*z), ld
        DU = dinv[:, None]*self.Ut
        Mt = np.eye(self.K)/tau**2 + self.Ut.T @ DU
        cf = cho_factor(Mt, lower=True)
        ld += self.K*np.log(tau**2) + 2*float(np.sum(np.log(np.diag(cf[0]))))

        def apply(z):
            dz = dinv[:, None]*z if z.ndim == 2 else dinv*z
            return dz - DU @ cho_solve(cf, DU.T @ z)
        return apply, ld

    def evidence(self, s, tau=1., h=1.):
        Ci, ld = self._parts(s, tau, h)
        Cu, Cy = Ci(self.ut), Ci(self.yt)
        Au = float(self.ut @ Cu)
        v0 = float(self.ut @ Cy)/Au
        q = Cy - Cu*v0
        r = self.yt - self.ut*v0
        N = len(self.y)
        return float(-0.5*ld - 0.5*np.log(Au) - 0.5*r @ q - 0.5*(N - 1)*np.log(2*np.pi) - np.log(self.hi - self.lo)
                     + self.ll_whiten), v0

    def fit(self, fit_h=False):
        """(s, tau, h) of the largest evidence on this shape"""
        n = 1 + int(self.K > 1) + int(fit_h)

        def unpack(x):
            s = np.exp(x[0])
            tau = np.exp(x[1]) if self.K > 1 else 1.
            h = np.exp(x[-1]) if fit_h else 1.
            return s, tau, h
        bnds = [np.log(S_RANGE)] + [np.log(TAU_RANGE)]*int(self.K > 1) + [np.log(H_RANGE)]*int(fit_h)

        def f(x):
            try:
                v = -self.evidence(*unpack(x))[0]
            except (np.linalg.LinAlgError, ValueError):
                return 1e30
            return v if np.isfinite(v) else 1e30
        best = None
        for s0 in (0.1, 0.3, 1.):                       # a few starts: the strength can trade off against tau
            x0 = np.r_[np.log(s0), [np.log(0.1)] if self.K > 1 else [], [0.] if fit_h else []]
            r = minimize(f, x0, method='Nelder-Mead', bounds=bnds,
                         options=dict(xatol=1e-4, fatol=1e-4, maxiter=2000))
            if best is None or r.fun < best.fun:
                best = r
        s, tau, h = unpack(best.x)
        ll, v0 = self.evidence(s, tau, h)
        return dict(s=float(s), tau=float(tau) if self.K > 1 else None, h=float(h), ll=ll, v0=v0, beta=self.beta,
                    ell=self.ell, n=n)

    def posterior(self, f, Lc, w, a_km2):
        """posterior mean and s.d. of VIS per cell, the domain levels v0 + delta_d, and the residual (nT)"""
        s, tau, h = f['s'], f['tau'] or 0., f['h']
        Ci, _ = self._parts(s, tau if self.K > 1 else 1., h)
        Cu = Ci(self.ut)
        Au = float(self.ut @ Cu)
        v0 = float(self.ut @ Ci(self.yt))/Au
        qt = Ci(self.yt - self.ut*v0)
        q = self.V @ qt                                  # C^-1 (y - u v0) in data space
        Gq = self.G.T @ q
        n = len(self.label)
        pat = np.empty(n)
        for b in self.blocks:
            pat[b] = self.R(b) @ Gq[b]
        S = parameterise.indicator(self.label) if self.K > 1 else None
        delta = tau**2*(S.T @ Gq) if S is not None else np.zeros(1)
        v = v0 + s**2*pat + (S @ delta if S is not None else 0.)
        # s.d.: diag(Sigma - Sigma G' C^-1 G Sigma) + the part of v0, chunked over cells
        GS = s**2*self.pattern_columns() + (tau**2*(self.G @ S) @ S.T if S is not None else 0.)   # G Sigma
        Y = self.V.T @ GS
        del GS
        prior = s**2 + (tau**2 if S is not None else 0.)                       # diag(Sigma): R_kk = 1
        red = np.empty(n)
        bt = np.empty(n)
        for k in range(0, n, 2000):
            Yk = Y[:, k:k + 2000]
            CY = Ci(Yk)
            red[k:k + 2000] = np.sum(Yk*CY, 0)
            bt[k:k + 2000] = 1. - CY.T @ self.ut
        sd = np.sqrt(np.maximum(prior - red + bt**2/Au, 0.))
        resid_w = self.y - self.u*v0 - self.G @ (v - v0)
        resid = (Lc @ resid_w if Lc is not None else resid_w)/w
        level = v0 + (S @ delta if S is not None else 0.)
        return dict(v=v, sd=sd, level=level, v0=v0, delta=delta, resid=resid)


def search(make, beta_grid, ell_grid, fit_h, say, name):
    """the shape search: beta at the middle ell, then ell at the best beta (one step further out at an end); only the
    best decomposition is kept in memory. Returns ((fit, Shape) of the best shape, {shape: fit})"""
    tried, best = {}, [None, None]

    def at(beta, ell):
        key = (float(beta), float(ell))
        if key not in tried:
            t = time.time()
            sh = make(beta, ell)
            f = sh.fit(fit_h)
            tried[key] = f
            say(f'{name}: beta {beta:.2f}, ell {ell:.0f} km: log evidence {f["ll"]:.1f}, s {f["s"]:.3f}'
                + (f', tau {f["tau"]:.3f}' if f['tau'] is not None else '') + (f', h {f["h"]:.3f}' if fit_h else '')
                + f' SI km, v0 {f["v0"]:.3f} ({(time.time() - t)/60:.1f} min)')
            if best[0] is None or f['ll'] > best[0]['ll']:
                best[:] = [f, sh]
            del sh
        return tried[key]
    ell_mid = ell_grid[len(ell_grid)//2]
    b_best = max(beta_grid, key=lambda b: at(b, ell_mid)['ll'])
    e_best = max(ell_grid, key=lambda e: at(b_best, e)['ll'])
    if e_best == ell_grid[0]:
        at(b_best, ell_grid[0]/2)
    elif e_best == ell_grid[-1]:
        at(b_best, ell_grid[-1]*2)
    return (best[0], best[1]), {f'{k[0]:.2f},{k[1]:.0f}': v for k, v in tried.items()}


def run(cfg, cache=None, log=print):
    cache = cache or Cache()
    t0 = time.time()
    say = lambda msg: log(f'[{(time.time() - t0)/60:5.1f} min] {msg}')
    me, par, tg = cfg['method'], cfg['parameterisation'], cfg['target']
    pb = gcm.global_problem(cfg, cache, say)
    cells, band, w = pb.cells, pb.band, pb.w
    G = pb.B
    G *= np.sqrt(pb.a)[None, :]                          # whitened coefficients of each cell's VIS
    u, y = pb.u, pb.y
    Lc, ll_whiten = None, 0.
    eb = cfg['errors'].get('background')
    if eb:
        C_oc, rms1, _ = gcm._ocean_covariance(cfg, band, w, pb.F, cache, say)
        rms = float(np.atleast_1d(eb['rms_nT'])[0])
        Lc = cholesky(np.eye(len(y)) + (rms/rms1)**2*C_oc, lower=True)
        del C_oc
        ll_whiten = -float(np.sum(np.log(np.diag(Lc))))
        G = solve_triangular(Lc, G, lower=True, overwrite_b=True)
        u, y = solve_triangular(Lc, u, lower=True), solve_triangular(Lc, y, lower=True)
        say(f'whitened by the ocean-model error at {rms:g} nT')
    names = me.get('layouts', ['one', 'groups', 'provinces', 'voronoi_groups', 'voronoi_provinces'])
    lays, coverage = parameterise.layouts(cells, par['tables'], cache, tg['area'], names)
    say(f'{len(lays)} layouts: ' + ', '.join(f'{n} ({len(d)})' for n, (_, d) in lays.items()))
    lo, hi = me.get('bounds', [0., 4.])
    fit_h = me.get('noise', 'fixed') == 'fitted'
    xyz = graph.unit(cells.clon, cells.clat)
    window = df.cap_window(4*np.pi*df.R_KM**2/(12*cells.nside**2))
    beta_grid = me.get('beta_grid', [2.0, 2.5, 3.0])
    ell_grid = me.get('ell_grid_km', [350., 700., 1400.])
    ranking, best = {}, None
    for name, (label, dnames) in lays.items():
        make = lambda beta, ell: Shape(G, u, y, label, xyz, window, beta, ell, lo, hi, ll_whiten)
        (f, sh), profile = search(make, beta_grid, ell_grid, fit_h, say, name)
        ranking[name] = dict(f, n_domains=len(dnames), profile=profile)
        say(f'{name}: best beta {f["beta"]:.2f}, ell {f["ell"]:.0f} km, log evidence {f["ll"]:.1f}')
        if best is None or f['ll'] > best[1]['ll']:
            best = (name, f, sh, label, dnames)
        else:
            del sh
    top = max(ranking, key=lambda n: ranking[n]['ll'])
    for n in ranking:
        ranking[n]['delta_log_evidence'] = ranking[n]['ll'] - ranking[top]['ll']
    say('ranking: ' + ', '.join(f'{n} {ranking[n]["delta_log_evidence"]:+.1f}'
                                for n in sorted(ranking, key=lambda n: -ranking[n]['ll'])))
    name, f, sh, label, dnames = best
    post = sh.posterior(f, Lc, w, cells.area_km2)
    rms_br = float(np.sqrt(np.sum(band.mean_square_br()*post['resid']**2)))
    neg = float(cells.area_km2[post['v'] < 0].sum()/cells.area_km2.sum())
    say(f'{name}: v0 {post["v0"]:.3f}, RMS Br misfit {rms_br:.2f} nT, cells below zero {neg:.1%}')
    domains = {}
    if sh.K > 1:
        area = np.bincount(label, cells.area_km2, sh.K)
        domains = {dn: dict(level=float(post['v0'] + post['delta'][j]), area_km2=float(area[j]))
                   for j, dn in enumerate(dnames)}
    cell_vars = dict(vis=post['v'], vis_sd=post['sd'], vis_domain=post['level'], domain=label.astype(float))
    if pb.truth is not None:
        cell_vars['truth'] = pb.truth
    data_nT = pb.y/w
    row_vars = dict(data=data_nT, model=data_nT - post['resid'], residual=post['resid'])
    attrs = dict(layout=name, layouts=ranking, domains=domains, v0=post['v0'], beta=f['beta'], ell_km=f['ell'],
                 strength=f['s'], tau=f['tau'] if f['tau'] is not None else -1., noise_scale=f['h'],
                 log_evidence=f['ll'], rms_br_misfit=rms_br, negative_area=neg, n_data=len(band),
                 node_coverage=coverage, ocean_error_rms_nT=float(np.atleast_1d(eb['rms_nT'])[0]) if eb else 0.,
                 panel_titles={'vis_domain': f'Domain levels ({name} layout)'})
    return results.make(cfg, cells, cell_vars, row_vars, band, **attrs)
