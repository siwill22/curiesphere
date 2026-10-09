"""
DF (domain fractal, CONTEXT.md; ADR 0007): VIS = the mean of the cell's domain + a Gaussian fractal pattern inside each
domain, the patterns independent between domains, from a local fit with measurement noise only.

    v = S theta + u,    u_k ~ N(0, s_k^2 R_k),    yw = Gw v + e,    e ~ N(0, h^2 I)

S: the indicator columns of a domain layout. R_k: the correlation of the cell averages of an isotropic field on the
sphere with angular spectrum S_l ~ (l(l+1) + (R/ell)^2)^(-beta/2) (a 2-D power spectrum k^-beta above the largest
patch size ell), restricted to the cells of domain k. Cell averages: a spherical cap of the cell's area as the window,
so the correlation is a Legendre series, positive definite on the sphere for any beta. theta: uniform on the box
[lo, hi] per domain, integrated out as a Gaussian integral over all of R^K (a domain whose best mean falls outside the
box is reported, not given zero mass). h: 1 (the error model as it is: the data are whitened for noise only) or fitted.
Hyperparameters (log s_k, beta, log ell, and log h when fitted) maximise the log evidence (exact gradients in s_k
and h, central differences in beta and ell), with Laplace s.d. from the curvature.

Config keys used, besides those of problem.local (weighting 'full', model_error rms 0):
  [parameterisation] kind = 'domains', table (as PD)
  [method] bounds ([lo, hi] of the means), beta_range, ell_range (km), s_range (SI km), noise ('fixed' or 'fitted'),
           hyper_sd (Laplace s.d. of the hyperparameters), layouts = {one = true, merge = <FD-merge result>,
           merge_settings = [i, ...]} (further layouts, ranked by the evidence with the table's)
"""
import json
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, block_diag
from scipy.optimize import minimize

from .. import errors, parameterise, problem, results
from ..cache import Cache
from ..geometry import area_name
from ..graph import unit

R_KM = 6371.2
LMAX = 2000                                   # degrees of the pattern (cell windows make the series converge)
N_TAB = 4001                                  # tabulated correlation, uniform in angle
JITTER = 1e-8
FD_STEP = 1e-3                                # central differences in beta and log ell


def cap_window(area_km2, lmax=LMAX):
    """W_l of the average over a spherical cap of the given area (W_0 = 1)"""
    x0 = 1 - area_km2/(2*np.pi*R_KM**2)
    P = np.empty(lmax + 2)
    P[0], P[1] = 1., x0
    for l in range(1, lmax + 1):
        P[l + 1] = ((2*l + 1)*x0*P[l] - l*P[l - 1])/(l + 1)
    l = np.arange(1, lmax + 1)
    return np.r_[1., (P[l - 1] - P[l + 1])/((2*l + 1)*(1 - x0))]


def spectrum(beta, ell, lmax=LMAX):
    l = np.arange(lmax + 1)
    return (l*(l + 1) + (R_KM/ell)**2)**(-beta/2)


def correlation(beta, ell, psi, window):
    """correlation of the cell averages at angular distances psi (radians)"""
    l = np.arange(len(window))
    c = (2*l + 1)*spectrum(beta, ell, len(window) - 1)*window**2
    return errors.legendre_sum(c, np.cos(psi))/np.sum(c)


class DomainFractal:
    """the DF model of one problem and one layout (label: domain index of each cell, 0..K-1)"""

    def __init__(self, Gw, yw, label, clon, clat, cell_area_km2, lo, hi):
        self.Gw, self.yw, self.label = Gw, yw, np.asarray(label)
        self.K = int(self.label.max()) + 1
        self.S = parameterise.indicator(self.label)
        self.U = Gw @ self.S
        self.lo, self.hi = (np.broadcast_to(np.asarray(b, float), (self.K,)) for b in (lo, hi))
        self.blocks = [np.flatnonzero(self.label == k) for k in range(self.K)]
        xyz = unit(clon, clat)
        self.psi = [np.arccos(np.clip(xyz[b] @ xyz[b].T, -1., 1.)) for b in self.blocks]
        self.psi_max = max(float(p.max()) for p in self.psi)
        self.window = cap_window(cell_area_km2)
        self._cache = {}

    def factors(self, beta, ell):
        """(M = Gw blockdiag(L_k), [L_k]) with R_k = L_k L_k'"""
        key = (round(float(beta), 12), round(float(ell), 9))
        if key not in self._cache:
            if len(self._cache) >= 6:
                self._cache.pop(next(iter(self._cache)))
            grid = np.linspace(0., self.psi_max*(1 + 1e-9) + 1e-12, N_TAB)
            tab = correlation(beta, ell, grid, self.window)
            M = np.empty_like(self.Gw)
            Ls = []
            for b, p in zip(self.blocks, self.psi):
                R = np.interp(p, grid, tab)
                L = cholesky(R + JITTER*np.eye(len(b)), lower=True)
                M[:, b] = self.Gw[:, b] @ L
                Ls.append(L)
            self._cache[key] = (M, Ls)
        return self._cache[key]

    def unpack(self, x, fit_h):
        s = np.exp(x[:self.K])
        beta, ell = x[self.K], np.exp(x[self.K + 1])
        h = np.exp(x[self.K + 2]) if fit_h else 1.
        return s, beta, ell, h

    def _solve(self, s, beta, ell, h):
        M, Ls = self.factors(beta, ell)
        Ms = M*s[self.label]
        N = len(self.yw)
        C = Ms @ Ms.T
        C[np.diag_indices(N)] += h**2
        cf = cho_factor(C, lower=True)
        CiU, Ciy = cho_solve(cf, self.U), cho_solve(cf, self.yw)
        A = self.U.T @ CiU
        cA = cho_factor(A, lower=True)
        theta = cho_solve(cA, self.U.T @ Ciy)
        q = Ciy - CiU @ theta                       # P y, P = C^-1 - C^-1 U A^-1 U' C^-1
        r = self.yw - self.U @ theta
        ll = (-np.sum(np.log(np.diag(cf[0]))) - np.sum(np.log(np.diag(cA[0]))) - 0.5*r @ q
              - 0.5*(N - self.K)*np.log(2*np.pi) - np.sum(np.log(self.hi - self.lo)))
        return dict(ll=float(ll), M=M, Ls=Ls, cf=cf, CiU=CiU, cA=cA, theta=theta, q=q)

    def evidence(self, x, fit_h=False):
        return self._solve(*self.unpack(x, fit_h))['ll']

    def gradient(self, x, fit_h=False):
        """(log evidence, gradient in x)"""
        s, beta, ell, h = self.unpack(x, fit_h)
        o = self._solve(s, beta, ell, h)
        M, CiU, q = o['M'], o['CiU'], o['q']
        N = len(self.yw)
        P = cho_solve(o['cf'], np.eye(N)) - CiU @ cho_solve(o['cA'], CiU.T)
        tr = np.bincount(self.label, np.einsum('ij,ij->j', P @ M, M), self.K)
        Mq = np.bincount(self.label, (M.T @ q)**2, self.K)
        g = np.zeros(len(x))
        g[:self.K] = s**2*(Mq - tr)
        for j in (self.K, self.K + 1):
            e = np.zeros(len(x))
            e[j] = FD_STEP
            g[j] = (self.evidence(x + e, fit_h) - self.evidence(x - e, fit_h))/(2*FD_STEP)
        if fit_h:
            g[self.K + 2] = h**2*(q @ q - np.trace(P))
        return o['ll'], g

    def bounds(self, s_range, beta_range, ell_range, fit_h):
        b = [tuple(np.log(s_range))]*self.K + [tuple(beta_range), tuple(np.log(ell_range))]
        return b + [(np.log(0.2), np.log(5.))] if fit_h else b

    def fit(self, s_range=(1e-3, 10.), beta_range=(2., 3.), ell_range=(50., 5000.), fit_h=False, x0=None):
        if x0 is None:
            x0 = np.r_[np.full(self.K, np.log(0.3)), np.mean(beta_range), np.log(500.), [0.] if fit_h else []]
        bnds = self.bounds(s_range, beta_range, ell_range, fit_h)
        f = lambda x: tuple(-a for a in self.gradient(x, fit_h))
        res = minimize(f, x0, jac=True, method='L-BFGS-B', bounds=bnds, options=dict(maxiter=2000, ftol=1e-11, gtol=1e-6))
        s, beta, ell, h = self.unpack(res.x, fit_h)
        at = np.array([np.isclose(v, lo, atol=1e-6) or np.isclose(v, hi, atol=1e-6) for v, (lo, hi) in zip(res.x, bnds)])
        return dict(x=res.x, s=s, beta=float(beta), ell=float(ell), h=float(h), ll=float(-res.fun), fit_h=fit_h,
                    at_bound=at, n_eval=int(res.nfev), converged=bool(res.success), message=str(res.message))

    def hyper_sd(self, fit, step=1e-3):
        """Laplace s.d. of the hyperparameters x (the inverse Hessian of -log evidence; NaN where singular)"""
        x, n = fit['x'], len(fit['x'])
        H = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = step
            H[:, j] = -(self.gradient(x + e, fit['fit_h'])[1] - self.gradient(x - e, fit['fit_h'])[1])/(2*step)
        H = 0.5*(H + H.T)
        try:
            return np.sqrt(np.where(np.diag(np.linalg.inv(H)) > 0, np.diag(np.linalg.inv(H)), np.nan))
        except np.linalg.LinAlgError:
            return np.full(n, np.nan)

    def posterior(self, fit, area):
        """posterior mean and s.d. of v, the process means theta, and the area mean of v over each domain"""
        o = self._solve(fit['s'], fit['beta'], fit['ell'], fit['h'])
        M, Ls, q = o['M'], o['Ls'], o['q']
        n = len(self.label)
        Lb = np.zeros((n, n))
        for b, L in zip(self.blocks, Ls):
            Lb[np.ix_(b, b)] = L
        d2 = fit['s'][self.label]**2
        SG = Lb @ (d2[:, None]*M.T)                         # Sigma Gw'
        v = self.S @ o['theta'] + SG @ q
        Sigma = Lb @ (d2[:, None]*Lb.T)
        CiSG = cho_solve(o['cf'], SG.T)
        B = self.S - SG @ o['CiU']
        cov_theta = cho_solve(o['cA'], np.eye(self.K))
        Cov = Sigma - SG @ CiSG + B @ cov_theta @ B.T
        c = self.S*area[:, None]
        c /= c.sum(0)
        cov_avg = c.T @ Cov @ c
        return dict(v=v, v_sd=np.sqrt(np.maximum(np.diag(Cov), 0)), theta=o['theta'],
                    theta_sd=np.sqrt(np.diag(cov_theta)), avg=c.T @ v, avg_sd=np.sqrt(np.maximum(np.diag(cov_avg), 0)),
                    ll=o['ll'])

    def no_variation(self, h=1.):
        """log evidence and chi2 with no pattern inside the domains (PD under this error model, means unbounded)"""
        N = len(self.yw)
        cA = cho_factor(self.U.T @ self.U/h**2, lower=True)
        theta = cho_solve(cA, self.U.T @ self.yw/h**2)
        r = self.yw - self.U @ theta
        ll = (-N*np.log(h) - np.sum(np.log(np.diag(cA[0]))) - 0.5*r @ r/h**2 - 0.5*(N - self.K)*np.log(2*np.pi)
              - np.sum(np.log(self.hi - self.lo)))
        return float(ll), float(r @ r), theta


def layouts(cells, cfg, cache, say):
    """dict name -> (label, domain names): the table's layout first, then those of [method.layouts]"""
    par, spec = cfg['parameterisation'], cfg['method'].get('layouts', {})
    label, params, coverage = parameterise.domains(cells, par['table'], cache, area_name(cfg['target']['area']))
    used = np.unique(label)
    if len(used) < len(params):
        say(f'domains holding no cell (left out): {[params[k] for k in set(range(len(params))) - set(used)]}')
    out = {'polygons': (np.searchsorted(used, label), [params[k] for k in used])}
    if spec.get('merge'):
        ds = results.load(spec['merge'])
        rows = json.loads(ds.attrs['settings'])
        for i in spec.get('merge_settings', range(len(rows))):
            r = rows[i]
            lab = np.unique(ds[f'domain_{i}'].values, return_inverse=True)[1]
            out[f'merging lam x{r["lam_rel"]:.2g} gamma {r["gamma_rel"]:g} (K {lab.max() + 1})'] = \
                (lab, [f'unit {k}' for k in range(lab.max() + 1)])
    if spec.get('one'):
        out['one domain'] = (np.zeros(len(cells), int), ['all'])
    return out, coverage


def run(cfg, cache=None, log=print):
    cache = cache or Cache()
    t0 = time.time()
    say = lambda msg: log(f'[{(time.time() - t0)/60:5.1f} min] {msg}')
    pb = problem.local(cfg, cache, say)
    cells, me, par = pb.cells, cfg['method'], cfg['parameterisation']
    if par.get('kind') != 'domains':
        raise ValueError('DF needs [parameterisation] kind = "domains"')
    if cfg['errors'].get('model_error', {}).get('rms_nT', 0.) != 0:
        raise ValueError('DF takes measurement noise only (ADR 0007): set [errors] model_error rms_nT = 0')
    lo, hi = me.get('bounds', [0., 4.])
    opts = dict(s_range=tuple(me.get('s_range', [1e-3, 10.])), beta_range=tuple(me.get('beta_range', [2., 3.])),
                ell_range=tuple(me.get('ell_range', [50., 5000.])))
    fit_h = me.get('noise', 'fixed') == 'fitted'
    cell_km2 = 4*np.pi*R_KM**2/(12*cells.nside**2)
    lays, coverage = layouts(cells, cfg, cache, say)
    say(f'{len(lays)} layouts: {list(lays)}')
    ranking, main = {}, None
    for name, (label, names) in lays.items():
        m = DomainFractal(pb.Gw, pb.yw, label, cells.clon, cells.clat, cell_km2, lo, hi)
        f = m.fit(fit_h=fit_h, **opts)
        ll0, chi0, _ = m.no_variation()
        ranking[name] = dict(K=int(m.K), ll=f['ll'], ll_no_variation=ll0, beta=f['beta'], ell_km=f['ell'], h=f['h'],
                             converged=f['converged'], n_eval=f['n_eval'])
        say(f'{name}: K {m.K}, log evidence {f["ll"]:.1f} (no variation {ll0:.1f}), beta {f["beta"]:.2f}, '
            f'ell {f["ell"]:.0f} km, h {f["h"]:.2f}, {f["n_eval"]} evaluations' + ('' if f['converged'] else ' NOT CONVERGED'))
        if main is None:
            main = (name, m, f, names)
    name, m, f, names = main
    post = m.posterior(f, cells.area_km2)
    sd = m.hyper_sd(f) if me.get('hyper_sd', True) else np.full(len(f['x']), np.nan)
    say(f'posterior and hyperparameter s.d. of {name!r}')
    check = None
    if not fit_h:                                         # the noise scale fitted as a check
        fh = m.fit(fit_h=True, x0=np.r_[f['x'], 0.], **opts)
        check = dict(h=fh['h'], ll=fh['ll'], beta=fh['beta'], ell_km=fh['ell'])
        say(f'check with the noise scale fitted: h {fh["h"]:.3f}, log evidence {fh["ll"]:.1f} (+{fh["ll"] - f["ll"]:.1f})')
    v, label = post['v'], m.label
    area = cells.area_km2
    neg = np.bincount(label, area*(v < 0), m.K)/np.bincount(label, area, m.K)
    out_box = [names[k] for k in range(m.K) if not (lo <= post['theta'][k] <= hi)]
    r = pb.yw - pb.Gw @ v
    chi2 = float(r @ r)
    N = len(pb.yw)
    ll0, chi0, _ = m.no_variation()
    K = m.K
    table = {names[k]: dict(mean=float(post['avg'][k]), mean_sd=float(post['avg_sd'][k]),
                            theta=float(post['theta'][k]), theta_sd=float(post['theta_sd'][k]),
                            strength=float(f['s'][k]), log_strength_sd=float(sd[k]),
                            strength_at_bound=bool(f['at_bound'][k]), share_below_zero=float(neg[k]),
                            area_km2=float(area[label == k].sum())) for k in range(K)}
    say(f'{K} domains, {N} data: chi2/N {chi2/N:.2f} (no variation {chi0/N:.2f}); beta {f["beta"]:.2f} '
        f'+- {sd[K]:.2f}, ell {f["ell"]:.0f} km (log s.d. {sd[K + 1]:.2f}); strengths at the lower bound: '
        f'{int(f["at_bound"][:K].sum())}; cells below zero {np.mean(v < 0):.1%}; means outside the box: {out_box}')
    cell_vars = dict(vis=v, vis_sd=post['v_sd'], vis_domain=post['avg'][label], strength=f['s'][label],
                     domain=label.astype(float))
    if pb.truth is not None:
        cell_vars['truth'] = pb.truth
    model = pb.G @ v
    attrs = dict(domains=table, layout=name, layouts=ranking, beta=f['beta'], beta_sd=float(sd[K]), ell_km=f['ell'],
                 log_ell_sd=float(sd[K + 1]), noise_scale=f['h'], noise_check=check, log_evidence=f['ll'],
                 log_evidence_no_variation=ll0, chi2_per_datum=chi2/N, chi2_per_datum_no_variation=chi0/N,
                 n_domains=K, node_coverage=coverage, bounds=[lo, hi], means_outside_box=out_box,
                 cells_below_zero=float(np.mean(v < 0)),
                 panel_titles={'vis_domain': 'VIS averaged over each domain', 'strength': 'Strength of variation in each domain (1 s.d.)'},
                 **pb.info)
    return results.make(cfg, cells, cell_vars, points=(pb.plon, pb.plat),
                        point_vars=dict(data=pb.y, model=model, residual=pb.y - model), **attrs)
