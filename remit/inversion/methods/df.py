"""
DF (domain fractal, CONTEXT.md; ADR 0007): VIS = the mean of the cell's domain + a Gaussian fractal pattern inside each
domain, the patterns independent between domains, from a local fit with measurement noise only.

    v = S theta + u,    u_k ~ N(0, s_k^2 R_k),    yw = Gw v + e,    e ~ N(0, h^2 I)

S: the indicator columns of a domain layout. R_k: the correlation of the cell averages of an isotropic field on the
sphere with angular spectrum S_l ~ (l(l+1) + (R/ell)^2)^(-beta/2) (a 2-D power spectrum k^-beta above the largest
patch size ell), restricted to the cells of domain k. Cell averages: a spherical cap of the cell's area as the window,
so the correlation is a Legendre series, positive definite on the sphere for any beta. theta, means = 'box': uniform on
the box [lo, hi] per domain, integrated out as a Gaussian integral over all of R^K (a domain whose best mean falls
outside the box is reported, not given zero mass). means = 'gaussian' (ADR 0007 amendment, as GCM's domain levels in
ADR 0008): theta_k = v0 + delta_k with delta_k ~ N(0, tau^2) and v0 uniform on the box; delta integrated out adds
tau^2 U U' to the data covariance (U = Gw S), so tau is a hyperparameter like an extra error (the last of sig).
h: 1 (the error model as it is: the data are whitened for noise only) or fitted.
Hyperparameters (log s_k, beta, log ell, and log h when fitted) maximise the log evidence (exact gradients in s_k
and h, central differences in beta and ell), with Laplace s.d. from the curvature.

Config keys used, besides those of problem.local (weighting 'full', model_error rms 0):
  [parameterisation] kind = 'domains', table (as PD)
  [method] means ('box' or 'gaussian'), tau_range (SI km, for 'gaussian'), bounds ([lo, hi] of the means), beta_range, ell_range (km), s_range (SI km), noise ('fixed' or 'fitted'),
           hyper_sd (Laplace s.d. of the hyperparameters), layouts = {one = true, merge = <FD-merge result>,
           merge_settings = [i, ...], controls = true, polygons = false} (further layouts, ranked by the evidence with
           the table's; controls: random Voronoi tiles, as many as the table's domains, blind to geology; polygons =
           false leaves the table's own layout out, the first remaining layout is then mapped),
           ring_error = {beyond_km = 50} (an extra error at the data points farther than beyond_km from every cell
           centre, i.e. the ring of points around the target: correlated with the field model's own spectrum in the
           band, its RMS sigma_ring (nT) fitted; for errors of the background there, which no domain can explain),
           beta_profile = [beta, ...] (the log evidence of the first layout with beta fixed at each value),
           sample = {chains, steps, burn, n_post} (sample the hyperparameters of the first layout by adaptive Metropolis,
           prior uniform in x on the bounds; maps and domain values then average over n_post samples), seed
Outputs: the posterior mean map (vis), its s.d., the domain averages, the strengths, and the most probable map with
VIS >= 0 (vis_bounded: a box QP on the Gaussian posterior at the best-fit hyperparameters).
"""
import json
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, block_diag
from scipy.optimize import minimize

from .. import errors, parameterise, problem, results, sampling
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

    def __init__(self, Gw, yw, label, clon, clat, cell_area_km2, lo, hi, extra=(), means='box',
                 tau_range=(0.01, 4.)):
        self.Gw, self.yw, self.label = Gw, yw, np.asarray(label)
        self.extra = list(extra)                    # error shapes E_j (data x data), each scaled by a fitted sigma_j
        self.extra_ranges = [(0.1, 500.)]*len(self.extra)
        self.extra_start = [30.]*len(self.extra)
        self.K = int(self.label.max()) + 1
        self.S = parameterise.indicator(self.label)
        self.Udom = Gw @ self.S
        self.gaussian = means == 'gaussian' and self.K > 1
        if self.gaussian:                           # theta_k = v0 + delta_k: v0 the only design column, delta in C
            self.Sd = np.ones((len(self.label), 1))
            self.extra.append(self.Udom @ self.Udom.T)
            self.extra_ranges.append(tuple(tau_range))
            self.extra_start.append(0.3)
        else:
            self.Sd = self.S
        self.U = Gw @ self.Sd
        self.p = self.U.shape[1]
        self.lo, self.hi = (np.broadcast_to(np.asarray(b, float), (self.p,)) for b in (lo, hi))
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
        """x = log s_k (K), beta, log ell, [log h if fit_h], log sigma_j (one per extra error shape)"""
        s = np.exp(x[:self.K])
        beta, ell = x[self.K], np.exp(x[self.K + 1])
        h = np.exp(x[self.K + 2]) if fit_h else 1.
        sig = np.exp(np.asarray(x[self.K + 2 + int(fit_h):]))
        return s, beta, ell, h, sig

    def _solve(self, s, beta, ell, h, sig=()):
        M, Ls = self.factors(beta, ell)
        Ms = M*s[self.label]
        N = len(self.yw)
        C = Ms @ Ms.T
        C[np.diag_indices(N)] += h**2
        for sj, E in zip(sig, self.extra):
            C += sj**2*E
        cf = cho_factor(C, lower=True)
        CiU, Ciy = cho_solve(cf, self.U), cho_solve(cf, self.yw)
        A = self.U.T @ CiU
        cA = cho_factor(A, lower=True)
        theta = cho_solve(cA, self.U.T @ Ciy)
        q = Ciy - CiU @ theta                       # P y, P = C^-1 - C^-1 U A^-1 U' C^-1
        r = self.yw - self.U @ theta
        ll = (-np.sum(np.log(np.diag(cf[0]))) - np.sum(np.log(np.diag(cA[0]))) - 0.5*r @ q
              - 0.5*(N - self.p)*np.log(2*np.pi) - np.sum(np.log(self.hi - self.lo)))
        return dict(ll=float(ll), M=M, Ls=Ls, cf=cf, CiU=CiU, cA=cA, theta=theta, q=q)

    def evidence(self, x, fit_h=False):
        return self._solve(*self.unpack(x, fit_h))['ll']

    def gradient(self, x, fit_h=False):
        """(log evidence, gradient in x)"""
        s, beta, ell, h, sig = self.unpack(x, fit_h)
        o = self._solve(s, beta, ell, h, sig)
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
        for j, (sj, E) in enumerate(zip(sig, self.extra)):
            g[self.K + 2 + int(fit_h) + j] = sj**2*(q @ E @ q - np.sum(P*E))
        return o['ll'], g

    def bounds(self, s_range, beta_range, ell_range, fit_h):
        b = [tuple(np.log(s_range))]*self.K + [tuple(beta_range), tuple(np.log(ell_range))]
        return b + [(np.log(0.2), np.log(5.))]*int(fit_h) + [tuple(np.log(r)) for r in self.extra_ranges]

    @property
    def tau(self):
        """index of tau in sig (the spread of the domain means), or None"""
        return len(self.extra) - 1 if self.gaussian else None

    def fit(self, s_range=(0.01, 10.), beta_range=(2., 3.), ell_range=(50., 5000.), fit_h=False, x0=None):
        if x0 is None:
            x0 = np.r_[np.full(self.K, np.log(0.3)), np.mean(beta_range), np.log(500.), [0.] if fit_h else [],
                       np.log(self.extra_start)]
        bnds = self.bounds(s_range, beta_range, ell_range, fit_h)
        x0 = np.clip(x0, [b[0] for b in bnds], [b[1] for b in bnds])
        f = lambda x: tuple(-a for a in self.gradient(x, fit_h))
        res = minimize(f, x0, jac=True, method='L-BFGS-B', bounds=bnds, options=dict(maxiter=2000, ftol=1e-11, gtol=1e-6))
        s, beta, ell, h, sig = self.unpack(res.x, fit_h)
        at = np.array([lo < hi and (np.isclose(v, lo, atol=1e-6) or np.isclose(v, hi, atol=1e-6))
                       for v, (lo, hi) in zip(res.x, bnds)])
        return dict(x=res.x, s=s, beta=float(beta), ell=float(ell), h=float(h), sig=sig, ll=float(-res.fun), fit_h=fit_h,
                    at_bound=at, n_eval=int(res.nfev), converged=bool(res.success), message=str(res.message))

    def hessian(self, fit, step=1e-3):
        """Hessian of -log evidence in x (central differences of the gradient)"""
        x, n = fit['x'], len(fit['x'])
        H = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = step
            H[:, j] = -(self.gradient(x + e, fit['fit_h'])[1] - self.gradient(x - e, fit['fit_h'])[1])/(2*step)
        return 0.5*(H + H.T)

    @staticmethod
    def laplace_sd(H):
        """Laplace s.d. of the hyperparameters x (the inverse Hessian; NaN where singular)"""
        try:
            d = np.diag(np.linalg.inv(H))
            return np.sqrt(np.where(d > 0, d, np.nan))
        except np.linalg.LinAlgError:
            return np.full(len(H), np.nan)

    def hyper_sd(self, fit, step=1e-3):
        return self.laplace_sd(self.hessian(fit, step))

    def chain(self, fit, bnds, n_steps, seed, cov0, start_sd=1.):
        """adaptive Metropolis (Haario et al. 2001) over x, prior uniform in x on the bounds (log-uniform strengths,
        uniform beta); starts from the fit displaced by start_sd x the proposal s.d. (overdispersed chains).
        Returns (samples, log evidences, acceptance rate)."""
        rng = np.random.default_rng(seed)
        lo, hi = np.array(bnds, float).T
        free = hi > lo
        d = int(free.sum())
        C0 = cov0[np.ix_(free, free)]
        x = fit['x'].copy()
        if start_sd:
            for _ in range(100):
                y = x.copy()
                y[free] += start_sd*np.linalg.cholesky(C0) @ rng.standard_normal(d)
                if np.all((y >= lo) & (y <= hi)):
                    x = y
                    break
        ll = self.evidence(x, fit['fit_h'])
        X, LL, acc = np.empty((n_steps, len(x))), np.empty(n_steps), 0
        Lp, log_scale, recent = np.linalg.cholesky(C0), np.log(2.38**2/d), []
        for i in range(n_steps):
            if i >= 1000 and i % 200 == 0:                  # adapt the shape to the chain so far
                Lp = np.linalg.cholesky(np.cov(X[i//2:i][:, free].T) + 1e-8*np.eye(d))
            y = x.copy()
            y[free] += np.exp(0.5*log_scale)*(Lp @ rng.standard_normal(d))
            ok = False
            if np.all((y >= lo) & (y <= hi)):
                ly = self.evidence(y, fit['fit_h'])
                if np.log(rng.uniform()) < ly - ll:
                    x, ll, ok = y, ly, True
            acc += ok
            log_scale += (float(ok) - 0.234)/np.sqrt(i + 1)*(i < n_steps//2)    # scale to ~23% acceptance (burn-in)
            X[i], LL[i] = x, ll
        return X, LL, acc/n_steps

    def fit_at(self, x, fit_h):
        s, beta, ell, h, sig = self.unpack(x, fit_h)
        return dict(x=x, s=s, beta=float(beta), ell=float(ell), h=float(h), sig=sig, fit_h=fit_h)

    def posterior(self, fit, area, full=False):
        """posterior mean and s.d. of v, the process means theta, and the area mean of v over each domain (and the
        covariance of v when full)"""
        sig = fit.get('sig', ())
        o = self._solve(fit['s'], fit['beta'], fit['ell'], fit['h'], sig)
        M, Ls, q = o['M'], o['Ls'], o['q']
        n = len(self.label)
        Lb = np.zeros((n, n))
        for b, L in zip(self.blocks, Ls):
            Lb[np.ix_(b, b)] = L
        d2 = fit['s'][self.label]**2
        SG = Lb @ (d2[:, None]*M.T)                         # Sigma Gw'
        Sigma = Lb @ (d2[:, None]*Lb.T)
        if self.gaussian:                                   # the domain offsets delta are part of the random field
            t2 = float(sig[self.tau])**2
            SG = SG + t2*(self.S @ self.Udom.T)
            Sigma = Sigma + t2*(self.S @ self.S.T)
        v = self.Sd @ o['theta'] + SG @ q
        CiSG = cho_solve(o['cf'], SG.T)
        B = self.Sd - SG @ o['CiU']
        cov_theta = cho_solve(o['cA'], np.eye(self.p))
        Cov = Sigma - SG @ CiSG + B @ cov_theta @ B.T
        if self.gaussian:                                   # theta_k = v0 + delta_k
            Bt = 1. - t2*(self.Udom.T @ o['CiU'])
            theta = o['theta'][0] + t2*(self.Udom.T @ q)
            cov_th = t2*np.eye(self.K) - t2**2*(self.Udom.T @ cho_solve(o['cf'], self.Udom)) + Bt @ cov_theta @ Bt.T
        else:
            theta, cov_th = o['theta'], cov_theta
        c = self.S*area[:, None]
        c /= c.sum(0)
        cov_avg = c.T @ Cov @ c
        out = dict(v=v, v_sd=np.sqrt(np.maximum(np.diag(Cov), 0)), theta=theta,
                   theta_sd=np.sqrt(np.maximum(np.diag(cov_th), 0)), avg=c.T @ v,
                   avg_sd=np.sqrt(np.maximum(np.diag(cov_avg), 0)), ll=o['ll'], v0=float(o['theta'][0]))
        if full:
            out['cov'] = Cov
        return out

    def no_variation(self, h=1., sig=()):
        """log evidence and chi2 with no pattern inside the domains (PD under this error model, means unbounded)"""
        if len(sig):
            o = self._solve(np.full(self.K, 1e-9), 2.5, 500., h, sig)
            r = self.yw - self.U @ o['theta']
            return o['ll'], float(r @ r), o['theta']
        N = len(self.yw)
        cA = cho_factor(self.U.T @ self.U/h**2, lower=True)
        theta = cho_solve(cA, self.U.T @ self.yw/h**2)
        r = self.yw - self.U @ theta
        ll = (-N*np.log(h) - np.sum(np.log(np.diag(cA[0]))) - 0.5*r @ r/h**2 - 0.5*(N - self.p)*np.log(2*np.pi)
              - np.sum(np.log(self.hi - self.lo)))
        return float(ll), float(r @ r), theta


def proposal_cov(H, bnds):
    """first proposal covariance of the sampler: the inverse Hessian at the best fit (correlations kept: beta and ell
    trade off), its eigenvalues floored so no direction is wider than a quarter of the widest bound, and each
    variance capped at (its bound width / 4)^2"""
    width = np.array([b[1] - b[0] for b in bnds])
    w, V = np.linalg.eigh(0.5*(H + H.T))
    C = (V/np.maximum(w, 1/(width.max()/4)**2)) @ V.T
    d = np.sqrt(np.diag(C))
    cap = np.minimum(1, (np.maximum(width, 1e-12)/4)/d)
    return C*cap[:, None]*cap[None, :]


def bounded_map(v, Cov, lo=0.):
    """the most probable map with v >= lo under the Gaussian posterior N(v, Cov): a box QP with H = Cov^-1"""
    from .. import core
    n = len(v)
    cf = cho_factor(Cov + 1e-10*np.trace(Cov)/n*np.eye(n), lower=True)
    H = cho_solve(cf, np.eye(n))
    H = 0.5*(H + H.T)
    return core.box_qp(H, H @ v, lo, np.inf, v=np.maximum(v, lo), tol=1e-10)


def mixture(m, fits, area):
    """posterior summaries averaged over hyperparameter samples: means averaged, variances = mean of the variances
    + variance of the means"""
    ps = [m.posterior(f, area) for f in fits]
    out = {}
    for k, ksd in (('v', 'v_sd'), ('avg', 'avg_sd'), ('theta', 'theta_sd')):
        mu = np.array([p[k] for p in ps])
        var = np.array([p[ksd]**2 for p in ps])
        out[k], out[ksd] = mu.mean(0), np.sqrt(var.mean(0) + mu.var(0))
    return out


def rhat(chains):
    """Gelman-Rubin potential scale reduction per parameter (chains: n_chains x n x d)"""
    c = np.asarray(chains)
    n = c.shape[1]
    W = c.var(1, ddof=1).mean(0)
    B = n*c.mean(1).var(0, ddof=1)
    return np.sqrt(((n - 1)/n*W + B/n)/np.where(W > 0, W, np.nan))


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
    if spec.get('controls'):
        k = len(out['polygons'][1])
        lab = np.unique(parameterise.voronoi(cells, k, parameterise.VORONOI_SEED + k), return_inverse=True)[1]
        out[f'random tiles (K {lab.max() + 1})'] = (lab, [f'tile {j}' for j in range(lab.max() + 1)])
    if spec.get('one'):
        out['one domain'] = (np.zeros(len(cells), int), ['all'])
    if spec.get('polygons') is False:                  # e.g. the controls alone, run beside the polygons
        out.pop('polygons')
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
    means = me.get('means', 'box')
    opts = dict(s_range=tuple(me.get("s_range", [0.01, 10.])), beta_range=tuple(me.get('beta_range', [2., 3.])),
                ell_range=tuple(me.get('ell_range', [50., 5000.])))
    fit_h = me.get('noise', 'fixed') == 'fitted'
    cell_km2 = 4*np.pi*R_KM**2/(12*cells.nside**2)
    lays, coverage = layouts(cells, cfg, cache, say)
    say(f'{len(lays)} layouts: {list(lays)}')
    xyz_c, xyz_p = unit(cells.clon, cells.clat), unit(pb.plon, pb.plat)
    dist_km = np.arccos(np.clip((xyz_p @ xyz_c.T).max(1), -1., 1.))*R_KM          # to the nearest cell centre
    ring = dist_km > me.get('ring_error', {}).get('beyond_km', 50.)
    extra = []
    if me.get('ring_error'):
        Wd = pb.W.toarray() if hasattr(pb.W, 'toarray') else np.asarray(pb.W)
        Cr = errors.point_shape(pb.spectra['model'], pb.band, pb.plon[ring], pb.plat[ring])
        extra = [Wd[:, ring] @ Cr @ Wd[:, ring].T]
        say(f'ring error at {ring.sum()} of {len(ring)} points')
    ranking, main = {}, None
    for name, (label, names) in lays.items():
        m = DomainFractal(pb.Gw, pb.yw, label, cells.clon, cells.clat, cell_km2, lo, hi, extra, means,
                          tuple(me.get('tau_range', [0.01, 4.])))
        f = m.fit(fit_h=fit_h, **opts)
        ll0, chi0, _ = m.no_variation(sig=f['sig'])
        tau = float(f['sig'][m.tau]) if m.gaussian else None
        ranking[name] = dict(K=int(m.K), ll=f['ll'], ll_no_variation=ll0, beta=f['beta'], ell_km=f['ell'], h=f['h'],
                             sigma_ring_nT=[float(v) for v in f['sig'][:len(extra)]], tau=tau,
                             converged=f['converged'], n_eval=f['n_eval'])
        say(f'{name}: K {m.K}, log evidence {f["ll"]:.1f} (no variation {ll0:.1f}), beta {f["beta"]:.2f}, '
            f'ell {f["ell"]:.0f} km, h {f["h"]:.2f}' + (f', ring error {f["sig"][0]:.1f} nT' if extra else '') +
            (f', spread of the domain means {tau:.2f} SI km' if m.gaussian else '') +
            f', {f["n_eval"]} evaluations' + ('' if f['converged'] else ' NOT CONVERGED'))
        if main is None:
            main = (name, m, f, names)
    name, m, f, names = main
    profile = []
    for b in me.get('beta_profile', []):
        fb = m.fit(fit_h=fit_h, x0=np.r_[f['x'][:m.K], b, f['x'][m.K + 1:]], **{**opts, 'beta_range': (b, b)})
        profile.append(dict(beta=float(b), ll=fb['ll'], ell_km=fb['ell'], h=fb['h'],
                            sigma_ring_nT=[float(v) for v in fb['sig']]))
        say(f'beta fixed at {b:.2f}: log evidence {fb["ll"]:.1f} ({fb["ll"] - f["ll"]:+.1f}), ell {fb["ell"]:.0f} km'
            + (f', ring error {fb["sig"][0]:.1f} nT' if extra else ''))
    area = cells.area_km2
    post = m.posterior(f, area, full=True)
    spec = me.get('sample') or {}
    spec = spec if spec.get('steps', 1) > 0 else {}                 # steps = 0: no sampling
    H = m.hessian(f) if me.get('hyper_sd', True) or spec else None
    sd = m.laplace_sd(H) if H is not None else np.full(len(f['x']), np.nan)
    say(f'posterior and hyperparameter s.d. of {name!r}')
    sampled = None
    if spec:
        bnds = m.bounds(opts['s_range'], opts['beta_range'], opts['ell_range'], fit_h)
        width = np.array([b[1] - b[0] for b in bnds])
        cov0 = proposal_cov(H, bnds)
        n_ch, steps = int(spec.get('chains', 3)), int(spec.get('steps', 10000))
        seed = int(me.get('seed', 20261009))
        out = sampling.chains(lambda c: m.chain(f, bnds, steps, seed + c, cov0), n_ch, cfg['compute'].get('workers', 3))
        k0 = int(steps*spec.get('burn', 1/3))
        X = np.array([o[0][k0:] for o in out])
        free = width > 0
        R = rhat(X[:, :, free])
        pooled = X.reshape(-1, X.shape[2])
        pick = pooled[np.linspace(0, len(pooled) - 1, int(spec.get('n_post', 150))).astype(int)]
        mix = mixture(m, [m.fit_at(x, fit_h) for x in pick], area)
        post = {**post, **mix}
        q = lambda j: (float(np.mean(pooled[:, j])), float(np.std(pooled[:, j])))
        Rf = np.full(len(free), np.nan)
        Rf[free] = R
        big = np.bincount(m.label, area, m.K) >= np.median(np.bincount(m.label, area, m.K))
        say(f'R-hat: beta {Rf[m.K]:.2f}, ell {Rf[m.K + 1]:.2f}, extra errors {np.round(Rf[m.K + 2:], 2).tolist()}, '
            f'strengths of the larger half of the domains max {np.nanmax(Rf[:m.K][big]):.2f}, smaller half max '
            f'{np.nanmax(Rf[:m.K][~big]):.2f}; strengths with R-hat > 1.1: '
            f'{[(names[k], round(float(np.exp(pooled[:, k]).mean()), 3)) for k in range(m.K) if Rf[k] > 1.1]}')
        sampled = dict(chains=n_ch, steps=steps, burn=k0, acceptance=[float(o[2]) for o in out], rhat=Rf.tolist(),
                       rhat_max=float(np.nanmax(R)), rhat_beta=float(R[np.flatnonzero(free).tolist().index(m.K)])
                       if free[m.K] else None, beta=q(m.K), log_ell=q(m.K + 1),
                       log_strength_mean=pooled[:, :m.K].mean(0).tolist(), log_strength_sd=pooled[:, :m.K].std(0).tolist(),
                       log_sigma_ring=q(m.K + 2 + int(fit_h)) if extra else None,
                       log_tau=q(m.K + 2 + int(fit_h) + m.tau) if m.gaussian else None)
        sd = pooled.std(0)
        say(f'sampled: {n_ch} chains x {steps} steps, acceptance {np.round(sampled["acceptance"], 2).tolist()}, '
            f'R-hat max {sampled["rhat_max"]:.2f}; beta {sampled["beta"][0]:.2f} +- {sampled["beta"][1]:.2f}, '
            f'ell {np.exp(sampled["log_ell"][0]):.0f} km (log s.d. {sampled["log_ell"][1]:.2f})')
    vb = bounded_map(post['v'], post['cov'])
    say(f'bounded map (VIS >= 0): moves cells by up to {np.max(abs(vb - post["v"])):.2f} SI km')
    check = None
    if not fit_h:                                         # the noise scale fitted as a check
        fh = m.fit(fit_h=True, x0=np.r_[f['x'][:m.K + 2], 0., f['x'][m.K + 2:]], **opts)
        check = dict(h=fh['h'], ll=fh['ll'], beta=fh['beta'], ell_km=fh['ell'])
        say(f'check with the noise scale fitted: h {fh["h"]:.3f}, log evidence {fh["ll"]:.1f} (+{fh["ll"] - f["ll"]:.1f})')
    v, label = post['v'], m.label
    neg = np.bincount(label, area*(v < 0), m.K)/np.bincount(label, area, m.K)
    out_box = [names[k] for k in range(m.K) if not (lo <= post['theta'][k] <= hi)]
    tau = float(f['sig'][m.tau]) if m.gaussian else None
    r = pb.yw - pb.Gw @ v
    chi2 = float(r @ r)
    N = len(pb.yw)
    ll0, chi0, _ = m.no_variation(sig=f['sig'])
    K = m.K
    res_pts = pb.y - pb.G @ v
    rms = dict(onshore=float(np.sqrt(np.mean(res_pts[~ring]**2))), ring=float(np.sqrt(np.mean(res_pts[ring]**2))))
    say(f'residual RMS: onshore {rms["onshore"]:.1f} nT, ring {rms["ring"]:.1f} nT'
        + (f'; ring error {f["sig"][0]:.1f} nT' if extra else ''))
    table = {names[k]: dict(mean=float(post['avg'][k]), mean_sd=float(post['avg_sd'][k]),
                            theta=float(post['theta'][k]), theta_sd=float(post['theta_sd'][k]),
                            strength=float(f['s'][k]), log_strength_sd=float(sd[k]),
                            strength_at_bound=bool(f['at_bound'][k]), share_below_zero=float(neg[k]),
                            area_km2=float(area[label == k].sum())) for k in range(K)}
    say(f'{K} domains, {N} data: chi2/N {chi2/N:.2f} (no variation {chi0/N:.2f}); beta {f["beta"]:.2f} '
        f'+- {sd[K]:.2f}, ell {f["ell"]:.0f} km (log s.d. {sd[K + 1]:.2f}); strengths at the lower bound: '
        f'{int(f["at_bound"][:K].sum())}; cells below zero {np.mean(v < 0):.1%}; means outside the box: {out_box}'
        + (f'; v0 {post["v0"]:.2f}, spread of the domain means {tau:.2f} SI km' if m.gaussian else ''))
    cell_vars = dict(vis=v, vis_sd=post['v_sd'], vis_domain=post['avg'][label], vis_bounded=vb,
                     strength=f['s'][label], domain=label.astype(float))
    if pb.truth is not None:
        cell_vars['truth'] = pb.truth
    model = pb.G @ v
    attrs = dict(domains=table, layout=name, layouts=ranking, beta=f['beta'], beta_sd=float(sd[K]), ell_km=f['ell'],
                 log_ell_sd=float(sd[K + 1]), noise_scale=f['h'], noise_check=check, log_evidence=f['ll'],
                 sigma_ring_nT=[float(v) for v in f['sig'][:len(extra)]], residual_rms_nT=rms, beta_profile=profile,
                 sampled=sampled, means=means, tau=tau, v0=post.get('v0'),
                 log_evidence_no_variation=ll0, chi2_per_datum=chi2/N, chi2_per_datum_no_variation=chi0/N,
                 n_domains=K, node_coverage=coverage, bounds=[lo, hi], means_outside_box=out_box,
                 cells_below_zero=float(np.mean(v < 0)),
                 panel_titles={'vis_domain': 'VIS averaged over each domain', 'vis_bounded': 'VIS, most probable map with VIS >= 0',
                               'strength': 'Strength of variation in each domain (1 s.d.)'},
                 **pb.info)
    return results.make(cfg, cells, cell_vars, points=(pb.plon, pb.plat),
                        point_vars=dict(data=pb.y, model=model, residual=pb.y - model), **attrs)
