"""
Linear-Gaussian VIS models on the cells: v = B theta + u, with a Gaussian Markov random field (GMRF)
for the part u that the design columns B do not explain, and the regularised maps (L2, TV) that are
the same model with fixed penalties.

Prior on u
  precision Q = diag(a/ell^2) + L, with a the cell areas (mean 1), L the cell-graph Laplacian with
  weights 1/d^2 (d: great-circle distance between centres, km) and ell (km) a correlation length
  (per cell: the class of the cell); R = the correlation matrix of Q^-1 (unit marginal variance), and
  Cov(u) = S R S with S = diag(s) (s: standard deviation, SI.km, per class)
Data: dw = Gw v + n, n ~ N(0, h^2 I) (whitened units), so dw ~ N(Gw B theta, C),
  C = Gw S R S Gw' + h^2 I (N x N, N = 845); theta (uniform on a box) integrated out by
  linear.log_evidence with this C. Hyperparameters (h, s, ell) maximise the evidence.
Maps
  posterior mean and standard deviation of v (Gaussian, no positivity)
  MAP with v >= 0: min |dw - Gw v|^2/h^2 + (v - v0)' P (v - v0), solved as one bounded least-squares
  problem on stacked rows (P = rows' rows): the GMRF prior (P = Cov(u)^-1, v0 = B theta), L2
  (P = beta (diag(a)/ell^2 + D'D)) or TV (IRLS weights on the rows of D)
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import csc_matrix, diags, issparse, vstack
from scipy.sparse.linalg import splu

from .linear import log_evidence, posterior_theta, box_qp, SparsePlusLowRank

H_BOUNDS, S_BOUNDS, ELL_BOUNDS = (0.2, 10.), (0.005, 5.), (20., 5000.)
ELL_GRID = np.geomspace(40., 3000., 12)
N_CACHE = 40


def precision(a, Lw, ell_cell):
    """Q = diag(a/ell^2) + Lw (dense)"""
    return np.diag(a/np.asarray(ell_cell, float)**2) + Lw


class Model:
    """the data covariance and evidence of v = B theta + u for one problem (Gw, dw, cells);
    cls: class of each cell (classes share s and ell; one class if None)"""

    def __init__(self, Gw, dw, a, Lw, cls=None, n_cache=N_CACHE):
        self.Gw, self.dw, self.a, self.Lw = Gw, dw, a/np.mean(a), csc_matrix(Lw)
        self.cls = np.zeros(Gw.shape[1], int) if cls is None else np.asarray(cls)
        self.nc = self.cls.max() + 1
        self.n_cache = n_cache
        self._cache, self._R = {}, (None, None)

    def _ell(self, ell):
        return np.broadcast_to(np.asarray(ell, float), (self.nc,))

    def correlation(self, ell):
        """R for ell (per class), from a sparse LU of Q (the last one kept)"""
        ell = self._ell(ell)
        key = tuple(np.round(ell, 8))
        if self._R[0] != key:
            lu = splu(csc_matrix(diags(self.a/ell[self.cls]**2) + self.Lw))
            Qi = lu.solve(np.eye(len(self.a)))
            d = 1/np.sqrt(np.diag(Qi))
            self._R = (key, Qi*d[:, None]*d[None, :])
        return self._R[1]

    def blocks(self, ell):
        """Gw_a R_ab Gw_b' for each pair of classes (cached per ell, n_cache kept)"""
        key = tuple(np.round(self._ell(ell), 8))
        if key not in self._cache:
            if len(self._cache) >= self.n_cache:
                self._cache.pop(next(iter(self._cache)))
            R = self.correlation(ell)
            blocks = {}
            for i in range(self.nc):
                GRi = self.Gw[:, self.cls == i] @ R[self.cls == i, :]
                for j in range(self.nc):
                    blocks[(i, j)] = GRi[:, self.cls == j] @ self.Gw[:, self.cls == j].T
            self._cache[key] = blocks
        return self._cache[key]

    def cov(self, h, s, ell):
        s = np.broadcast_to(np.asarray(s, float), (self.nc,))
        B = self.blocks(ell)
        C = h**2*np.eye(len(self.dw))
        for (i, j), b in B.items():
            C += s[i]*s[j]*b
        return C

    def prior_cov(self, s, ell):
        """Cov(u) = S R S"""
        R = self.correlation(ell)
        sc = np.broadcast_to(np.asarray(s, float), (self.nc,))[self.cls]
        return R*sc[:, None]*sc[None, :]

    def evidence(self, B, lo, hi, h, s, ell, dw=None):
        dw = self.dw if dw is None else dw
        return log_evidence(self.Gw @ B, dw, lo, hi, h=1., C=self.cov(h, s, ell))

    def _fit_hs(self, B, lo, hi, ell, x0, dw):
        """maximise over log h, log s (one s) at fixed ell"""
        def f(x):
            ll = self.evidence(B, lo, hi, *np.exp(x), ell, dw=dw)[0]
            return -ll if np.isfinite(ll) else 1e12
        r = minimize(f, np.log(x0), method='Nelder-Mead', bounds=[np.log(H_BOUNDS), np.log(S_BOUNDS)],
                     options=dict(xatol=1e-3, fatol=1e-3))
        return -r.fun, np.exp(r.x), r.nfev

    def fit(self, B, lo, hi, per_class=False, x0=(1., 0.3), dw=None, maxiter=3000, start=None):
        """maximise the evidence over the hyperparameters.
        shared (one h, s, ell): profile over ell (ELL_GRID, then a bounded 1-D search around the best
        node; the cached blocks are reused across models), inner Nelder-Mead over log h, log s.
        per_class: Nelder-Mead over log h, log s_c, log ell_c, from start (a shared fit) if given"""
        n_eval = 0
        if not per_class:
            prof = []
            for ell in ELL_GRID:
                ll, x, ne = self._fit_hs(B, lo, hi, ell, x0, dw)
                prof.append((ll, x)); n_eval += ne
            k = int(np.argmax([p[0] for p in prof]))
            a, b = np.log(ELL_GRID[max(k - 1, 0)]), np.log(ELL_GRID[min(k + 1, len(ELL_GRID) - 1)])
            best = {}

            def g(le):
                ll, x, ne = self._fit_hs(B, lo, hi, np.exp(le), prof[k][1], dw)
                best[le] = (ll, x)
                return -ll
            minimize_scalar(g, bounds=(a, b), method='bounded', options=dict(xatol=0.02))
            le = max(best, key=lambda q: best[q][0])
            (h, s), ell = best[le][1], np.exp(le)
            s, ell = np.array([s]), np.array([ell])
            converged = True
        else:
            k = self.nc
            if start is None:
                start = self.fit(B, lo, hi, dw=dw)
            x_start = np.log(np.r_[start['h'], np.full(k, start['s'][0]), np.full(k, start['ell'][0])])
            bounds = [np.log(H_BOUNDS)] + [np.log(S_BOUNDS)]*k + [np.log(ELL_BOUNDS)]*k

            def f(x):
                e = np.exp(x)
                ll = self.evidence(B, lo, hi, e[0], e[1:1+k], e[1+k:], dw=dw)[0]
                return -ll if np.isfinite(ll) else 1e12
            r = minimize(f, x_start, method='Nelder-Mead', bounds=bounds,
                         options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-3, adaptive=True))
            e = np.exp(r.x)
            h, s, ell, n_eval, converged = e[0], e[1:1+k], e[1+k:], r.nfev, bool(r.success)
        ll, chi2, theta = self.evidence(B, lo, hi, h, s, ell, dw=dw)
        return dict(h=h, s=s, ell=ell, ll=ll, theta=theta, n_eval=n_eval, converged=converged)

    def hyper_sd(self, B, lo, hi, fit, dw=None, step=0.05):
        """standard deviations of log h, log s, log ell (Laplace: the inverse Hessian of -log evidence,
        by central differences in the log parameters, at the fitted values)"""
        k = len(fit['s'])
        x0 = np.log(np.r_[fit['h'], fit['s'], fit['ell']])
        f = lambda x: -self.evidence(B, lo, hi, np.exp(x[0]), np.exp(x[1:1+k]), np.exp(x[1+k:]), dw=dw)[0]
        n = len(x0)
        Hm = np.zeros((n, n))
        f0 = f(x0)
        for i in range(n):
            for j in range(i, n):
                ei, ej = np.eye(n)[i]*step, np.eye(n)[j]*step
                if i == j:
                    Hm[i, i] = (f(x0 + ei) - 2*f0 + f(x0 - ei))/step**2
                else:
                    Hm[i, j] = Hm[j, i] = (f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej))/(4*step**2)
        try:
            return np.sqrt(np.maximum(np.diag(np.linalg.inv(Hm)), 0))
        except np.linalg.LinAlgError:
            return np.full(n, np.nan)

    def posterior(self, B, h, s, ell, dw=None, functionals=None):
        """posterior mean and sd of theta, of v, and of linear functionals c'v (dict name: c)
        (flat prior on theta, no positivity)"""
        return self._posterior(B, self.cov(h, s, ell), self.prior_cov(s, ell), dw, functionals)

    def _posterior(self, B, C, Su, dw=None, functionals=None):
        """the same for any data covariance C and prior covariance Su of u; also returns
        alpha = C^-1 (dw - Gw B theta), so that a part u_k of u (Cov Su_k) has mean Su_k Gw' alpha"""
        dw = self.dw if dw is None else dw
        GB = self.Gw @ B
        theta, Ct = posterior_theta(GB, dw, C)
        cf = cho_factor(C)
        K = Su @ self.Gw.T                                  # n x N
        alpha = cho_solve(cf, dw - GB @ theta)
        v = B @ theta + K @ alpha
        KCi = cho_solve(cf, K.T).T                          # Su G' C^-1
        M = B - KCi @ GB                                    # sensitivity of v to theta
        var = np.diag(Su) - np.sum(KCi*K, 1) + np.sum((M @ Ct)*M, 1)
        fun = {}
        for name, c in (functionals or {}).items():
            Kc, Mc = K.T @ c, M.T @ c
            fun[name] = (float(c @ v), float(np.sqrt(max(c @ Su @ c - Kc @ cho_solve(cf, Kc) + Mc @ Ct @ Mc, 0))))
        return dict(theta=theta, theta_sd=np.sqrt(np.diag(Ct)), theta_cov=Ct, v=v, v_sd=np.sqrt(np.maximum(var, 0)),
                    functionals=fun, alpha=alpha)

    # ---------------------------------------------------------- two scales (one class)
    # u = u1 + u2, independent GMRFs: broad (s1, ell1) and detail (s2, ell2), ell2 < ell1;
    # C = h^2 I + s1^2 Gw R(ell1) Gw' + s2^2 Gw R(ell2) Gw'
    def cov2(self, h, s1, ell1, s2, ell2):
        return h**2*np.eye(len(self.dw)) + s1**2*self.blocks(ell1)[(0, 0)] + s2**2*self.blocks(ell2)[(0, 0)]

    def prior_cov2(self, s1, ell1, s2, ell2):
        return s1**2*self.correlation(ell1).copy() + s2**2*self.correlation(ell2)

    def evidence2(self, B, lo, hi, h, s1, ell1, s2, ell2, dw=None):
        dw = self.dw if dw is None else dw
        return log_evidence(self.Gw @ B, dw, lo, hi, h=1., C=self.cov2(h, s1, ell1, s2, ell2))

    def fit2(self, B, lo, hi, start, ell1_grid=(500., 1000., 2000., 4000.), ell2_grid=(40., 80., 160.), dw=None):
        """maximise the evidence of the two-scale model: (ell1, ell2) on a grid (ell2 < ell1), inner
        Nelder-Mead over log h, log s1, log s2 from a one-scale fit (start). Returns the best and the grid."""
        assert self.nc == 1
        h0, s0 = start['h'], start['s'][0]
        grid = []
        for ell1 in ell1_grid:
            for ell2 in ell2_grid:
                if ell2 >= ell1:
                    continue

                def f(x):
                    ll = self.evidence2(B, lo, hi, np.exp(x[0]), np.exp(x[1]), ell1, np.exp(x[2]), ell2, dw=dw)[0]
                    return -ll if np.isfinite(ll) else 1e12
                r = minimize(f, np.log([h0, 0.9*s0, 0.4*s0]), method='Nelder-Mead',
                             bounds=[np.log(H_BOUNDS), np.log(S_BOUNDS), np.log(S_BOUNDS)],
                             options=dict(xatol=1e-2, fatol=1e-2))
                grid.append((ell1, ell2, -r.fun, *np.exp(r.x)))
        g = np.array(grid)
        ell1, ell2, ll, h, s1, s2 = g[np.argmax(g[:, 2])]
        theta = self.evidence2(B, lo, hi, h, s1, ell1, s2, ell2, dw=dw)[2]
        return dict(h=h, s1=s1, ell1=ell1, s2=s2, ell2=ell2, ll=ll, theta=theta, grid=g)

    def posterior2(self, B, fit, dw=None, functionals=None):
        """posterior of the two-scale model; also the means of the broad part (B theta + u1) and the detail u2"""
        a = (fit['s1'], fit['ell1'], fit['s2'], fit['ell2'])
        p = self._posterior(B, self.cov2(fit['h'], *a), self.prior_cov2(*a), dw, functionals)
        p['v_detail'] = fit['s2']**2*self.correlation(fit['ell2']) @ (self.Gw.T @ p['alpha'])
        p['v_broad'] = p['v'] - p['v_detail']
        return p

    def prior_rows(self, s, ell):
        """rows P^1/2 with P = Cov(u)^-1"""
        ellc = np.broadcast_to(np.asarray(ell, float), (self.nc,))[self.cls]
        Q = precision(self.a, self.Lw, ellc)
        Qi_diag = np.diag(cho_solve(cho_factor(Q), np.eye(len(Q))))
        sc = np.broadcast_to(np.asarray(s, float), (self.nc,))[self.cls]
        return cholesky(Q)*(np.sqrt(Qi_diag)/sc)[None, :]


def map_estimate(Gw, dw, rows, v0, h=1., lo=0., hi=np.inf, v_start=None, GtG=None):
    """min |dw - Gw v|^2/h^2 + |rows (v - v0)|^2 with lo <= v <= hi (GtG = Gw'Gw, if precomputed).
    With sparse rows and fewer than a third as many data as cells, H = rows'rows + Gw'Gw/h^2 is kept as
    sparse plus low rank (exact, much faster for many cells); otherwise it is formed densely."""
    P = rows.T @ rows
    if issparse(rows) and Gw.shape[0] < Gw.shape[1]/3:
        H = SparsePlusLowRank(P, Gw/h)
    else:
        P = P.toarray() if issparse(P) else P
        H = (Gw.T @ Gw if GtG is None else GtG)/h**2 + P
    g = Gw.T @ dw/h**2 + P @ v0
    if np.all(np.isneginf(lo)) and np.all(np.isposinf(hi)):
        return H.solver()(g) if isinstance(H, SparsePlusLowRank) else cho_solve(cho_factor(H), g)
    return box_qp(H, g, lo, hi, v_start)


def tikhonov_rows(a, D, ell, beta, w=None):
    """rows (sparse) of the L2 penalty beta (|diag(sqrt(a))/ell v|^2 + |W^1/2 D v|^2) (TV: IRLS weights w)"""
    Dw = D if w is None else diags(np.sqrt(w)) @ D
    return (np.sqrt(beta)*vstack([diags(np.sqrt(a/np.mean(a))/ell), Dw])).tocsr()


def tikhonov(Gw, dw, a, D, ell, beta, v_ref, p=2, n_irls=15, eps_rel=0.01, v_start=None, GtG=None):
    """L2 (p=2) or TV (p=1, IRLS from the L2 map) with v >= 0; TV weights rms/sqrt((Dv)^2 + eps^2),
    eps = eps_rel x rms, rms = the RMS of D v of the L2 map (so w = 1 where |Dv| = rms)"""
    GtG = Gw.T @ Gw if GtG is None else GtG
    v0 = np.full(Gw.shape[1], v_ref) if np.ndim(v_ref) == 0 else v_ref
    v = map_estimate(Gw, dw, tikhonov_rows(a, D, ell, beta), v0, v_start=v_start, GtG=GtG)
    if p == 2:
        return v
    rms = np.sqrt(np.mean((D @ v)**2))
    eps = eps_rel*rms
    for _ in range(n_irls):
        w = rms/np.sqrt((D @ v)**2 + eps**2)
        v_new = map_estimate(Gw, dw, tikhonov_rows(a, D, ell, beta, w), v0, v_start=v, GtG=GtG)
        done = np.max(np.abs(v_new - v)) < 1e-4
        v = v_new
        if done:
            break
    return v


def discrepancy(solve, Gw, dw, chi2_target, log_beta=(0., 21.), n_bisect=16):
    """beta (bisection on log beta) at which the misfit of solve(beta) equals chi2_target"""
    lo, hi = log_beta
    sol = {}
    for _ in range(n_bisect):
        mid = 0.5*(lo + hi)
        v = solve(np.exp(mid))
        chi2 = float(np.sum((Gw @ v - dw)**2))
        sol[mid] = (v, chi2)
        lo, hi = (mid, hi) if chi2 < chi2_target else (lo, mid)
    best = min(sol, key=lambda k: abs(sol[k][1] - chi2_target))
    return np.exp(best), sol[best][0], sol[best][1]
