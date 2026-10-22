"""
The GMRF prior on cells (CONTEXT.md: CM, CR) and its evidence, on the linear-Gaussian core.

    v = Bd theta + u,    u ~ N(0, S R S),    yw = Gw v + e,    e ~ N(0, h^2 I)

Bd: a design (cells x p: one level, class means, covariate slopes), theta flat or uniform on a box. R: the correlation
matrix of the precision diag(a/ell^2) + L (a the cell areas, mean 1; L the cell-graph Laplacian with weights 1/d^2,
d the great-circle distance between centres, km), ell per cell (the class of the cell); S = diag(s) per class.
With R = L_R L_R' and s = s0 r (r = 1 for class 0) this is the core's model with B = Gw diag(r) L_R, U = Gw Bd,
sigma = h, tau = s0, alpha^2 = h^2/s0^2.
Hyperparameters maximise the evidence: shared (one s and ell) by profiling ell (a grid, then a bounded search) with h
and s0 at their best; per class by Nelder-Mead over log ell_c and log r_c from a shared fit, h and s0 again at their
best inside.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.optimize import minimize, minimize_scalar

from . import core

ELL_GRID = np.geomspace(40., 3000., 12)
ELL_BOUNDS, R_BOUNDS = (20., 5000.), (0.01, 100.)
N_PARTS = 48                                # cached (L, B, eigendecomposition) sets, ~25 MB each for 1400 cells


class GMRF:
    """the GMRF model of one problem: Gw (data x cells), yw, cell areas, Laplacian Lw, cell classes (one if None)"""

    def __init__(self, Gw, yw, area, Lw, cls=None):
        self.Gw, self.yw, self.a, self.Lw = Gw, yw, area/np.mean(area), Lw
        self.cls = np.zeros(Gw.shape[1], int) if cls is None else np.asarray(cls)
        self.nc = self.cls.max() + 1
        self._L = (None, None)
        self._parts = {}                    # (ell, r) -> (L, B, eigendecomposition): shared by every design

    def _per_class(self, x):
        return np.broadcast_to(np.asarray(x, float), (self.nc,))

    def factor(self, ell):
        """L_R, the Cholesky factor of the correlation matrix, for ell (km, one or per class)"""
        ell = self._per_class(ell)
        key = tuple(np.round(ell, 10))
        if self._L[0] != key:
            Qi = cho_solve(cho_factor(np.diag(self.a/ell[self.cls]**2) + self.Lw), np.eye(len(self.a)))
            d = 1/np.sqrt(np.diag(Qi))
            self._L = (key, cholesky(Qi*d[:, None]*d[None, :], lower=True))
        return self._L[1]

    def prior_factor(self, ell, r=1.):
        """diag(r) L_R: the factor of S R S / s0^2"""
        return self._per_class(r)[self.cls][:, None]*self.factor(ell)

    def model(self, Bd, ell, r=1.):
        key = (tuple(np.round(self._per_class(ell), 10)), tuple(np.round(self._per_class(r), 10)))
        if key not in self._parts:
            if len(self._parts) >= N_PARTS:
                self._parts.pop(next(iter(self._parts)))
            L = self.prior_factor(ell, r)
            lr = core.LinearRidge(self.Gw @ L, self.Gw @ Bd, 'eigh', level='marginal')
            self._parts[key] = (L, lr.B, (lr.lam, lr.E))
            return lr, L
        L, B, eig = self._parts[key]
        return core.LinearRidge(B, self.Gw @ Bd, 'eigh', eig=eig, level='marginal'), L

    def evidence(self, Bd, h, s, ell, box=None):
        """log evidence at explicit hyperparameters (s per class)"""
        s = self._per_class(s)
        lr, _ = self.model(Bd, ell, s/s[0])
        alpha2 = h**2/s[0]**2
        ll = lr.log_evidence(self.yw/h, alpha2, False) - (len(self.yw) - lr.p)*np.log(h)     # sigma = h, by scaling
        if box is not None:
            theta = lr.solve(self.yw, alpha2)['theta']
            lo, hi = (np.broadcast_to(np.asarray(b, float), (lr.p,)) for b in box)
            if (theta < lo).any() or (theta > hi).any():
                return -np.inf
            ll -= np.sum(np.log(hi - lo))
        return ll

    def best(self, Bd, ell, r=1., box=None):
        """(log evidence, h, s0, alpha^2, ridge, L) at ell and r, h and s0 at their best"""
        lr, L = self.model(Bd, ell, r)
        alpha2 = lr.by_evidence(self.yw, fit_sigma=True)
        h = lr.sigma(lr.solve(self.yw, alpha2), True)
        return lr.log_evidence(self.yw, alpha2, True, box), h, h/np.sqrt(alpha2), alpha2, lr, L

    def fit(self, Bd, box=None, length='evidence'):
        """shared hyperparameters: dict(h, s (per class), ell (per class), ll, alpha2, ridge, L)"""
        def out(ell, b):
            ll, h, s0, alpha2, lr, L = b
            return dict(h=h, s=self._per_class(s0).copy(), ell=self._per_class(ell).copy(), ll=ll, alpha2=alpha2,
                        ridge=lr, L=L)
        if length != 'evidence':
            return out(float(length), self.best(Bd, float(length), box=box))
        prof = [self.best(Bd, e, box=box)[0] for e in ELL_GRID]
        k = int(np.argmax(prof))
        lo, hi = np.log(ELL_GRID[max(k - 1, 0)]), np.log(ELL_GRID[min(k + 1, len(ELL_GRID) - 1)])
        best = {}

        def f(le):
            best[le] = self.best(Bd, np.exp(le), box=box)
            return -best[le][0]
        minimize_scalar(f, bounds=(lo, hi), method='bounded', options=dict(xatol=0.02))
        le = max(best, key=lambda q: best[q][0])
        return out(np.exp(le), best[le])

    def fit_per_class(self, Bd, box=None, start=None, maxiter=2000):
        """s and ell per class: Nelder-Mead over log ell_c and log(s_c/s_0) from a shared fit"""
        start = start or self.fit(Bd, box)
        k = self.nc
        x0 = np.log(np.r_[np.broadcast_to(np.asarray(start['ell'], float)[:k], (k,)), np.ones(k - 1)])
        bounds = [np.log(ELL_BOUNDS)]*k + [np.log(R_BOUNDS)]*(k - 1)
        memo = {}

        def f(x):
            b = self.best(Bd, np.exp(x[:k]), np.r_[1., np.exp(x[k:])], box)
            memo[tuple(x)] = b
            return -b[0] if np.isfinite(b[0]) else 1e12
        res = minimize(f, x0, method='Nelder-Mead', bounds=bounds,
                       options=dict(maxiter=maxiter, xatol=1e-3, fatol=1e-3, adaptive=True))
        ll, h, s0, alpha2, lr, L = memo.get(tuple(res.x)) or self.best(Bd, np.exp(res.x[:k]), np.r_[1., np.exp(res.x[k:])], box)
        return dict(h=h, s=s0*np.r_[1., np.exp(res.x[k:])], ell=np.exp(res.x[:k]), ll=ll, alpha2=alpha2, ridge=lr, L=L,
                    n_eval=res.nfev, converged=bool(res.success))

    def hyper_sd(self, Bd, fit, box=None, step=0.05):
        """s.d. of log h, log s_c, log ell_c (Laplace: the inverse Hessian of -log evidence by central differences)"""
        k = len(fit['s'])
        x0 = np.log(np.r_[fit['h'], fit['s'], fit['ell']])
        f = lambda x: -self.evidence(Bd, np.exp(x[0]), np.exp(x[1:1 + k]), np.exp(x[1 + k:]), box)
        n = len(x0)
        H = np.zeros((n, n))
        f0 = f(x0)
        for i in range(n):
            for j in range(i, n):
                ei, ej = np.eye(n)[i]*step, np.eye(n)[j]*step
                if i == j:
                    H[i, i] = (f(x0 + ei) - 2*f0 + f(x0 - ei))/step**2
                else:
                    H[i, j] = H[j, i] = (f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej))/(4*step**2)
        try:
            return np.sqrt(np.maximum(np.diag(np.linalg.inv(H)), 0))
        except np.linalg.LinAlgError:
            return np.full(n, np.nan)

    def posterior(self, Bd, fit, functionals=None):
        """posterior mean and s.d. of theta, of v and of linear functionals c'v (dict name: c); no bounds"""
        lr, L, alpha2, h = fit['ridge'], fit['L'], fit['alpha2'], fit['h']
        sol = lr.solve(self.yw, alpha2)
        Bd = np.asarray(Bd, float).reshape(len(L), -1)
        v = Bd @ sol['theta'] + L @ sol['x']
        cov_theta = h**2*np.linalg.inv(sol['P'])
        fun = {}
        for name, c in (functionals or {}).items():
            sd = lr.posterior_sd_map(alpha2, h, (c @ L)[None, :], (c @ Bd)[None, :])[0]
            fun[name] = (float(c @ v), float(sd))
        return dict(theta=sol['theta'], theta_sd=np.sqrt(np.diag(cov_theta)), theta_cov=cov_theta, v=v,
                    v_sd=lr.posterior_sd_map(alpha2, h, L, Bd), functionals=fun, chi2=sol['chi2'])
