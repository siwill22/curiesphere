"""
The linear-Gaussian core, for the damped fits with a free uniform level (GCM and its kin):

    y = v0 u + B x + e,    x ~ N(0, tau^2 I),    e ~ N(0, sigma^2 I),    alpha^2 = sigma^2 / tau^2

y and B are whitened by the fixed part of the error model (unit noise when sigma = 1), x = sqrt(a)(m - v0) are the
area-scaled departures of the cell values m from the level v0, and v0 (the column u) is undamped, estimated in
closed form. The MAP x minimises |y - v0 u - B x|^2 + alpha^2 |x|^2.

Two routes give the same answers (ADR 0006):
  'eigh'      one eigendecomposition of the smaller of B'B and BB' (float64, CPU), after which any alpha costs
              matrix-vector products: the whole evidence curve and the posterior s.d. of every cell
  'cholesky'  a Cholesky factorisation of BB' + alpha^2 I (data space) or B'B + alpha^2 I (cell space) per alpha
B'B or BB' may be formed on the GPU in float32 (ADR 0005); the factorisations are always float64 on the CPU.

Damping choices: 'evidence' (largest marginal likelihood, sigma fitted or fixed), 'discrepancy' (chi^2 = N, sigma
fixed), or a number (alpha^2).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import minimize_scalar

ALPHA2_FLOOR = {'numpy': 1e-10,    # alpha^2 is searched above this x the largest eigenvalue: above the rounding
                'mps': 1e-4}       # of B'B or BB' (float64; float32 on the GPU, ~1e-5 relative)


def _largest_eigenvalue(C, n=50):
    """by power iteration (only the scale of the alpha search depends on it)"""
    v = np.random.default_rng(0).standard_normal(len(C))
    for _ in range(n):
        v = C @ v
        v /= np.linalg.norm(v)
    return float(v @ C @ v)


def gram(B, side, backend='numpy'):
    """B'B (side 'cells') or BB' (side 'data'), float64; 'mps' forms it in float32 on the GPU"""
    if backend == 'mps':
        import torch
        if torch.backends.mps.is_available():
            Bt = torch.as_tensor(B, dtype=torch.float32, device='mps')
            C = Bt.T @ Bt if side == 'cells' else Bt @ Bt.T
            torch.mps.synchronize()
            return C.cpu().double().numpy()
    return B.T @ B if side == 'cells' else B @ B.T


class LevelRidge:
    """the model above for one B and u; route 'eigh' or 'cholesky'"""

    def __init__(self, B, u, route='eigh', products='numpy', eig=None):
        self.B, self.u = B, u
        self.N, self.K = B.shape
        self.side = 'data' if self.N <= self.K else 'cells'
        self.route = route
        self.floor = ALPHA2_FLOOR['mps' if products == 'mps' else 'numpy']
        if route == 'eigh':
            self.lam, self.Q = eig if eig is not None else self._decompose(products)
            self.q = self.Q.T @ (B.T @ u)
            self.lam_max = self.lam.max()
        elif route == 'cholesky':
            self.C = gram(B, self.side, products)
            self.lam_max = _largest_eigenvalue(self.C)
            self._f = {}
        else:
            raise ValueError(f'route {route!r}')

    def _decompose(self, products):
        """lam > 0 and Q (cells x r, orthonormal) spanning the row space of B; the other directions have lam = 0"""
        C = gram(self.B, self.side, products)
        lam, V = eigh(C, overwrite_a=True, check_finite=False)
        keep = lam > lam.max()*1e-13
        if self.side == 'data':
            return lam[keep], (self.B.T @ V[:, keep])/np.sqrt(lam[keep])
        return lam[keep], V[:, keep]

    # ---------------------------------------------------------------- solutions
    def solve(self, y, alpha2):
        """MAP for whitened data y: x, v0, chi2 (|r|^2), J (|r|^2 + alpha^2 |x|^2), precision of v0 (sigma = 1)"""
        if self.route == 'eigh':
            p = self.Q.T @ (self.B.T @ y)
            den = self.lam + alpha2
            prec = self.u @ self.u - np.sum(self.q**2/den)
            v0 = (self.u @ y - np.sum(p*self.q/den))/prec
            x = self.Q @ ((p - v0*self.q)/den)
        else:
            f = self._factor(alpha2)
            if self.side == 'data':
                Ciy, Ciu = cho_solve(f, y), cho_solve(f, self.u)
                v0 = (self.u @ Ciy)/(self.u @ Ciu)
                prec = alpha2*(self.u @ Ciu)
                x = self.B.T @ (Ciy - v0*Ciu)
            else:
                Bty, Btu = self.B.T @ y, self.B.T @ self.u
                Hy, Hu = cho_solve(f, Bty), cho_solve(f, Btu)
                prec = self.u @ self.u - Btu @ Hu
                v0 = (self.u @ y - Btu @ Hy)/prec
                x = Hy - v0*Hu
        r = y - v0*self.u - self.B @ x
        chi2 = float(r @ r)
        return dict(x=x, v0=float(v0), chi2=chi2, J=chi2 + alpha2*float(x @ x), prec_v0=float(prec), r=r)

    def _factor(self, alpha2):
        if alpha2 not in self._f:
            self._f = {alpha2: cho_factor(self.C + alpha2*np.eye(len(self.C)), check_finite=False)}
        return self._f[alpha2]

    def logdet(self, alpha2):
        """log det(I + BB'/alpha^2) = log det(I + B'B/alpha^2)"""
        if self.route == 'eigh':
            return float(np.sum(np.log1p(self.lam/alpha2)))
        f = self._factor(alpha2)
        return float(2*np.sum(np.log(np.diag(f[0]))) - len(self.C)*np.log(alpha2))

    def log_evidence(self, y, alpha2, fit_sigma=True):
        """log marginal likelihood up to a constant, v0 at its estimate; sigma at its maximum (J/N) or 1"""
        J = self.solve(y, alpha2)['J']
        if fit_sigma:
            return -0.5*(self.N*np.log(J/self.N) + self.logdet(alpha2))
        return -0.5*(J + self.logdet(alpha2))

    def sigma(self, sol, fit_sigma):
        return float(np.sqrt(sol['J']/self.N)) if fit_sigma else 1.

    def posterior_sd_x(self, alpha2, sigma=1., chunk=2000):
        """posterior s.d. of each x_k given v0"""
        if self.route == 'eigh':
            s = self.lam/(alpha2*(self.lam + alpha2))
            var = np.concatenate([1/alpha2 - (self.Q[k:k + chunk]**2) @ s for k in range(0, self.K, chunk)])
        elif self.side == 'cells':
            var = np.diag(cho_solve(self._factor(alpha2), np.eye(self.K)))
        else:
            raise NotImplementedError('posterior s.d. on the data-space Cholesky route: use route "eigh"')
        return sigma*np.sqrt(np.maximum(var, 0.))

    # ---------------------------------------------------------------- choosing alpha
    def bounds(self):
        return np.log10(self.lam_max*self.floor), np.log10(self.lam_max) + 4

    def by_target(self, y, key, target, n=50):
        """alpha^2 at which solve(y)[key] reaches target (increasing with alpha), bisection in log alpha^2"""
        lo, hi = self.bounds()
        for _ in range(n if self.route == 'eigh' else 20):
            mid = 0.5*(lo + hi)
            lo, hi = (mid, hi) if key(self.solve(y, 10**mid)) < target else (lo, mid)
        return 10**(0.5*(lo + hi))

    def by_evidence(self, y, fit_sigma=True, n_grid=29):
        """alpha^2 of the largest evidence: a grid in log alpha^2, then a bounded search around its best point"""
        lo, hi = self.bounds()
        grid = np.linspace(lo, hi, n_grid if self.route == 'eigh' else 9)
        k = int(np.argmax([self.log_evidence(y, 10**g, fit_sigma) for g in grid]))
        a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        res = minimize_scalar(lambda g: -self.log_evidence(y, 10**g, fit_sigma), bounds=(a, b), method='bounded',
                              options=dict(xatol=1e-3 if self.route == 'eigh' else 2e-2))
        return 10**res.x
