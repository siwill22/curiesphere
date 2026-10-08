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

The level in the evidence: 'profile' takes v0 at its estimate (GCM); 'marginal' integrates it out under a flat prior,
which adds -1/2 log(precision of v0) and leaves N - 1 degrees of freedom for sigma (the cell maps' GMRF, as the
first Australian build). The posterior s.d. of any linear map v = v0 + M x includes the level's uncertainty.

Bounded MAP (positivity or any box): min |y - A v|^2 + |rows (v - v_ref)|^2 with lo <= v <= hi, by projected Newton;
with sparse rows and few data the Hessian is kept as sparse plus low rank.
"""
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.optimize import minimize_scalar
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import splu

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

    def __init__(self, B, u, route='eigh', products='numpy', eig=None, level='profile'):
        if level not in ('profile', 'marginal'):
            raise ValueError(f'level {level!r}')
        self.B, self.u, self.level = B, u, level
        self.N, self.K = B.shape
        self.side = 'data' if self.N <= self.K else 'cells'
        self.route = route
        self.floor = ALPHA2_FLOOR['mps' if products == 'mps' else 'numpy']
        if route == 'eigh':
            self.lam, self.E = eig if eig is not None else self._decompose(products)
            self.lam_max = self.lam.max()
            if self.side == 'data':
                self.c = self.E.T @ u                       # u along the data-space eigenvectors
                self.u_res = u - self.E @ self.c            # and outside them (computed directly: no cancellation)
                self.q = np.sqrt(self.lam)*self.c           # = Q'B'u
            else:
                self.q = self.E.T @ (B.T @ u)
            self._Q = None
        elif route == 'cholesky':
            self.C = gram(B, self.side, products)
            self.lam_max = _largest_eigenvalue(self.C)
            self._f = {}
        else:
            raise ValueError(f'route {route!r}')

    @property
    def Q(self):
        """orthonormal cell-space eigenvectors (cells x r): E itself on the cell side, B'E/sqrt(lam) on the data side"""
        if self._Q is None:
            self._Q = self.E if self.side == 'cells' else (self.B.T @ self.E)/np.sqrt(self.lam)
        return self._Q

    def _decompose(self, products):
        """lam > 0 and E, the orthonormal eigenvectors of BB' (data side) or B'B (cell side) with those eigenvalues;
        the other directions have lam = 0"""
        C = gram(self.B, self.side, products)
        lam, V = eigh(C, overwrite_a=True, check_finite=False)
        keep = lam > lam.max()*1e-13
        return lam[keep], V[:, keep]

    # ---------------------------------------------------------------- solutions
    def solve(self, y, alpha2):
        """MAP for whitened data y: x, v0, chi2 (|r|^2), J (|r|^2 + alpha^2 |x|^2), precision of v0 (sigma = 1)"""
        # the level's precision u'(I + BB'/alpha^2)^-1 u and its right-hand side. On the data side every weight is at
        # most 1 and the part of u outside the eigenvectors is computed directly, so nothing cancels or is amplified
        # (u is often nearly in the range of B, where u'u - sum q^2/(lam + alpha^2) loses all its digits).
        if self.route == 'eigh':
            den = self.lam + alpha2
            if self.side == 'data':
                py = self.E.T @ y
                g = alpha2/den
                prec = self.u_res @ self.u_res + np.sum(self.c**2*g)
                v0 = (self.u_res @ y + np.sum(self.c*py*g))/prec
                x = self.B.T @ (self.E @ ((py - v0*self.c)/den))
            else:
                p = self.Q.T @ (self.B.T @ y)
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
                Hy, Hu = cho_solve(f, self.B.T @ y), cho_solve(f, self.B.T @ self.u)
                up = self.u - self.B @ Hu
                prec = up @ up + alpha2*(Hu @ Hu)
                v0 = (up @ y)/prec
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

    def _dof(self):
        return self.N - (self.level == 'marginal')

    def log_evidence(self, y, alpha2, fit_sigma=True):
        """log marginal likelihood up to a constant: v0 at its estimate ('profile') or integrated out ('marginal');
        sigma at its maximum (J/dof) or 1"""
        sol = self.solve(y, alpha2)
        J, n = sol['J'], self._dof()
        extra = np.log(sol['prec_v0']) if self.level == 'marginal' else 0.
        if fit_sigma:
            return -0.5*(n*np.log(J/n) + self.logdet(alpha2) + extra)
        return -0.5*(J + self.logdet(alpha2) + extra)

    def sigma(self, sol, fit_sigma):
        return float(np.sqrt(sol['J']/self._dof())) if fit_sigma else 1.

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

    def posterior_sd_map(self, alpha2, sigma, M):
        """posterior s.d. of v = v0 + M x (M: n x K), v0 integrated out"""
        if self.route == 'eigh':
            den = self.lam + alpha2
            MQ = M @ self.Q
            c = self.Q @ (self.q/den)
            var = np.sum(MQ**2/den, 1) + (np.sum(M**2, 1) - np.sum(MQ**2, 1))/alpha2
        else:
            Ai = (cho_solve(self._factor(alpha2), np.eye(self.K)) if self.side == 'cells' else
                  (np.eye(self.K) - self.B.T @ cho_solve(self._factor(alpha2), self.B))/alpha2)
            c = Ai @ (self.B.T @ self.u)
            var = np.sum((M @ Ai)*M, 1)
        prec = self.solve(np.zeros(self.N), alpha2)['prec_v0']
        return sigma*np.sqrt(np.maximum(var + (1 - M @ c)**2/prec, 0.))

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


# -------------------------------------------------------------------- bounded MAP
class SparsePlusLowRank:
    """H = P + A'A with P sparse and positive definite (n x n) and A dense (k x n), k << n: products, the diagonal and
    solves on any subset F of the variables (Woodbury: a sparse factorisation of P_FF and a k x k system)"""

    def __init__(self, P, A):
        self.P, self.A = csc_matrix(P), np.asarray(A)
        self.diag = self.P.diagonal() + np.einsum('ij,ij->j', self.A, self.A)

    def __matmul__(self, x):
        return self.P @ x + self.A.T @ (self.A @ x)

    def solver(self, F=None):
        """x -> H_FF^-1 x"""
        P, A = (self.P, self.A) if F is None else (self.P[F][:, F].tocsc(), self.A[:, F])
        lu = splu(P)
        Z = lu.solve(np.asfortranarray(A.T))
        C = cho_factor(np.eye(A.shape[0]) + A @ Z)
        return lambda r: (lambda y: y - Z @ cho_solve(C, A @ y))(lu.solve(r))


def box_qp(H, g, lo=0., hi=np.inf, v=None, tol=1e-8, max_iter=1000):
    """min 1/2 v'Hv - g'v with lo <= v <= hi (H positive definite): projected Newton (Bertsekas 1982).
    Variables within eps of a bound with the gradient pushing outward are held (eps = the size of the projected-
    gradient step); the rest take a Newton step (exact Cholesky on the free block), the held ones a diagonally scaled
    gradient step; Armijo backtracking along the projection arc, with the decrease computed from the step (a
    difference of two large objective values loses the last digits). Stops when the projected gradient (KKT
    residual) is below tol x max|g|. H: a dense matrix or a SparsePlusLowRank."""
    n = len(g)
    lo, hi = np.broadcast_to(lo, (n,)).astype(float), np.broadcast_to(hi, (n,)).astype(float)
    structured = isinstance(H, SparsePlusLowRank)
    dH = H.diag.copy() if structured else np.diag(H).copy()
    decrease = lambda s, grad: -(grad @ s) - 0.5*s @ (H @ s)
    if v is None:
        v = H.solver()(g) if structured else np.linalg.solve(H, g)
    v = np.clip(v, lo, hi)
    scale = tol*max(np.max(np.abs(g)), 1e-300)
    converged = False
    for it in range(max_iter):
        grad = H @ v - g
        pgrad = v - np.clip(v - grad/dH, lo, hi)
        kkt = np.max(np.abs(np.clip(v - grad, lo, hi) - v))
        if kkt <= scale:
            converged = True
            break
        eps = min(1e-3*(1 + np.max(np.abs(v))), np.max(np.abs(pgrad)))
        held = ((v <= lo + eps) & (grad > 0)) | ((v >= hi - eps) & (grad < 0))
        F = ~held
        d = np.where(held, -grad/dH, 0.)
        if F.any():
            d[F] = -(H.solver(F)(grad[F]) if structured else cho_solve(cho_factor(H[np.ix_(F, F)]), grad[F]))
        t = 1.
        while True:
            w = np.clip(v + t*d, lo, hi)
            r = decrease(w - v, grad)
            if r >= 1e-4*(-grad @ (w - v)) and r >= 0:
                break
            t *= 0.5
            if t < 1e-12:
                break
        if t < 1e-12:
            w = np.clip(v - grad/dH, lo, hi)
            if decrease(w - v, grad) < 0:
                break
        v = w
    if not converged:
        warnings.warn(f'box_qp: not converged after {it + 1} iterations (KKT residual '
                      f'{np.max(np.abs(np.clip(v - (H @ v - g), lo, hi) - v)):.3g}, tolerance {scale:.3g})')
    return v


def bounded_map(A, y, rows, v_ref, lo=0., hi=np.inf, v_start=None, AtA=None):
    """min |y - A v|^2 + |rows (v - v_ref)|^2 with lo <= v <= hi (AtA = A'A, if precomputed). With sparse rows and
    fewer than a third as many data as cells, H = rows'rows + A'A is kept as sparse plus low rank (exact)"""
    P = rows.T @ rows
    if issparse(rows) and A.shape[0] < A.shape[1]/3:
        H = SparsePlusLowRank(P, A)
    else:
        P = P.toarray() if issparse(P) else P
        H = (A.T @ A if AtA is None else AtA) + P
    g = A.T @ y + P @ v_ref
    if np.all(np.isneginf(lo)) and np.all(np.isposinf(hi)):
        return H.solver()(g) if isinstance(H, SparsePlusLowRank) else cho_solve(cho_factor(H), g)
    return box_qp(H, g, lo, hi, v_start)
