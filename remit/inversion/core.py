"""
The linear-Gaussian core, for damped fits with undamped design terms (GCM, CM, CR):

    y = U theta + B x + e,    x ~ N(0, tau^2 I),    e ~ N(0, sigma^2 I),    alpha^2 = sigma^2 / tau^2

y and B are whitened by the fixed part of the error model (unit noise when sigma = 1); x are the damped departures
(GCM: x = sqrt(a)(m - v0), area-scaled; CM and CR: the GMRF in its Cholesky coordinates); theta are undamped
(flat prior), estimated in closed form: one level v0 (U = u, GCM and CM) or a design (class means and covariate
slopes, CR). The MAP minimises |y - U theta - B x|^2 + alpha^2 |x|^2.

Two routes give the same answers (ADR 0006):
  'eigh'      one eigendecomposition of the smaller of B'B and BB' (float64, CPU), after which any alpha costs
              matrix-vector products: the whole evidence curve and the posterior s.d. of every cell
  'cholesky'  a Cholesky factorisation of BB' + alpha^2 I (data space) or B'B + alpha^2 I (cell space) per alpha
B'B or BB' may be formed on the GPU in float32 (ADR 0005); the factorisations are always float64 on the CPU.

Damping choices: 'evidence' (largest marginal likelihood, sigma fitted or fixed), 'discrepancy' (chi^2 = N, sigma
fixed), or a number (alpha^2).

theta in the evidence: 'profile' takes it at its estimate (GCM); 'marginal' integrates it out under a flat prior (or
uniform on a box), which adds -1/2 log det(precision of theta) and leaves N - p degrees of freedom for sigma (CM,
CR, as the first Australian build). The posterior s.d. of any linear map v = D theta + M x includes theta's
uncertainty.

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


class LinearRidge:
    """y = U theta + B x + e for one B and design U (N x p, or a single column u); route 'eigh' or 'cholesky'.
    theta is undamped (flat prior); with p = 1 the solution also carries v0 and prec_v0 (the level of GCM and CM)."""

    def __init__(self, B, u, route='eigh', products='numpy', eig=None, level='profile'):
        if level not in ('profile', 'marginal'):
            raise ValueError(f'level {level!r}')
        self.B, self.level = B, level
        self.U = np.asarray(u, float).reshape(len(B), -1)
        self.p = self.U.shape[1]
        self.N, self.K = B.shape
        self.side = 'data' if self.N <= self.K else 'cells'
        self.route = route
        self.floor = ALPHA2_FLOOR['mps' if products == 'mps' else 'numpy']
        if route == 'eigh':
            self.lam, self.E = eig if eig is not None else self._decompose(products)
            self.lam_max = self.lam.max()
            if self.side == 'data':
                self.c = self.E.T @ self.U                  # U along the data-space eigenvectors (r x p)
                self.U_res = self.U - self.E @ self.c       # and outside them (computed directly: no cancellation)
            else:
                self.q = self.E.T @ (B.T @ self.U)          # Q'B'U (r x p)
            self._Q = None
        elif route == 'cholesky':
            self.C = gram(B, self.side, products)
            self.lam_max = _largest_eigenvalue(self.C)
            self._f = {}
        else:
            raise ValueError(f'route {route!r}')

    @property
    def u(self):
        return self.U[:, 0] if self.p == 1 else self.U

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
    def _design(self, alpha2):
        """P = U'(I + BB'/alpha^2)^-1 U (p x p, the precision of theta for sigma = 1) and the map y -> U'(...)^-1 y.
        On the data side every weight is at most 1 and the part of U outside the eigenvectors is computed directly,
        so nothing cancels or is amplified (U is often nearly in the range of B, where U'U - U'B(B'B + alpha^2)^-1 B'U
        loses all its digits)."""
        if self.route == 'eigh':
            den = self.lam + alpha2
            if self.side == 'data':
                g = alpha2/den
                P = self.U_res.T @ self.U_res + self.c.T @ (g[:, None]*self.c)
                rhs = lambda y: self.U_res.T @ y + self.c.T @ (g*(self.E.T @ y))
            else:
                P = self.U.T @ self.U - self.q.T @ (self.q/den[:, None])
                rhs = lambda y: self.U.T @ y - self.q.T @ ((self.Q.T @ (self.B.T @ y))/den)
        else:
            f = self._factor(alpha2)
            if self.side == 'data':
                CiU = cho_solve(f, self.U)
                P = alpha2*(self.U.T @ CiU)
                rhs = lambda y: alpha2*(CiU.T @ y)
            else:
                HU = cho_solve(f, self.B.T @ self.U)
                Up = self.U - self.B @ HU
                P = Up.T @ Up + alpha2*(HU.T @ HU)
                rhs = lambda y: Up.T @ y
        return 0.5*(P + P.T), rhs

    def _x(self, z, alpha2):
        """(B'B + alpha^2 I)^-1 B'z"""
        if self.route == 'eigh':
            den = self.lam + alpha2
            if self.side == 'data':
                return self.B.T @ (self.E @ ((self.E.T @ z)/den))
            return self.Q @ ((self.Q.T @ (self.B.T @ z))/den)
        f = self._factor(alpha2)
        return self.B.T @ cho_solve(f, z) if self.side == 'data' else cho_solve(f, self.B.T @ z)

    def solve(self, y, alpha2):
        """MAP for whitened data y: x, theta, chi2 (|r|^2), J (|r|^2 + alpha^2 |x|^2), P (precision of theta, sigma = 1);
        for p = 1 also v0 and prec_v0"""
        P, rhs = self._design(alpha2)
        theta = np.linalg.solve(P, rhs(y))
        x = self._x(y - self.U @ theta, alpha2)
        r = y - self.U @ theta - self.B @ x
        chi2 = float(r @ r)
        out = dict(x=x, theta=theta, chi2=chi2, J=chi2 + alpha2*float(x @ x), P=P, r=r)
        if self.p == 1:
            out.update(v0=float(theta[0]), prec_v0=float(P[0, 0]))
        return out

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
        return self.N - (self.p if self.level == 'marginal' else 0)

    def log_evidence(self, y, alpha2, fit_sigma=True, box=None):
        """log marginal likelihood. 'profile': theta at its estimate, up to a constant (GCM). 'marginal': theta
        integrated out under a flat prior, complete with its constants, so models with different designs compare
        directly; box = (lo, hi) per column makes the prior uniform on that box (-sum log(hi - lo); -inf if the
        estimate leaves it). sigma at its maximum (J/dof) or 1."""
        sol = self.solve(y, alpha2)
        J, n = sol['J'], self._dof()
        if self.level == 'profile':
            if fit_sigma:
                return -0.5*(n*np.log(J/n) + self.logdet(alpha2))
            return -0.5*(J + self.logdet(alpha2))
        ll = -0.5*(self.logdet(alpha2) + np.linalg.slogdet(sol['P'])[1])
        ll += -0.5*n*(np.log(2*np.pi*J/n) + 1) if fit_sigma else -0.5*(n*np.log(2*np.pi) + J)
        if box is not None:
            lo, hi = (np.broadcast_to(np.asarray(b, float), (self.p,)) for b in box)
            if (sol['theta'] < lo).any() or (sol['theta'] > hi).any():
                return -np.inf
            ll -= np.sum(np.log(hi - lo))
        return float(ll)

    def sigma(self, sol, fit_sigma):
        return float(np.sqrt(sol['J']/self._dof())) if fit_sigma else 1.

    def posterior_sd_x(self, alpha2, sigma=1., chunk=2000):
        """posterior s.d. of each x_k given theta"""
        if self.route == 'eigh':
            s = self.lam/(alpha2*(self.lam + alpha2))
            var = np.concatenate([1/alpha2 - (self.Q[k:k + chunk]**2) @ s for k in range(0, self.K, chunk)])
        elif self.side == 'cells':
            var = np.diag(cho_solve(self._factor(alpha2), np.eye(self.K)))
        else:
            raise NotImplementedError('posterior s.d. on the data-space Cholesky route: use route "eigh"')
        return sigma*np.sqrt(np.maximum(var, 0.))

    def posterior_sd_map(self, alpha2, sigma, M, D=None):
        """posterior s.d. of v = D theta + M x (M: n x K; D: n x p, default ones), theta integrated out"""
        D = np.ones((len(M), self.p)) if D is None else np.asarray(D, float).reshape(len(M), -1)
        if self.route == 'eigh' and self.side == 'data':
            den = self.lam + alpha2
            MBE = (M @ self.B.T) @ self.E                    # (B'B + alpha^2)^-1 = (I - B'E diag(1/den) E'B)/alpha^2
            var = (np.sum(M**2, 1) - np.sum(MBE**2/den, 1))/alpha2
            CU = self.B.T @ (self.E @ (self.c/den[:, None]))  # (B'B + alpha^2)^-1 B'U
        elif self.route == 'eigh':
            den = self.lam + alpha2
            MQ = M @ self.Q
            var = np.sum(MQ**2/den, 1)
            CU = self.Q @ (self.q/den[:, None])
        else:
            Ai = (cho_solve(self._factor(alpha2), np.eye(self.K)) if self.side == 'cells' else
                  (np.eye(self.K) - self.B.T @ cho_solve(self._factor(alpha2), self.B))/alpha2)
            var = np.sum((M @ Ai)*M, 1)
            CU = Ai @ (self.B.T @ self.U)
        P = self._design(alpha2)[0]
        S = D - M @ CU                                         # sensitivity of v to theta
        return sigma*np.sqrt(np.maximum(var + np.sum(S*np.linalg.solve(P, S.T).T, 1), 0.))

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
        """alpha^2 of the largest evidence: a grid in log alpha^2, then a bounded search around its best point
        (the box of the prior on theta does not depend on alpha, so it is left out here)"""
        lo, hi = self.bounds()
        grid = np.linspace(lo, hi, n_grid if self.route == 'eigh' else 9)
        k = int(np.argmax([self.log_evidence(y, 10**g, fit_sigma) for g in grid]))
        a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        res = minimize_scalar(lambda g: -self.log_evidence(y, 10**g, fit_sigma), bounds=(a, b), method='bounded',
                              options=dict(xatol=1e-3 if self.route == 'eigh' else 2e-2))
        return 10**res.x


LevelRidge = LinearRidge                     # the one-column case (GCM, CM)


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


def regression_evidence(A, y, lo, hi):
    """log evidence of y = A theta + e alone (no damped part): theta uniform on the box [lo, hi], e ~ N(0, h^2 I) with
    h integrated out under p(h) ~ 1/h:
        -sum log(hi - lo) - 1/2 log|A'A| + log Gamma((N-K)/2) - (N-K)/2 log(pi chi2)
    -inf if the least-squares theta leaves the box (or A'A is singular)"""
    from scipy.special import gammaln
    N, K = A.shape
    M = A.T @ A
    try:
        cf = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return -np.inf
    theta = np.linalg.solve(M, A.T @ y)
    lo, hi = (np.broadcast_to(np.asarray(b, float), (K,)) for b in (lo, hi))
    if (theta < lo).any() or (theta > hi).any():
        return -np.inf
    chi2 = float(np.sum((A @ theta - y)**2))
    return float(-np.sum(np.log(hi - lo)) - np.sum(np.log(np.diag(cf))) + gammaln((N - K)/2)
                 - (N - K)/2*np.log(np.pi*chi2))


# -------------------------------------------------------------------- extremal bounds
def extremes(A, y, c, chi2_target, lo=0., hi=4., D=None, rough_max=None, solver='CLARABEL'):
    """the smallest and largest c'v over all models with |A v - y|^2 <= chi2_target and lo <= v <= hi (and |D v| <=
    rough_max if given): a second-order cone programme, solved exactly by an interior-point method (cvxpy, Clarabel;
    SCS if that fails). With more data than cells the misfit is first reduced exactly by a QR factorisation,
    |A v - y|^2 = |R v - Q'y|^2 + |y - QQ'y|^2. Returns {'min': ..., 'max': ...}, each value, v, chi2, status, rough."""
    import cvxpy as cp
    v = cp.Variable(A.shape[1])
    M, b, target = A, y, chi2_target
    if A.shape[0] > A.shape[1]:
        Qm, M = np.linalg.qr(A)
        b = Qm.T @ y
        target = chi2_target - float(np.sum((y - Qm @ b)**2))
        if target <= 0:
            raise ValueError(f'bounds: no model reaches chi2 {chi2_target:.0f} (the part of the data outside the span '
                             f'of A alone exceeds it)')
    cons = [cp.norm(M @ v - b) <= np.sqrt(target), v >= lo, v <= hi]
    if rough_max is not None:
        cons.append(cp.norm(D @ v) <= rough_max)
    out = {}
    for name, sense in (('min', cp.Minimize), ('max', cp.Maximize)):
        prob = cp.Problem(sense(c @ v), cons)
        try:
            prob.solve(solver=solver)
        except cp.error.SolverError:
            prob.solve(solver='SCS', eps=1e-7, max_iters=200000)
        if prob.status not in ('optimal', 'optimal_inaccurate'):
            raise ValueError(f'bounds: {prob.status} (no model within the bounds reaches chi2 {chi2_target:.0f}?)')
        x = np.clip(v.value, lo, hi)
        out[name] = dict(value=float(c @ x), v=x, chi2=float(np.sum((A @ x - y)**2)), status=prob.status,
                         rough=float(np.linalg.norm(D @ x)) if D is not None else None)
    return out
