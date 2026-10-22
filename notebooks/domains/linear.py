"""
Linear models with the parameters integrated out analytically: the evidence shared by the domain
samplers (voronoi, lvoronoi, partition), the covariate regressions, the hypothesis comparison and the
random-field models (gmrf).

Model: dw = GR theta + n, n ~ N(0, h^2 C) (C = I unless given), theta uniform on the box [lo, hi]
(one interval per column, or the same for all). The Gaussian integral is taken over all of R^K and a
model whose least-squares theta leaves the box gets zero mass, so
    log p(dw | model, h) = -sum log(hi - lo) - 1/2 log|GR'GR| - (N-K)/2 log(2 pi h^2) - chi2/(2 h^2)
                           - 1/2 log|C|   (GR, dw and chi2 in the space whitened by C)
and with h unknown (Jeffreys, p(h) ~ 1/h) integrated out,
    log p = -sum log(hi - lo) - 1/2 log|GR'GR| + log Gamma((N-K)/2) - (N-K)/2 log(pi chi2) - 1/2 log|C|
Constants in K are dropped consistently, so evidences of models with different K compare directly.
"""
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.special import gammaln

V_LO, V_HI = 0., 4.              # VIS bounds (SI.km) of the domain values


def whiten(C, *arrays):
    """Cholesky factor L of C (lower) and L^-1 applied to each array; also 1/2 log|C|"""
    L = np.linalg.cholesky(C)
    return L, [solve_triangular(L, a, lower=True) for a in arrays], float(np.sum(np.log(np.diag(L))))


def log_evidence(GR, dw, lo=V_LO, hi=V_HI, h=None, pen=0., C=None):
    """log evidence, chi2 and least-squares theta of the model dw = GR theta (+ noise).
    pen: a penalty added to chi2 in the likelihood term (lvoronoi's boundary length); C: noise
    covariance shape (h^2 C); returns (-inf, inf, theta or None) for a singular or out-of-box model"""
    half_logdet_C = 0.
    if C is not None:
        _, (GR, dw), half_logdet_C = whiten(C, GR, dw)
    K = GR.shape[1]
    M = GR.T @ GR
    try:
        cf = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return -np.inf, np.inf, None
    theta = np.linalg.solve(M, GR.T @ dw)
    if (theta < lo).any() or (theta > hi).any():
        return -np.inf, np.inf, theta
    chi2 = float(np.sum((GR @ theta - dw)**2))
    N = len(dw)
    base = -np.sum(np.broadcast_to(np.log(np.subtract(hi, lo)), (K,))) - np.sum(np.log(np.diag(cf))) - half_logdet_C
    if h is None:
        ll = base + gammaln((N - K)/2) - (N - K)/2*np.log(np.pi*(chi2 + pen))
    else:
        ll = base - (N - K)/2*np.log(2*np.pi*h**2) - (chi2 + pen)/(2*h**2)
    return float(ll), chi2, theta


def posterior_theta(GR, dw, C):
    """Gaussian posterior mean and covariance of theta (flat prior) for dw ~ N(GR theta, C)"""
    _, (GRw, dww), _ = whiten(C, GR, dw)
    M = cho_factor(GRw.T @ GRw)
    return cho_solve(M, GRw.T @ dww), cho_solve(M, np.eye(GR.shape[1]))


class SparsePlusLowRank:
    """H = P + A'A with P sparse and positive definite (n x n) and A dense (k x n), k << n: products, the
    diagonal and solves on any subset F of the variables (Woodbury: a sparse factorisation of P_FF and a
    k x k system), without forming H. Used by box_qp for many cells and few data combinations."""

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
    Variables within eps of a bound with the gradient pushing outward are held (eps = the size of the
    projected-gradient step, so the held set is right near the solution); the rest take a Newton step
    (exact Cholesky on the free block), the held ones a diagonally scaled gradient step; Armijo
    backtracking along the projection arc, with the decrease f(v) - f(v + s) = -grad's - s'Hs/2 computed
    from the step (a difference of two large objective values loses the last digits and stalls the
    search short of the tolerance). Stops when the projected gradient (KKT residual) is below
    tol x max|g|. H: a dense matrix or a SparsePlusLowRank."""
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
        pgrad = v - np.clip(v - grad/dH, lo, hi)                     # scaled projected-gradient step
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
            # Armijo along the arc: sufficient decrease relative to the first-order model
            if r >= 1e-4*(-grad @ (w - v)) and r >= 0:
                break
            t *= 0.5
            if t < 1e-12:
                break
        if t < 1e-12:
            # Newton direction failed: fall back to a projected scaled-gradient step
            w = np.clip(v - grad/dH, lo, hi)
            if decrease(w - v, grad) < 0:
                break
        v = w
    if not converged:
        warnings.warn(f'box_qp: not converged after {it + 1} iterations (KKT residual '
                      f'{np.max(np.abs(np.clip(v - (H @ v - g), lo, hi) - v)):.3g}, tolerance {scale:.3g})')
    return v
