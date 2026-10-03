"""
Error model and whitening (docs/domain-inversion-framework.md, step 2).

Covariance of the data errors at the points:
    C = sigma_n^2 * Cn + s^2 * Cm
  Cn  data noise: the shape of an isotropic field with the Lowes spectrum of LCS-1 - MF7
  Cm  model error: the shape of an isotropic field with a chosen spectrum (see error_model.py)
Both shapes are normalised to a mean variance of 1 at the points, so sigma_n and s are RMS
values in nT. Both are estimated from the data (ml_amplitudes): the noise-shaped term then stands
for all short-wavelength error, data noise and fine-scale model error together.

Covariance of an isotropic field from its spectrum: for Br at r = a with Lowes spectrum R_l
(mean square of the full field vector per degree), each coefficient has variance
R_l / ((l+1)(2l+1)), and the addition theorem for Schmidt harmonics gives
    cov(Br(x), Br(y)) = sum_l (l+1) R_l / (2l+1) * F(l)^2 * P_l(cos theta)
with F(l) the LCS-1 filter applied to data and model alike.

Whitening for any s at once: with the Cholesky factor Cn + nugget = U U' and the eigenvectors Q
of U^-1 Cm U^-T (eigenvalues lam), T = Q' U^-1 turns C into the diagonal
sigma_n^2 (1 + nugget) + s^2 lam. The nugget (a small diagonal term, relative to the mean
variance) keeps Cn invertible: the points (~55 km) oversample a field band-limited to degree 133.
"""
import numpy as np
from scipy.linalg import cholesky, solve_triangular, eigh

from .basis import unit, band_filter


def lowes(c):
    l = np.arange(c.shape[1])
    return (l+1)*(c**2).sum((0, 2))


def cov_coeffs(R_l, lmin, lmax_fit):
    """Legendre coefficients c_l of the Br covariance from a Lowes spectrum, band and filter applied"""
    l = np.arange(lmax_fit+1)
    F = band_filter(lmax_fit)
    c = (l+1)*np.asarray(R_l)[:lmax_fit+1]/(2*l+1)*F**2
    c[:lmin] = 0
    return c


def legendre_sum(c, x):
    """sum_l c_l P_l(x) for an array x (Bonnet recurrence, in place over x)"""
    p0, p1 = np.ones_like(x), x.copy()
    out = c[0]*p0 + c[1]*p1
    for l in range(1, len(c)-1):
        p0, p1 = p1, ((2*l+1)*x*p1 - l*p0)/(l+1)
        out += c[l+1]*p1
    return out


def cov_matrix(c, plon, plat):
    """covariance matrix at the points, normalised to a mean variance of 1"""
    xyz = unit(plon, plat)
    x = np.clip(xyz @ xyz.T, -1., 1.)
    C = legendre_sum(c, x)
    return C/np.mean(np.diag(C))


def cov_function(c, theta):
    """covariance against angular distance (radians), normalised to 1 at zero lag"""
    return legendre_sum(c, np.cos(theta))/np.sum(c)


def empirical_cov(r, plon, plat, edges_km, R_km=6371.):
    """binned mean of r_i r_j against great-circle distance (pairs i <= j); bin centres, values, counts"""
    xyz = unit(plon, plat)
    d = np.arccos(np.clip(xyz @ xyz.T, -1., 1.))*R_km
    iu = np.triu_indices(len(r))
    k = np.digitize(d[iu], edges_km) - 1
    ok = (k >= 0) & (k < len(edges_km)-1)
    prod = np.outer(r, r)[iu]
    n = np.bincount(k[ok], minlength=len(edges_km)-1)
    s = np.bincount(k[ok], weights=prod[ok], minlength=len(edges_km)-1)
    return 0.5*(edges_km[1:] + edges_km[:-1]), s/np.maximum(n, 1), n


def whitening(Cn, Cm, nugget):
    """T and lam such that T (sn^2 (Cn + nugget I) + s^2 Cm) T' = diag(sn^2 + s^2 lam) for all sn, s"""
    U = cholesky(Cn + nugget*np.eye(len(Cn)), lower=True)
    A = solve_triangular(U, Cm, lower=True)
    W = solve_triangular(U, A.T, lower=True)              # U^-1 Cm U^-T (symmetric)
    lam, Q = eigh(0.5*(W + W.T))
    T = solve_triangular(U, np.eye(len(Cn)), lower=True)  # U^-1
    return Q.T @ T, np.clip(lam, 0., None)


def neg_loglik(s2, rw, lam, sn2):
    """-log likelihood (constants dropped) of whitened residuals rw for model-error variance s2"""
    v = sn2 + s2*lam
    return 0.5*np.sum(np.log(v) + rw**2/v)


def ml_model_error(rw, lam, sn2):
    """maximum-likelihood s^2 for whitened residuals (1-D search on a log grid, then refined)"""
    grid = np.logspace(0, 5, 200)
    s2 = grid[np.argmin([neg_loglik(g, rw, lam, sn2) for g in grid])]
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda q: neg_loglik(np.exp(q), rw, lam, sn2),
                          bracket=(np.log(s2) - 0.1, np.log(s2) + 0.1))
    return float(np.exp(res.x))


def ml_amplitudes(rw, lam, start=(20., 30.)):
    """maximum-likelihood (sigma_n^2, s^2) for whitened residuals: both amplitudes free. The
    noise-shaped term then stands for all short-wavelength error (data noise and fine-scale model
    error), the model-error term for the field-shaped part."""
    from scipy.optimize import minimize
    f = lambda q: neg_loglik(np.exp(q[1]), rw, lam, np.exp(q[0]))
    res = minimize(f, np.log(np.square(start)), method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-6))
    return tuple(float(x) for x in np.exp(res.x))
