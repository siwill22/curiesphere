"""
The error model (CONTEXT.md): noise and model error, each a per-degree spectrum with an amplitude.

  noise          the Lowes spectrum of (field model - F x reference model), degrees >= 16 (the reference's
                 lowest), F the field model's resolution filter so its damping is not counted as error
  model error    optional: the field model's own spectral shape, with a fixed amplitude (RMS Br, nT)
  amplitudes     'fixed' (the spectrum as it is) or 'evidence' (a scale fitted with the model, by the core)

For coefficient data an isotropic field is diagonal: each coefficient of degree l has variance
R_l/((l+1)(2l+1)) for a Lowes spectrum R_l.

For Br at points the covariance is dense: by the addition theorem for Schmidt harmonics,
    cov(Br(x), Br(y)) = sum_l (l+1) R_l/(2l+1) P_l(cos theta)      (degrees of the band)
Both terms are used as shapes of unit variance with an RMS amplitude each (nT): C = sn^2 Cn + s^2 Cm.
The points (~55 km) oversample a band-limited field, so the data are compressed onto the leading eigenvectors P of
C (a fraction `keep` of its trace) and whitened there. In that subspace a Cholesky factor of Cn (+ a small nugget
that keeps it invertible) and the eigenvectors of Cm relative to it give T with
    T P' (sn^2 (Cn + nugget I) + s^2 Cm) P T' = diag(sn^2 (1 + nugget) + s^2 lam)       for any sn, s.
Weightings of the compressed data:
  'subspace'  P'/sn: the subspace only, errors treated as white with the noise RMS (the first Australian build)
  'full'      diag(sn^2 (1 + nugget) + s^2 lam)^-1/2 T P': whitened by the whole error model
"""
import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular

LMIN_REFERENCE = 16


def lowes(c):
    """Lowes-Mauersberger spectrum R_l of a (2, l, m) coefficient array"""
    l = np.arange(c.shape[1])
    return (l + 1)*(c**2).sum((0, 2))


def noise_spectrum(c, ref, F):
    """R_l of c - F ref, zero below the reference's lowest degree"""
    diff = c - ref*F[None, :, None]
    diff[:, :LMIN_REFERENCE] = 0
    return lowes(diff)


def coefficient_variance(R):
    l = np.arange(len(R))
    return R/((l + 1.)*(2*l + 1.))


def scaled_to_rms_br(R, band, rms):
    """the spectrum R rescaled so that its field in the band has RMS Br = rms (nT)"""
    var = coefficient_variance(R)[band.l]
    return R*rms**2/np.sum(band.mean_square_br()*var)


def legendre_sum(c, x):
    """sum_l c_l P_l(x) for an array x (Bonnet recurrence)"""
    p0, p1 = np.ones_like(x), x.copy()
    out = c[0]*p0 + c[1]*p1
    for l in range(1, len(c) - 1):
        p0, p1 = p1, ((2*l + 1)*x*p1 - l*p0)/(l + 1)
        out += c[l + 1]*p1
    return out


def point_shape(R, band, plon, plat):
    """the covariance of Br at the points of an isotropic field with spectrum R in the band, scaled to unit variance"""
    from .graph import unit
    l = np.arange(band.lmax + 1)
    c = (l + 1)*np.asarray(R, float)[:band.lmax + 1]/(2*l + 1)
    c[:band.lmin] = 0
    xyz = unit(plon, plat)
    return legendre_sum(c, np.clip(xyz @ xyz.T, -1., 1.))/np.sum(c)


def point_whitening(Cn, Cm, sn, s, keep, nugget):
    """P (points x k, orthonormal), T (k x k) and lam of the compressed whitening above"""
    w, V = eigh(sn**2*Cn + s**2*Cm)
    k = int(np.searchsorted(np.cumsum(w[::-1])/w.sum(), keep)) + 1
    P = V[:, ::-1][:, :k]
    U = cholesky(P.T @ Cn @ P + nugget*np.eye(k), lower=True)
    A = solve_triangular(U, P.T @ Cm @ P, lower=True)
    W = solve_triangular(U, A.T, lower=True)                       # U^-1 Cm U^-T
    lam, Q = eigh(0.5*(W + W.T))
    return P, Q.T @ solve_triangular(U, np.eye(k), lower=True), np.clip(lam, 0., None)


def weights(P, T, lam, sn, s, nugget, weighting):
    """the weighting matrix (k x points) of compressed data"""
    if weighting == 'subspace':
        return P.T/sn
    if weighting == 'full':
        return (T @ P.T)/np.sqrt(sn**2*(1 + nugget) + s**2*lam)[:, None]
    raise ValueError(f'weighting {weighting!r}')
