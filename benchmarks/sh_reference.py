"""
Independent reference for remit's forward transform: Gauss coefficients of a thin
magnetised shell by direct quadrature of the potential integral.

For VIM M (A, per unit area of a shell of radius r_s) the external potential is
    V(r) = mu0/(4 pi) int M . grad'(1/|r - r'|) dS'.
Expanding 1/|r - r'| with the Schmidt addition theorem and matching to
V = a sum (a/r)^(l+1) (g cos m phi + h sin m phi) P_lm gives, for reference radius a,

  g_lm = mu0/(4 pi a) int rho^(l+1) [ l M_r P_lm cos m phi + M_theta dP_lm/dtheta cos m phi
                                       - M_phi (m/sin theta) P_lm sin m phi ] dOmega
  h_lm = mu0/(4 pi a) int rho^(l+1) [ l M_r P_lm sin m phi + M_theta dP_lm/dtheta sin m phi
                                       + M_phi (m/sin theta) P_lm cos m phi ] dOmega

with rho = r_s/a (1 on the reference sphere). This uses no remit code: the Schmidt
functions come from pyshtools (PlmSchmidt_d1), the longitude sums are explicit
cos/sin products (no FFT), and the latitude quadrature uses Driscoll & Healy (1994)
weights computed from their formula.

Checks built into the formula:
  - radial M0 S_lm         -> g_lm = mu0 M0 l/((2l+1) a)
  - poloidal M0 grad_h S_lm -> g_lm = mu0 M0 l(l+1)/((2l+1) a)
  - toroidal M0 r x grad_h S_lm -> 0
  - Runcorn: M proportional to an internal potential field (l+1) Y r_hat - grad_h Y -> 0
"""
import numpy as np
import pyshtools

MU0 = pyshtools.constants.mu0.value


def dh_weights(n):
    """Driscoll & Healy (1994) weights for nodes theta_j = pi j/n, j = 0..n-1:
    sum_j w_j f(theta_j) = int_0^pi f sin(theta) dtheta, exact for band-limited f.
    Also returns w_j / sin(theta_j), which is finite (and zero) at the pole."""
    th = np.pi*np.arange(n)/n
    k = np.arange(n//2)
    s = (np.sin(np.outer(th, 2*k + 1))/(2*k + 1)).sum(1)
    return (4./n)*np.sin(th)*s, (4./n)*s


def _grid(lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    n = len(lat)
    assert lat[0] == 90 and len(lon) == 2*n and np.allclose(np.diff(lat), -180/n), 'DH2 grid expected'
    colat = np.radians(90 - lat)
    phi = np.radians(lon)
    w, wsin = dh_weights(n)
    return colat, phi, w, wsin, 2*np.pi/len(lon)


def _index(lmax):
    """degree and order of each pyshtools PlmSchmidt index k = l(l+1)/2 + m"""
    ll = np.concatenate([np.full(l + 1, l) for l in range(lmax + 1)])
    mm = np.concatenate([np.arange(l + 1) for l in range(lmax + 1)])
    return ll, mm


def _legendre(lmax, theta):
    p, dz = pyshtools.legendre.PlmSchmidt_d1(lmax, np.cos(theta))
    return p, -np.sin(theta)*dz                       # dP/dtheta = -sin(theta) dP/dz


def gauss_coeffs(lat, lon, mr, mt, mp, a, lmax):
    """Gauss coefficients (nT, pyshtools (2, lmax+1, lmax+1) layout, reference radius a)
    of VIM components (mr, mt, mp) in A on a DH2 grid (lat from 90, lon from 0) on the
    sphere of radius a."""
    colat, phi, w, wsin, dphi = _grid(lat, lon)
    m = np.arange(lmax + 1)
    cos_m, sin_m = np.cos(np.outer(phi, m))*dphi, np.sin(np.outer(phi, m))*dphi
    Cr, Sr = mr @ cos_m, mr @ sin_m                   # longitude sums, (nlat, lmax+1)
    Ct, St = mt @ cos_m, mt @ sin_m
    Cp, Sp = mp @ cos_m, mp @ sin_m
    ll, mm = _index(lmax)
    g = np.zeros(len(ll))
    h = np.zeros(len(ll))
    for i in range(1, len(colat)):                    # the pole row has zero weight
        p, dp = _legendre(lmax, colat[i])
        g += w[i]*(ll*p*Cr[i, mm] + dp*Ct[i, mm]) - wsin[i]*mm*p*Sp[i, mm]
        h += w[i]*(ll*p*Sr[i, mm] + dp*St[i, mm]) + wsin[i]*mm*p*Cp[i, mm]
    out = np.zeros((2, lmax + 1, lmax + 1))
    out[0, ll, mm] = g
    out[1, ll, mm] = h
    out[1, :, 0] = 0.
    return out*MU0/(4*np.pi*a)*1e9


def gauss_coeffs_at_radius(lat, lon, mr, mt, mp, rho, a, lmax):
    """As gauss_coeffs, for sources at radius rho*a varying from node to node
    (VIM per unit area at the source radius), with the exact factor rho^(l+1)
    applied node by node. Cost grows as lmax^2 x grid size."""
    colat, phi, w, wsin, dphi = _grid(lat, lon)
    nlat = len(colat)
    P = np.zeros(((lmax + 1)*(lmax + 2)//2, nlat))
    dP = np.zeros_like(P)
    for i in range(1, nlat):
        P[:, i], dP[:, i] = _legendre(lmax, colat[i])
    out = np.zeros((2, lmax + 1, lmax + 1))
    rho_l = rho.copy()                                # rho^(l+1), starting at l = 0
    for l in range(lmax + 1):
        m = np.arange(l + 1)
        cos_m, sin_m = np.cos(np.outer(phi, m))*dphi, np.sin(np.outer(phi, m))*dphi
        fr, ft, fp = mr*rho_l, mt*rho_l, mp*rho_l
        Cr, Sr = fr @ cos_m, fr @ sin_m
        Ct, St = ft @ cos_m, ft @ sin_m
        Cp, Sp = fp @ cos_m, fp @ sin_m
        k = l*(l + 1)//2 + m
        Pl, dPl = P[k].T, dP[k].T                     # (nlat, l+1)
        out[0, l, :l+1] = (w[:, None]*(l*Pl*Cr + dPl*Ct) - wsin[:, None]*m*Pl*Sp).sum(0)
        out[1, l, :l+1] = (w[:, None]*(l*Pl*Sr + dPl*St) + wsin[:, None]*m*Pl*Cp).sum(0)
        rho_l = rho_l*rho
    out[1, :, 0] = 0.
    return out*MU0/(4*np.pi*a)*1e9
