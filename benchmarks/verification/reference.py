"""
The reference method of the verification suite: a direct sum of point dipoles in Cartesian coordinates, sharing no
code with remit's transform or with the cell columns.

    B(x) = mu0/4pi sum_j [3 (m_j . R) R / |R|^5 - m_j / |R|^3],   R = x - s_j

A magnetisation grid is turned into dipoles with one dipole per grid node: moment m_j = VIM_j r_j^2 w_j dlon, where
w_j are the Driscoll & Healy (1994) latitude weights, computed here from their formula (sh_reference.dh_weights). The
sum is exact on the sphere: no flat-Earth approximation enters, and sources may sit at any radius.

Also here: Driscoll-Healy grids of any size, unit vectors, the observation grid of the maps, and the synthesis of
remit's Gauss coefficients on that grid.
"""
import os
import sys
from types import SimpleNamespace

import numba
import numpy as np
import pyshtools

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from sh_reference import dh_weights                     # noqa: E402

R0 = 6371000.
MU0 = pyshtools.constants.mu0.value
THREADS = int(os.environ.get('VERIFY_THREADS', '3'))    # keep the machine usable (at most 3 threads)


def unit_vectors(colat, lon):
    """r, theta, phi unit vectors (..., 3) in Cartesian coordinates; angles in radians"""
    st, ct, sp, cp = np.sin(colat), np.cos(colat), np.sin(lon), np.cos(lon)
    rh = np.stack([st*cp, st*sp, ct], -1)
    th = np.stack([ct*cp, ct*sp, -st], -1)
    ph = np.stack([-sp, cp, np.zeros_like(sp)], -1)
    return rh, th, ph


def dh_grid(n):
    """Driscoll-Healy grid of n x 2n nodes (pyshtools DH2: north pole in, south pole out, lon 0 in, 360 out):
    lat, lon (degrees), colat and lon (radians, 2-D), unit vectors and the solid angle of each node"""
    g = SimpleNamespace(n=n)
    g.lat = 90 - np.arange(n)*180/n
    g.lon = np.arange(2*n)*360/(2*n)
    g.colat, g.lonr = np.meshgrid(np.radians(90 - g.lat), np.radians(g.lon), indexing='ij')
    g.rh, g.th, g.ph = unit_vectors(g.colat, g.lonr)
    w, _ = dh_weights(n)
    g.cell = w[:, None]*np.radians(360/(2*n))*np.ones((1, 2*n))      # solid angle per node (sr)
    g.lmax_exact = n//2 - 1                                          # highest degree the grid samples exactly
    return g


def obs_grid(step=1.0):
    """observation points of the maps: a Driscoll-Healy grid of the given step (degrees), the nodes pyshtools
    synthesises on, so remit's field needs no interpolation"""
    return dh_grid(int(round(180/step)))


# ---------------------------------------------------------------- the dipole sum

@numba.njit(parallel=True, cache=True)
def _dipole_sum(obs, src, mom, out):
    """out[i] += sum_j field of dipole mom[j] at src[j], observed at obs[i] (Cartesian, SI, without mu0/4pi).
    Observers are processed in blocks so each pass over the sources serves 16 of them."""
    nobs, nsrc, nb = obs.shape[0], src.shape[0], 16
    for b in numba.prange((nobs + nb - 1)//nb):
        i0 = b*nb
        i1 = min(i0 + nb, nobs)
        acc = np.zeros((nb, 3))
        for j in range(nsrc):
            sx, sy, sz = src[j, 0], src[j, 1], src[j, 2]
            mx, my, mz = mom[j, 0], mom[j, 1], mom[j, 2]
            for i in range(i0, i1):
                rx, ry, rz = obs[i, 0] - sx, obs[i, 1] - sy, obs[i, 2] - sz
                inv2 = 1.0/(rx*rx + ry*ry + rz*rz)
                inv3 = inv2*np.sqrt(inv2)
                t = 3.0*(mx*rx + my*ry + mz*rz)*inv2
                acc[i - i0, 0] += (t*rx - mx)*inv3
                acc[i - i0, 1] += (t*ry - my)*inv3
                acc[i - i0, 2] += (t*rz - mz)*inv3
        for i in range(i0, i1):
            out[i, 0] += acc[i - i0, 0]
            out[i, 1] += acc[i - i0, 1]
            out[i, 2] += acc[i - i0, 2]


class DipoleSum:
    """the field of magnetisation on a grid, at observation points (unit vectors obs_rh, obs_th, obs_ph and radii)"""

    def __init__(self, obs, altitudes):
        """obs: a grid from obs_grid; altitudes (m) above R0"""
        rh, th, ph = (u.reshape(-1, 3) for u in (obs.rh, obs.th, obs.ph))
        self.shape, self.altitudes = obs.colat.shape, list(altitudes)
        self.xyz = np.concatenate([(R0 + h)*rh for h in altitudes])
        self.rh, self.th, self.ph = (np.concatenate([u]*len(altitudes)) for u in (rh, th, ph))
        self.B = np.zeros_like(self.xyz)
        numba.set_num_threads(THREADS)

    def add(self, g, mr, mt, mp, r_s, sel=None):
        """VIM components (A) on grid g, sources at radius r_s (m, scalar or grid); sel limits the nodes"""
        r_s = np.broadcast_to(r_s, mr.shape)
        m = mr[..., None]*g.rh + mt[..., None]*g.th + mp[..., None]*g.ph
        m *= (r_s**2*g.cell)[..., None]
        keep = np.any(m != 0, axis=-1) if sel is None else sel & np.any(m != 0, axis=-1)
        self.add_dipoles((r_s[..., None]*g.rh)[keep], m[keep])

    def add_dipoles(self, xyz, moment):
        """point dipoles at Cartesian positions xyz (m) with moments (A m^2)"""
        _dipole_sum(self.xyz, np.ascontiguousarray(xyz, float), np.ascontiguousarray(moment, float), self.B)

    def br(self):
        """Br (nT) as (altitude, lat, lon) grids"""
        B = self.B*MU0/(4*np.pi)*1e9
        return (B*self.rh).sum(1).reshape(len(self.altitudes), *self.shape)


def remit_br(coeffs, obs, altitudes, lmax):
    """Br (nT) of Gauss coefficients (r0 = R0, degrees 1..lmax) on the observation grid, as (altitude, lat, lon)"""
    c = pyshtools.SHMagCoeffs.from_array(np.array(coeffs[:, :lmax+1, :lmax+1]), r0=R0)
    n = obs.n
    out = []
    for h in altitudes:
        # synthesise on a DH grid at least as fine as lmax needs, then take the observation nodes
        k = max(1, int(np.ceil(2*(lmax + 1)/n)))
        grid = c.expand(a=R0 + h, lmax=k*n//2 - 1, lmax_calc=lmax, extend=False).rad.data
        out.append(grid[::k, ::k])
    return np.array(out)
