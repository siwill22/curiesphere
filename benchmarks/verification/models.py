"""
Test magnetisations for the verification suite, on the 0.1 deg Driscoll-Healy grid that the production forward
models use (1800 x 3600; dh_grid(1800)). Each model returns the vertically integrated magnetisation (VIM, A) as
(r, theta, phi) grids for sources on the R0 sphere, and a map to show what the model is.

  dipoles      seven point dipoles in different places and directions, including the grid seam at lon 0, the
               dateline, the rows next to both poles, and induced (IGRF) directions
  checkerboard 20 deg squares in lon/lat, VIS +1 / -1 SI.km. On the sphere the squares narrow toward the poles, so
               the pole rows carry many sign changes in a small area
  healpix      HEALPix NSIDE 4 tiles (equal area, about 15 deg), alternating within each of the 12 base faces
  uniform      VIS 1 SI.km everywhere. Magnetisation induced by an internal field in a spherical shell produces no
               field outside it (Runcorn's theorem), so both methods must give zero
  band_limited a model low-passed in its Cartesian components to degree L-1: the field outside is then band-limited
               to degree L, and remit at lmax L must match the dipole sum to rounding

Induced magnetisation here is built independently of remit: VIS [SI.km] x 1e3 x IGRF-13 [nT] x 1e-9 / mu0.
"""
import functools

import numpy as np
import pyshtools
from astropy_healpix import healpy as hp

from reference import MU0, R0

MOMENT = 2.5e14            # A m^2: about 50 nT of Br at 100 km above a radial dipole
VIM_PER_VIS = 1e3*1e-9/MU0  # A per (SI.km x nT)

# label, latitude, longitude, direction
DIPOLES = [
    ('equator at lon 0 (the grid seam)', 0.0, 0.0, 'radial'),
    ('30N 90E', 30.0, 90.0, 'north'),
    ('30S 180E (the dateline)', -30.0, 180.0, 'east'),
    ('next to the north pole', 89.9, 45.0, 'east'),
    ('next to the south pole', -89.9, 270.0, 'north'),
    ('60S 300E', -60.0, 300.0, 'induced'),
    ('45N 250E', 45.0, 250.0, 'induced'),
]


@functools.lru_cache(maxsize=1)
def igrf(n):
    """IGRF-13 (r, theta, phi) in nT on the n x 2n DH grid"""
    b = pyshtools.datasets.Earth.IGRF_13().expand(lmax=n//2 - 1, extend=False)
    return [x.data for x in (b.rad, b.theta, b.phi)]


def induced(g, vis):
    return [vis*VIM_PER_VIS*b for b in igrf(g.n)]


def dipoles(g):
    """VIM grids of the planted dipoles, and their node indices and moment vectors (r, theta, phi)"""
    mr, mt, mp = (np.zeros(g.colat.shape) for _ in range(3))
    table = []
    B = igrf(g.n)
    for label, lat, lon, direction in DIPOLES:
        i, j = int(np.argmin(np.abs(g.lat - lat))), int(np.argmin(np.abs(g.lon - lon)))
        if direction == 'induced':
            u = np.array([b[i, j] for b in B])
            u /= np.linalg.norm(u)
        else:
            u = dict(radial=[1., 0, 0], north=[0., -1, 0], east=[0., 0, 1])[direction]
        m = MOMENT*np.asarray(u)
        for grid, mk in zip((mr, mt, mp), m):
            grid[i, j] += mk/(R0**2*g.cell[i, j])    # VIM such that VIM r^2 w dlon is the moment
        table.append(dict(label=label, lat=g.lat[i], lon=g.lon[j], direction=direction, i=i, j=j))
    return [mr, mt, mp], table


def checkerboard(g, size=20.):
    sign = (np.floor(np.degrees(np.pi/2 - g.colat)/size) + np.floor(np.degrees(g.lonr)/size)) % 2
    return np.where(sign == 0, 1., -1.)


def healpix_tiles(g, nside=4):
    """+1 / -1 alternating within each HEALPix base face (nested x + y parity)"""
    pix = hp.ang2pix(nside, np.degrees(g.lonr).ravel(), 90 - np.degrees(g.colat).ravel(), lonlat=True, nest=True)
    p = pix % (nside*nside)
    x = sum(((p >> (2*b)) & 1) << b for b in range(8))          # de-interleave the nested index
    y = sum(((p >> (2*b + 1)) & 1) << b for b in range(8))
    return np.where((x + y) % 2 == 0, 1., -1.).reshape(g.colat.shape)


def band_limited(g, vim, L):
    """VIM low-passed to degree L-1 in Cartesian components, so its external field stops at degree L"""
    xyz = [sum(c*u[..., k] for c, u in zip(vim, (g.rh, g.th, g.ph))) for k in range(3)]
    low = [pyshtools.expand.MakeGridDH(pyshtools.expand.SHExpandDH(c, sampling=2, lmax_calc=L - 1),
                                       lmax=g.n//2 - 1, sampling=2) for c in xyz]
    return [sum(low[k]*u[..., k] for k in range(3)) for u in (g.rh, g.th, g.ph)]
