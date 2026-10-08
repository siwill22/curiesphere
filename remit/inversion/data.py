"""
Data for inversions. Global fits (ADR 0002) use the Gauss coefficients of a field model in a degree band, with the
field model's resolution filter applied to the columns and the background; local fits (Br at points) come later.

  field models  'lcs1' (LCS_mod.cof), 'mf7' (MF7_mod.cof, degree <= 133) or the path of a Swarm .shc file
                (e.g. MLI_SHA_2E); all referenced to r0 = R0
  filter        'lcs1': F(l) of depth.lcs1_filter fitted to LCS-1's damping; 'none'
  band          the rows of a coefficient fit: g_lm for lmin <= l <= lmax, then h_lm for m > 0
  background    the fixed sources: the VIS of a model ('Hemant2005+slabs') with the free cells' grid nodes removed
                (in a global fit all continental crust, which leaves oceanic VIS and slabs under ocean-age crust),
                VIS spread over the HM05 oceanic crust as in depth.vis_coeffs, plus ocean remanence (an
                OCEAN_MODEL_LIST model, by default GK07 with the tuned near-ridge enhancement)
"""
import functools
import os

import numpy as np
import pyshtools

from remit import depth
from remit.data.models import DATA_DIR, OCEAN_MODEL_LIST, load_lcs, load_vis_model
from . import geometry
from .columns import packed_lm

BACKGROUND_VERSION = 1


def read_shc(path, lmax):
    """Gauss coefficients (2, lmax+1, lmax+1) of a Swarm-format .shc file (n, m, value; m < 0 for h), moved from
    6371.2 km to r0 = R0 by (a/R0)^(n+2)"""
    c = np.zeros((2, lmax + 1, lmax + 1))
    with open(os.path.expanduser(path)) as f:
        for line in f:
            p = line.split()
            if line.startswith('#') or len(p) != 3:
                continue
            n, m = int(p[0]), int(p[1])
            if n > lmax:
                break
            c[0 if m >= 0 else 1, n, abs(m)] = float(p[2])
    n = np.arange(lmax + 1)
    return c*((6371.2e3/depth.R0)**(n + 2))[None, :, None]


def field_coeffs(spec, lmax):
    """Gauss coefficients (2, lmax+1, lmax+1) of a field model, zero-padded to lmax"""
    if spec == 'lcs1':
        c = load_lcs(lmin=0, lmax=depth.LMAX).coeffs
    elif spec == 'mf7':
        c, _ = pyshtools.shio.shread(f'{DATA_DIR}/shc/MF7_mod.cof', lmax=133)       # r0 taken as 6371.0 km
    else:
        return read_shc(spec, lmax)
    out = np.zeros((2, lmax + 1, lmax + 1))
    k = min(lmax, c.shape[1] - 1) + 1
    out[:, :k, :k] = c[:, :k, :k]
    return out


@functools.lru_cache(maxsize=1)
def _lh():
    return depth.fit_lh()


def resolution_filter(kind, lmax):
    if kind == 'lcs1':
        return depth.lcs1_filter(np.arange(lmax + 1), _lh())
    if kind == 'none':
        return np.ones(lmax + 1)
    raise ValueError(f'filter {kind!r}')


class Band:
    """the rows of a coefficient fit, degrees lmin-lmax: g_lm (all m), then h_lm (m > 0)"""

    def __init__(self, lmin, lmax):
        self.lmin, self.lmax = lmin, lmax
        L, M = packed_lm(lmax)
        self.sel_g = (L >= lmin)
        self.sel_h = self.sel_g & (M > 0)
        self.l = np.concatenate([L[self.sel_g], L[self.sel_h]])
        self.m = np.concatenate([M[self.sel_g], M[self.sel_h]])
        self.gh = np.concatenate([np.zeros(self.sel_g.sum(), int), np.ones(self.sel_h.sum(), int)])

    def __len__(self):
        return len(self.l)

    def rows(self, c):
        """band rows of a (2, l, m) coefficient array"""
        return np.where(self.gh == 0, c[0, self.l, self.m], c[1, self.l, self.m])

    def columns(self, cols):
        """band rows (n_rows, n_cells) of packed columns (2, n_cells, n_lm), n_lm of any lmax >= the band's"""
        n = len(self.sel_g)
        return np.concatenate([cols[0][:, :n][:, self.sel_g], cols[1][:, :n][:, self.sel_h]], axis=1).T

    def coeffs(self, values):
        """(2, lmax+1, lmax+1) coefficient array with the band rows set to values"""
        c = np.zeros((2, self.lmax + 1, self.lmax + 1))
        c[self.gh, self.l, self.m] = values
        return c

    def mean_square_br(self):
        """mean square of Br over the sphere at r0 per squared coefficient, per row (Schmidt harmonics)"""
        return (self.l + 1.)**2/(2*self.l + 1)


def background(cache, mask, mask_name, lmax=depth.LMAX, vis_model='Hemant2005+slabs', remanence='GK07',
               near_ridge=True):
    """Gauss coefficients (2, lmax+1, lmax+1) of the fixed sources, with the nodes of mask (the free cells) removed"""
    spec = dict(grid=geometry.AGE_GRID, mask=mask_name, lmax=lmax, vis_model=vis_model, remanence=remanence,
                near_ridge=near_ridge, version=BACKGROUND_VERSION)

    def compute():
        g = geometry.grid()
        r_top = depth.top_of_crust_radius(g.ocean)
        vis = load_vis_model(name=vis_model, match=(g.ocean.lon, g.ocean.lat, g.ocean.age))
        vis.vis = np.where(mask, 0., vis.vis)
        c = depth.vis_coeffs(g.ocean, vis, r_top, lmax=lmax).coeffs
        if remanence:
            params = OCEAN_MODEL_LIST[remanence]
            params = depth.with_near_ridge(params) if near_ridge else params
            c = c + depth.remanent_coeffs(g.ocean, r_top, params, lmax=lmax)[0].coeffs
        return c
    return np.asarray(cache.array('background', spec, 'coeffs', compute))
