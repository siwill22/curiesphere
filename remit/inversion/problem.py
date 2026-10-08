"""
The problem of a local fit (ADR 0002), built from a config; methods estimate cell values from it.

  cells      HEALPix cells of the target area (partial at its edge), optionally coarsened (sums of four children)
  G          Br (nT per SI.km) of each cell at the data points, degrees of the band, resolution filter applied
  data       Br of the field model at the points less the background's (filtered); or, for synthetic data,
             G truth plus a draw from the error model
  weighting  W (k x points) from the compressed error model (errors.py); Gw = W G, yw = W data

Config keys used:
  [target] area (an outline file), nside;  [geometry] layers (one), direction, lmax;  [compute] columns_backend,
  coarsen;  [fit] buffer_km, data_nside, background {vis_model, remanence, near_ridge};
  [data] model ('lcs1', 'mf7', a .shc path, or 'synthetic'), band, filter; synthetic: based_on, truth, seed;
  [errors] weighting ('subspace' or 'full'), keep, nugget;
  [errors.noise] reference (field model), rms_nT (a number, 'points': the RMS of the difference at the points, or
                 'spectrum': its global RMS);
  [errors.model_error] spectrum ('field': the field model's own), rms_nT (0 for none)
"""
import hashlib
import os
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from . import columns, data, errors, geometry, truths
from .cache import save_atomic

WHITENING_VERSION = 1


@dataclass
class Local:
    cells: geometry.Cells
    fine: geometry.Cells
    parent: np.ndarray              # coarse index of each fine cell (None if not coarsened)
    mask: np.ndarray
    plon: np.ndarray
    plat: np.ndarray
    G: np.ndarray                   # points x cells
    y: np.ndarray                   # data at the points (nT), background removed
    W: np.ndarray                   # k x points
    truth: np.ndarray = None
    info: dict = field(default_factory=dict)

    @property
    def Gw(self):
        return self.W @ self.G

    @property
    def yw(self):
        return self.W @ self.y


def coarsened(cells, n):
    """cells coarsened n times, and the coarse index of each original cell (None for n = 0)"""
    parent = None
    for _ in range(n):
        cells, p = geometry.coarsen(cells)
        parent = p if parent is None else p[parent]
    return cells, parent


def _hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
    return h.hexdigest()[:16]


def local(cfg, cache, say=print):
    tg, geo, comp, fit, d, er = cfg['target'], cfg['geometry'], cfg['compute'], cfg['fit'], cfg['data'], cfg['errors']
    if len(geo['layers']) != 1:
        raise NotImplementedError('one layer')
    t0 = time.time()
    mask = geometry.target_mask(tg['area'], cache)
    name = geometry.area_name(tg['area'])
    fine = geometry.make_cells(mask, tg['nside'])
    store = columns.ColumnStore(fine, name, geo['layers'][0], geo['direction'], geo['lmax'],
                                comp['columns_backend'], cache).build(say)
    band = data.Band(*d['band'])
    plon, plat = data.points(mask, fit['data_nside'], fit['buffer_km'])
    Y = data.synthesis(plon, plat, band)
    F = data.resolution_filter(d.get('filter', 'lcs1'), geo['lmax'])
    Fb = F[band.l]

    cells, parent = coarsened(fine, comp.get('coarsen', 0))
    G = np.zeros((len(plon), len(cells)))
    for idx in store.blocks:
        Gb = Y @ (Fb[:, None]*band.columns(store.get(idx)))
        if parent is None:
            G[:, idx] = Gb
        else:
            G += (sparse.csr_matrix((np.ones(len(idx)), (parent[idx], np.arange(len(idx)))),
                                    shape=(len(cells), len(idx))) @ Gb.T).T

    # error model
    synthetic = d['model'] == 'synthetic'
    model = d.get('based_on', 'lcs1') if synthetic else d['model']
    c_model = data.field_coeffs(model, geo['lmax'])
    nz, me = er['noise'], er.get('model_error', {})
    c_ref = data.field_coeffs(nz['reference'], geo['lmax'])
    R_noise = errors.noise_spectrum(c_model, c_ref, F)
    sn = nz.get('rms_nT', 'points')
    if sn == 'points':
        sn = float(np.sqrt(np.mean((Y @ band.rows(c_model - F[None, :, None]*c_ref))**2)))
    elif sn == 'spectrum':
        sn = float(np.sqrt(np.sum(band.mean_square_br()*errors.coefficient_variance(R_noise)[band.l])))
    s = float(me.get('rms_nT', 0.))
    if me.get('spectrum', 'field') != 'field':
        raise NotImplementedError(f'model-error spectrum {me["spectrum"]!r}')
    R_model = errors.lowes(c_model)
    keep, nugget = er.get('keep', 0.999), er.get('nugget', 1e-6)
    spec = dict(points=_hash(plon, plat), band=[band.lmin, band.lmax], noise=_hash(R_noise[:band.lmax + 1]),
                model=_hash(R_model[:band.lmax + 1]), sn=sn, s=s, keep=keep, nugget=nugget, version=WHITENING_VERSION)

    def whitening():
        say(f'error model at {len(plon)} points (dense, compressed)')
        Cn = errors.point_shape(R_noise, band, plon, plat)
        Cm = errors.point_shape(R_model, band, plon, plat) if s > 0 else np.zeros_like(Cn)
        return errors.point_whitening(Cn, Cm, sn, s, keep, nugget)
    files = [os.path.join(cache.entry('whitening', spec), f'{k}.npy') for k in ('P', 'T', 'lam')]
    if not all(os.path.exists(f) for f in files):
        for f, a in zip(files, whitening()):
            save_atomic(f, lambda p: np.save(p, a))
    P, T, lam = (np.load(f) for f in files)
    W = errors.weights(P, T, lam, sn, s, nugget, er.get('weighting', 'subspace'))

    # data
    truth = None
    if synthetic:
        truth_fine = truths.truth(d['truth'], fine)
        truth = truth_fine if parent is None else np.bincount(parent, truth_fine*fine.area_km2)/cells.area_km2
        z = np.random.default_rng(d['seed']).standard_normal(len(lam))*np.sqrt(sn**2*(1 + nugget) + s**2*lam)
        y = G @ truth + P @ np.linalg.solve(T, z)
    else:
        bg = fit['background']
        C_bg = data.background(cache, mask, name, geo['lmax'], bg['vis_model'], bg['remanence'], bg['near_ridge'])
        y = Y @ band.rows(c_model) - Y @ (Fb*band.rows(C_bg))
    info = dict(n_points=len(plon), n_data=len(lam), noise_rms=sn, model_error_rms=s,
                weighting=er.get('weighting', 'subspace'))
    say(f'{len(cells)} cells (NSIDE {cells.nside}), {len(plon)} points compressed to {len(lam)} data; noise RMS '
        f'{sn:.1f} nT, model error RMS {s:.1f} nT ({time.time() - t0:.0f} s)')
    return Local(cells, fine, parent, mask, plon, plat, G, y, W, truth, info)
