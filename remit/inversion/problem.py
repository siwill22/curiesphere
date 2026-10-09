"""
The problem of a local fit (ADR 0002), built from a config; methods estimate cell values from it.

  cells      HEALPix cells of the target area (partial at its edge), optionally coarsened (sums of four children)
  G          Br (nT per SI.km) of each cell at the data points, degrees of the band, resolution filter applied
  data       Br of the field model at the points less the background's (filtered); or, for synthetic data,
             G truth plus a draw from the error model
  weighting  W (k x points) from the compressed error model (errors.py), or 'diagonal': every point with weight
             1/(noise RMS), no compression (correlated errors ignored, as the first polygon-domain inversion);
             Gw = W G, yw = W data
  draws      realisations of the error model at the points: random-phase fields with the noise and model-error
             spectra in the band, each scaled to its RMS (for synthetic data and noise ensembles)

Config keys used:
  [target] area (an outline file), nside;  [geometry] layers (one), direction, lmax;  [compute] columns_backend,
  coarsen;  [fit] buffer_km, data_nside, background {vis_model, remanence, near_ridge};
  [data] model ('lcs1', 'mf7', a .shc path, or 'synthetic'), band, filter; synthetic: based_on, truth, seed;
  [errors] weighting ('subspace', 'full' or 'diagonal'), keep, nugget;
  [errors.noise] reference (field model), rms_nT (a number, 'points': the RMS of the difference at the points, or
                 'spectrum': its global RMS);
  [errors.model_error] spectrum ('field': the field model's own), rms_nT (0 for none)
"""
import functools
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
    W: object                       # k x points (dense, or sparse for 'diagonal')
    truth: np.ndarray = None
    info: dict = field(default_factory=dict)
    draw: object = None             # draw(rng, model_error=True) -> an error-model realisation at the points (nT)
    band: object = None
    spectra: dict = None            # Lowes spectra of the noise and of the field model ('noise', 'model')

    @functools.cached_property
    def Gw(self):
        return np.asarray(self.W @ self.G)

    @property
    def yw(self):
        return np.asarray(self.W @ self.y)


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
    sn = float(sn)
    s = float(me.get('rms_nT', 0.))
    if me.get('spectrum', 'field') != 'field':
        raise NotImplementedError(f'model-error spectrum {me["spectrum"]!r}')
    R_model = errors.lowes(c_model)
    keep, nugget = er.get('keep', 0.999), er.get('nugget', 1e-6)
    weighting = er.get('weighting', 'subspace')
    if weighting == 'diagonal':
        W, n_data = sparse.diags(np.full(len(plon), 1/sn)), len(plon)
    else:
        spec = dict(points=_hash(plon, plat), band=[band.lmin, band.lmax], noise=_hash(R_noise[:band.lmax + 1]),
                    model=_hash(R_model[:band.lmax + 1]), sn=sn, s=s, keep=keep, nugget=nugget,
                    version=WHITENING_VERSION)

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
        W, n_data = errors.weights(P, T, lam, sn, s, nugget, weighting), len(lam)

    sd_rows = [np.sqrt(errors.coefficient_variance(R)[band.l]) for R in (R_noise, R_model)]
    sd_rows = [sd*rms/np.sqrt(np.sum(band.mean_square_br()*sd**2)) for sd, rms in zip(sd_rows, (sn, s))]

    def draw(rng, model_error=True):
        e = Y @ (sd_rows[0]*rng.standard_normal(len(band)))
        return e + Y @ (sd_rows[1]*rng.standard_normal(len(band))) if model_error and s > 0 else e

    # data
    truth = None
    if synthetic:
        truth_fine = truths.truth(d['truth'], fine)
        truth = truth_fine if parent is None else np.bincount(parent, truth_fine*fine.area_km2)/cells.area_km2
        y = G @ truth + draw(np.random.default_rng(d['seed']))
    else:
        bg = fit['background']
        C_bg = data.background(cache, mask, name, geo['lmax'], bg['vis_model'], bg['remanence'], bg['near_ridge'])
        y = Y @ band.rows(c_model) - Y @ (Fb*band.rows(C_bg))
    info = dict(n_points=len(plon), n_data=n_data, noise_rms=sn, model_error_rms=s, weighting=weighting)
    say(f'{len(cells)} cells (NSIDE {cells.nside}), {len(plon)} points, {n_data} data ({weighting}); noise RMS '
        f'{sn:.1f} nT, model error RMS {s:.1f} nT ({time.time() - t0:.0f} s)')
    return Local(cells, fine, parent, mask, plon, plat, G, y, W, truth, info, draw, band,
                 dict(noise=R_noise, model=R_model))
