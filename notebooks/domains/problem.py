"""
The cached problem of a region (made by build_region.py and error_model.py), loaded once for the run
scripts: G, data, the compressing and weighting transform T, and the cells.

Weightings (as in run_merge.py)
  points  orthonormal rows spanning the error-model subspace, scaled by sigma = RMS(LCS-1 - MF7)
          (A2: point weighting; the default)
  noise   the error-model eigenvectors scaled by sigma
  model   the eigenvectors scaled by the fitted noise + model-error variance
Whitened quantities: Gw = T G, dw_real = T (d_lcs - b), noise_w = T (noise realisations at sigma).

Bands (build_bands.py): band=<name> (or env BAND) loads the variant in cache/bands/<name> (another data
model and/or degree range: its own data, G, noise and error model; cells shared). Its 'cache' is the
band directory (outputs go there); 'shared' is always the region cache (covariates, synthetic truths).
"""
import os
import tomllib

import numpy as np


def load(path, weighting='points', band=None):
    """path: the region's config.toml (or its directory); returns a dict of arrays"""
    reg = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    cfg = {}
    if os.path.exists(f'{reg}/config.toml'):
        with open(f'{reg}/config.toml', 'rb') as f:
            cfg = tomllib.load(f)
    band = band if band is not None else os.environ.get('BAND') or None
    shared = f'{reg}/cache'
    C = f'{shared}/bands/{band}' if band else shared
    E = np.load(f'{C}/errors.npz')
    z = np.load(f'{C}/data.npz')
    G = np.load(f'{C}/G.npz')['G']
    cells = np.load(f'{shared}/cells.npz')
    d = z['d'] if 'd' in z else z['d_lcs']                              # the data model at the points
    sigma = float(z['sigma']) if 'sigma' in z else float(np.sqrt(np.mean((z['d_lcs'] - z['d_mf7'])**2)))
    if weighting == 'model':
        T = E['T']/np.sqrt(float(E['sn2']) + float(E['s2'])*E['lam'])[:, None]
    elif weighting == 'noise':
        T = E['T']/sigma
    else:
        T = np.linalg.svd(E['T'].T, full_matrices=False)[0].T/sigma
    noise = np.load(f'{C}/noise.npz')['noise']
    noise = noise*sigma/np.sqrt(np.mean(noise**2))
    return dict(reg=reg, cache=C, shared=shared, band=band, cfg=cfg, E=E, z=z, G=G, cells=cells, sigma=sigma,
                T=T, Gw=T @ G, dw_real=T @ (d - z['b']), noise=noise, noise_w=(T @ noise.T).T,
                edges=cells['edges'], clon=cells['clon'], clat=cells['clat'], area=cells['area_km2'])
