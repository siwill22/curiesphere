"""
Build the shared engine for a region (docs/domain-inversion-framework.md, step 1): extent, cells,
cell columns, data points, background, data, noise. Every step is cached in <region>/cache/,
so the script can be rerun to resume.

Run from notebooks/ (conda env pygmt17), with single-threaded BLAS for the parallel cell columns:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    PYTHONPATH=..:$PYTHONPATH python build_region.py regions/australia/config.toml
"""
import os
import sys
import tomllib

import numpy as np
from astropy_healpix import healpy as hp

from domains import region, basis, outputs

cfg_path = sys.argv[1]
with open(cfg_path, 'rb') as f:
    cfg = tomllib.load(f)
REG = os.path.dirname(os.path.abspath(cfg_path))
CACHE = f'{REG}/cache'
os.makedirs(CACHE, exist_ok=True)
lmin, lmax_fit = cfg['lmin'], cfg['lmax_fit']

ocean, glon, glat, age = basis.ocean_grid()

# 1. extent
f = f'{CACHE}/extent.npz'
if not os.path.exists(f):
    ext = region.extent_grid(f'{REG}/{cfg["outline"]}', glon, glat, age)
    np.savez_compressed(f, ext=ext)
ext = np.load(f)['ext']
print(f'extent: {ext.sum()} nodes, '
      f'{(np.cos(np.radians(glat))[:, None]*ext).sum()*basis.KM2_PER_NODE/1e6:.2f} M km2')

# 2. cells, cell graph, coarse parents
f = f'{CACHE}/cells.npz'
if not os.path.exists(f):
    cell, cell_ids, area = basis.make_cells(ext, glon, glat, cfg['nside_cell'])
    clon, clat = (np.asarray(x, float) for x in hp.pix2ang(cfg['nside_cell'], cell_ids, lonlat=True))
    edges, bridges = outputs.cell_edges(cell, clon, clat)
    parent, parent_ids = basis.coarse_parents(cell_ids, cfg['nside_cell'])
    np.savez_compressed(f, cell=cell, cell_ids=cell_ids, area_km2=area, nside=cfg['nside_cell'], clon=clon,
                        clat=clat, edges=edges, bridges=bridges, parent=parent, parent_ids=parent_ids)
z = np.load(f)
cell, cell_ids = z['cell'], z['cell_ids']
print(f'{len(cell_ids)} cells (NSIDE {cfg["nside_cell"]}), {len(z["parent_ids"])} coarse cells '
      f'(NSIDE {cfg["nside_cell"]//2}), {len(z["edges"])} edges, {len(z["bridges"])} bridges')

# 3. data points, background, data
f = f'{CACHE}/data.npz'
if not os.path.exists(f):
    plon, plat = basis.data_points(ext, glon, glat, cfg['nside_data'], cfg['buffer_km'])
    C_bg, b, d_lcs, d_mf7 = basis.background_and_data(ext, ocean, plon, plat, lmin, lmax_fit)
    np.save(f'{CACHE}/C_bg.npy', C_bg)
    np.savez(f, plon=plon, plat=plat, b=b, d_lcs=d_lcs, d_mf7=d_mf7)
z = np.load(f)
plon, plat = z['plon'], z['plat']
print(f'{len(plon)} data points; RMS(LCS-1 - MF7) = {np.sqrt(np.mean((z["d_lcs"] - z["d_mf7"])**2)):.2f} nT')

# 4. noise realisations (unscaled)
f = f'{CACHE}/noise.npz'
if not os.path.exists(f):
    np.savez(f, noise=basis.noise_realisations(plon, plat, lmin, lmax_fit, cfg['n_noise'], cfg['seed']))

# 5. cell columns and G (columns of unchanged cells can be reused from an earlier build: reuse_cells
#    in the config names that build's cells.npz and column directory, relative to the region)
if 'reuse_cells' in cfg:
    n = basis.reuse_cell_columns(f'{REG}/{cfg["reuse_cells"]["cells_npz"]}', f'{REG}/{cfg["reuse_cells"]["dir"]}',
                                 cell, cell_ids, f'{CACHE}/cells')
    print(f'{n} cell columns reused')
basis.compute_cell_columns(cell, cell_ids, ocean, f'{CACHE}/cells', cfg['base_depth_km'], cfg['lmax_cell'],
                           int(os.environ.get('N_WORKERS', 8)))
f = f'{CACHE}/G.npz'
if not os.path.exists(f):
    G = basis.cell_matrix(f'{CACHE}/cells', cell_ids, plon, plat, lmin, lmax_fit)
    np.savez(f, G=G, plon=plon, plat=plat, cell_ids=cell_ids)
print(f'G: {np.load(f)["G"].shape} (points x cells)')
