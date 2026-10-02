"""
Field of each small cell of uniform VIS: the basis for inversions where the domains form from
the data (merging cells, Voronoi), with no polygon boundaries imposed inside the continent.

Cells are HEALPix NSIDE pixels (equal area, ~100 km at NSIDE 64) over the same outer extent as
invert_domains.py: the union of its domain cells, i.e. non-ocean-age crust of Australia and its
margins, with PNG cut off. A cell's VIS is 1 SI.km spread uniformly from the top of basement to
BASE_DEPTH below the ellipsoid, as in invert_domains.py. The field of any partition into domains
is then the sum of its cells' columns.

Each cell's Gauss coefficients are saved as they finish (resumable). The final step assembles Br
at the data points of invert_domains.py and checks that the cells summed over each of its
domains reproduce that domain's column.

Run from notebooks/ (conda env pygmt17), after invert_domains.py:
    PYTHONPATH=..:$PYTHONPATH python cell_basis.py
For another region, after region_setup.py: REGION=north_america ... python cell_basis.py
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import time
import multiprocessing

from astropy_healpix import healpy as hp
import numpy as np
import pyshtools
from scipy.spatial import cKDTree

import depth_models as dm
from remit.data.models import load_ocean_age_model
from remit.utils.region import generate_healpix_points

REGION = os.environ.get('REGION', 'australia')     # another region: set up by region_setup.py
OUT = 'inversion' if REGION == 'australia' else f'inversion_{REGION}'
CELL_DIR = f'{OUT}/cells'
os.makedirs(CELL_DIR, exist_ok=True)
NSIDE_CELL = 64
LMAX_CELL = 150
LMIN, LMAX_FIT = 16, 133
BASE_DEPTH = 30e3
BUFFER_KM, NSIDE_DATA = 300., 128           # data points, as in invert_domains.py
N_WORKERS = int(os.environ.get('N_WORKERS', 8))
R0 = dm.R0

ocean = load_ocean_age_model()
glon, glat = np.asarray(ocean.lon), np.asarray(ocean.lat)
pgrid = np.load(f'{OUT}/param_grid.npz')['pgrid']
free = pgrid >= 0
LO, LA = np.meshgrid(glon, glat)

# cell index of every grid node in the extent
pix = np.full(pgrid.shape, -1, dtype=np.int64)
pix[free] = hp.ang2pix(NSIDE_CELL, LO[free], LA[free], lonlat=True)
cell_ids = np.unique(pix[free])
cell = np.full(pgrid.shape, -1, dtype=np.int32)
cell[free] = np.searchsorted(cell_ids, pix[free])
n_cell = len(cell_ids)
cell_area = np.bincount(cell[free], weights=np.cos(np.radians(LA[free])), minlength=n_cell)
km2 = (np.radians(0.1)*R0/1e3)**2
print(f'{n_cell} cells (NSIDE {NSIDE_CELL}, full cell {4*np.pi*(R0/1e3)**2/(12*NSIDE_CELL**2):.0f} km2); '
      f'cells under half full: {(cell_area*km2 < 0.5*4*np.pi*(R0/1e3)**2/(12*NSIDE_CELL**2)).sum()}')
np.savez(f'{OUT}/cells.npz', cell=cell, cell_ids=cell_ids, area_km2=cell_area*km2, nside=NSIDE_CELL)

r_top = dm.top_of_basement_radius(ocean)
r_base = dm.ellipsoid_radius(glat)[:, None]*np.ones_like(r_top) - BASE_DEPTH
to_gmm = dm.induced_to_gmm(glon, glat)


def compute(i):
    f = f'{CELL_DIR}/{cell_ids[i]}.npy'
    if not os.path.exists(f):
        c = dm.slab_vis_coeffs(np.where(cell == i, 1., 0.), r_top, r_base, to_gmm, lmax=LMAX_CELL).coeffs
        np.save(f + '.tmp.npy', c)
        os.replace(f + '.tmp.npy', f)
    return i


if __name__ == '__main__':
    todo = [i for i in range(n_cell) if not os.path.exists(f'{CELL_DIR}/{cell_ids[i]}.npy')]
    print(f'{len(todo)} cells to compute with {N_WORKERS} workers', flush=True)
    t = time.time()
    with multiprocessing.get_context('fork').Pool(N_WORKERS) as pool:   # workers share the grids
        for n, _ in enumerate(pool.imap_unordered(compute, todo), 1):
            if n % 50 == 0 or n == len(todo):
                el = time.time() - t
                print(f'  {n}/{len(todo)} cells, {el/60:.0f} min, ~{el/n*(len(todo)-n)/60:.0f} min left', flush=True)

    # Br at the data points (same selection as invert_domains.py)
    hlon, hlat = generate_healpix_points(NSIDE_DATA)
    hlon = np.mod(hlon, 360.)
    unit = lambda lon, lat: np.stack([np.cos(np.radians(lat))*np.cos(np.radians(lon)),
                                      np.cos(np.radians(lat))*np.sin(np.radians(lon)), np.sin(np.radians(lat))], -1)
    chord, _ = cKDTree(unit(LO[free], LA[free])).query(unit(hlon, hlat))
    keep = 2*np.arcsin(np.clip(chord/2, 0, 1))*R0/1e3 <= BUFFER_KM
    plon, plat = hlon[keep], hlat[keep]
    z = np.load(f'{OUT}/columns.npz')
    assert np.array_equal(z['plon'], plon), 'data points differ from invert_domains.py'

    F = dm.lcs1_filter(np.arange(LMAX_CELL+1), dm.fit_lh())

    def br(c, lmax_fit):
        c = np.array(c)[:, :lmax_fit+1, :lmax_fit+1]*F[None, :lmax_fit+1, None]
        c[:, :LMIN] = 0
        return np.asarray(pyshtools.SHMagCoeffs.from_array(c, r0=R0).expand(
            a=R0, lat=plat, lon=plon, lmax_calc=lmax_fit))[:, 0]

    C = np.array([np.load(f'{CELL_DIR}/{cid}.npy') for cid in cell_ids])

    def synthesis(lmax_fit):
        """matrices Yc, Ys (points x (l, m)) with Br = Yc @ g + Ys @ h at r = R0 (Schmidt, as pyshtools)"""
        z = np.sin(np.radians(plat))
        P = np.array([pyshtools.legendre.PlmSchmidt(lmax_fit, zi) for zi in z])     # [pt, lm]
        lm = [(l, m) for l in range(lmax_fit+1) for m in range(l+1)]
        L = np.array([l for l, _ in lm]); M = np.array([m for _, m in lm])
        assert P.shape[1] == len(lm)
        phi = np.radians(plon)[:, None]*M[None, :]
        w = (L+1)*F[L]*(L >= LMIN)
        return P*w*np.cos(phi), P*w*np.sin(phi), L, M

    G_out = {}
    for lmax_fit in (133, 150):
        Yc, Ys, L, M = synthesis(lmax_fit)
        G_out[lmax_fit] = Yc @ C[:, 0, L, M].T + Ys @ C[:, 1, L, M].T
        for i in [0, n_cell//2, n_cell-1]:                     # check against pyshtools expand
            ref = br(C[i], lmax_fit)
            assert np.abs(G_out[lmax_fit][:, i] - ref).max() < 1e-8*np.abs(ref).max(), 'synthesis mismatch'
    G133, G150 = G_out[133], G_out[150]
    print('synthesis matrix agrees with pyshtools expand')
    np.savez(f'{OUT}/cell_columns.npz', G133=G133, G150=G150, plon=plon, plat=plat, cell_ids=cell_ids)

    # check: cells summed over each domain of invert_domains.py reproduce the domain's column
    C_dom = np.load(f'{OUT}/column_coeffs.npz')['C_cols'][:, :, :LMAX_CELL+1, :LMAX_CELL+1]
    worst = 0.
    for k in range(C_dom.shape[0]):
        members = np.unique(cell[pgrid == k])
        # a cell straddling a domain boundary belongs to both, so compare on whole-domain grids instead
        whole = np.all([np.all(pgrid[cell == m] == k) for m in members])
        if whole:
            s = C[members].sum(0)
            worst = max(worst, np.abs(s - C_dom[k]).max()/np.abs(C_dom[k]).max())
    total = C.sum(0)
    rel_total = np.abs(total - C_dom.sum(0)).max()/np.abs(C_dom.sum(0)).max()
    print(f'check, all cells vs all domains: max rel diff {rel_total:.1e}; '
          f'domains made of whole cells: max rel diff {worst:.1e}')
    print(f'Br columns saved: {G133.shape[1]} cells x {G133.shape[0]} points')
