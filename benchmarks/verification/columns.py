"""
Verification of the inversions' forward model, the cell columns (remit.inversion.columns), against the direct dipole
sum of reference.py.

A column is the Gauss coefficients of 1 SI.km of VIS spread uniformly through the layer (top of basement to 30 km
below the ellipsoid) in one HEALPix NSIDE 64 cell, magnetised along IGRF-13. The dipole sum puts one dipole per grid
node at each of several radii through the layer, with moment VIM (w_q/2) r_q^2 w dlon.

Cases
  cells          six single cells, VIS 1 SI.km each: the grid seam at lon 0, next to the north pole, Antarctica at
                 85S, the dateline, central Australia, deep ocean basement in the North Atlantic. The dipole sum
                 uses 32 Gauss-Legendre radii through the layer where the columns use 8, so the radial integral is
                 checked as well. Columns at lmax 185 (production), 400 and 899; and the columns at lmax 185 against
                 one global remit transform per cell (columns.transform_columns), coefficient by coefficient.
  cont_checker   every continental cell (22,591) from the inversions' own cached column store (float64; and the
                 float32 GPU store, compared with the float64 one), VIS 1.5 / 0.5 in a HEALPix NSIDE 4 checkerboard
  cont_geology   the same with the Hemant & Maus VIS averaged over each cell
                 For both, the dipole sum uses the columns' 8 radii, and remit's depth series (remit.depth
                 .slab_vis_coeffs, lmax 600) gives the field above degree 185.

As in forward.py, columns(lmax) - dipole sum must equal minus the field of the same sources above lmax.
Run from the repository root (about 1.5 h at 3 threads; dipole sums cached):
    PYTHONPATH=benchmarks/verification:benchmarks:$PYTHONPATH OMP_NUM_THREADS=3 \
        nice -n 15 conda run -n curiesphere python benchmarks/verification/columns.py [case ...]
Outputs: benchmarks/verification/results/columns_<case>.nc and columns_summary.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from astropy_healpix import healpy as hp

import forward
import models
from forward import ALTITUDES, G, cached, log, rows, summarise, save
from reference import R0, DipoleSum, obs_grid, remit_br
from remit import depth
from remit.inversion import columns, geometry
from remit.inversion.cache import Cache

LAYER = {'top': 'basement', 'base': 30.0}
DIRECTION = 'induced_igrf13'
NSIDE = 64
CELLS = [('equator at lon 0 (the grid seam)', 0.0, 0.0),
         ('next to the north pole', 89.6, 0.0),
         ('Antarctica, 85S', -85.0, 120.0),
         ('30S on the dateline', -30.0, 180.0),
         ('central Australia', -25.0, 135.0),
         ('North Atlantic (deep basement)', 25.0, 320.0)]


def coeffs_array(packed, lmax):
    """packed (2, n_lm) Gauss coefficients -> pyshtools layout (2, lmax+1, lmax+1)"""
    L, M = columns.packed_lm(lmax)
    c = np.zeros((2, lmax + 1, lmax + 1))
    c[0, L, M], c[1, L, M] = packed[0], packed[1]
    return c


def dipole_layer(obs, vis_nodes, mask, n_gl):
    """dipole sum of VIS (DH2 grid, SI.km) in the layer at n_gl Gauss-Legendre radii, induced by IGRF-13 (built
    independently of remit); mask (the full grid) selects the nodes"""
    g = geometry.grid()
    r_top, r_base = (r[:-1, :-1] for r in geometry.layer_radii(LAYER, mask, g))
    mask = mask[:-1, :-1]
    vim = models.induced(G, vis_nodes)
    ds = DipoleSum(obs, ALTITUDES)
    x, wq = np.polynomial.legendre.leggauss(n_gl)
    for xq, w in zip(x, wq):
        r = r_top + (r_base - r_top)*(xq + 1)/2
        ds.add(G, *(c*w/2 for c in vim), r, sel=mask)
    return ds.br()


# ---------------------------------------------------------------- single cells

def run_cells():
    g = geometry.grid()
    LO, LA = np.meshgrid(g.lon, g.lat)
    pix = [hp.ang2pix(NSIDE, lon, lat, lonlat=True) for _, lat, lon in CELLS]
    mask = np.isin(hp.ang2pix(NSIDE, LO, LA, lonlat=True), pix)
    mask[-1, :] = mask[:, -1] = False                      # the DH sums leave out the south-pole row and lon 360
    cells = geometry.make_cells(mask, NSIDE, g)
    idx = np.arange(len(cells))
    obs = obs_grid(0.5)                                    # a cell's field is a few degrees across
    L_list = [185, 400, 899]

    S = {L: columns.Setup(LAYER, DIRECTION, L, mask, g) for L in L_list}
    cols = {L: cached(f'columns_cells_node_sum_{L}', lambda L=L: columns.node_sum(cells.cell, idx, S[L]))
            for L in L_list}
    ref = cached('columns_cells_transform_185', lambda: columns.transform_columns(cells.cell, idx, S[185], g))
    per_cell = np.abs(cols[185] - ref).max(axis=(0, 2))/np.abs(ref).max(axis=(0, 2))
    order = np.searchsorted(cells.ids, pix)                # cells are numbered by HEALPix id, not as in CELLS
    for k, (label, *_) in zip(order, CELLS):
        log(f'  {label}: columns vs transform, max relative difference {per_cell[k]:.1e}')
    rows.append(dict(case='cells', altitude_km=np.nan, lmax=185, check='columns vs remit transform (max rel)',
                     value=per_cell.max()))

    dip = cached('columns_cells_dipole_step0.5', lambda: dipole_layer(obs, (cells.cell >= 0)[:-1, :-1]*1., mask, 32))
    rem = {L: remit_br(coeffs_array(cols[L].sum(1), L), obs, ALTITUDES, L) for L in L_list}
    summarise('cells', dip, rem, L_list)
    vis = np.where(mask[:-1, :-1], 1., 0.)
    extra = dict(dipoles=pd.DataFrame(dict(label=[c[0] for c in CELLS], lat=cells.clat[order], lon=cells.clon[order],
                                           direction='cell')))
    save('cells', obs, dip, rem, ('VIS', 'SI·km', vis), extra, prefix='columns')


# ---------------------------------------------------------------- all continental cells

def continental_vis(pattern, cells, g):
    """VIS per cell: a HEALPix NSIDE 4 checkerboard of 1.5 / 0.5, or Hemant & Maus averaged over the cell"""
    if pattern == 'checker':
        pix = hp.ang2pix(4, cells.clon, cells.clat, lonlat=True, nest=True) % 16
        x = sum(((pix >> (2*b)) & 1) << b for b in range(2))
        y = sum(((pix >> (2*b + 1)) & 1) << b for b in range(2))
        return np.where((x + y) % 2 == 0, 1.5, 0.5)
    from remit.data.models import load_vis_model
    vis = np.asarray(load_vis_model(name='Hemant2005+slabs', match=(g.ocean.lon, g.ocean.lat, g.ocean.age)).vis, float)
    m = cells.cell >= 0
    w = np.cos(np.radians(g.lat))[:, None]*np.ones_like(vis)
    return np.bincount(cells.cell[m], weights=(np.nan_to_num(vis)*w)[m], minlength=len(cells)) / \
        np.bincount(cells.cell[m], weights=w[m], minlength=len(cells))


def run_continents(pattern):
    case = f'cont_{pattern}'
    g = geometry.grid()
    mask = geometry.continental_crust(g)
    cells = geometry.make_cells(mask, NSIDE, g)
    v = continental_vis(pattern, cells, g)
    vis = np.where(mask, v[np.maximum(cells.cell, 0)], 0.)
    obs = obs_grid(1.0)
    L, L_hi = 185, 600

    def column_coeffs(backend):
        store = columns.ColumnStore(cells, 'continental_crust', LAYER, DIRECTION, L, backend, Cache()).build(log)
        acc = np.zeros((2, (L + 1)*(L + 2)//2))
        for idx in store.blocks:
            acc += np.einsum('c,kcj->kj', v[idx], store.get(idx))
        return coeffs_array(acc, L)

    c64 = cached(f'columns_{case}_float64', lambda: column_coeffs('numpy'))
    c32 = cached(f'columns_{case}_float32', lambda: column_coeffs('mps'))
    r_top, r_base = geometry.layer_radii(LAYER, mask, g)
    c_hi = cached(f'columns_{case}_transform_{L_hi}', lambda: depth.slab_vis_coeffs(
        vis, r_top, r_base, depth.induced_to_gmm(g.lon, g.lat), lmax=L_hi, n_gl=columns.N_GL).coeffs)
    for name, c in (('float64', c64), ('float32', c32)):
        rel = np.abs(c - c_hi[:, :L + 1, :L + 1]).max()/np.abs(c_hi[:, :L + 1, :L + 1]).max()
        log(f'  {name} columns vs remit transform: max relative difference {rel:.1e}')
        rows.append(dict(case=case, altitude_km=np.nan, lmax=L, check=f'{name} columns vs remit transform (max rel)',
                         value=rel))

    dip = cached(f'columns_{case}_dipole', lambda: dipole_layer(obs, vis[:-1, :-1], mask, columns.N_GL))
    rem = {L: remit_br(c64, obs, ALTITUDES, L), L_hi: remit_br(c_hi, obs, ALTITUDES, L_hi)}
    summarise(case, dip, rem, [L, L_hi])
    r32 = remit_br(c32, obs, ALTITUDES, L)
    for ia, h in enumerate(ALTITUDES):                     # the GPU store against the float64 one, field by field
        d = r32[ia] - rem[L][ia]
        rows.append(dict(case=case, altitude_km=np.nan, lmax=L,
                         check=f'float32 minus float64 columns, Br at {h/1e3:.0f} km (max, relative to the RMS field)',
                         value=np.abs(d).max()/np.sqrt(np.mean(dip[ia]**2))))
    show = ('VIS', 'SI·km', np.where(mask[:-1, :-1], vis[:-1, :-1], np.nan))
    save(case, obs, dip, rem, show, {}, prefix='columns')


CASES = ['cells', 'cont_checker', 'cont_geology']

if __name__ == '__main__':
    for case in sys.argv[1:] or CASES:
        log(case)
        run_cells() if case == 'cells' else run_continents(case.removeprefix('cont_'))
        df = pd.DataFrame(rows)
        f = os.path.join(forward.RESULTS, 'columns_summary.csv')
        if os.path.exists(f):
            old = pd.read_csv(f)
            df = pd.concat([old[~old.case.str.startswith(case)], df])
        df.to_csv(f, index=False, float_format='%.3e')
        rows.clear()
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(pd.read_csv(os.path.join(forward.RESULTS, 'columns_summary.csv')).to_string(index=False))
    log('done')
