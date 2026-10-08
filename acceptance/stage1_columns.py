"""
Stage 1 acceptance, columns (docs/inversion-toolbox-spec.md): the column store for all continental crust
(NSIDE 64, top of basement to 30 km, IGRF-13, lmax 185) against the existing columns:
  - GCM-GLC node-sum columns (notebooks/regions/global_continents/cache/G, basis.direct_columns): every cell
  - Australian transform columns (notebooks/regions/australia/cache/cells_l185, one global transform per cell):
    the cells whose grid nodes are the same in both (interior cells; edge cells of the Australian extent differ)
and the mode agreement of the columns: the explore store (GPU float32) against the final store (float64).

Run from the repository root (conda env curiesphere):
    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 15 python acceptance/stage1_columns.py
"""
import os
import sys

import numpy as np

from remit.inversion import cache, columns, geometry

NB = 'notebooks/regions'
LAYER, DIRECTION, LMAX = {'top': 'basement', 'base': 30.0}, 'induced_igrf13', 185
store_cache = cache.Cache()
cells = geometry.make_cells(geometry.continental_crust(), 64)
final = columns.ColumnStore(cells, 'continental_crust', LAYER, DIRECTION, LMAX, 'numpy', store_cache).build()
print(f'{len(cells)} cells; final store {final.dir}')

# GCM-GLC columns (blocks of 200 cells in the same cell order: same mask, same NSIDE)
old = np.load(f'{NB}/global_continents/cache/cells.npz')
assert np.array_equal(old['cell_ids'], cells.ids) and np.array_equal(old['cell'], cells.cell)
worst = 0.
for b in range(len(final.blocks)):
    ref = np.load(f'{NB}/global_continents/cache/G/{b:04d}.npy')
    new = final.get(final.blocks[b])
    worst = max(worst, np.abs(new - ref).max()/np.abs(ref).max())
print(f'GCM-GLC columns, all {len(cells)} cells: max relative difference {worst:.1e}')
ok = worst < 1e-12

# Australian transform columns, for the cells with the same grid nodes
aus = np.load(f'{NB}/australia/cache/cells.npz')
acell, aids = aus['cell'], aus['cell_ids']
same = []
for k, cid in enumerate(aids):
    j = np.searchsorted(cells.ids, cid)
    if j < len(cells) and cells.ids[j] == cid and np.array_equal(acell == k, cells.cell == j):
        same.append((k, j))
same = same[::max(1, len(same)//200)]                              # a sample of ~200 interior cells
L, M = columns.packed_lm(LMAX)
worst = 0.
for k, j in same:
    ref = np.load(f'{NB}/australia/cache/cells_l185/{aids[k]}.npy')[:, L, M]
    worst = max(worst, np.abs(final.get([j])[:, 0] - ref).max()/np.abs(ref).max())
print(f'Australian transform columns, {len(same)} interior cells: max relative difference {worst:.1e}')
ok &= worst < 1e-10

# mode agreement of the columns
explore = columns.ColumnStore(cells, 'continental_crust', LAYER, DIRECTION, LMAX, 'mps', store_cache).build()
if explore.backend == 'mps':
    worst = 0.
    for b in range(len(final.blocks)):
        a, e = final.get(final.blocks[b]), explore.get(explore.blocks[b])
        worst = max(worst, np.abs(a - e).max()/np.abs(a).max())
    print(f'explore (GPU float32) against final columns: max relative difference {worst:.1e}')
    ok &= worst < 1e-5
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
