"""
Covariates on the cells of a region (domains/covariates.py): crustal thickness, LAB depth, heat flow
(main and alternative products), the province class, age and group of each cell, and the H&M (+ slabs)
VIS per cell (for its mean only).
Output: covariates.npz in the region cache.

Run from notebooks/ (conda env pygmt17):
    PYTHONPATH=..:$PYTHONPATH python run_covariates.py regions/australia/config.toml
"""
import sys

import numpy as np

from domains import covariates, problem

pb = problem.load(sys.argv[1])
cells = pb['cells']
grids = covariates.grid_covariates(cells, pb['cfg']['covariates'])
prov = covariates.province_covariates(pb['clon'], pb['clat'])
hm = covariates.hm_cell_vis(cells)
for k, v in grids.items():
    print(f'{k:22s} min {np.nanmin(v):8.2f}  median {np.nanmedian(v):8.2f}  max {np.nanmax(v):8.2f}  '
          f'missing {np.sum(~np.isfinite(v))}')
print(f'cells mostly on AusMoho2012: {np.mean(grids["crust_from_ausmoho"] > 0.5):.0%}')
for c, name in enumerate(covariates.CLASS_NAMES):
    print(f'class {c} {name:24s} {np.sum(prov["cls"] == c)} cells')
print(f'undated cells {int(prov["undated"].sum())}; age range {prov["age"].min():.0f}-{prov["age"].max():.0f} Ma')
print(f'H&M (+ slabs) cell VIS: area-weighted mean {np.average(hm, weights=pb["area"]):.3f} SI.km')
np.savez(f'{pb["cache"]}/covariates.npz', **grids, **prov, hm_vis=hm)
