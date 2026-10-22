"""
A synthetic truth with detail at the scales that higher degrees should resolve (the future-data test):
  provinces    the hast_var truth of realistic_synthetic.npz (one value per Hasterok province, times
               smooth variation of 25% with ~200 km correlation length)
  fine detail  times lognormal variation of 20% with ~50 km correlation length (about one cell)
  bodies       N_BODIES discs, diameters uniform 50-150 km, VIS raised by BODY_VIS (SI.km; e.g. a mafic
               intrusion of susceptibility 0.03 SI through 30 km of crust), centres at least 150 km inside
               the cell area and apart from each other by the sum of the radii plus 100 km
Values clipped to 0.01-4 SI.km, as the other truths.
Spherical geometry: all distances are great-circle distances between cell centres (unit vectors).
Output: multiscale_truth.npz in the region cache (truth, its parts, body of each cell (-1: none), body
centres and diameters), read by run_linear_models.py and run_realistic_synthetic.py as case 'hast_ms'.

Run from notebooks/ (conda env pygmt17):
    PYTHONPATH=..:$PYTHONPATH python make_multiscale_truth.py regions/australia_fine/config.toml
"""
import sys

import numpy as np

from domains import problem
from domains.basis import unit

N_BODIES = 40
DIAMETER_KM = (50., 150.)
BODY_VIS = 0.8
FINE_SD, FINE_KM = 0.20, 50.
SEED = 20261008
R_KM = 6371.

pb = problem.load(sys.argv[1])
C = pb['cache']
rs = np.load(f'{C}/realistic_synthetic.npz', allow_pickle=True)
area, edges = pb['area'], pb['edges']
xyz = unit(pb['clon'], pb['clat'])
dist = np.arccos(np.clip(xyz @ xyz.T, -1, 1))*R_KM
n = len(area)
rng = np.random.default_rng(SEED)

# fine detail: exponential covariance along great circles
fine = np.linalg.cholesky(np.exp(-dist/FINE_KM) + 1e-8*np.eye(n)) @ rng.standard_normal(n)
base = rs['truth_hast_var']*np.exp(FINE_SD*fine - 0.5*FINE_SD**2)

# bodies: centres away from the edge of the cell area (cells with fewer neighbours than the most common count)
deg = np.bincount(edges.ravel(), minlength=n)
edge_cells = deg < np.bincount(deg).argmax()
inner = dist[:, edge_cells].min(1) >= 150.
body = -np.ones(n, int)
centres, diam = [], []
for i in rng.permutation(np.flatnonzero(inner)):
    if len(centres) == N_BODIES:
        break
    d_k = rng.uniform(*DIAMETER_KM)
    if all(dist[i, j] >= 0.5*(d_k + dj) + 100. for j, dj in zip(centres, diam)):
        inside = np.flatnonzero(dist[i] <= 0.5*d_k)
        inside = inside if len(inside) else np.array([i])
        body[inside] = len(centres)
        centres.append(i)
        diam.append(d_k)
truth = np.clip(base + BODY_VIS*(body >= 0), 0.01, 4.)

print(f'{n} cells; {len(centres)} bodies, diameters {min(diam):.0f}-{max(diam):.0f} km, '
      f'{np.bincount(body[body >= 0]).min()}-{np.bincount(body[body >= 0]).max()} cells each; '
      f'truth {truth.min():.2f}-{truth.max():.2f} SI.km, area mean {np.average(truth, weights=area):.2f}')
rms = lambda v: pb['sigma']*np.sqrt(np.sum((pb['Gw'] @ v)**2)/len(pb['z']['plon']))
print(f'field of the truth in the data subspace (first build): {rms(truth):.0f} nT '
      f'(hast_var {rms(rs["truth_hast_var"]):.0f} nT)')
np.savez(f'{C}/multiscale_truth.npz', truth=truth, provinces=rs['truth_hast_var'], fine=fine, body=body,
         body_centre=np.array(centres), body_diameter_km=np.array(diam), body_vis=BODY_VIS, fine_sd=FINE_SD,
         fine_km=FINE_KM, prov=rs['prov'])
print(f'saved {C}/multiscale_truth.npz')
