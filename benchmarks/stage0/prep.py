"""Stage 0 inputs (pygmt17): node geometry, Legendre rows, test cells and reference columns from basis.direct_columns."""
import os, time, numpy as np, pyshtools
from domains import basis
OUT = os.environ.get('STAGE0_DIR', os.path.expanduser('~/.cache/curiesphere/stage0'))
LMAX = 185
ocean = basis.ocean_grid()[0]
z = np.load('regions/global_continents/cache/cells.npz')
cell, clat = z['cell'], z['clat']
S = basis.direct_setup(ocean, 30., LMAX)
rng = np.random.default_rng(1)
test = np.unique(np.concatenate([rng.choice(len(clat), 200, replace=False), [np.argmin(clat), np.argmax(clat)]]))
t = time.time(); ref = basis.direct_columns(cell, test, S); dt = time.time() - t
rows_all = np.unique(np.nonzero(cell >= 0)[0])
P = np.zeros((S['nth'], 2, len(S['L'])))
for i in rows_all:
    p, dp = basis._legendre_row(S, i); P[i, 0], P[i, 1] = p, dp
nodes = sum((cell == c).sum() for c in test)
os.makedirs(OUT, exist_ok=True)
np.savez(f'{OUT}/inputs.npz', cell=cell, test=test, ref=ref, t_ref=dt, nodes_test=nodes, n_nodes_all=(cell >= 0).sum(),
         mr=S['comps'][0], mt=S['comps'][1], mp=S['comps'][2], r_top=S['r_top'], r_base=S['r_base'], w=S['w'], wsin=S['wsin'],
         dphi=S['dphi'], nth=S['nth'], x=S['x'], wq=S['wq'], L=S['L'], M=S['M'], R0=basis.R0,
         K=1e9*pyshtools.constants.mu0.value/(4*np.pi*basis.R0))
np.save(f'{OUT}/P.npy', P.astype(np.float32)); np.save(f'{OUT}/P64.npy', P)
print(f'{len(test)} test cells, {nodes} nodes: reference direct_columns {dt:.1f} s = {dt/nodes*1e3:.3f} ms/node; '
      f'all nodes {(cell >= 0).sum()} -> {dt/nodes*(cell >= 0).sum()/60:.0f} min single-threaded')
