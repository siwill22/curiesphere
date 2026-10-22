"""
Stage 1 acceptance, GCM (docs/inversion-toolbox-spec.md): the toolbox's GCM-GLC against the first GCM run
(notebooks/run_global_continents.py, results in notebooks/regions/global_continents/cache/lcs1_16-133/results.npz),
and the mode agreement of explore against final.

  background   data.background against the first run's C_bg.npy
  real data    evidence and discrepancy damping: v0, noise scale, prior s.d., VIS and its s.d.
  synthetic    uniform 1.0 and 1.5, uniform 1.0 with an ocean-model error, 0.04 x Crust1.0: v0 and VIS
  modes        explore (NSIDE 32 cells summed from NSIDE 64, GPU float32 columns and products, Cholesky) against
               final, both on the NSIDE 32 cells: differences in units of the explore result's posterior s.d.
               Reported for review, no pass/fail (ADR 0006).

Run from the repository root (conda env curiesphere), after the configs in inversions/ have been run or not
(missing results are computed; existing ones in inversions/results/ are reused):
    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 15 python acceptance/stage1_gcm.py
"""
import os
import sys

import numpy as np

from remit.inversion import cache, config, data, geometry, results
from remit.inversion.methods import gcm

OLD = 'notebooks/regions/global_continents/cache'
old = np.load(f'{OLD}/lcs1_16-133/results.npz')
ok = True


def check(label, value, expected, tol):
    global ok
    good = abs(value - expected) <= tol
    ok &= good
    print(f'  {label:42s} {value:10.4f}  expected {expected:10.4f} +- {tol:g}  {"ok" if good else "FAIL"}')


def result(name):
    path = f'inversions/results/{name}.nc'
    if not os.path.exists(path):
        results.save(gcm.run(config.load(f'inversions/{name}.toml')), path)
    return results.load(path)


print('background')
mask = geometry.continental_crust()
bg = data.background(cache.Cache(), mask, 'continental_crust')
ref = np.load(f'{OLD}/C_bg.npy')
check('max relative difference from the first run', np.abs(bg - ref).max()/np.abs(ref).max(), 0., 1e-8)

print('real data')
for name, key in [('gcm_glc_lcs1_16-133', 'real_evid'), ('gcm_glc_lcs1_16-133_disc', 'real_disc')]:
    ds = result(name)
    print(f' {name}')
    check('v0 (SI.km)', ds.attrs['v0'], float(old[f'{key}_v0']), 1e-4)
    check('noise scale sigma', ds.attrs['sigma'], float(old[f'{key}_sigma']), 1e-3)
    check('alpha^2 (relative)', ds.attrs['alpha2']/float(old[f'{key}_alpha2']), 1., 1e-2)
    sd = old[f'{key}_sd']
    check('max |VIS - first run| / s.d.', np.max(np.abs(ds.vis.values - old[f'{key}_m'])/sd), 0., 1e-2)
    check('max relative difference of the s.d.', np.max(np.abs(ds.vis_sd.values/sd - 1)), 0., 1e-2)

print('synthetic data')
for name, key in [('u1.0', 'syn_u1.0'), ('u1.5', 'syn_u1.5'), ('u1.0_oceanerr', 'syn_u1.0_oceanerr'),
                  ('crust', 'syn_crust')]:
    ds = result(f'gcm_glc_syn_{name}')
    a = ds.area_km2.values
    check(f'{name}: v0', ds.attrs['v0'], float(old[f'{key}_evid_v0']), 2e-3)
    check(f'{name}: RMS(VIS - truth)', np.sqrt(np.average((ds.vis.values - ds.truth.values)**2, weights=a)),
          np.sqrt(np.average((old[f'{key}_evid_m'] - old[f'{key}_truth'])**2, weights=a)), 1e-2)

print('mode agreement (explore against final, NSIDE 32 cells)')
fin = result('gcm_glc_lcs1_16-133')
exp = result('gcm_glc_lcs1_16-133_explore')
cells64 = geometry.make_cells(mask, 64)
cells32, parent = geometry.coarsen(cells64)
assert np.array_equal(cells32.ids, exp.cell_id.values)
a64 = fin.area_km2.values
fin32 = np.bincount(parent, fin.vis.values*a64)/np.bincount(parent, a64)
d = np.abs(exp.vis.values - fin32)/exp.vis_sd.values
print(f'  v0: explore {exp.attrs["v0"]:.4f}, final {fin.attrs["v0"]:.4f}; '
      f'prior s.d. explore {exp.attrs["tau"]:.3f}, final {fin.attrs["tau"]:.3f} SI.km (per own cell size)')
print(f'  |VIS explore - VIS final (area mean of the four children)| / explore s.d.: '
      f'median {np.median(d):.2f}, 95% {np.percentile(d, 95):.2f}, max {d.max():.2f}')
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
