"""
Stage 3 acceptance (docs/inversion-toolbox-spec.md): FD-merge for Australia against the first build
(notebooks/regions/australia/cache: merge_points.npz, 'fine' mode, and merge_points_sweep.npz, the gamma sweep), and
the mode agreement of explore against final.

  estimator   on the first build's whitened data with its own noise realisations (null runs 0-39, bootstrap 40-69):
              lambda_95, the null check, the partition of every setting (adjusted Rand index, K, chi2) and the
              bootstrap; the sweep's settings at that lambda_95
  end to end  the toolbox's configs, with its own noise draws: lambda_95 (a calibration on 20 random null sets, so it
              varies), K and the agreement of the partitions with the first build's
  modes       explore (NSIDE 32 cells) against final: K and partition agreement (final labels moved to the coarse
              cells by majority area). Reported for review, no pass/fail (ADR 0006).

Run from the repository root (conda env curiesphere):
    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 15 python acceptance/stage3.py
"""
import json
import os
import sys

import numpy as np

from remit.inversion import cache, config, geometry, results
from remit.inversion.methods import merge

OLD = 'notebooks/regions/australia/cache'
ok = True


def check(label, value, expected, tol):
    global ok
    good = abs(value - expected) <= tol
    ok &= good
    print(f'  {label:58s} {value:11.5g}  expected {expected:9.4g} +- {tol:g}  {"ok" if good else "FAIL"}', flush=True)


def result(name):
    path = f'inversions/results/{name}.nc'
    if not os.path.exists(path):
        results.save(merge.run(config.load(f'inversions/{name}.toml')), path)
    return results.load(path)


oc, od, E = np.load(f'{OLD}/cells.npz'), np.load(f'{OLD}/data.npz'), np.load(f'{OLD}/errors.npz')
P0 = np.linalg.svd(E['T'].T, full_matrices=False)[0]
sigma = float(np.sqrt(np.mean((od['d_lcs'] - od['d_mf7'])**2)))
Gw0 = P0.T @ np.load(f'{OLD}/G.npz')['G']/sigma
dw0 = P0.T @ (od['d_lcs'] - od['b'])/sigma
noise = np.load(f'{OLD}/noise.npz')['noise']
noise_w = (P0.T @ (noise*sigma/np.sqrt(np.mean(noise**2))).T).T/sigma
edges0 = oc['edges']
old = {k: np.load(f'{OLD}/merge_points{k}.npz', allow_pickle=True) for k in ('', '_sweep')}
cfg = {k: config.load(f'inversions/merge_aus_lcs1_16-133{k}.toml') for k in ('', '_sweep')}

print('estimator on the first build\'s data and noise realisations')
out = merge.calibrated(Gw0, dw0, edges0, list(noise_w[:70]), cfg['']['method'], say=lambda m: print('   ', m, flush=True))
check('lambda_95 (relative)', out['lam95']/float(old['']['fine_lam95']), 1, 1e-6)
check('null check: runs with K = 1', np.sum(out['null_K'] == 1), np.sum(old['']['fine_null_K'] == 1), 0)
for k in ('', '_sweep'):
    if k:
        me = dict(cfg[k]['method'], lambda_95=out['lam95'])
        o = merge.calibrated(Gw0, dw0, edges0, list(noise_w[:70]), me, say=lambda m: print('   ', m, flush=True))
    else:
        o = out
    aris = [merge.ari(r['label'], old[k]['fine_label'][i]) for i, r in enumerate(o['real'])]
    dK = [r['K'] - int(old[k]['fine_K'][i]) for i, r in enumerate(o['real'])]
    check(f'merge_points{k}: smallest ARI over the {len(aris)} settings', min(aris), 1, 1e-9)
    check(f'merge_points{k}: largest |K - first|', np.abs(dK).max(), 0, 0)
    check(f'merge_points{k}: bootstrap K differing from the first', np.sum(o['boot'][0]['K'] != old[k]['fine_boot0_K']), 0, 0)
    check(f'merge_points{k}: max |boundary probability - first|',
          np.abs(o['boot'][0]['prob'] - old[k]['fine_boot0_prob']).max(), 0, 1e-9)

print('end to end (toolbox configs, toolbox noise draws)')
for k in ('', '_sweep'):
    ds = result(f'merge_aus_lcs1_16-133{k}')
    rows = json.loads(ds.attrs['settings'])
    print(f'  merge_aus_lcs1_16-133{k}: lambda_95 {ds.attrs["lambda_95"]:.1f} (first {float(old[k]["fine_lam95"]):.1f}); '
          f'null runs with K = 1: {np.sum(np.array(json.loads(ds.attrs["null_K"])) == 1)}/20')
    for i, r in enumerate(rows):
        a = merge.ari(ds[f'domain_{i}'].values, old[k]['fine_label'][i])
        print(f'    lambda x{r["lam_rel"]:.2f}, gamma {r["gamma_rel"]:.3f}: K {r["K"]:2d} (first {int(old[k]["fine_K"][i]):2d}), '
              f'ARI with the first {a:.2f}')
    b = json.loads(ds.attrs['bootstrap'])[0]
    print(f'    bootstrap {b["setting"]}: K median {np.median(b["K"]):.0f} (first {np.median(old[k]["fine_boot0_K"]):.0f})')

print('mode agreement (explore against final)')
fin, exp = result('merge_aus_lcs1_16-133'), result('merge_aus_lcs1_16-133_explore')
cells = geometry.make_cells(geometry.target_mask(cfg['']['target']['area'], cache.Cache()), 64)
cells32, parent = geometry.coarsen(cells)
for i, r in enumerate(json.loads(fin.attrs['settings'])):
    lab = fin[f'domain_{i}'].values.astype(int)
    A = np.zeros((len(cells32), lab.max() + 1))
    np.add.at(A, (parent, lab), cells.area_km2)
    re = json.loads(exp.attrs['settings'])[i]
    print(f'  lambda x{r["lam_rel"]:.2f}, gamma {r["gamma_rel"]:.3f}: K explore {re["K"]:2d}, final {r["K"]:2d}; ARI '
          f'{merge.ari(exp[f"domain_{i}"].values, A.argmax(1)):.2f}')
print(f'  lambda_95: explore {exp.attrs["lambda_95"]:.1f}, final {fin.attrs["lambda_95"]:.1f}')
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
