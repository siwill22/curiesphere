"""
Null calibration of DF scoring FD-merge's path (df_aus_lcs1_16-133_gcmbg_gm_mpath.toml): the partitions come from the
same data that score them, so the log evidence rises with K whatever the truth. Here the truth has no boundaries: data
drawn from DF's one-domain fit to LCS-1 (its v0, strength, beta, ell and ring error; noise as the error model). Each
null data set gets the same merge path and DF scoring, plus the 39 polygons and the random tiles, so the gain over one
domain that chance alone gives at each K can be set against the real data's.
Run from the repo root (env curiesphere): python acceptance/df_merge_path_null.py [n_draws] [workers] [config stem]
(default stem df_aus_lcs1_16-133_gcmbg_gm_mpath, scored at 2-100 domains; other stems use their config's K list).
Writes inversions/results/<stem>_null.json (Australia: df_aus_mpath_null.json, the first run).
"""
import json
import multiprocessing as mp
import sys
import time

import numpy as np

from remit.inversion import config, errors, problem
from remit.inversion.cache import Cache
from remit.inversion.graph import unit
from remit.inversion.methods import df

N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
STEM = sys.argv[3] if len(sys.argv) > 3 else 'df_aus_lcs1_16-133_gcmbg_gm_mpath'
AUS = STEM == 'df_aus_lcs1_16-133_gcmbg_gm_mpath'
OUT = 'inversions/results/' + ('df_aus_mpath_null.json' if AUS else f'{STEM}_null.json')

cfg = config.load(f'inversions/{STEM}.toml')
if AUS:
    cfg['method']['layouts']['merge_path']['K'] = [2, 4, 8, 14, 20, 30, 50, 100]
KS = cfg['method']['layouts']['merge_path']['K']
me = cfg['method']
cache = Cache()
pb = problem.local(cfg, cache, print)
cells = pb.cells
lo, hi = me['bounds']
opts = dict(s_range=tuple(me['s_range']), beta_range=tuple(me['beta_range']), ell_range=tuple(me['ell_range']))
cell_km2 = 4*np.pi*df.R_KM**2/(12*cells.nside**2)

xyz_c, xyz_p = unit(cells.clon, cells.clat), unit(pb.plon, pb.plat)
ring = np.arccos(np.clip((xyz_p @ xyz_c.T).max(1), -1., 1.))*df.R_KM > me['ring_error']['beyond_km']
Wd = pb.W.toarray() if hasattr(pb.W, 'toarray') else np.asarray(pb.W)
Cr = errors.point_shape(pb.spectra['model'], pb.band, pb.plon[ring], pb.plat[ring])
extra = [Wd[:, ring] @ Cr @ Wd[:, ring].T]


def model(yw, label):
    return df.DomainFractal(pb.Gw, yw, label, cells.clon, cells.clat, cell_km2, lo, hi, extra, 'gaussian',
                            tuple(me.get('tau_range', [0.01, 4.])))


# the truth: DF's one-domain fit to the real data
one = np.zeros(len(cells), int)
m1 = model(pb.yw, one)
f1 = m1.fit(**opts)
o = m1._solve(f1['s'], f1['beta'], f1['ell'], 1., f1['sig'])
Lc, mean = np.tril(o['cf'][0]), m1.U @ o['theta']
print(f'one-domain fit to LCS-1: v0 {o["theta"][0]:.3f}, s {f1["s"][0]:.3f}, beta {f1["beta"]:.2f}, '
      f'ell {f1["ell"]:.0f} km, ring error {f1["sig"][0]:.1f} nT, log evidence {f1["ll"]:.1f}', flush=True)


def draw(i):
    t = time.time()
    rng = np.random.default_rng(20261012 + i)
    y = mean + Lc @ rng.standard_normal(len(mean))
    lays, _ = df.layouts(cells, cfg, cache, lambda s: None, pb.Gw, y)
    ll = {}
    for name, (label, _) in lays.items():
        ll[name] = model(y, label).fit(**opts)['ll']
    print(f'draw {i}: ' + ', '.join(f'{k} {v - ll["one domain"]:+.0f}' for k, v in ll.items() if k != 'one domain')
          + f' ({(time.time() - t)/60:.1f} min)', flush=True)
    return ll


if __name__ == '__main__':
    if WORKERS == 1:                   # in-process: forking after threaded BLAS can deadlock the child
        out = [draw(i) for i in range(N_DRAWS)]
    else:
        with mp.get_context('fork').Pool(WORKERS) as pool:
            out = pool.map(draw, range(N_DRAWS))
    json.dump(dict(truth=dict(v0=float(o['theta'][0]), s=float(f1['s'][0]), beta=f1['beta'], ell_km=f1['ell'],
                              sigma_ring_nT=float(f1['sig'][0])), K=KS, draws=out), open(OUT, 'w'), indent=1)
    print('wrote', OUT)
