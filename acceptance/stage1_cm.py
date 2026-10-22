"""
Stage 1 acceptance, CM (docs/inversion-toolbox-spec.md): the toolbox's CM-AUS against the first Australian build
(notebooks/regions/australia/cache: extent, cells, data, G, C_bg, errors; GMRF fit in linear_models.npz; L2 and TV
maps in map_sweep.npz), and the mode agreement of explore against final.

  problem      target mask, cells, cell graph, data points, G, background against the first build's caches;
               the compressed data subspace (cosines of the principal angles)
  estimators   on the first build's own whitened data (Gw, dw rebuilt from its caches as notebooks/domains/problem.py
               does): the GMRF evidence, its fit (h, s, ell, mean, s.d.) and the L2 and TV maps at the same beta
  end to end   the toolbox's CM-AUS run against the same results, in units of the first build's GMRF s.d.; it differs
               by design in the data (the field model is not filtered again) and slightly in the noise RMS
  modes        explore (NSIDE 32 cells summed from NSIDE 64, GPU float32 columns) against final on the NSIDE 32 cells,
               in units of the explore result's s.d. Reported for review, no pass/fail (ADR 0006).

Run from the repository root (conda env curiesphere):
    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 15 python acceptance/stage1_cm.py
"""
import os
import sys
import time

import numpy as np

from remit.inversion import cache, config, data, geometry, gmrf, graph, results
from remit.inversion.methods import cm

OLD = 'notebooks/regions/australia/cache'
CFG = 'inversions/cm_aus_lcs1_16-133.toml'
ok = True


def check(label, value, expected, tol):
    global ok
    good = abs(value - expected) <= tol
    ok &= good
    print(f'  {label:52s} {value:12.6g}  expected {expected:12.6g} +- {tol:g}  {"ok" if good else "FAIL"}', flush=True)


def result(name):
    path = f'inversions/results/{name}.nc'
    if not os.path.exists(path):
        results.save(cm.run(config.load(f'inversions/{name}.toml')), path)
    return results.load(path)


C = cache.Cache()
cfg = config.load(CFG)
oc, od, E = np.load(f'{OLD}/cells.npz'), np.load(f'{OLD}/data.npz'), np.load(f'{OLD}/errors.npz')
lm, sw = np.load(f'{OLD}/linear_models.npz'), np.load(f'{OLD}/map_sweep.npz')

print('problem')
mask = geometry.target_mask(cfg['target']['area'], C)
check('target nodes differing from the extent', (mask != np.load(f'{OLD}/extent.npz')['ext']).sum(), 0, 0)
cells = geometry.make_cells(mask, 64)
check('cells: ids differ', 0 if np.array_equal(cells.ids, oc['cell_ids']) else 1, 0, 0)
check('cells: max area difference (km2)', np.abs(cells.area_km2 - oc['area_km2']).max(), 0, 1e-6)
edges, _ = graph.cell_edges(cells)
check('cell graph: edges differ', 0 if np.array_equal(edges, oc['edges']) else 1, 0, 0)
plon, plat = data.points(mask, 128, 300.)
check('points: max position difference (deg)', max(np.abs(plon - od['plon']).max(), np.abs(plat - od['plat']).max()), 0, 0)

ds = result('cm_aus_lcs1_16-133')
from remit.inversion import problem
pb = problem.local(cfg, C, lambda m: None)
G0 = np.load(f'{OLD}/G.npz')['G']
check('G: max relative difference', np.abs(pb.G - G0).max()/np.abs(G0).max(), 0, 1e-10)
bg = data.background(C, mask, geometry.area_name(cfg['target']['area']))
ref = np.load(f'{OLD}/C_bg.npy')
check('background: max relative difference', np.abs(bg - ref).max()/np.abs(ref).max(), 0, 1e-8)
P_old = np.linalg.svd(E['T'].T, full_matrices=False)[0]
cos = np.linalg.svd(P_old.T @ (pb.W.T*pb.info['noise_rms']), compute_uv=False)
check('data subspace: dimension', pb.W.shape[0], P_old.shape[1], 0)
check('data subspace: smallest cosine of principal angles', cos.min(), 1, 1e-3)
y_old = od['d_lcs'] - od['b']
print(f'  (data: RMS difference {np.sqrt(np.mean((pb.y - y_old)**2)):.2f} nT of {np.sqrt(np.mean(y_old**2)):.1f} nT, from '
      f'not filtering the field model again; noise RMS {pb.info["noise_rms"]:.2f} nT against {np.sqrt(np.mean((od["d_lcs"] - od["d_mf7"])**2)):.2f})')

print('estimators on the first build\'s whitened data')
sigma = float(np.sqrt(np.mean((od['d_lcs'] - od['d_mf7'])**2)))
T = P_old.T/sigma
Gw, dw = T @ G0, T @ y_old
check('whitened data against linear_models real_dw (max abs)', np.abs(dw - lm['real_dw']).max(), 0, 1e-8)
area = oc['area_km2']
Lw = graph.laplacian(edges, 1/graph.edge_lengths_km(edges, oc['clon'], oc['clat'])**2, len(area))
D = graph.difference_operator(edges, oc['clon'], oc['clat'])
t = time.time()
gm = gmrf.GMRF(Gw, dw, area, Lw)
one = np.ones((len(area), 1))
fit = gm.fit(one, box=([0.], [4.]))
ell, ll, alpha2, h = float(fit['ell'][0]), fit['ll'], fit['alpha2'], fit['h']
H, S, EL = float(lm['real_gp_h']), float(lm['real_gp_s']), float(lm['real_gp_ell'])
solver_old, L_old = gm.model(one, EL)
print(f'  GMRF fit in {time.time() - t:.0f} s: noise x {h:.4f}, VIS s.d. {h/np.sqrt(alpha2):.4f} SI.km, length {ell:.0f} km '
      f'(first build {H:.4f}, {S:.4f}, {EL:.0f} km)')
check('GMRF noise scale h (relative)', h/H, 1, 2e-3)
check('log evidence at the first build\'s h, s, ell against its own', gm.evidence(one, H, S, EL, ([0.], [4.])),
      float(lm['real_gp_ll']), 1e-6)
check('log evidence: toolbox maximum - the first build\'s', ll - float(lm['real_gp_ll']), 0.5, 0.5)
sol = solver_old.solve(dw, H**2/S**2)
v = sol['v0'] + L_old @ sol['x']
sd = solver_old.posterior_sd_map(H**2/S**2, H, L_old)
check('GMRF mean at the first build\'s h, s, ell: max |diff| / s.d.', np.max(np.abs(v - lm['real_gp_v'])/lm['real_gp_sd']), 0, 1e-3)
check('GMRF s.d. at the first build\'s h, s, ell: max rel. diff', np.max(np.abs(sd/lm['real_gp_sd'] - 1)), 0, 1e-4)
hm = data.hm_cell_vis(cells)
hm_old = np.load(f'{OLD}/covariates.npz')['hm_vis']
check('H&M cell VIS: max difference (SI.km)', np.nanmax(np.abs(hm - hm_old)), 0, 1e-9)
hm_mean = float(np.average(hm, weights=area))
GtG = Gw.T @ Gw
for pen, ek, n_bis in (('l2', '200', 16), ('l2', 'gp', 16), ('tv', 'gp', 12)):
    key = f'real_{pen}_ell{ek}_b1'
    e, b = float(sw[f'{key}_ell']), float(sw[f'{key}_beta'])
    t = time.time()
    vm = cm.penalised(Gw, dw, area, D, e, b, hm_mean, pen, GtG=GtG)
    check(f'{pen} map, length {e:.0f} km, same beta: max |diff| (SI.km)', np.abs(vm - sw[f'{key}_v']).max(), 0, 1e-3)
    if pen == 'l2':
        b2 = cm.discrepancy(lambda q: cm.penalised(Gw, dw, area, D, e, q, hm_mean, pen, GtG=GtG), Gw, dw,
                            len(dw)*float(lm['real_gp_h'])**2, n_bisect=n_bis)[0]
        check(f'{pen} map, length {e:.0f} km: beta by discrepancy (relative)', b2/b, 1, 1e-6)
    print(f'    ({time.time() - t:.0f} s)')

print('end to end (toolbox CM-AUS against the first build)')
gs = lm['real_gp_sd']
print(f'  GMRF: noise x {ds.attrs["noise_scale_h"]:.3f} (first build {float(lm["real_gp_h"]):.3f}), VIS s.d. '
      f'{ds.attrs["prior_sd_s"]:.3f} ({float(lm["real_gp_s"]):.3f}), length {ds.attrs["length_km"]:.0f} km '
      f'({float(lm["real_gp_ell"]):.0f}), v0 {ds.attrs["v0"]:.3f}')
d = np.abs(ds.vis.values - lm['real_gp_v'])/gs
print(f'  GMRF mean: |difference| / first-build s.d.: median {np.median(d):.3f}, max {d.max():.3f}')
check('GMRF mean: max |difference| / first-build s.d.', d.max(), 0, 0.25)
for lab, key in (('l2', 'real_l2_ellgp_b1'), ('tv', 'real_tv_ellgp_b1'), ('l2_200km', 'real_l2_ell200_b1')):
    dv = ds[f'vis_{lab}'].values - sw[f'{key}_v']
    print(f'  {lab}: RMS difference {np.sqrt(np.average(dv**2, weights=area)):.3f} SI.km, max {np.abs(dv).max():.3f} '
          f'(map RMS {np.sqrt(np.average(sw[f"{key}_v"]**2, weights=area)):.2f})')

print('mode agreement (explore against final, NSIDE 32 cells)')
exp = result('cm_aus_lcs1_16-133_explore')
cells32, parent = geometry.coarsen(cells)
assert np.array_equal(cells32.ids, exp.cell_id.values)
fin32 = np.bincount(parent, ds.vis.values*area)/np.bincount(parent, area)
d = np.abs(exp.vis.values - fin32)/exp.vis_sd.values
print(f'  GMRF: explore length {exp.attrs["length_km"]:.0f} km, v0 {exp.attrs["v0"]:.3f}; final {ds.attrs["length_km"]:.0f} km, '
      f'v0 {ds.attrs["v0"]:.3f}')
print(f'  |VIS explore - VIS final (area mean of the four children)| / explore s.d.: '
      f'median {np.median(d):.2f}, 95% {np.percentile(d, 95):.2f}, max {d.max():.2f}')
for lab in ('l2', 'tv'):
    f32 = np.bincount(parent, ds[f'vis_{lab}'].values*area)/np.bincount(parent, area)
    dv = exp[f'vis_{lab}'].values - f32
    print(f'  {lab}: RMS difference {np.sqrt(np.average(dv**2, weights=cells32.area_km2)):.3f} SI.km')
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
