"""
Tune the GK07 ocean magnetization model for the best pattern match to LCS-1.

Method (agreed 2026-09-29):
  - objective: pooled Pearson correlation of Br over equal-area (HEALPix) points
    in the 10 ocean regions of gis/OceanMasks.geojson, excluding the two Pacific
    Triangles; Br at 0 km (r0 = 6371.0 km), spherical-harmonic degrees 16-100.
    300 km, same band, is reported as a check.
  - free: the 3 GK07 layer magnetizations w_k (A/m), P, lambda, and one global
    effective source depth d for the remanence. The induced part (VIS) is fixed.
  - staged, split by seafloor age:
      stage 1 (w, d):          crust >= 20 Ma and > 100 km (great circle) from crust < 20 Ma
      stage 2 (lambda, P):     the other dated points in the 10 regions (young crust + buffer)
      stage 3:                 short joint polish on all points
  - no separate amplitude step: the RMS ratio to LCS-1 is reported.

Everything tuned except lambda and d is linear, so the script caches Br at the
sample points for a set of unit basis fields and recombines them:
    Br = VIS + sum_k w_k [ B0_k + P * BP_k(lambda) ]
with the depth factor ((r0-d)/r0)^(l+2) applied to the remanent coefficients.

Result: the pattern match gains little over GK07 as published; layers 1 and 2 are
not separable and lambda is unresolved. GK07 was kept, and the near-ridge terms
were retuned separately in tune_near_ridge.py (see
docs/review-2026-09-units-and-geometry.md).

Run from notebooks/:
    conda run -n pygmt17 python tune_ocean_model.py
Outputs go to notebooks/tuning/ (the .npz caches are not tracked).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import pygmt
import pygplates
import pyshtools
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import cKDTree

sys.path.insert(0, '..')
from remit.data.models import load_ocean_age_model, load_vis_model, create_vim, load_lcs
from remit.utils.region import load_mask_features, generate_healpix_points

OUT = 'tuning'
os.makedirs(OUT, exist_ok=True)

FIT_REGIONS = ['CentralAtlantic', 'SouthAtlanticN', 'SouthAtlanticS', 'SouthwestIndianOcean',
               'Wharton-BayofBengal', 'SoutheastIndianOcean', 'PacificAntarcticRidge',
               'EastPacificRidgeS', 'EastPacificRidgeN', 'NEPacific']
REPORT_ONLY = ['PacificTriangleS', 'PacificTriangleN']
LMIN, LMAX = 16, 100
R0 = 6371000.                     # remit and the paper's LCS-1 both use r0 = 6371.0 km
ALTITUDES = {'0km': 0., '300km': 300e3}
AGE_CUT = 20.                     # Ma
BUFFER_KM = 100.
NSIDE = 128                       # HEALPix, ~55 km spacing
LAYERS = [0, 500, 1500, 6500]     # GK07 layer boundaries, m
GK07 = {'w': np.array([5., 2.3, 1.2]), 'P': 5., 'lmbda': 3.}
LAMBDAS = [0.5, 1., 2., 3., 5.]   # Ma; capped so the 20 Ma cut exceeds 4 x lambda
DEPTHS_KM = [0., 2., 4., 6., 8., 10.]
W_MAX, P_MAX = 50., 50.


# ============================================================ 1. sample points
lon, lat = generate_healpix_points(NSIDE)
lon = np.mod(lon, 360.)
xyz = np.stack([np.cos(np.radians(lat))*np.cos(np.radians(lon)),
                np.cos(np.radians(lat))*np.sin(np.radians(lon)),
                np.sin(np.radians(lat))], 1)

masks = load_mask_features('./gis/OceanMasks.geojson')
region = np.full(len(lon), '', dtype=object)
for _, feature in masks.iterrows():
    if feature.NAME not in FIT_REGIONS + REPORT_ONLY:
        continue
    vertices = list(zip(feature.geometry.exterior.coords.xy[1], feature.geometry.exterior.coords.xy[0]))
    polygon = pygplates.PolygonOnSphere(vertices)
    # prefilter with a spherical cap around the polygon, then test exactly
    c = polygon.get_boundary_centroid().to_xyz()
    radius = max(pygplates.GeometryOnSphere.distance(polygon.get_boundary_centroid(), pygplates.PointOnSphere(v))
                 for v in vertices)
    candidates = np.where((xyz @ np.array(c) >= np.cos(radius)) & (region == ''))[0]
    inside = [i for i in candidates if polygon.is_point_in_polygon(pygplates.PointOnSphere(lat[i], lon[i]))]
    region[inside] = feature.NAME
keep = region != ''
lon, lat, xyz, region = lon[keep], lat[keep], xyz[keep], region[keep]

ocean = load_ocean_age_model()
glat, glon = np.asarray(ocean.lat), np.asarray(ocean.lon)          # 0.1 deg, lat 90 -> -90, lon 0 -> 360
age_grid = np.asarray(ocean.age)
dlat, dlon = abs(glat[1] - glat[0]), abs(glon[1] - glon[0])
ilat = np.clip(np.round((glat[0] - lat)/dlat).astype(int), 0, len(glat)-1)
ilon = np.clip(np.round((lon - glon[0])/dlon).astype(int), 0, len(glon)-1)
age = age_grid[ilat, ilon]

# great-circle distance from each point to the nearest crust younger than AGE_CUT
young = np.argwhere(age_grid < AGE_CUT)
ylat, ylon = np.radians(glat[young[:, 0]]), np.radians(glon[young[:, 1]])
young_xyz = np.stack([np.cos(ylat)*np.cos(ylon), np.cos(ylat)*np.sin(ylon), np.sin(ylat)], 1)
chord, _ = cKDTree(young_xyz).query(xyz)
dist_young_km = 2*np.arcsin(np.clip(chord/2, 0, 1))*R0/1e3

fit = np.isin(region, FIT_REGIONS)
dated = ~np.isnan(age)
stage1 = fit & dated & (age >= AGE_CUT) & (dist_young_km > BUFFER_KM)
stage2 = fit & dated & ~stage1
allfit = stage1 | stage2
print(f'points: {len(lon)} in 12 regions; stage 1 {stage1.sum()}, stage 2 {stage2.sum()}, '
      f'undated in fit regions {(fit & ~dated).sum()} (excluded), Pacific Triangles {(~fit).sum()} (report only)')
np.savez(f'{OUT}/points.npz', lon=lon, lat=lat, region=region.astype(str), age=age,
         dist_young_km=dist_young_km, stage1=stage1, stage2=stage2)

fig = pygmt.Figure()
fig.basemap(region='g', projection='W0/18c', frame='afg')
fig.coast(land='gray80', shorelines='0.2p,gray50', resolution='l', area_thresh=5000)
for sel, colour, label in [(stage1, 'dodgerblue', f'stage 1: age >= {AGE_CUT:.0f} Ma and > {BUFFER_KM:.0f} km from younger crust'),
                           (stage2, 'orange', f'stage 2: crust < {AGE_CUT:.0f} Ma plus {BUFFER_KM:.0f} km buffer'),
                           (fit & ~dated, 'black', 'undated (excluded)'),
                           (~fit, 'gray40', 'Pacific Triangles (report only)')]:
    fig.plot(x=lon[sel], y=lat[sel], style='c0.03c', fill=colour, label=label)
fig.legend(position='JBC+jTC+o0c/0.4c', box='+gwhite+p0.5p')
fig.savefig(f'{OUT}/01-point-sets.png', dpi=200)


# ============================================================ 2. basis columns
l = np.arange(LMAX+1)

def br_at_points(c, altitude, depth_km=0.):
    c = c*(((R0 - depth_km*1e3)/R0)**(l+2))[None, :, None]
    x = pyshtools.SHMagCoeffs.from_array(c, r0=R0)
    return np.asarray(x.expand(a=R0+altitude, lat=lat, lon=lon, lmax_calc=LMAX))[:, 0]

def gauss(gmm):
    _, c = gmm.transform(lmax=LMAX)
    c = np.array(c.coeffs)
    c[:, :LMIN, :] = 0
    return c

def remanent_coeffs(k, P, lmbda):
    weights = [0., 0., 0.]
    weights[k] = 1.
    return gauss(create_vim(ocean, None, seafloor_layer='2d', layer_boundary_depths=LAYERS,
                            layer_weights=weights, MagMax=None, P=P, lmbda=lmbda, Mtrm=1, Mcrm=0))

cache = f'{OUT}/basis_columns.npz'
if os.path.exists(cache):
    z = np.load(cache)
    assert np.array_equal(z['lon'], lon) and np.array_equal(z['lat'], lat), 'point set changed; delete the cache'
    VIS, B0, BP, LCS = z['VIS'], z['B0'], z['BP'], z['LCS']
    C_VIS, C0, CP = z['C_VIS'], z['C0'], z['CP']
else:
    t = time.time()
    vis = load_vis_model(name='Hemant2005+slabs', match=(ocean.lon, ocean.lat, ocean.age))
    C_VIS = gauss(create_vim(None, vis))
    C0 = np.array([remanent_coeffs(k, 0., 3.) for k in range(3)])                     # lambda unused at P = 0
    CP = np.array([[remanent_coeffs(k, 1., lm) - C0[k] for k in range(3)] for lm in LAMBDAS])
    print(f'basis coefficients: {time.time()-t:.0f} s', flush=True)
    lcs = load_lcs(lmin=LMIN, lmax=LMAX)
    na, nd, nl = len(ALTITUDES), len(DEPTHS_KM), len(LAMBDAS)
    VIS = np.array([br_at_points(C_VIS, a) for a in ALTITUDES.values()])                 # [alt, pt]
    LCS = np.array([br_at_points(np.array(lcs.coeffs), a) for a in ALTITUDES.values()])  # [alt, pt]
    B0 = np.array([[[br_at_points(C0[k], a, d) for k in range(3)] for d in DEPTHS_KM]
                   for a in ALTITUDES.values()])                                         # [alt, d, k, pt]
    BP = np.array([[[[br_at_points(CP[j, k], a, d) for k in range(3)] for j in range(nl)] for d in DEPTHS_KM]
                   for a in ALTITUDES.values()])                                         # [alt, d, lam, k, pt]
    np.savez(cache, lon=lon, lat=lat, VIS=VIS, B0=B0, BP=BP, LCS=LCS, C_VIS=C_VIS, C0=C0, CP=CP)
    print(f'basis columns cached: {time.time()-t:.0f} s', flush=True)


# ============================================================ 3. model and objective
def model(ia, idep, ilam, w, P):
    """Br at all points: VIS + sum_k w_k (B0_k + P BP_k(lambda)), remanence at depth DEPTHS_KM[idep]"""
    return VIS[ia] + np.tensordot(w, B0[ia, idep] + P*BP[ia, idep, ilam], 1)

def corr(y, x):
    y, x = y - y.mean(), x - x.mean()
    return (y @ x)/np.sqrt((y @ y)*(x @ x))

def fit_w(y, sel, idep, ilam, P, w0):
    res = minimize(lambda w: -corr(y[sel], model(0, idep, ilam, w, P)[sel]), w0,
                   method='L-BFGS-B', bounds=[(0., W_MAX)]*3)
    return res.x, -res.fun

def fit_P(y, sel, idep, ilam, w):
    res = minimize_scalar(lambda P: -corr(y[sel], model(0, idep, ilam, w, P)[sel]),
                          bounds=(0., P_MAX), method='bounded')
    return res.x, -res.fun

def fit_joint(y, sel, idep, ilam, w0, P0):
    res = minimize(lambda p: -corr(y[sel], model(0, idep, ilam, p[:3], p[3])[sel]), np.r_[w0, P0],
                   method='L-BFGS-B', bounds=[(0., W_MAX)]*3 + [(0., P_MAX)])
    return res.x[:3], res.x[3], -res.fun

ilam_gk07 = LAMBDAS.index(GK07['lmbda'])


def tune(y, s1, s2, verbose=True):
    """stages 1-3 against observations y (Br at all points, 0 km); returns the chosen parameters"""
    out = {}
    # stage 1: w and d on old crust, lambda and P at GK07 values
    s1_results = [(idep,) + fit_w(y, s1, idep, ilam_gk07, GK07['P'], GK07['w']) for idep in range(len(DEPTHS_KM))]
    idep, w, r1 = max(s1_results, key=lambda t: t[2])
    out['stage1'] = dict(d=DEPTHS_KM[idep], w=w, r=r1, by_depth=[(DEPTHS_KM[i], r) for i, _, r in s1_results])
    # stage 2: lambda and P on young crust + buffer, w and d fixed
    s2_results = [(ilam,) + fit_P(y, s2, idep, ilam, w) for ilam in range(len(LAMBDAS))]
    ilam, P, r2 = max(s2_results, key=lambda t: t[2])
    out['stage2'] = dict(lmbda=LAMBDAS[ilam], P=P, r=r2, by_lambda=[(LAMBDAS[i], p, r) for i, p, r in s2_results])
    # stage 3: joint polish of w and P on all points, at the chosen and neighbouring lambda and d
    allsel = s1 | s2
    base = dict(w=w, P=P, idep=idep, ilam=ilam, r_all=corr(y[allsel], model(0, idep, ilam, w, P)[allsel]))
    best = base
    for jd in range(max(idep-1, 0), min(idep+2, len(DEPTHS_KM))):
        for jl in range(max(ilam-1, 0), min(ilam+2, len(LAMBDAS))):
            wj, Pj, rj = fit_joint(y, allsel, jd, jl, w, P)
            r1j = corr(y[s1], model(0, jd, jl, wj, Pj)[s1])
            r2j = corr(y[s2], model(0, jd, jl, wj, Pj)[s2])
            # keep only if the all-point score improves and neither stage subset loses more than 0.005
            if rj > best['r_all'] and r1j >= r1 - 0.005 and r2j >= r2 - 0.005:
                best = dict(w=wj, P=Pj, idep=jd, ilam=jl, r_all=rj)
    out['stage3'] = dict(accepted=best is not base, w=best['w'], P=best['P'], d=DEPTHS_KM[best['idep']],
                         lmbda=LAMBDAS[best['ilam']], r=best['r_all'])
    out['final'] = best
    if verbose:
        print(f"  stage 1: d = {out['stage1']['d']:.0f} km, w = {np.round(w, 2)} A/m, r = {r1:.4f}")
        print('           r by d: ' + ', '.join(f'{d:.0f} km {r:.4f}' for d, r in out['stage1']['by_depth']))
        print(f"  stage 2: lambda = {out['stage2']['lmbda']} Ma, P = {P:.2f}, r = {r2:.4f}")
        print('           by lambda: ' + ', '.join(f'{lm} Ma (P {p:.1f}) {r:.4f}' for lm, p, r in out['stage2']['by_lambda']))
        s3 = out['stage3']
        print(f"  stage 3: {'accepted' if s3['accepted'] else 'no improvement kept'}: d = {s3['d']:.0f} km, "
              f"lambda = {s3['lmbda']} Ma, w = {np.round(s3['w'], 2)}, P = {s3['P']:.2f}, r(all) = {s3['r']:.4f}")
    return out


# ============================================================ 4. checks before fitting
print('\nChecks')
# (a) basis linearity: GK07 built the normal way vs the basis combination (d = 0)
vis_model = load_vis_model(name='Hemant2005+slabs', match=(ocean.lon, ocean.lat, ocean.age))
C_gk07 = gauss(create_vim(ocean, vis_model, seafloor_layer='2d', layer_boundary_depths=LAYERS,
                          layer_weights=list(GK07['w']), MagMax=None, P=GK07['P'], lmbda=GK07['lmbda'],
                          Mtrm=1, Mcrm=0))
direct = br_at_points(C_gk07, 0.)
combined = model(0, 0, ilam_gk07, GK07['w'], GK07['P'])
print(f'  basis linearity (GK07): max |direct - basis| / rms = {np.abs(direct - combined).max()/direct.std():.1e}')

# (b) basis collinearity on stage-1 points (0 km, d = 0, GK07 lambda and P)
cols = np.stack([VIS[0]] + [B0[0, 0, k] + GK07['P']*BP[0, 0, ilam_gk07, k] for k in range(3)], 1)[stage1]
cols = (cols - cols.mean(0))/cols.std(0)
print('  column correlations [VIS, layer 1, layer 2, layer 3] on stage-1 points:')
print('  ' + np.array2string(np.corrcoef(cols.T), precision=3).replace('\n', '\n  '))
print(f'  condition number (standardised columns) = {np.linalg.cond(cols):.1f}')

# (c) recovery: GK07's own synthetic field as the observation
print('  recovery test (observation = GK07 synthetic):')
rec = tune(combined, stage1, stage2, verbose=True)


# ============================================================ 5. fit to LCS-1
y0 = LCS[0]
baseline = {name: corr(y0[sel], combined[sel]) for name, sel in [('stage1', stage1), ('stage2', stage2), ('all', allfit)]}
print('\nGK07 baseline (0 km, l 16-100): ' + ', '.join(f'{k} r = {v:.4f}' for k, v in baseline.items()))
print('\nTuning against LCS-1')
result = tune(y0, stage1, stage2)
final = result['final']
tuned = model(0, final['idep'], final['ilam'], final['w'], final['P'])


# ============================================================ 6. leave-one-region-out
print('\nLeave-one-region-out (refit without the region; score the held-out region)')
rows = []
for name in FIT_REGIONS:
    out_sel = region == name
    o = tune(y0, stage1 & ~out_sel, stage2 & ~out_sel, verbose=False)['final']
    held = out_sel & allfit
    m = model(0, o['idep'], o['ilam'], o['w'], o['P'])
    rows.append(dict(left_out=name, d_km=DEPTHS_KM[o['idep']], lmbda=LAMBDAS[o['ilam']], P=o['P'],
                     w1=o['w'][0], w2=o['w'][1], w3=o['w'][2],
                     r_heldout_tuned=corr(y0[held], m[held]), r_heldout_gk07=corr(y0[held], combined[held])))
loro = pd.DataFrame(rows)
loro['gain'] = loro.r_heldout_tuned - loro.r_heldout_gk07
loro.to_csv(f'{OUT}/leave_one_region_out.csv', index=False)
print(loro.round(3).to_string(index=False))
print(f"  parameter spread (std): w = {loro[['w1', 'w2', 'w3']].std().round(2).tolist()}, "
      f"P = {loro.P.std():.2f}, lambda values {sorted(set(loro.lmbda))}, d values {sorted(set(loro.d_km))}")
print(f'  held-out gain over GK07: mean {loro.gain.mean():+.4f}, std {loro.gain.std():.4f}, '
      f'positive in {(loro.gain > 0).sum()} of {len(loro)} regions')


# ============================================================ 7. report
print('\nPer-region correlation (reported, not fitted), l 16-100')
rep = []
for name in FIT_REGIONS + REPORT_ONLY:
    sel = (region == name) & dated if name in FIT_REGIONS else (region == name)
    row = dict(region=name, n=int(sel.sum()))
    for ia, alt in enumerate(ALTITUDES):
        gk = model(ia, 0, ilam_gk07, GK07['w'], GK07['P'])
        tu = model(ia, final['idep'], final['ilam'], final['w'], final['P'])
        row[f'r_gk07_{alt}'] = corr(LCS[ia][sel], gk[sel])
        row[f'r_tuned_{alt}'] = corr(LCS[ia][sel], tu[sel])
    rep.append(row)
per_region = pd.DataFrame(rep)
per_region.to_csv(f'{OUT}/per_region_correlation.csv', index=False)
print(per_region.round(3).to_string(index=False))

print('\nPooled summary')
summary = []
for ia, alt in enumerate(ALTITUDES):
    gk = model(ia, 0, ilam_gk07, GK07['w'], GK07['P'])
    tu = model(ia, final['idep'], final['ilam'], final['w'], final['P'])
    for name, sel in [('stage1', stage1), ('stage2', stage2), ('all', allfit)]:
        summary.append(dict(altitude=alt, points=name, r_gk07=corr(LCS[ia][sel], gk[sel]),
                            r_tuned=corr(LCS[ia][sel], tu[sel]),
                            rms_ratio_gk07=gk[sel].std()/LCS[ia][sel].std(),
                            rms_ratio_tuned=tu[sel].std()/LCS[ia][sel].std()))
summary = pd.DataFrame(summary)
summary.to_csv(f'{OUT}/summary.csv', index=False)
print(summary.round(3).to_string(index=False))

params = pd.DataFrame([dict(model='GK07', w1=5., w2=2.3, w3=1.2, P=GK07['P'], lmbda=GK07['lmbda'], d_km=0.),
                       dict(model='GK07_TUNED', w1=final['w'][0], w2=final['w'][1], w3=final['w'][2],
                            P=final['P'], lmbda=LAMBDAS[final['ilam']], d_km=DEPTHS_KM[final['idep']])])
params.to_csv(f'{OUT}/parameters.csv', index=False)
print('\nParameters (layer magnetizations in A/m; layers 0-0.5, 0.5-1.5, 1.5-6.5 km)')
print(params.round(3).to_string(index=False))
