"""
Tune the GK07 near-ridge enhancement (P, lambda) for pattern and amplitude.

Method (agreed 2026-09-30):
  - model: GK07 with realistic source depth (depth_models.py), all other GK07
    parameters unchanged; TRM = Mtrm tp (1 + P exp(-t/lambda)) is linear in P,
    so Br = B0 + P * BP(lambda) with B0 the P = 0 model (remanence + VIS)
  - points: the young-crust set of tune_ocean_model.py (crust < 20 Ma plus a
    100 km great-circle buffer, in the 10 non-Triangle regions)
  - Br at 0 km (r0 sphere), degrees 16-100
  - objective: for each lambda, P is set so that RMS(model) = RMS(LCS-1) on the
    points; the (P, lambda) with the highest pooled correlation is chosen
  - lambda grid 0.25-5 Ma (4 lambda stays within the 20 Ma cut)
  - checks: 300 km, all points, per-region values, leave-one-region-out

Run from notebooks/:
    conda run -n pygmt17 python tune_near_ridge.py
Outputs go to notebooks/tuning/near_ridge_*.
"""
import os

import numpy as np
import pandas as pd
import pyshtools

import depth_models as dm
from basis_models import GK07
from remit.data.models import load_ocean_age_model, load_vis_model, load_lcs

OUT = 'tuning'
LMIN, LMAX = 16, 100
ALTITUDES = {'0km': 0., '300km': 300e3}
LAMBDAS = [0.25, 0.5, 1., 1.5, 2., 3., 4., 5.]
PAPER, P_TUNED_CORR = (5., 3.), 2.3            # (P, lambda) of the paper; P from 2026-09-29 tuning
FIT_REGIONS = ['CentralAtlantic', 'SouthAtlanticN', 'SouthAtlanticS', 'SouthwestIndianOcean',
               'Wharton-BayofBengal', 'SoutheastIndianOcean', 'PacificAntarcticRidge',
               'EastPacificRidgeS', 'EastPacificRidgeN', 'NEPacific']

# ------------------------------------------------------------ 1. points
pts = np.load(f'{OUT}/points.npz')
infit = np.isin(pts['region'], FIT_REGIONS)
young = pts['stage2'] & infit
allpts = (pts['stage1'] | pts['stage2']) & infit
lat, lon, region = pts['lat'], pts['lon'], pts['region']
print(f'points: young-crust {young.sum()}, all {allpts.sum()}')

# ------------------------------------------------------------ 2. basis fields (cached)
cache = f'{OUT}/near_ridge_basis.npz'
if not os.path.exists(cache):
    ocean = load_ocean_age_model()
    vis = load_vis_model(name='Hemant2005+slabs', match=(ocean.lon, ocean.lat, ocean.age))
    r_top = dm.top_of_crust_radius(ocean)
    params = {k: v for k, v in GK07.items()}
    coeffs = {'VIS': dm.vis_coeffs(ocean, vis, r_top).coeffs}
    coeffs['R_P0'] = dm.remanent_coeffs(ocean, r_top, dict(params, P=0.))[0].coeffs
    for lam in LAMBDAS:
        c1 = dm.remanent_coeffs(ocean, r_top, dict(params, P=1., lmbda=lam))[0].coeffs
        coeffs[f'BP_{lam}'] = c1 - coeffs['R_P0']
        print(f'  basis lambda = {lam} Ma done', flush=True)
    np.savez(cache, **coeffs)
C = dict(np.load(cache))


def br(c, alt):
    c = pyshtools.SHMagCoeffs.from_array(c.copy(), r0=dm.R0).pad(LMAX)
    c.coeffs[:, :LMIN, :] = 0
    return np.asarray(c.expand(a=dm.R0 + alt, lat=lat, lon=lon))[:, 0]


lcs = load_lcs(lmin=LMIN, lmax=LMAX)
L = {a: np.asarray(lcs.expand(a=lcs.r0 + h, lat=lat, lon=lon))[:, 0] for a, h in ALTITUDES.items()}
B0 = {a: br(C['VIS'] + C['R_P0'], h) for a, h in ALTITUDES.items()}
BP = {(a, lam): br(C[f'BP_{lam}'], h) for a, h in ALTITUDES.items() for lam in LAMBDAS}

# check: the basis reproduces the full depth-resolved GK07 of depth_models (P = 5, lambda = 3)
full = br(dm.depth_model_coeffs(['GK07'])['GK07'][0].coeffs, 0.)
comb = B0['0km'] + 5.*BP[('0km', 3.)]
print(f'basis check (GK07, P=5, lambda=3): max|diff|/rms = {np.abs(full-comb).max()/full.std():.1e}')


def stats(P, lam, sel, alt='0km'):
    m = B0[alt][sel] + P*BP[(alt, lam)][sel]
    o = L[alt][sel]
    return np.corrcoef(m, o)[0, 1], m.std()/o.std()


def rms_match_P(lam, sel):
    """P >= 0 with RMS(model) = RMS(LCS-1) on sel (demeaned); None if P = 0 is already too strong"""
    b0, bp, o = [x - x.mean() for x in (B0['0km'][sel], BP[('0km', lam)][sel], L['0km'][sel])]
    a, b, c = bp @ bp, 2*(b0 @ bp), b0 @ b0 - o @ o
    disc = b*b - 4*a*c
    if disc < 0:
        return None
    roots = [(-b + s*np.sqrt(disc))/(2*a) for s in (1, -1)]
    roots = [p for p in roots if p >= 0]
    return max(roots) if roots else None


def fit(sel):
    rows = []
    for lam in LAMBDAS:
        P = rms_match_P(lam, sel)
        if P is not None:
            rows.append((lam, P, stats(P, lam, sel)[0]))
    lam, P, r = max(rows, key=lambda x: x[2])
    return lam, P, r, rows


# ------------------------------------------------------------ 3. fit
lam_b, P_b, r_b, rows = fit(young)
print('\nRMS-matched P and correlation on the young-crust points (0 km, l 16-100):')
for lam, P, r in rows:
    print(f'  lambda {lam:5.2f} Ma   P {P:6.2f}   r {r:.4f}')
print(f'chosen: lambda = {lam_b} Ma, P = {P_b:.2f}, r = {r_b:.4f}')

# ------------------------------------------------------------ 4. leave-one-region-out
loro = []
for reg in FIT_REGIONS:
    held = young & (region == reg)
    if held.sum() == 0:
        continue
    lam, P, r, _ = fit(young & (region != reg))
    loro.append(dict(held_out=reg, lmbda=lam, P=P, r_train=r,
                     r_held=stats(P, lam, held)[0], rms_held=stats(P, lam, held)[1],
                     r_held_chosen=stats(P_b, lam_b, held)[0]))
loro = pd.DataFrame(loro)
loro.to_csv(f'{OUT}/near_ridge_leave_one_region_out.csv', index=False, float_format='%.4f')
print('\nleave-one-region-out:')
print(loro.to_string(index=False, float_format='%.3f'))

# ------------------------------------------------------------ 5. report
cases = {'paper (P 5, lambda 3)': PAPER, 'P 2.3, lambda 3': (P_TUNED_CORR, 3.),
         f'tuned (P {P_b:.2f}, lambda {lam_b})': (P_b, lam_b), 'P 0': (0., 3.)}
rows = []
for name, (P, lam) in cases.items():
    for setname, sel in [('young', young), ('all', allpts)]:
        for alt in ALTITUDES:
            r, q = stats(P, lam, sel, alt)
            rows.append(dict(model=name, points=setname, altitude=alt, r=r, rms_ratio=q))
summary = pd.DataFrame(rows)
summary.to_csv(f'{OUT}/near_ridge_summary.csv', index=False, float_format='%.4f')
print('\n' + summary.to_string(index=False, float_format='%.3f'))

per = []
for reg in FIT_REGIONS:
    sel = young & (region == reg)
    for name, (P, lam) in cases.items():
        r, q = stats(P, lam, sel)
        per.append(dict(region=reg, model=name, n=int(sel.sum()), r=r, rms_ratio=q))
per = pd.DataFrame(per)
per.to_csv(f'{OUT}/near_ridge_per_region.csv', index=False, float_format='%.4f')
print('\nper region, young crust, 0 km:')
print(per.pivot(index='region', columns='model', values='rms_ratio').to_string(float_format='%.2f'))
print(per.pivot(index='region', columns='model', values='r').to_string(float_format='%.3f'))
