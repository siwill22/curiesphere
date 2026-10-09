"""
The future-data test: what do higher degrees with lower noise add to the maps of a synthetic geology?
Runs (build_bands.py, synthetic only, regions/australia_fine): today's noise to degree 133 and 185, and
future data (noise at most 10% or 30% of the signal amplitude) to degree 133, 185 and 300.
Truths: hast (provinces), hast_var (+ ~200 km variation), hast_ms (make_multiscale_truth.py: + ~50 km
detail + small bodies). Maps: GP (one scale), GP (two scales), L2, TV (run_linear_models.py) and merging
at lambda_95 (run_realistic_synthetic.py).

Measures
  RMS error         area-weighted, SI.km
  share by scale    the VIS split into scales by Gaussian smoothing along great circles, with widths at
                    which the smoothing halves degree 100 and degree 200 (75 and 37 km): broad (beyond
                    ~degree 100, > ~400 km), middle (~degree 100-200) and fine (~degree 200 to the cell size,
                    ~50 km); share = 1 - |error in the scale|^2 / |truth in the scale|^2 (area weights);
                    1 is perfect, 0 no better than a flat map at that scale, negative worse
  boundaries        merging only: mean great-circle distance (km) from each true boundary (province or
                    body edge) to the nearest recovered boundary ('missed') and back ('spurious')
  bodies            hast_ms: recovered contrast (body mean minus the ring 50-150 km outside it) over the
                    true contrast, by body size; 'found' = more than half the contrast recovered
Output: future_summary.npz in the region cache and the printed tables.

Run from notebooks/ (conda env pygmt17):
    PYTHONPATH=..:$PYTHONPATH python compare_future.py regions/australia_fine/config.toml
"""
import os
import sys

import numpy as np

from domains import problem
from domains.basis import unit

BANDS = {'syn_16-133': 'today, to 133', 'syn_16-185': 'today, to 185', 'syn_f10_16-133': 'future 10%, to 133',
         'syn_f10_16-185': 'future 10%, to 185', 'syn_f10_16-300': 'future 10%, to 300',
         'syn_f30_16-300': 'future 30%, to 300'}
CASES = ['hast', 'hast_var', 'hast_ms']
METHODS = {'gp': 'GP', 'gp2': 'GP 2 scales', 'l2': 'L2', 'tv': 'TV', 'merge': 'merging'}
SIZES = [(50., 83.), (83., 117.), (117., 150.)]
R_KM = 6371.

pb = problem.load(sys.argv[1])
C = pb['cache']
area, edges = pb['area'], pb['edges']
xyz = unit(pb['clon'], pb['clat'])
dist = np.arccos(np.clip(xyz @ xyz.T, -1, 1))*R_KM
ms = np.load(f'{C}/multiscale_truth.npz')
prov, body = ms['prov'], ms['body']


def smoother(degree):
    """area-weighted Gaussian smoothing along great circles that halves the given degree"""
    sig = R_KM*np.sqrt(2*np.log(2)/(degree*(degree + 1)))
    W = np.exp(-0.5*(dist/sig)**2)*area[None, :]
    return W/W.sum(1, keepdims=True)


S100, S200 = smoother(100), smoother(200)


def scales(v):
    a, b = S100 @ v, S200 @ v
    return {'broad': a, 'middle': b - a, 'fine': v - b}


def share(rec, truth):
    out = {}
    ts, rs = scales(truth), scales(rec)
    for k in ts:
        t = ts[k] - (np.average(ts[k], weights=area) if k == 'broad' else 0.)
        out[k] = 1 - np.sum(area*(rs[k] - ts[k])**2)/np.sum(area*t**2)
    return out


mid = xyz[edges[:, 0]] + xyz[edges[:, 1]]
mid /= np.linalg.norm(mid, axis=1, keepdims=True)


def boundary_distance(lab_true, lab_rec):
    """mean distance (km) from true boundary edges to the nearest recovered one, and back"""
    t = lab_true[edges[:, 0]] != lab_true[edges[:, 1]]
    r = lab_rec[edges[:, 0]] != lab_rec[edges[:, 1]]
    if not r.any():
        return np.nan, np.nan
    d = np.arccos(np.clip(mid[t] @ mid[r].T, -1, 1))*R_KM
    return float(d.min(1).mean()), float(d.min(0).mean())


true_lab = {'hast': prov, 'hast_var': prov, 'hast_ms': np.where(body >= 0, prov.max() + 1 + body, prov)}
rings = []
for k, (i, dk) in enumerate(zip(ms['body_centre'], ms['body_diameter_km'])):
    inside = body == k
    ring = (dist[i] >= 0.5*dk + 50.) & (dist[i] <= 0.5*dk + 150.) & (body < 0)
    rings.append((inside, ring, dk))


def body_ratios(rec, truth):
    c = lambda v, I, R: np.average(v[I], weights=area[I]) - np.average(v[R], weights=area[R])
    return np.array([c(rec, I, R)/c(truth, I, R) for I, R, _ in rings])


diam = np.array([r[2] for r in rings])
save, rows = {}, []
for b, label in BANDS.items():
    f_lm, f_rs = f'{C}/bands/{b}/linear_models.npz', f'{C}/bands/{b}/realistic_synthetic.npz'
    lm = np.load(f_lm) if os.path.exists(f_lm) else {}
    rsy = np.load(f_rs, allow_pickle=True) if os.path.exists(f_rs) else {}
    for case in CASES:
        truth = ms['truth'] if case == 'hast_ms' else (lm[f'{case}_truth'] if f'{case}_truth' in lm else None)
        if truth is None and f'truth_{case}' in rsy:
            truth = rsy[f'truth_{case}']
        if truth is None:
            continue
        for m in METHODS:
            key = f'{case}_n1_l1.00_v' if m == 'merge' else f'{case}_{m}_v'
            src = rsy if m == 'merge' else lm
            if key not in src:
                continue
            v = src[key]
            r = dict(band=b, case=case, method=m, rms=float(np.sqrt(np.average((v - truth)**2, weights=area))),
                     **share(v, truth))
            if m == 'merge':
                lab = src[f'{case}_n1_l1.00_label']
                r['K'] = int(src[f'{case}_n1_l1.00_K'])
                r['missed_km'], r['spurious_km'] = boundary_distance(true_lab[case], lab)
            if case == 'hast_ms':
                br = body_ratios(v, truth)
                save[f'{b}|{case}|{m}|body_ratio'] = br
                for lo, hi in SIZES:
                    sel = (diam >= lo) & (diam < hi + 1e-9)
                    r[f'found {lo:.0f}-{hi:.0f} km'] = float(np.mean(br[sel] > 0.5))
            save[f'{b}|{case}|{m}|v'] = v
            rows.append(r)

for case in CASES:
    print(f'\n== {case}: share of the true VIS recovered, by scale (broad / middle / fine), and RMS error (SI.km)')
    print(f'{"run":22s}' + ''.join(f'{METHODS[m]:>24s}' for m in METHODS))
    for b, label in BANDS.items():
        cells_ = []
        for m in METHODS:
            r = next((r for r in rows if r['band'] == b and r['case'] == case and r['method'] == m), None)
            cells_.append(f'{r["broad"]:5.2f}/{r["middle"]:5.2f}/{r["fine"]:5.2f} {r["rms"]:.3f}' if r else '-')
        print(f'{label:22s}' + ''.join(f'{c:>24s}' for c in cells_))
    print(f'merging: domains K; boundaries missed / spurious (mean distance, km)')
    for b, label in BANDS.items():
        r = next((r for r in rows if r['band'] == b and r['case'] == case and r['method'] == 'merge'), None)
        if r:
            print(f'  {label:22s} K {r["K"]:3d}; {r["missed_km"]:5.0f} / {r["spurious_km"]:5.0f} km')
print(f'\n== hast_ms: bodies found (more than half the true contrast recovered), by diameter '
      f'({", ".join(f"{lo:.0f}-{hi:.0f} km: {np.sum((diam >= lo) & (diam < hi + 1e-9))}" for lo, hi in SIZES)} bodies)')
print(f'{"run":22s}' + ''.join(f'{METHODS[m]:>18s}' for m in METHODS))
for b, label in BANDS.items():
    cells_ = []
    for m in METHODS:
        r = next((r for r in rows if r['band'] == b and r['case'] == 'hast_ms' and r['method'] == m), None)
        cells_.append('/'.join(f'{r[f"found {lo:.0f}-{hi:.0f} km"]:.2f}' for lo, hi in SIZES) if r else '-')
    print(f'{label:22s}' + ''.join(f'{c:>18s}' for c in cells_))

cols = sorted({k for r in rows for k in r} - {'band', 'case', 'method'})
save.update(rows_band=np.array([r['band'] for r in rows]), rows_case=np.array([r['case'] for r in rows]),
            rows_method=np.array([r['method'] for r in rows]), cols=np.array(cols),
            values=np.array([[r.get(c, np.nan) for c in cols] for r in rows], float),
            bands=np.array(list(BANDS)), band_labels=np.array(list(BANDS.values())), body_diameter_km=diam)
np.savez(f'{C}/future_summary.npz', **save)
print(f'\nsaved {C}/future_summary.npz')
