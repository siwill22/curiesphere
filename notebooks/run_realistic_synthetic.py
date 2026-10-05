"""
How many domains can satellite data recover from a realistic, complicated geology, and what do they mean?

Truth: every cell takes the VIS of its Hasterok et al. (2022) province (53 provinces over the Australia
cells). Province values are drawn from a lognormal fitted to the H&M (+ slabs) VIS averaged over the
cells (its statistics only, not its map), with the spread of the log values scaled (CONTRAST_SCALE) so
that the true field is as strong as the real one (RMS of LCS-1 minus background in the data subspace,
less the noise): the H&M map is smooth, so its own spread gives a field about half as strong as
observed. Two versions:
  hast        one value per province
  hast_var    plus smooth variation within provinces (lognormal, 25% standard deviation, ~200 km
              correlation length along great circles)
Data: the truth's field at the data points plus realistic noise (the random-phase LCS-1 - MF7 noise
realisations of build_region.py, scaled to sigma = RMS(LCS-1 - MF7)), at 1x and 2x sigma (2x is about
the misfit of the real-data models; env NOISES, default 1). Weighting as for the real data (A2: point weighting on the
compressed data).
Method: merging (merge.run, fine mode) at the real-data settings (lambda_95, gamma 0.01 lambda_95) and
at lambda_95/3; bootstrap: N_BOOT reruns on the fitted model + fresh noise.
Benchmark: the best map of the truth with the same number of domains, made directly from the true VIS
(Ward clustering of neighbouring cells, area weights).
Output: realistic_synthetic.npz in the region cache.

Run from notebooks/ (conda env pygmt17), lightly:
    N_WORKERS=3 nice -n 15 env PYTHONPATH=..:$PYTHONPATH python run_realistic_synthetic.py regions/australia/config.toml
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '1')
import sys
import time
import multiprocessing

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering

from domains import merge, outputs
from domains.basis import unit
from domains.region import cell_provinces
from remit.data.models import load_vis_model

REG = os.path.dirname(os.path.abspath(sys.argv[1]))
C = f'{REG}/cache'
N_BOOT = int(os.environ.get('N_BOOT', 15))
N_WORKERS = int(os.environ.get('N_WORKERS', 3))
MUS = [0.03, 0.1, 0.3]
SEED = 20261005

E = np.load(f'{C}/errors.npz')
z = np.load(f'{C}/data.npz')
G = np.load(f'{C}/G.npz')['G']
cells = np.load(f'{C}/cells.npz')
edges, area = cells['edges'], cells['area_km2']
sigma = float(np.sqrt(np.mean((z['d_lcs'] - z['d_mf7'])**2)))
T = np.linalg.svd(E['T'].T, full_matrices=False)[0].T/sigma
Gw = T @ G
noise = np.load(f'{C}/noise.npz')['noise']
noise = noise*sigma/np.sqrt(np.mean(noise**2))
lam95 = float(np.load(f'{C}/merge_points_sweep.npz')['fine_lam95'])
SETTINGS = [(1., 0.01), (1/3, 0.01)]                       # (lambda / lambda_95, gamma / lambda_95)

# provinces and H&M statistics
prov_key, prov_table = cell_provinces(cells['clon'], cells['clat'])
prov = np.unique(prov_key, return_inverse=True)[1]
cell, glat_n, glon_n = cells['cell'], cells['cell'].shape[0], cells['cell'].shape[1]
glon, glat = np.linspace(0, 360, glon_n), np.linspace(90, -90, glat_n)
hm = load_vis_model(name='Hemant2005+slabs')
lo, la = np.meshgrid(glon, glat)
from scipy.interpolate import RegularGridInterpolator
f = RegularGridInterpolator((hm.lat[::-1] if hm.lat[0] > hm.lat[-1] else hm.lat, hm.lon),
                            hm.vis[::-1] if hm.lat[0] > hm.lat[-1] else hm.vis, bounds_error=False, fill_value=None)
inside = cell >= 0
w = np.cos(np.radians(la[inside]))
hm_cell = np.bincount(cell[inside], weights=w*f(np.stack([la[inside], lo[inside] % 360], 1)))/np.bincount(cell[inside], weights=w)
lmu, lsd = np.average(np.log(np.clip(hm_cell, 1e-3, None)), weights=area), None
lsd = np.sqrt(np.average((np.log(np.clip(hm_cell, 1e-3, None)) - lmu)**2, weights=area))

rng = np.random.default_rng(SEED)
zp = rng.standard_normal(prov.max() + 1)
xyz = unit(cells['clon'], cells['clat'])
dist = np.arccos(np.clip(xyz @ xyz.T, -1, 1))*6371.
Ccov = np.exp(-dist/200.) + 1e-8*np.eye(len(dist))
field = np.linalg.cholesky(Ccov) @ rng.standard_normal(len(dist))
n_pts = len(z['plon'])
rms_nT = lambda w: sigma*np.sqrt(np.sum(w**2)/n_pts)                 # RMS at the points of a subspace vector
real_signal = np.sqrt(rms_nT(T @ (z['d_lcs'] - z['b']))**2 - sigma**2)
scales = np.linspace(0.5, 5, 91)
strength = [rms_nT(Gw @ np.clip(np.exp(lmu + s_*lsd*zp), 0.01, 4.)[prov]) for s_ in scales]
CONTRAST_SCALE = float(scales[np.argmin(np.abs(np.array(strength) - real_signal))])
v_prov = np.clip(np.exp(lmu + CONTRAST_SCALE*lsd*zp), 0.01, 4.)
truths = {'hast': v_prov[prov], 'hast_var': np.clip(v_prov[prov]*np.exp(0.25*field - 0.5*0.25**2), 0.01, 4.)}
NOISES = [float(x) for x in os.environ.get('NOISES', '1').split(',')]
data = {}
j = 0
for name in truths:
    for nl in NOISES:
        data[(name, nl)] = Gw @ truths[name] + nl*(T @ noise[j])
        j += 1
P = merge.setup(Gw, edges)
conn = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(area),)*2)
conn = conn + conn.T


def best_map(truth, K):
    """best K-domain map of the true VIS itself: Ward clustering of neighbouring cells (cells are of
    near-equal area at NSIDE 64, so no weights); returns the labels and the domain-mean map"""
    lab = AgglomerativeClustering(n_clusters=K, linkage='ward', connectivity=conn).fit_predict(truth[:, None])
    means = np.bincount(lab, weights=truth*area)/np.bincount(lab, weights=area)
    return lab, means[lab]


def run(args):
    dw, lr, g = args
    t = time.time()
    r = merge.run(P, dw, lr*lam95, g*lam95, MUS)
    r['seconds'] = time.time() - t
    return r


if __name__ == '__main__':
    t0 = time.time()
    print(f'{prov.max()+1} provinces; H&M cell VIS: lognormal median {np.exp(lmu):.2f} SI.km, log sd {lsd:.2f}; '
          f'contrast scale {CONTRAST_SCALE:.2f} (true field {rms_nT(Gw @ truths["hast"]):.0f} nT, real signal '
          f'{real_signal:.0f} nT); province values {v_prov.min():.2f}-{v_prov.max():.2f}', flush=True)
    keys = [(name, nl, lr, g) for name, nl in data for lr, g in SETTINGS]
    with multiprocessing.get_context('fork').Pool(N_WORKERS) as pool:
        res = dict(zip(keys, pool.map(run, [(data[(n, nl)], lr, g) for n, nl, lr, g in keys])))
        boot_jobs = []
        for (name, nl) in data:
            r0 = res[(name, nl, 1., 0.01)]
            pred = Gw @ r0['v']
            boot_jobs += [((name, nl), (pred + nl*(T @ noise[40 + b]), 1., 0.01)) for b in range(N_BOOT)]
        boot = pool.map(run, [b[1] for b in boot_jobs])
    save = dict(prov=prov, prov_key=prov_key, v_prov=v_prov, edges=edges, settings=np.array(SETTINGS),
                noises=NOISES, lam95=lam95, hm_lmu=lmu, hm_lsd=lsd,
                contrast_scale=CONTRAST_SCALE, real_signal=real_signal)
    for name, tv in truths.items():
        save[f'truth_{name}'] = tv
        for nl in NOISES:
            for lr, g in SETTINGS:
                r = res[(name, nl, lr, g)]
                K = r['K']
                blab, bmap = best_map(tv, K)
                rec = r['v']
                wr2 = lambda m: 1 - np.average((tv - m)**2, weights=area)/np.average((tv - np.average(tv, weights=area))**2, weights=area)
                avg_in = (np.bincount(r['label'], weights=tv*area)/np.bincount(r['label'], weights=area))[r['label']]
                key = f'{name}_n{nl:g}_l{lr:.2f}'
                print(f'{key}: K {K}; share of the true VIS variation captured: recovered {wr2(rec):.2f}, '
                      f'truth averaged in the recovered domains {wr2(avg_in):.2f}, best possible with {K} domains '
                      f'{wr2(bmap):.2f}; fit RMS {sigma*np.sqrt(r["chi2"]/len(z["plon"])):.1f} nT ({r["seconds"]:.0f} s)', flush=True)
                save.update({f'{key}_label': r['label'], f'{key}_v': rec, f'{key}_K': K, f'{key}_chi2': r['chi2'],
                             f'{key}_best_label': blab, f'{key}_best_map': bmap, f'{key}_avg_in': avg_in})
            labs = np.array([b['label'] for (k, _), b in zip(boot_jobs, boot) if k == (name, nl)])
            Ks = np.array([b['K'] for (k, _), b in zip(boot_jobs, boot) if k == (name, nl)])
            save[f'{name}_n{nl:g}_boot_prob'] = outputs.boundary_probability(labs, edges)
            save[f'{name}_n{nl:g}_boot_K'] = Ks
            print(f'{name}_n{nl:g} bootstrap ({N_BOOT}): K {sorted(Ks)}', flush=True)
        for nl in NOISES:
            save[f'data_{name}_n{nl:g}'] = data[(name, nl)]
    np.savez(f'{C}/realistic_synthetic.npz', **save)
    print(f'done ({(time.time()-t0)/60:.0f} min)')
