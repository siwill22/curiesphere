"""
Export the Australian inversion results for the inversion explorer (GeodeViewers/CuriesphereInversion):
static files only, read by the browser.

  index.json            manifest: grid, cells (positions, classes, covariates), cases, design columns,
                        functionals, the catalogue of results, regression / classes / hypotheses /
                        bounds / null-test tables, colour maps, coastline, info text
  cells.u16             0.1 deg index raster of the cells over the region (65535 = no cell)
  results.f32           every map (1406 cell values each), in catalogue order
  labels.i16            partitions of the domain results (for boundaries), in label-catalogue order
  brs.f32               Br (nT) at the data points of each map that has one, in catalogue order
  points.f32            data points: lon, lat, LCS-1, MF7, background (5 x 7681)
  explore/columns.f64   the design columns B (n x ncol) of domains.design, row-major
  explore/ell_<k>.f64   per correlation length: eigenvalues of K = Gw R Gw' (N), Bhat = U' Gw B (ncol x N),
                        dhat = U' dw (ncase x N), c'Rc for each functional
  explore/ell_<k>.f32   W = R Gw' U (n x N), row-major
  test/fixture.json     Python values (gmrf.Model) for the browser parity test

With these the browser evaluates the GP / regression model exactly for any noise level h, VIS standard
deviation s, correlation length on the grid and set of design columns (see the viewer's js/lingauss.js).

Run from notebooks/ (conda env pygmt17), after run_linear_models.py, run_bounds.py, run_map_sweep.py,
run_null_regression.py and run_covariates.py (and compare_bands.py for the degree ranges, if run):
    PYTHONPATH=..:$PYTHONPATH python export_viewer_inversion.py regions/australia/config.toml [out_dir]
"""
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(k, '1')
import sys
import json
import subprocess
import datetime

import numpy as np

from domains import bounds, covariates, gmrf, graph, problem
from domains.design import design, GROUPS
from domains.plotting import REGION_AUSTRALIA

OUT = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else '~/GIT/GeodeViewers/CuriesphereInversion/data')
REGION = REGION_AUSTRALIA
COLS = ['intercept', 'class', 'crust', 'lab', 'heat_flow', 'heat_flow_seisheat', 'heat_flow_heatr', 'age',
        'undated', 'craton']

pb = problem.load(sys.argv[1])
C = pb['cache']
Gw, G, area, cells = pb['Gw'], pb['G'], pb['area'], pb['cells']
N, n = Gw.shape
z = pb['z']
load = lambda name, **kw: dict(np.load(f'{C}/{name}.npz', **kw))
cov = load('covariates')
lm = load('linear_models')
bd = load('bounds')
ms = load('map_sweep')
nr = load('null_regression')
rs = load('realistic_synthetic', allow_pickle=True)
mp = load('merge_points_sweep', allow_pickle=True)
Lw = graph.laplacian(pb['edges'], 1/graph.edge_lengths_km(pb['edges'], pb['clon'], pb['clat'])**2, n)
CASES = [str(c) for c in lm['cases']]
os.makedirs(f'{OUT}/explore', exist_ok=True)
os.makedirs(f'{OUT}/test', exist_ok=True)
f32 = lambda a: np.asarray(a, np.float32)
fl = lambda a, d=4: [round(float(x), d) for x in np.ravel(a)]


# ------------------------------------------------------------------ grid and cells
cell = cells['cell']
glat = np.linspace(90, -90, cell.shape[0]); glon = np.linspace(0, 360, cell.shape[1])
jx = np.nonzero((glon >= REGION[0] - 1e-9) & (glon <= REGION[1] + 1e-9))[0]
iy = np.nonzero((glat <= REGION[3] + 1e-9) & (glat >= REGION[2] - 1e-9))[0]
sub = cell[np.ix_(iy, jx)]
raster = np.where(sub >= 0, sub, 65535).astype(np.uint16)
raster.tofile(f'{OUT}/cells.u16')
grid = dict(lon0=float(glon[jx[0]]), lat0=float(glat[iy[0]]), dlon=0.1, dlat=-0.1, nx=len(jx), ny=len(iy))

# functionals (area mean, craton means, craton - rest)
craton = np.isin(cov['group'], covariates.CRATON_GROUPS)
FUN = {'area mean': bounds.mean_functional(area, np.ones(n, bool)),
       **{g.split()[0][0] + 'AC mean': bounds.mean_functional(area, cov['group'] == g) for g in covariates.CRATON_GROUPS},
       'craton - rest': bounds.mean_functional(area, craton) - bounds.mean_functional(area, ~craton)}

# ------------------------------------------------------------------ results catalogue
results, blobs, br_blobs, label_blobs, label_cat = [], [], [], [], []


def add(case, family, method, label, v, kind='vis', params=None, labels=None, chi2=None, br=True, note=None):
    rid = f'{case}:{family}:{method}:{label}'
    entry = dict(id=rid, case=case, family=family, method=method, label=label, kind=kind, params=params or {},
                 index=len(results))
    v = np.asarray(v, float)
    entry['stats'] = dict(min=round(float(np.nanmin(v)), 4), max=round(float(np.nanmax(v)), 4),
                          mean=round(float(np.average(v, weights=area)), 4))
    if chi2 is not None:
        entry['chi2'] = round(float(chi2), 1)
    if note:
        entry['note'] = note
    if labels is not None:
        entry['labels'] = len(label_cat)
        label_cat.append(rid)
        label_blobs.append(np.unique(labels, return_inverse=True)[1].astype(np.int16))
    if br and kind == 'vis':
        entry['br'] = len(br_blobs)
        br_blobs.append(f32(G @ v))
    results.append(entry)
    blobs.append(f32(v))
    return entry


def edge_prob_to_cells(prob, edges):
    out = np.zeros(n)
    np.maximum.at(out, edges[:, 0], prob); np.maximum.at(out, edges[:, 1], prob)
    return out


case_meta = {
    'cov_truth': dict(label='Synthetic: covariates + smooth noise', kind='synthetic'),
    'hast': dict(label='Synthetic: one value per province', kind='synthetic'),
    'hast_var': dict(label='Synthetic: provinces + variation within', kind='synthetic'),
    'real': dict(label='Real data (LCS-1)', kind='real'),
}
for case in CASES:
    k = lambda s: f'{case}_{s}'
    chi = lambda v: float(np.sum((Gw @ v - lm[k('dw')])**2))
    if k('truth') in lm:
        add(case, 'truth', 'truth', 'True VIS', lm[k('truth')])
    add(case, 'gp', 'GP mean', f'GP mean (length {float(lm[k("gp_ell")]):.0f} km)', lm[k('gp_v')], chi2=chi(lm[k('gp_v')]),
        params=dict(h=float(lm[k('gp_h')]), s=float(lm[k('gp_s')]), ell=float(lm[k('gp_ell')])))
    add(case, 'gp', 'GP sd', 'GP uncertainty (1 sd)', lm[k('gp_sd')], kind='sd', br=False)
    add(case, 'gp', 'GP MAP', 'GP, positive VIS (MAP)', lm[k('gp_map')], chi2=chi(lm[k('gp_map')]))
    for m, mname in (('l2', 'L2 (smooth)'), ('tv', 'TV (blocky)')):
        for ek in [str(e) for e in ms['ells']]:
            for r in ms['beta_rel']:
                key = f'{case}_{m}_ell{ek}_b{r:g}'
                if f'{key}_v' not in ms:
                    continue
                ell = float(ms[f'{key}_ell'])
                add(case, m, mname, f'{mname}, length {ell:.0f} km, beta x{r:g}', ms[f'{key}_v'], chi2=ms[f'{key}_chi2'],
                    params=dict(ell=ell, beta=float(ms[f'{key}_beta']), beta_rel=float(r), ell_choice=ek))
    if case == 'real':
        for s, v, lab, K, c2 in zip(mp['settings'], mp['fine_v'], mp['fine_label'], mp['fine_K'], mp['fine_chi2']):
            add(case, 'merging', 'merging', f'Merging lambda x{s[0]:.2g}, gamma {s[1]:g} (K {K})', v, labels=lab,
                chi2=c2, params=dict(lam_rel=float(s[0]), gamma_rel=float(s[1]), K=int(K)))
        add(case, 'merging', 'boundary probability', 'Merging: bootstrap boundary probability',
            edge_prob_to_cells(mp['fine_boot0_prob'], pb['edges']), kind='prob', br=False)
    elif case in ('hast', 'hast_var'):
        for lr in ('1.00', '0.33'):
            add(case, 'merging', 'merging', f'Merging lambda x{lr} (K {int(rs[f"{case}_n1_l{lr}_K"])})',
                rs[f'{case}_n1_l{lr}_v'], labels=rs[f'{case}_n1_l{lr}_label'], params=dict(lam_rel=float(lr)))
        add(case, 'merging', 'boundary probability', 'Merging: bootstrap boundary probability',
            edge_prob_to_cells(rs[f'{case}_n1_boot_prob'], pb['edges']), kind='prob', br=False)
    for q in [str(x) for x in bd['quantities']]:
        for st in ['pos'] + [f'r{r:g}' for r in bd['rough']]:
            for e in ('min', 'max'):
                key = f'{case}_{q}_{st}'
                if f'{key}_v{e}' not in bd:
                    continue
                tag = 'positivity only' if st == 'pos' else f'roughness <= {st[1:]}x L2'
                add(case, 'bounds', f'extremal {e}', f'{q} {e} ({tag}): {float(bd[f"{key}_{e}"]):.2f}', bd[f'{key}_v{e}'],
                    params=dict(quantity=q, setting=st, end=e, value=float(bd[f'{key}_{e}'])))

# domain samplers on their own synthetic truths (white noise h = 1, 2 in whitened units)
vs = load('voronoi_synthetic'); ps = load('partition_synthetic'); lv = load('lvoronoi_synthetic_guided_g0.003')
for truth_name, hs in (('vor6', (1, 2)), ('real7', (1,))):
    lab, vals = vs[f'truth_{truth_name}_label'], vs[f'truth_{truth_name}_v']
    for h in hs:
        case = f'{truth_name}_h{h}'
        case_meta[case] = dict(label=f'Synthetic ({"6 Voronoi domains" if truth_name == "vor6" else "7 domains of the real-data merging"}), '
                                     f'white noise x{h}', kind='synthetic', samplers_only=True)
        add(case, 'truth', 'truth', 'True VIS', vals[lab], labels=lab)
        for c in (0, 1):
            key = f'{truth_name}_h{h}_c{c}'
            if f'{key}_vmean' in vs:
                add(case, 'voronoi', 'Voronoi sampler', f'Voronoi sampler, chain {c}: posterior mean', vs[f'{key}_vmean'],
                    labels=vs[f'{key}_label_last'])
                add(case, 'voronoi', 'boundary probability', f'Voronoi sampler, chain {c}: boundary probability',
                    edge_prob_to_cells(vs[f'{key}_prob'], pb['edges']), kind='prob', br=False)
            if f'{key}_vmean' in lv:
                add(case, 'lvoronoi', 'labelled Voronoi', f'Labelled Voronoi, chain {c}: posterior mean', lv[f'{key}_vmean'],
                    labels=lv[f'{key}_label_last'])
        for g in [float(x) for x in ps['gammas']]:
            key = f'{truth_name}_h{h}_g{g:g}_c0'
            if f'{key}_vmean' in ps:
                add(case, 'partition', 'partition sampler', f'Partition sampler, gamma {g:g}: posterior mean',
                    ps[f'{key}_vmean'], labels=ps[f'{key}_label_last'], params=dict(gamma_rel=g))
                add(case, 'partition', 'boundary probability', f'Partition sampler, gamma {g:g}: boundary probability',
                    edge_prob_to_cells(ps[f'{key}_prob'], pb['edges']), kind='prob', br=False)

# inputs as results (case 'inputs')
for key, label, unit in (('crust', 'Crustal thickness', 'km'), ('lab', 'LAB depth', 'km'),
                         ('heat_flow', 'Heat flow (Lucazeau 2019)', 'mW/m2'),
                         ('heat_flow_seisheat', 'Heat flow (SeisHeat, tomography)', 'mW/m2'),
                         ('heat_flow_heatr', 'Heat flow (Heatr, 13 inputs)', 'mW/m2'),
                         ('age', 'Oldest magmatic age (undated: median)', 'Ma'), ('hm_vis', 'H&M (+ slabs) VIS', 'SI.km')):
    add('inputs', 'covariate', key, label, cov[key], kind='covariate', br=(key == 'hm_vis'), params=dict(unit=unit))
add('inputs', 'covariate', 'class', 'Province class', cov['cls'].astype(float), kind='class', br=False,
    labels=cov['cls'])
add('inputs', 'covariate', 'province', 'Hasterok province (boundaries)', np.zeros(n), kind='class', br=False,
    labels=cov['key'])

# degree ranges and data models (compare_bands.py): the maps of every band, and a table for the Degrees tab
MODEL_NAME = {'lcs1': 'LCS-1', '2e': 'MLI 2E'}
BAND_MAPS = (('gp_v', 'GP mean', 'vis'), ('gp_sd', 'GP uncertainty (1 sd)', 'sd'), ('l2_v', 'L2 (smooth)', 'vis'),
             ('tv_v', 'TV (blocky)', 'vis'))
band_table = None
if os.path.exists(f'{C}/bands_summary.npz'):
    bs = load('bands_summary')
    runs = []
    for b in [str(x) for x in bs['bands']]:
        data, lmin, lmax, noise, synth, Nb = [str(x) for x in bs[f'{b}|meta']]
        synth = synth == '1'
        label = (f'Synthetic only, degrees {lmin}–{lmax}' if synth else f'{MODEL_NAME[data]}, degrees {lmin}–{lmax}'
                 + (' (first build, noise vs MF7)' if noise == 'mf7' else ''))
        runs.append(dict(id=b, label=label, data=data, lmin=int(lmin), lmax=int(lmax), noise=noise, synthetic=synth, N=int(Nb)))
    band_table = dict(runs=runs, fun_names=[str(x) for x in bs['fun_names']], rms_methods=[str(x) for x in bs['rms_methods']],
                      cases={}, truth_fun={})
    for case in CASES:
        if f'{case}_truth' in lm:
            band_table['truth_fun'][case] = {q: round(float(FUN[q] @ lm[f'{case}_truth']), 4) for q in FUN}
        per = {}
        for r in runs:
            k = lambda s: f'{r["id"]}|{case}|{s}'
            if k('hyper') not in bs:
                continue
            e = dict(hyper=fl(bs[k('hyper')]), fun=[fl(x) for x in bs[k('fun')]],
                     bounds={}, maps={})
            if k('rms') in bs:
                e['rms'] = fl(bs[k('rms')])
            if k('cls') in bs:
                e['cls'] = [fl(x) for x in bs[k('cls')]]
            if k('hyp_ll') in bs:
                ll, nm = bs[k('hyp_ll')], bs[k('hyp_names')]
                o = [i for i in np.argsort(-ll) if np.isfinite(ll[i])][:6]
                e['hyp'] = [[str(nm[i]), round(float(ll[i] - ll[o[0]]), 1)] for i in o]
            for key in [x for x in bs if x.startswith(k('bounds|'))]:
                q, st = key.split('|')[-1].rsplit('_', 1)
                e['bounds'].setdefault(q, {})[st] = fl(bs[key])
            for m, mname, kind in BAND_MAPS:
                if k(m) in bs:
                    e['maps'][m] = add(case, 'band', mname, f'{mname}: {r["label"]}', bs[k(m)], kind=kind, br=False,
                                       params=dict(band=r['id']))['index']
            per[r['id']] = e
        if per:
            band_table['cases'][case] = per
    print(f'{len(runs)} bands exported', flush=True)

np.concatenate(blobs).astype(np.float32).tofile(f'{OUT}/results.f32')
np.concatenate(br_blobs).astype(np.float32).tofile(f'{OUT}/brs.f32')
np.concatenate(label_blobs).astype(np.int16).tofile(f'{OUT}/labels.i16')
np.stack([np.mod(z['plon'], 360), z['plat'], z['d_lcs'], z['d_mf7'], z['b']]).astype(np.float32).tofile(f'{OUT}/points.f32')
print(f'{len(results)} maps, {len(br_blobs)} Br sets, {len(label_cat)} partitions', flush=True)

# ------------------------------------------------------------------ explore: per-ell eigen-decompositions
B, lo, hi, names = design(COLS, cov, area)
B.astype(np.float64).tofile(f'{OUT}/explore/columns.f64')
col_meta = [dict(name=nm, lo=float(a), hi=float(b)) for nm, a, b in zip(names, lo, hi)]
col_groups = {}
for c in COLS:
    col_groups[c] = [i for i, nm in enumerate(names) if (nm.startswith('mean ') and c == 'class') or nm == c]
m1 = gmrf.Model(Gw, np.zeros(N), area, Lw, n_cache=1)
dws = np.stack([lm[f'{c}_dw'] for c in CASES])
Fmat = np.stack(list(FUN.values()))
ells = [float(e) for e in gmrf.ELL_GRID]
for i, ell in enumerate(ells):
    R = m1.correlation(ell)
    K = Gw @ R @ Gw.T
    lam, U = np.linalg.eigh(0.5*(K + K.T))
    lam = np.maximum(lam, 0.)
    W = R @ Gw.T @ U
    np.concatenate([lam, (U.T @ Gw @ B).T.ravel(), (dws @ U).ravel(), np.einsum('fi,ij,fj->f', Fmat, R, Fmat)]
                   ).astype(np.float64).tofile(f'{OUT}/explore/ell_{i}.f64')
    W.astype(np.float32).tofile(f'{OUT}/explore/ell_{i}.f32')
    print(f'  ell {ell:.0f} km exported', flush=True)

# parity fixture: Python evaluations to compare with js/lingauss.js
fixture = []
for (ci, i, h, s, cols) in [(3, 6, 6.8, 0.5, ['intercept']), (1, 4, 2.2, 0.5, ['class', 'crust', 'heat_flow']),
                            (0, 3, 2.0, 0.3, ['class', 'crust', 'age', 'undated', 'heat_flow', 'lab']),
                            (2, 8, 1.5, 0.8, ['intercept', 'craton'])]:
    idx = [j for c in cols for j in col_groups[c]]
    m = gmrf.Model(Gw, dws[ci], area, Lw, n_cache=1)
    Bs = B[:, idx]
    ll, chi2w, theta = m.evidence(Bs, lo[idx], hi[idx], h, s, ells[i])
    p = m.posterior(Bs, h, s, ells[i], functionals=FUN)
    v = p['v']
    fixture.append(dict(case=ci, ell_index=i, h=h, s=s, cols=idx, ll=ll, theta=fl(p['theta'], 12),
                        theta_sd=fl(p['theta_sd'], 12), v_head=fl(v[:40], 12), v_sd_head=fl(p['v_sd'][:40], 12),
                        v_mean=float(np.mean(v)), chi2=float(np.sum((Gw @ v - dws[ci])**2)),
                        fun=[[float(a), float(b)] for a, b in p['functionals'].values()]))
with open(f'{OUT}/test/fixture.json', 'w') as f:
    json.dump(fixture, f)

# ------------------------------------------------------------------ tables
tables = dict(regression={}, classes={}, hypotheses={}, bounds={}, gp_functionals={}, null={})
for case in CASES:
    k = lambda s: f'{case}_{s}'
    reg = {}
    for v in ('main', 'heat_flow_seisheat', 'heat_flow_heatr'):
        if k(f'reg_{v}_theta') in lm:
            reg[v] = dict(names=[str(x) for x in lm[k(f'reg_{v}_names')]], theta=fl(lm[k(f'reg_{v}_theta')]),
                          sd=fl(lm[k(f'reg_{v}_sd')]), ll=float(lm[k(f'reg_{v}_ll')]))
    tables['regression'][case] = reg
    if k('cls_mean') in lm:
        tables['classes'][case] = dict(names=list(covariates.CLASS_NAMES), mean=fl(lm[k('cls_mean')]),
                                       mean_sd=fl(lm[k('cls_mean_sd')]), s=fl(lm[k('cls_s')]), ell=fl(lm[k('cls_ell')], 0),
                                       log_sd=fl(lm[k('cls_log_sd')]), dll=float(lm[k('cls_ll')] - lm[k('cls_shared_ll')]),
                                       h=float(lm[k('cls_h')]))
    if k('hyp_ll') in lm:
        ll = lm[k('hyp_ll')]
        tables['hypotheses'][case] = dict(names=[str(x) for x in lm[k('hyp_names')]],
                                          ll=[None if not np.isfinite(x) else round(float(x), 2) for x in ll],
                                          ll_plain=[None if not np.isfinite(x) else round(float(x), 2) for x in lm[k('hyp_ll_plain')]],
                                          K=[int(x) for x in lm[k('hyp_K')]],
                                          hyper=[fl(r, 3) for r in lm[k('hyp_hyper')]])
    tables['gp_functionals'][case] = {str(nm): fl(val) for nm, val in zip(lm[k('gp_fun_names')], lm[k('gp_fun')])}
    bq = {}
    for q in [str(x) for x in bd['quantities']]:
        bq[q] = {}
        for st in ['pos'] + [f'r{r:g}' for r in bd['rough']]:
            key = f'{case}_{q}_{st}'
            if f'{key}_min' in bd:
                bq[q][st] = [round(float(bd[f'{key}_min']), 4), round(float(bd[f'{key}_max']), 4)]
        if k('truth') in lm:
            bq[q]['truth'] = round(float(FUN[q] @ lm[k('truth')]), 4)
    tables['bounds'][case] = bq
sl = [int(i) for i in nr['slopes']]
tables['bands'] = band_table
tables['null'] = dict(names=[str(nr['names'][i]) for i in sl], spread=fl(nr['theta'][:, sl].std(0, ddof=1)),
                      reported_sd=fl(nr['sd'][:, sl].mean(0)),
                      false_positive=fl(np.mean(np.abs(nr['theta'][:, sl]) > 2*nr['sd'][:, sl], 0), 3),
                      n_rep=int(nr['theta'].shape[0]))

# ------------------------------------------------------------------ colour maps, coastline
import matplotlib
cmaps = {name: (np.array(matplotlib.colormaps[name](np.linspace(0, 1, 256)))[:, :3]*255).round().astype(int).tolist()
         for name in ('viridis', 'magma', 'cividis', 'RdBu_r', 'Greys', 'Oranges')}
coast_file = f'{OUT}/coast_tmp.txt'
import pygmt
with pygmt.clib.Session() as lib:
    lib.call_module('coast', f'-R{REGION[0]}/{REGION[1]}/{REGION[2]}/{REGION[3]} -Di -W -M -A500 ->{coast_file}')
coast, seg = [], []
for line in open(coast_file):
    if line.startswith('>'):
        if len(seg) > 1:
            coast.append(seg)
        seg = []
    elif line.strip():
        x, y = map(float, line.split()[:2])
        seg.append([round(np.mod(x, 360), 3), round(y, 3)])
if len(seg) > 1:
    coast.append(seg)
os.remove(coast_file)

# ------------------------------------------------------------------ manifest
commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                        cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
from viewer_info import INFO      # noqa: E402  (plain-English text, notebooks/viewer_info.py)
manifest = dict(
    meta=dict(built=datetime.datetime.now().isoformat(timespec='minutes'), commit=commit, n_cells=n, n_dims=N,
              n_points=len(z['plon']), region=REGION, hm_mean=float(np.average(cov['hm_vis'], weights=area))),
    grid=grid,
    cells=dict(lon=fl(pb['clon'], 3), lat=fl(pb['clat'], 3), area=fl(area, 1), cls=[int(x) for x in cov['cls']],
               group=[str(x) for x in cov['group']], prov_type=[str(x) for x in cov['prov_type']],
               covariates={c: fl(cov[c], 3) for c in ('crust', 'lab', 'heat_flow', 'heat_flow_seisheat', 'heat_flow_heatr', 'age')},
               undated=[int(x) for x in cov['undated']], hm_vis=fl(cov['hm_vis'])),
    classes=list(covariates.CLASS_NAMES),
    cases=[dict(id=c, **case_meta[c]) for c in case_meta],
    explore=dict(cases=CASES, ells=ells, N=N, n=n, columns=col_meta, groups=col_groups, auto_groups=list(GROUPS),
                 functionals=list(FUN), functional_weights={k: fl(v, 8) for k, v in FUN.items()},
                 fitted={c: dict(h=float(lm[f'{c}_gp_h']), s=float(lm[f'{c}_gp_s']), ell=float(lm[f'{c}_gp_ell']))
                         for c in CASES}),
    results=results, label_catalogue=label_cat, tables=tables, colormaps=cmaps, coast=coast, info=INFO)
with open(f'{OUT}/index.json', 'w') as f:
    json.dump(manifest, f, separators=(',', ':'))
sizes = {fn: os.path.getsize(f'{OUT}/{fn}') for fn in ('index.json', 'results.f32', 'brs.f32', 'labels.i16', 'points.f32', 'cells.u16')}
print('wrote', OUT, {k: f'{v/1e6:.1f} MB' for k, v in sizes.items()})
