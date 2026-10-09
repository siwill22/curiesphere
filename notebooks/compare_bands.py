"""
How the results depend on the degree range and the data model: collects, for every band of [bands] in
the region config that has been run (build_bands.py, then run_linear_models.py and run_bounds.py with
BAND=<name>), the GP fit, the regional means with their uncertainty, the class means, the covariate
slopes, the best hypotheses, the synthetic map errors and the extremal bounds, and prints them side by
side. The first build (no band) is included as 'first build'.
Output: bands_summary.npz in the region cache (read by the viewer exporter).

Run from notebooks/ (conda env pygmt17):
    PYTHONPATH=..:$PYTHONPATH python compare_bands.py regions/australia/config.toml
"""
import os
import sys
import tomllib

import numpy as np

cfg_path = sys.argv[1]
with open(cfg_path, 'rb') as f:
    cfg = tomllib.load(f)
REG = os.path.dirname(os.path.abspath(cfg_path))
CACHE = f'{REG}/cache'
area = np.load(f'{CACHE}/cells.npz')['area_km2']
BANDS = {'first build': CACHE} | {b: f'{CACHE}/bands/{b}' for b in cfg['bands']['runs']}
BANDS = {b: d for b, d in BANDS.items() if os.path.exists(f'{d}/linear_models.npz')}
MAPS = ('gp_v', 'gp_sd', 'gp_map', 'l2_v', 'tv_v')

# check: the first build rebuilt as a band ('mf7' noise) must give the same GP fit; it is then left out
for b in [b for b in BANDS if cfg['bands']['runs'].get(b, {}).get('noise') == 'mf7']:
    a, c = np.load(f'{CACHE}/linear_models.npz'), np.load(f'{BANDS.pop(b)}/linear_models.npz')
    print(f'check {b}: GP map max difference from the first build {np.abs(a["real_gp_v"] - c["real_gp_v"]).max():.2e} SI.km, '
          f'log evidence {float(a["real_gp_ll"]):.3f} vs {float(c["real_gp_ll"]):.3f}')

rows, save = [], dict(bands=np.array(list(BANDS)), rms_methods=np.array(['gp_v', 'gp_map', 'l2_v', 'tv_v']))
for b, d in BANDS.items():
    lm = dict(np.load(f'{d}/linear_models.npz'))
    bd = dict(np.load(f'{d}/bounds.npz')) if os.path.exists(f'{d}/bounds.npz') else {}
    run = cfg['bands']['runs'].get(b, dict(data='lcs1', lmin=cfg['lmin'], lmax=cfg['lmax_fit'], noise='mf7'))
    E = np.load(f'{d}/errors.npz')
    save[f'{b}|meta'] = np.array([run['data'], run['lmin'], run['lmax'], run.get('noise', cfg['bands']['noise']),
                                  int(run.get('synthetic', False)), E['T'].shape[0]], dtype=str)
    for case in [str(c) for c in lm['cases']]:
        k = f'{case}_gp_h'
        if k not in lm:
            continue
        r = dict(band=b, case=case, N=E['T'].shape[0], h=float(lm[k]), s=float(lm[f'{case}_gp_s']),
                 ell=float(lm[f'{case}_gp_ell']), ll=float(lm[f'{case}_gp_ll']))
        names = [str(x) for x in lm[f'{case}_gp_fun_names']]
        fun = lm[f'{case}_gp_fun']
        r['fun'] = {nm: (float(m), float(s)) for nm, (m, s) in zip(names, fun)}
        truth = lm.get(f'{case}_truth')
        if truth is not None:
            r['rms'] = {m: float(np.sqrt(np.average((lm[f'{case}_{m}'] - truth)**2, weights=area)))
                        for m in ('gp_v', 'gp_map', 'l2_v', 'tv_v') if f'{case}_{m}' in lm}
            save[f'{b}|{case}|rms'] = np.array([r['rms'].get(m, np.nan) for m in ('gp_v', 'gp_map', 'l2_v', 'tv_v')])
        if f'{case}_cls_mean' in lm:
            r['cls'] = list(zip(lm[f'{case}_cls_mean'], lm[f'{case}_cls_mean_sd']))
            save[f'{b}|{case}|cls'] = np.array(r['cls'])
        if f'{case}_reg_main_theta' in lm:
            r['reg'] = {str(n): (float(t), float(s)) for n, t, s in
                        zip(lm[f'{case}_reg_main_names'], lm[f'{case}_reg_main_theta'], lm[f'{case}_reg_main_sd'])}
        if f'{case}_hyp_ll' in lm:
            ll, nm = lm[f'{case}_hyp_ll'], lm[f'{case}_hyp_names']
            o = np.argsort(-ll)
            r['hyp'] = [(str(nm[i]), float(ll[i] - ll[o[0]])) for i in o[:3]]
            save[f'{b}|{case}|hyp_names'], save[f'{b}|{case}|hyp_ll'] = nm, ll
        r['bounds'] = {key[len(case)+1:-4]: (float(bd[key]), float(bd[key[:-4] + '_max']))
                       for key in bd if key.startswith(f'{case}_') and key.endswith('_min')}
        rows.append(r)
        for m in MAPS:
            if f'{case}_{m}' in lm:
                save[f'{b}|{case}|{m}'] = lm[f'{case}_{m}']
        save[f'{b}|{case}|hyper'] = np.array([r['h'], r['s'], r['ell'], r['ll']])
        save[f'{b}|{case}|fun'] = np.array([r['fun'][n] for n in names])
        save['fun_names'] = np.array(names)
        for q, (lo, hi) in r['bounds'].items():
            save[f'{b}|{case}|bounds|{q}'] = np.array([lo, hi])
np.savez(f'{CACHE}/bands_summary.npz', **save)


def table(case, title, cols, w=16):
    sel = [r for r in rows if r['case'] == case]
    if not sel:
        return
    print(f'\n{title} [{case}]')
    print(f'{"band":18s}' + ''.join(f'{c:>{w}s}' for c, _ in cols))
    for r in sel:
        print(f'{r["band"]:18s}' + ''.join(f'{f(r):>{w}s}' for _, f in cols))


pm = lambda t: f'{t[0]:.2f}+-{t[1]:.2f}'
for case in ('real', 'hast', 'hast_var', 'cov_truth'):
    table(case, 'GP fit and regional means (SI.km)',
          [('dims', lambda r: f'{r["N"]}'), ('noise h', lambda r: f'{r["h"]:.2f}'), ('VIS sd s', lambda r: f'{r["s"]:.2f}'),
           ('length km', lambda r: f'{r["ell"]:.0f}')] +
          [(k, (lambda k: lambda r: pm(r['fun'][k]))(k)) for k in next(r['fun'] for r in rows)])
    if case != 'real':
        table(case, 'map error against the truth (area-weighted RMS, SI.km)',
              [(m, (lambda m: lambda r: f'{r.get("rms", {}).get(m, np.nan):.3f}')(m)) for m in ('gp_v', 'gp_map', 'l2_v', 'tv_v')])
    table(case, 'class means (SI.km)', [(f'class {k}', (lambda k: lambda r: pm(r['cls'][k]) if 'cls' in r else '-')(k)) for k in range(3)])
    table(case, 'bounds (SI.km): positivity only | roughness <= 1 x L2',
          [(q, (lambda q: lambda r: (f'{r["bounds"][q + "_pos"][0]:.2f}-{r["bounds"][q + "_pos"][1]:.2f}|'
                                     f'{r["bounds"][q + "_r1"][0]:.2f}-{r["bounds"][q + "_r1"][1]:.2f}')
                if q + '_pos' in r['bounds'] and q + '_r1' in r['bounds'] else '-')(q))
           for q in ('NAC mean', 'WAC mean', 'SAC mean', 'craton - rest')], w=24)
    sel = [r for r in rows if r['case'] == case and 'hyp' in r]
    if sel:
        print(f'\nbest hypotheses [{case}] (log evidence relative to the best)')
        for r in sel:
            print(f'{r["band"]:18s}' + ' | '.join(f'{n} ({d:+.1f})' for n, d in r['hyp']))
print(f'\nsaved {CACHE}/bands_summary.npz ({len(BANDS)} bands)')
