"""Maps of the future-data test (compare_future.py), for the multiscale synthetic truth (hast_ms):
  future_maps.png   top row: the true VIS (province and body outlines), its Br at the data points from
                    degrees 16-300 and from 16-133 (no noise), and the noise of today's data to 133;
                    then one row per run: the TV map, the merging map with its domain boundaries,
                    the error of the TV map (map - truth), and the Br misfit of the TV map (data - model)
Shared colour scales down each column. Spherical geometry: maps only (cell values on their 0.1 deg grid,
Br gridded from the data points).
Run from notebooks/: PYTHONPATH=..:$PYTHONPATH python regions/australia_fine/plot_future.py"""
import numpy as np
import pygmt

from domains import problem
from domains.plotting import cell_grid, point_grid, map_panel

OUT = 'regions/australia_fine'
CFG = f'{OUT}/config.toml'
RUNS = {'syn_16-133': "Today's data, degrees 16-133", 'syn_16-185': "Today's data, degrees 16-185",
        'syn_f10_16-133': 'Future (noise 10%), degrees 16-133', 'syn_f10_16-185': 'Future (noise 10%), degrees 16-185',
        'syn_f10_16-300': 'Future (noise 10%), degrees 16-300', 'syn_f30_16-300': 'Future (noise 30%), degrees 16-300'}
CASE = 'hast_ms'
NOISE_DRAW = 2                                                   # as run_linear_models.py for hast_ms
W = 5.0                                                          # panel width, cm

pb0 = problem.load(CFG)
cells, area = pb0['cells'], pb0['area']
ms = np.load(f'{pb0["cache"]}/multiscale_truth.npz')
truth = ms['truth']
true_lab = np.where(ms['body'] >= 0, ms['prov'].max() + 1 + ms['body'], ms['prov'])


def outline(label):
    """lon, lat of the boundary pixels of a cell labelling on the 0.1 deg grid"""
    g = cell_grid(label.astype(float), cells)
    a = g.values
    b = np.zeros(a.shape, bool)
    b[:, 1:] |= (a[:, 1:] != a[:, :-1]) & np.isfinite(a[:, 1:]) & np.isfinite(a[:, :-1])
    b[1:, :] |= (a[1:, :] != a[:-1, :]) & np.isfinite(a[1:, :]) & np.isfinite(a[:-1, :])
    i, j = np.nonzero(b)
    return g['lon'].values[j], g['lat'].values[i]


def draw_outline(fig, label, colour='black'):
    x, y = outline(label)
    fig.plot(x=x, y=y, style='s0.025c', fill=colour)


rms = lambda v: float(np.sqrt(np.average(v**2, weights=area)))
rows = []
for b, title in RUNS.items():
    pb = problem.load(CFG, band=b)
    lm = np.load(f'{pb["cache"]}/linear_models.npz')
    rs = np.load(f'{pb["cache"]}/realistic_synthetic.npz', allow_pickle=True)
    tv = lm[f'{CASE}_tv_v']
    obs = pb['G'] @ truth + pb['noise'][NOISE_DRAW]              # the data: the truth's field plus this run's noise
    rows.append(dict(title=title, tv=tv, merge=rs[f'{CASE}_n1_l1.00_v'], label=rs[f'{CASE}_n1_l1.00_label'],
                     K=int(rs[f'{CASE}_n1_l1.00_K']), misfit=obs - pb['G'] @ tv, sigma=pb['sigma'],
                     plon=pb['z']['plon'], plat=pb['z']['plat'],
                     br300=pb['G'] @ truth if b == 'syn_f10_16-300' else None,
                     br133=pb['G'] @ truth if b == 'syn_16-133' else None,
                     noise133=pb['noise'][NOISE_DRAW] if b == 'syn_16-133' else None))
    del pb

plon, plat = rows[0]['plon'], rows[0]['plat']
br300 = next(r['br300'] for r in rows if r['br300'] is not None)
br133 = next(r['br133'] for r in rows if r['br133'] is not None)
noise133 = next(r['noise133'] for r in rows if r['noise133'] is not None)
vis = dict(cmap='viridis', series=[0, 2, 0.05], label='VIS (SI.km)')
err = dict(cmap='vik', series=[-1, 1, 0.05], label='map - truth (SI.km)')
bmax = round(float(np.percentile(np.abs(br300), 98)), -1)
br = dict(cmap='vik', series=[-bmax, bmax, 1], label='Br (nT)')
res = dict(cmap='vik', series=[-60, 60, 1], label='data - model Br (nT)')

fig = pygmt.Figure()
pygmt.config(FONT_TITLE='7p', FONT_ANNOT_PRIMARY='6p', FONT_LABEL='6p', MAP_TITLE_OFFSET='2p')


def panel(grid, title, s, col, outline_label=None):
    if col:
        fig.shift_origin(xshift=f'{W + 0.9}c')
    map_panel(fig, grid, title, s['cmap'], s['series'], width=f'{W}c', label=s['label'])
    if outline_label is not None:
        draw_outline(fig, outline_label)


def next_row():
    fig.shift_origin(xshift=f'-{3*(W + 0.9)}c', yshift=f'-{W + 2.1}c')


panel(cell_grid(truth, cells), 'True VIS: 53 provinces, detail, 40 bodies', vis, 0, true_lab)
panel(point_grid(br300, plon, plat), 'Its Br, degrees 16-300 (no noise)', br, 1)
panel(point_grid(br133, plon, plat), 'Its Br, degrees 16-133 (no noise)', br, 2)
panel(point_grid(noise133, plon, plat), f"Noise of today's data to 133: {rows[0]['sigma']:.0f} nT", res, 3)
for r in rows:
    next_row()
    panel(cell_grid(r['tv'], cells), f'{r["title"]}: TV map', vis, 0)
    panel(cell_grid(r['merge'], cells), f'Merging: {r["K"]} domains' + (' (the cap)' if r['K'] == 100 else ''), vis, 1,
          r['label'])
    panel(cell_grid(r['tv'] - truth, cells), f'TV map error: RMS {rms(r["tv"] - truth):.2f} SI.km', err, 2, true_lab)
    panel(point_grid(r['misfit'], plon, plat),
          f'Br misfit of TV: RMS {np.sqrt(np.mean(r["misfit"]**2)):.0f} nT (noise {r["sigma"]:.0f} nT)', res, 3)
fig.savefig(f'{OUT}/future_maps.png', dpi=170)
print('wrote', f'{OUT}/future_maps.png')
