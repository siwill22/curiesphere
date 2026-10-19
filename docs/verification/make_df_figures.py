"""
Maps for the DF part of the verification document, from benchmarks/verification/results/df_v2.npz (df_checks.py v2).

  df_v2_maps.png   three of the drawn truths (rows): the truth, DF's map with the hyperparameters at their true values,
                   DF's map with them fitted, the fitted map minus the truth, its reported s.d., and the error in units
                   of the reported s.d.
  df_v2_spread.png how far DF's maps are from the truth, in units of the s.d. DF reports, over all truths and cells,
                   against the bell curve they should follow

Run from the repository root:
    conda run -n curiesphere python docs/verification/make_df_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pygmt

RES = Path('benchmarks/verification/results')
OUT = Path('docs/verification/figures')
REGION = [110, 157, -45, -8]
W = 4.6
BASE = 2.0                                  # cm below the lowest row, for its colour bar


def cell_grid(lon, lat, v, step=0.2, max_km=70.):
    """cell values on a lon/lat grid over REGION: each node takes its nearest cell centre (great-circle distance),
    blank beyond max_km (outside the target)"""
    import xarray as xr
    from scipy.spatial import cKDTree
    unit = lambda lo, la: np.c_[np.cos(np.radians(la))*np.cos(np.radians(lo)),
                                np.cos(np.radians(la))*np.sin(np.radians(lo)), np.sin(np.radians(la))]
    gx = np.arange(REGION[0], REGION[1] + step/2, step)
    gy = np.arange(REGION[2], REGION[3] + step/2, step)
    GX, GY = np.meshgrid(gx, gy)
    d, i = cKDTree(unit(lon, lat)).query(unit(GX.ravel(), GY.ravel()))
    g = np.where(d*6371. <= max_km, np.asarray(v)[i], np.nan).reshape(GX.shape)
    return xr.DataArray(g, coords=dict(lat=gy, lon=gx), dims=('lat', 'lon'))


def cells_map(fig, lon, lat, v, cpt, series, title, label, edge=None):
    proj = f'M{W}c'
    pygmt.makecpt(cmap=cpt, series=series, background=True)
    fig.basemap(region=REGION, projection=proj, frame=['af', f'+t{title}'])
    fig.grdimage(cell_grid(lon, lat, v), cmap=True, nan_transparent=True)
    fig.coast(shorelines='0.2p,gray30')
    fig.colorbar(position=f'JBC+w{0.8*W}c/0.13c+o0/0.3c+h+e', frame=[f'x+l{label}'])


def domain_edges(label, lon, lat):
    """cells with a neighbour (within 1.3 cell spacings) in another domain"""
    from scipy.spatial import cKDTree
    xyz = np.c_[np.cos(np.radians(lat))*np.cos(np.radians(lon)), np.cos(np.radians(lat))*np.sin(np.radians(lon)),
                np.sin(np.radians(lat))]
    d = 1.3*np.median(cKDTree(xyz).query(xyz, 2)[0][:, 1])
    t = cKDTree(xyz)
    return np.array([np.any(label[t.query_ball_point(p, d)] != label[i]) for i, p in enumerate(xyz)])


def maps(z):
    lon, lat = z['clon'], z['clat']
    edge = domain_edges(z['label'], lon, lat)
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='7.5p', FONT_ANNOT_PRIMARY='5.5p', FONT_LABEL='6p', MAP_TITLE_OFFSET='3p',
                 MAP_FRAME_TYPE='plain')
    n = len(z['v_true'])
    for k in range(n):
        vt, vf, vx, sd = z['v_true'][k], z['v_fit'][k], z['v_fixed'][k], z['v_sd'][k]
        err = vf - vt
        panels = [(vt, 'batlow', [0, 1.6], f'Truth {k + 1}', 'VIS (SI km)'),
                  (vx, 'batlow', [0, 1.6], 'DF, true settings', 'VIS (SI km)'),
                  (vf, 'batlow', [0, 1.6], 'DF, fitted settings', 'VIS (SI km)'),
                  (err, 'vik', [-0.4, 0.4], 'DF minus truth', 'SI km'),
                  (sd, 'lajolla', [0, 0.3], 'Reported s.d.', 'SI km'),
                  (err/sd, 'vik', [-3, 3], 'Error / reported s.d.', 'number of s.d.')]
        for j, (v, cpt, series, title, label) in enumerate(panels):
            with fig.shift_origin(xshift=f'{j*(W + 0.4)}c', yshift=f'{BASE + (n - 1 - k)*(0.9*W + 1.6)}c'):
                cells_map(fig, lon, lat, v, cpt, series, title, label)
    fig.savefig(OUT / 'df_v2_maps.png', dpi=170)
    print('df_v2_maps.png')


def spread(z):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True, sharey=True)
    x = np.linspace(-4.5, 4.5, 400)
    bell = np.exp(-x**2/2)/np.sqrt(2*np.pi)
    for ax, (key, what) in zip(axs, [('z_fixed', 'settings held at their true values'),
                                     ('z_fit', 'settings fitted, as in a real run')]):
        v = z[key].ravel()
        ax.hist(v, bins=np.linspace(-4.5, 4.5, 61), density=True, color='0.7', label='all cells of all truths')
        ax.plot(x, bell, 'k', lw=1.2, label='what it should be')
        ax.set_title(f'DF, {what}\nwithin 1 s.d.: {np.mean(np.abs(v) < 1):.0%} (should be 68%), '
                     f'within 2 s.d.: {np.mean(np.abs(v) < 2):.0%} (95%)', fontsize=8.5, loc='left')
        ax.set_xlabel('(DF map minus truth) / reported s.d.')
    axs[0].legend(fontsize=7.5, frameon=False)
    fig.savefig(OUT / 'df_v2_spread.png', dpi=170)
    print('df_v2_spread.png')


def checker_maps():
    """rows: tile sizes; columns: the truth, then DF's map minus the truth for box and Gaussian means, without and with
    noise"""
    import json
    z = np.load(RES / 'df_checker.npz')
    info = {(r['tile_nside'], r['noise'], r['means']): r for r in json.load(open(RES / 'df_checker.json'))}
    lon, lat = z['clon'], z['clat']
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='7.5p', FONT_SUBTITLE='6.5p', FONT_ANNOT_PRIMARY='5.5p', FONT_LABEL='6p',
                 MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE='plain')
    sizes = [4, 8, 16]
    for k, ns in enumerate(sizes):
        truth = z[f'{ns}_truth']
        panels = [(truth, 'batlow', [0, 1.6], f'Truth: {info[(ns, False, "box")]["tiles"]} tiles', None, 'VIS (SI km)')]
        for noisy in (False, True):
            for means in ('box', 'gaussian'):
                r = info[(ns, noisy, means)]
                err = z[f'{ns}_{int(noisy)}_{means}'][:, 0] - truth
                panels.append((err, 'vik', [-0.2, 0.2] if noisy else [-0.05, 0.05],
                               f'{means.capitalize()} means, {"noise" if noisy else "no noise"}',
                               f'DF minus truth, max {r["max_abs_error"]:.1e} SI km', 'SI km'))
        for j, (v, cpt, series, title, sub, label) in enumerate(panels):
            with fig.shift_origin(xshift=f'{j*(W + 0.5)}c', yshift=f'{BASE + (len(sizes) - 1 - k)*(0.9*W + 1.8)}c'):
                proj = f'M{W}c'
                pygmt.makecpt(cmap=cpt, series=series, background=True)
                fig.basemap(region=REGION, projection=proj, frame=['af', f'+t{title}' + (f'+s{sub}' if sub else '')])
                fig.grdimage(cell_grid(lon, lat, v), cmap=True, nan_transparent=True)
                fig.coast(shorelines='0.2p,gray30')
                fig.colorbar(position=f'JBC+w{0.8*W}c/0.13c+o0/0.3c+h+e', frame=[f'x+l{label}'])
    fig.savefig(OUT / 'df_checker.png', dpi=170)
    print('df_checker.png')


def shrink_figures():
    """the recovered contrast against tile size, and maps of DF's VIS at four sizes"""
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in json.load(open(RES / 'df_shrink.json')) if 'failed' not in r]
    setups = [('tiles', 'box', 'told the tiles, box means', '#1b6ca8'),
              ('tiles', 'gaussian', 'told the tiles, Gaussian means', '#d1495b'),
              ('one', 'box', 'not told (one domain)', '#2a2a2a')]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    for lay, means, name, colour in setups:
        for noisy, style in ((False, '-'), (True, '--')):
            sel = sorted((r for r in rows if r['layout'] == lay and r['means'] == means and r['noise'] == noisy),
                         key=lambda r: r['size'])
            if sel:
                ax.plot([r['size']*111 for r in sel], [r['contrast'] for r in sel], style, color=colour, marker='o',
                        ms=3.5, label=f'{name}, {"noise" if noisy else "no noise"}')
    ax.axhline(1, color='0.6', lw=0.8)
    ax.set_xscale('log')
    ax.minorticks_off()
    ax.set_xticks([170, 250, 330, 450, 550, 670, 900])
    ax.set_xticklabels(['170', '250', '330', '450', '550', '670', '900'])
    ax.set_xlabel('checkerboard square size (km, north-south)')
    ax.set_ylabel('recovered contrast / true contrast')
    ax.set_title('How much of the checkerboard DF recovers', loc='left', fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(OUT / 'df_shrink_contrast.png', dpi=170)
    print('df_shrink_contrast.png')

    z = np.load(RES / 'df_shrink.npz')
    lon, lat = z['clon'], z['clat']
    show = [8., 4., 2.5, 1.5]
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='7.5p', FONT_SUBTITLE='6.5p', FONT_ANNOT_PRIMARY='5.5p', FONT_LABEL='6p',
                 MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE='plain')
    for k, size in enumerate(show):
        i, j = np.floor(lat/size).astype(int), np.floor(np.mod(lon, 360)/size).astype(int)
        truth = np.where((i + j) % 2 == 0, 1.5, 0.5)
        panels = [(truth, f'Truth: {size:g}° squares', None)]
        for lay, means, name in (('tiles', 'gaussian', 'Told the squares'), ('one', 'box', 'Not told')):
            for noisy in (False, True):
                key = f'{size:g}_{lay}_{means}_{int(noisy)}'
                if key in z:
                    r = next(r for r in rows if r['size'] == size and r['layout'] == lay and r['means'] == means
                             and r['noise'] == noisy)
                    panels.append((z[key][:, 0], f'{name}, {"noise" if noisy else "no noise"}',
                                   f'contrast {r["contrast"]:.2f}'))
        for jj, (v, title, sub) in enumerate(panels):
            with fig.shift_origin(xshift=f'{jj*(W + 0.5)}c', yshift=f'{BASE + (len(show) - 1 - k)*(0.9*W + 1.8)}c'):
                pygmt.makecpt(cmap='batlow', series=[0, 2], background=True)
                fig.basemap(region=REGION, projection=f'M{W}c',
                            frame=['af', f'+t{title}' + (f'+s{sub}' if sub else '')])
                fig.grdimage(cell_grid(lon, lat, v), cmap=True, nan_transparent=True)
                fig.coast(shorelines='0.2p,gray30')
                fig.colorbar(position=f'JBC+w{0.8*W}c/0.13c+o0/0.3c+h+e', frame=['x+lVIS (SI km)'])
    fig.savefig(OUT / 'df_shrink_maps.png', dpi=170)
    print('df_shrink_maps.png')


def v1_maps():
    """DF's map against the independent solver's, at fixed settings (V1b)"""
    z = np.load(RES / 'df_v1.npz')
    lon, lat = z['clon'], z['clat']
    d = z['v_df'] - z['v_ref']
    edges = boundaries(lon, lat, z['label'])
    panels = [(z['truth'], 'batlow', [0, 2], 'The truth (synthetic)', None, 'VIS (SI km)'),
              (z['v_df'], 'batlow', [0, 2], 'DF', None, 'VIS (SI km)'),
              (z['v_ref'], 'batlow', [0, 2], 'Independent solver', None, 'VIS (SI km)'),
              (d, 'vik', [-2e-14, 2e-14], 'DF minus independent', f'max {np.abs(d).max():.1e} SI km', 'SI km'),
              (z['sd_df'], 'lajolla', [0, 0.4], 'DF: s.d.', None, 'SI km'),
              (z['sd_df']/z['sd_ref'] - 1, 'vik', [-5e-14, 5e-14], 'DF s.d. / independent s.d. - 1',
               f'max {np.abs(z["sd_df"]/z["sd_ref"] - 1).max():.1e}', '')]
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='7.5p', FONT_SUBTITLE='6.5p', FONT_ANNOT_PRIMARY='5.5p', FONT_LABEL='6p',
                 MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE='plain')
    for j, (v, cpt, series, title, sub, label) in enumerate(panels):
        with fig.shift_origin(xshift=f'{(j % 3)*(W + 0.5)}c', yshift=f'{BASE + (1 - j//3)*(0.9*W + 1.8)}c'):
            pygmt.makecpt(cmap=cpt, series=series, background=True)
            fig.basemap(region=REGION, projection=f'M{W}c', frame=['af', f'+t{title}' + (f'+s{sub}' if sub else '')])
            fig.grdimage(cell_grid(lon, lat, v), cmap=True, nan_transparent=True)
            if j < 3:
                fig.plot(x=edges[0], y=edges[1], style='p0.02c', fill='white')
            fig.coast(shorelines='0.2p,gray30')
            fig.colorbar(position=f'JBC+w{0.8*W}c/0.13c+o0/0.3c+h+e', frame=[f'x+l{label or "ratio - 1"}'])
    fig.savefig(OUT / 'df_v1_maps.png', dpi=170)
    print('df_v1_maps.png')


def boundaries(lon, lat, label):
    """grid nodes where the domain changes, from the nearest-cell grid of the labels"""
    g = cell_grid(lon, lat, label.astype(float))
    a = g.values
    edge = np.zeros(a.shape, bool)
    edge[:, 1:] |= (a[:, 1:] != a[:, :-1]) & np.isfinite(a[:, 1:]) & np.isfinite(a[:, :-1])
    edge[1:, :] |= (a[1:, :] != a[:-1, :]) & np.isfinite(a[1:, :]) & np.isfinite(a[:-1, :])
    GX, GY = np.meshgrid(g.lon.values, g.lat.values)
    return GX[edge], GY[edge]


def mismatch_maps():
    import json
    rows = json.load(open(RES / 'df_mismatch.json'))
    z = np.load(RES / 'df_mismatch.npz')
    lon, lat = z['clon'], z['clat']
    layouts = list(dict.fromkeys(r['layout'] for r in rows))
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='7.5p', FONT_SUBTITLE='6.5p', FONT_ANNOT_PRIMARY='5.5p', FONT_LABEL='6p',
                 MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE='plain')
    cases = [(8., False), (8., True), (4., False), (4., True)]
    for k, (size, noisy) in enumerate(cases):
        i, j = np.floor(lat/size).astype(int), np.floor(np.mod(lon, 360)/size).astype(int)
        truth = np.where((i + j) % 2 == 0, 1.5, 0.5)
        sel = {r['layout']: r for r in rows if r['size'] == size and r['noise'] == noisy}
        best = max(r['ll'] for r in sel.values())
        panels = [(truth, f'Truth: {size:g}° squares, {"noise" if noisy else "no noise"}', None, None)]
        for lay in layouts:
            r = sel[lay]
            edges = boundaries(lon, lat, z[f'label_{lay}']) if f'label_{lay}' in z else None
            name = lay.split(' (')[0]
            panels.append((z[f'{size:g}_{lay}_{int(noisy)}'][:, 0], f'DF with {name} ({r["domains"]})',
                           f'contrast {r["contrast"]:.2f}, log evidence {r["ll"] - best:+.0f}', edges))
        for jj, (v, title, sub, edges) in enumerate(panels):
            with fig.shift_origin(xshift=f'{jj*(W + 0.5)}c', yshift=f'{BASE + (len(cases) - 1 - k)*(0.9*W + 1.8)}c'):
                pygmt.makecpt(cmap='batlow', series=[0, 2], background=True)
                fig.basemap(region=REGION, projection=f'M{W}c',
                            frame=['af', f'+t{title}' + (f'+s{sub}' if sub else '')])
                fig.grdimage(cell_grid(lon, lat, v), cmap=True, nan_transparent=True)
                if edges is not None and len(edges[0]):
                    fig.plot(x=edges[0], y=edges[1], style='p0.02c', fill='white')
                fig.coast(shorelines='0.2p,gray30')
                fig.colorbar(position=f'JBC+w{0.8*W}c/0.13c+o0/0.3c+h+e', frame=['x+lVIS (SI km)'])
    fig.savefig(OUT / 'df_mismatch.png', dpi=170)
    print('df_mismatch.png')


def matching_figures():
    """maps for the first truth of three cases, and the evidence of every layout against the polygons"""
    import json
    rows = json.load(open(RES / 'df_matching.json'))
    z = np.load(RES / 'df_matching.npz')
    lon, lat = z['clon'], z['clat']
    edges = boundaries(lon, lat, z['label_polygons'])
    show = ['polygons', 'random tiles 1', 'one domain']
    cases = [(0.0, True, 'breaks only'), (0.3, True, 'breaks and steps'), (0.3, False, 'steps only')]
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='7.5p', FONT_SUBTITLE='6.5p', FONT_ANNOT_PRIMARY='5.5p', FONT_LABEL='6p',
                 MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE='plain')
    for k, (tau, restart, name) in enumerate(cases):
        sel = {r['layout']: r for r in rows if r['tau'] == tau and r['restart'] == restart and r['draw'] == 0}
        best = sel['polygons']['ll']
        sub = f'level s.d. {tau:g}, {"pattern restarts" if restart else "one pattern"}'
        panels = [(z[f'truth_{tau:g}_{int(restart)}'], f'Truth: {name}', sub, edges)]
        for lay in show:
            r = sel[lay]
            e = boundaries(lon, lat, z[f'label_{lay}']) if lay != 'one domain' else None
            panels.append((z[f'{tau:g}_{int(restart)}_{lay}'][:, 0], f'DF with {lay} ({r["domains"]})',
                           f'log evidence {r["ll"] - best:+.0f}, RMS error {r["rms_error"]:.2f}', e))
        for jj, (v, title, sub, e) in enumerate(panels):
            with fig.shift_origin(xshift=f'{jj*(W + 0.5)}c', yshift=f'{BASE + (len(cases) - 1 - k)*(0.9*W + 1.8)}c'):
                pygmt.makecpt(cmap='batlow', series=[0, 2], background=True)
                fig.basemap(region=REGION, projection=f'M{W}c', frame=['af', f'+t{title}+s{sub}'])
                fig.grdimage(cell_grid(lon, lat, v), cmap=True, nan_transparent=True)
                if e is not None and len(e[0]):
                    fig.plot(x=e[0], y=e[1], style='p0.02c', fill='white')
                fig.coast(shorelines='0.2p,gray30')
                fig.colorbar(position=f'JBC+w{0.8*W}c/0.13c+o0/0.3c+h+e', frame=['x+lVIS (SI km)'])
    fig.savefig(OUT / 'df_matching_maps.png', dpi=170)
    print('df_matching_maps.png')

    # log evidence of each layout minus that of the polygons, every truth; one column per case
    order = [(0.0, True), (0.1, True), (0.3, True), (0.6, True), (0.3, False)]
    names = ['0, breaks only', '0.1', '0.3', '0.6', '0.3, steps only']
    poly = {(r['tau'], r['restart'], r['draw']): r['ll'] for r in rows if r['layout'] == 'polygons'}
    d = [r['ll'] - poly[(r['tau'], r['restart'], r['draw'])] for r in rows]
    lo = min(d) - 30
    fig = pygmt.Figure()
    pygmt.config(FONT_ANNOT_PRIMARY='7p', FONT_LABEL='8p', FONT_TITLE='9p', MAP_FRAME_TYPE='plain',
                 MAP_LABEL_OFFSET='0.55c')
    fig.basemap(region=[0.4, len(order) + 0.6, lo, 20], projection='X13c/7c',
                frame=['WSne+tEvidence when the polygons are right', 'xa100+lpolygon level s.d. (SI km); pattern restarts at polygon edges unless marked',
                       'yaf+llog evidence minus polygons'])
    fig.plot(x=[0.4, len(order) + 0.6], y=[0, 0], pen='0.5p,gray40,-')
    fig.plot(x=[4.5, 4.5], y=[lo, 20], pen='0.4p,gray60')
    for name, sym, col, dx in (('random tiles', 'c0.16c', 'orange', -0.12), ('one domain', 's0.17c', 'steelblue', 0.12)):
        xs, ys = [], []
        for r, dd in zip(rows, d):
            if r['layout'].startswith(name):
                xs.append(order.index((r['tau'], r['restart'])) + 1 + dx + 0.04*(r['draw'] - 1))
                ys.append(dd)
        fig.plot(x=xs, y=ys, style=sym, fill=col, pen='0.3p,black', label=name)
    fig.text(x=list(range(1, len(order) + 1)), y=[lo]*len(order), text=names, font='7p', justify='CT',
             offset='0/-0.15c', no_clip=True)
    fig.legend(position='JBL+jBL+o0.2c', box='+gwhite+p0.3p')
    fig.savefig(OUT / 'df_matching_evidence.png', dpi=200)
    print('df_matching_evidence.png')

def sample_figures():
    """coverage with the settings fitted, sampled briefly (explore) and sampled in full; maps of the s.d."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z = np.load(RES / 'df_sample.npz')
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True, sharey=True)
    x = np.linspace(-4.5, 4.5, 400)
    bell = np.exp(-x**2/2)/np.sqrt(2*np.pi)
    for ax, (key, what) in zip(axs, [('z_fit', 'settings fitted'), ('z_explore', 'sampled, 2000 steps'),
                                     ('z_sampled', 'sampled, 12000 steps')]):
        v = z[key].ravel()
        ax.hist(v, bins=np.linspace(-4.5, 4.5, 61), density=True, color='0.7', label='all cells of all truths')
        ax.plot(x, bell, 'k', lw=1.2, label='what it should be')
        ax.set_title(f'DF, {what}\nwithin 1 s.d.: {np.mean(np.abs(v) < 1):.0%} (should be 68%), '
                     f'within 2 s.d.: {np.mean(np.abs(v) < 2):.0%} (95%)', fontsize=8.5, loc='left')
        ax.set_xlabel('(DF map minus truth) / reported s.d.')
    axs[0].legend(fontsize=7.5, frameon=False)
    fig.savefig(OUT / 'df_sample_spread.png', dpi=170)
    print('df_sample_spread.png')

    lon, lat = z['clon'], z['clat']
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='7.5p', FONT_ANNOT_PRIMARY='5.5p', FONT_LABEL='6p', MAP_TITLE_OFFSET='3p',
                 MAP_FRAME_TYPE='plain')
    n = len(z['v_true'])
    for k in range(n):
        vt, vf, vs, sf, ss = z['v_true'][k], z['v_fit'][k], z['v_sampled'][k], z['sd_fit'][k], z['sd_sampled'][k]
        panels = [(vt, 'batlow', [0, 1.6], f'Truth {k + 1}', 'VIS (SI km)'),
                  (vs, 'batlow', [0, 1.6], 'DF, settings sampled', 'VIS (SI km)'),
                  (sf, 'lajolla', [0, 0.3], 's.d., settings fitted', 'SI km'),
                  (ss, 'lajolla', [0, 0.3], 's.d., settings sampled', 'SI km'),
                  ((vf - vt)/sf, 'vik', [-3, 3], 'Error / s.d., fitted', 'number of s.d.'),
                  ((vs - vt)/ss, 'vik', [-3, 3], 'Error / s.d., sampled', 'number of s.d.')]
        for j, (v, cpt, series, title, label) in enumerate(panels):
            with fig.shift_origin(xshift=f'{j*(W + 0.4)}c', yshift=f'{BASE + (n - 1 - k)*(0.9*W + 1.6)}c'):
                cells_map(fig, lon, lat, v, cpt, series, title, label)
    fig.savefig(OUT / 'df_sample_maps.png', dpi=170)
    print('df_sample_maps.png')


if __name__ == '__main__':
    if 'sample' in sys.argv:
        sample_figures()
        sys.exit()
    if (RES / 'df_matching.json').exists():
        matching_figures()
    if 'matching' in sys.argv:
        sys.exit()
    if (RES / 'df_v1.npz').exists():
        v1_maps()
    if (RES / 'df_mismatch.json').exists():
        mismatch_maps()
    if (RES / 'df_shrink.json').exists():
        shrink_figures()
    if (RES / 'df_v2.npz').exists():
        z = np.load(RES / 'df_v2.npz')
        maps(z)
        spread(z)
    if (RES / 'df_checker.npz').exists():
        checker_maps()
    if (RES / 'df_sample.npz').exists():
        sample_figures()
