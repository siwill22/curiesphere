"""
Figures shared by the inversion benchmark pages: what each method's map looks like on the real data, and galleries of
the checkerboard tests with the methods side by side (score in the corner of each panel).

  method_df.png, method_cm.png, method_pd.png        the real Australian runs: VIS and its s.d.
  method_gcm.png, method_gcml.png, method_df_glc.png the real global runs: VIS and its s.d.
  methods_aus.png, methods_global.png                the same, all methods of a region side by side
  gallery_aus.png                                    checkerboards over Australia: truth, DF, CM (with noise)
  gallery_global.png                                 equal-area checkerboards: truth, GCM, DF-GLC (with noise)

Run from the repository root:
    PATH=~/anaconda3/envs/curiesphere/bin:$PATH conda run -n curiesphere python docs/verification/make_overview_figures.py
"""
import json

import numpy as np
import pygmt
import xarray as xr

from make_df_figures import BASE, OUT, REGION, RES, W, boundaries, cell_grid
from make_gcm_figures import WG, global_grid

RUNS = 'inversions/results'
NAMES = dict(df='DF (domain fractal)', cm='CM (cell map)', pd='PD (polygon domains)', gcm='GCM (global cell map)',
             df_glc='DF-GLC (global domain fractal)', gcml='GCM-L (global cell map, domain levels)')
AUS = dict(df='df_aus_lcs1_16-133', cm='cm_aus_lcs1_16-133', pd='pd_aus_lcs1_16-133')
GLOBAL = dict(gcm='gcm_glc_lcs1_16-133', gcml='gcm_glc_layouts_oc_lcs1_16-133', df_glc='df_glc_lcs1_16-133_groups')
SD_AUS, SD_GLOBAL = [0, 0.4], [0, 0.6]


def config():
    pygmt.config(FONT_TITLE='8p', FONT_SUBTITLE='6.5p', FONT_ANNOT_PRIMARY='5.5p', FONT_LABEL='6p',
                 MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE='plain')


def score(fig, text, corner='BR'):
    """the panel's score in a corner, as in scikit-learn's comparison galleries"""
    fig.text(position=corner, text=text, font='8p,Helvetica-Bold,black', fill='white@15', clearance='0.06c/0.04c',
             offset='-0.08c/0.08c' if corner == 'BR' else '0.08c/0.08c', justify=corner, no_clip=True)


def aus_panel(fig, lon, lat, v, cpt, series, title, label, edges=None, sub=None, ticks='af'):
    pygmt.makecpt(cmap=cpt, series=series, background=True)
    fig.basemap(region=REGION, projection=f'M{W}c', frame=[ticks] + ([f'+t{title}' + (f'+s{sub}' if sub else '')] if title else []))
    fig.grdimage(cell_grid(lon, lat, v), cmap=True, nan_transparent=True)
    if edges is not None:
        fig.plot(x=edges[0], y=edges[1], style='p0.02c', fill='white')
    fig.coast(shorelines='0.2p,gray30')
    if label:
        fig.colorbar(position=f'JBC+w{0.8*W}c/0.13c+o0/0.3c+h+e', frame=[f'x+l{label}'])


def global_edges(lon, lat, label):
    g = global_grid(lon, lat, label.astype(float))
    a = g.values
    edge = np.zeros(a.shape, bool)
    edge[:, 1:] |= (a[:, 1:] != a[:, :-1]) & np.isfinite(a[:, 1:]) & np.isfinite(a[:, :-1])
    edge[1:, :] |= (a[1:, :] != a[:-1, :]) & np.isfinite(a[1:, :]) & np.isfinite(a[:-1, :])
    GX, GY = np.meshgrid(g.lon.values, g.lat.values)
    return GX[edge], GY[edge]


def global_panel(fig, lon, lat, v, cpt, series, title, label, edges=None, sub=None):
    pygmt.makecpt(cmap=cpt, series=series, background=True)
    fig.basemap(region='g', projection=f'N0/{WG}c', frame=['f'] + ([f'+t{title}' + (f'+s{sub}' if sub else '')] if title else []))
    fig.coast(land='gray92', water='white')
    fig.grdimage(global_grid(lon, lat, v), cmap=True, nan_transparent=True)
    if edges is not None:
        fig.plot(x=edges[0], y=edges[1], style='p0.008c', fill='gray90')
    fig.coast(shorelines='0.1p,gray55', area_thresh=20000)
    if label:
        fig.colorbar(position=f'JBC+w{0.6*WG}c/0.13c+o0/0.25c+h+e', frame=[f'x+l{label}'])


def run(name):
    d = xr.open_dataset(f'{RUNS}/{name}.nc')
    lon, lat = d.lon.values, d.lat.values
    edges = d.domain.values if 'domain' in d else None
    return lon, lat, d.vis.values, d.vis_sd.values, edges


def method_maps():
    """one figure per method (VIS, s.d.) and one per region with the methods side by side"""
    for region, runs, panel, dx, dy, sd, bounds in (
            ('aus', AUS, aus_panel, W + 0.5, 0.9*W + 1.8, SD_AUS, boundaries),
            ('global', GLOBAL, global_panel, WG + 0.6, 0.5*WG + 2.0, SD_GLOBAL, global_edges)):
        data = {m: run(f) for m, f in runs.items()}
        edges = {m: bounds(lon, lat, lab) if lab is not None else None for m, (lon, lat, _, _, lab) in data.items()}
        for m, (lon, lat, v, s, _) in data.items():
            fig = pygmt.Figure()
            config()
            for j, (val, cpt, series, title, label, e) in enumerate(
                    [(v, 'batlow', [0, 2], 'VIS', 'VIS (SI km)', edges[m]),
                     (s, 'lajolla', sd, 'Its s.d. (uncertainty)', 's.d. (SI km)', None)]):
                with fig.shift_origin(xshift=f'{j*dx}c', yshift=f'{BASE}c'):
                    panel(fig, lon, lat, val, cpt, series, title, label, e)
            fig.savefig(OUT / f'method_{m}.png', dpi=170)
            print(f'method_{m}.png')
        fig = pygmt.Figure()
        config()
        for j, (m, (lon, lat, v, s, _)) in enumerate(data.items()):
            for k, (val, cpt, series, title, label, e) in enumerate(
                    [(v, 'batlow', [0, 2], NAMES[m], 'VIS (SI km)', edges[m]),
                     (s, 'lajolla', sd, f'{NAMES[m].split(" (")[0]}: s.d.', 's.d. (SI km)', None)]):
                with fig.shift_origin(xshift=f'{j*dx}c', yshift=f'{BASE + (1 - k)*dy}c'):
                    panel(fig, lon, lat, val, cpt, series, title, label, e)
        fig.savefig(OUT / f'methods_{region}.png', dpi=170)
        print(f'methods_{region}.png')


def gallery_aus():
    """rows: square sizes; columns: the truth, DF (one domain) and CM, both with noise; contrast in the corner"""
    df_rows = {r['size']: r for r in json.load(open(RES / 'df_shrink.json'))
               if r.get('layout') == 'one' and r.get('means') == 'box' and r.get('noise') is True}
    cm_rows = {r['size']: r for r in json.load(open(RES / 'cm_shrink.json')) if r['noise'] == 'noise'}
    zd, zc = np.load(RES / 'df_shrink.npz'), np.load(RES / 'cm_shrink.npz')
    lon, lat = zd['clon'], zd['clat']
    sizes = [8., 4., 2., 1.5]
    fig = pygmt.Figure()
    config()
    for k, size in enumerate(sizes):
        i, j = np.floor(lat/size).astype(int), np.floor(np.mod(lon, 360)/size).astype(int)
        truth = np.where((i + j) % 2 == 0, 1.5, 0.5)
        panels = [(truth, f'Truth: {size:g}° squares ({size*111:.0f} km)', None),
                  (zd[f'{size:g}_one_box_1'][:, 0], NAMES['df'] if k == 0 else '', df_rows[size]['contrast']),
                  (zc[f'{size:g}_noise'][:, 0], NAMES['cm'] if k == 0 else '', cm_rows[size]['contrast'])]
        for jj, (v, title, c) in enumerate(panels):
            with fig.shift_origin(xshift=f'{jj*(W + 0.3)}c', yshift=f'{BASE + (len(sizes) - 1 - k)*(0.9*W + 1.0)}c'):
                aus_panel(fig, lon, lat, v, 'batlow', [0, 2], title, 'VIS (SI km)' if k == len(sizes) - 1 else None,
                          ticks='f')
                if c is not None:
                    score(fig, f'{c:.2f}')
    fig.savefig(OUT / 'gallery_aus.png', dpi=170)
    print('gallery_aus.png')


def gallery_global():
    """rows: equal-area tile sizes; columns: the truth, GCM and DF-GLC (one domain), both with noise"""
    g = {r['nside']: r for r in json.load(open(RES / 'gcm_shrink.json')) if r['noise'] == 'noise'}
    d = {r['nside']: r for r in json.load(open(RES / 'df_glc_shrink.json')) if r['noise'] == 'noise'}
    zg, zd = np.load(RES / 'gcm_shrink.npz'), np.load(RES / 'df_glc_shrink.npz')
    lon, lat = zg['clon'], zg['clat']
    nsides = [4, 8, 16, 32]
    fig = pygmt.Figure()
    config()
    for k, ns in enumerate(nsides):
        panels = [(zg[f'{ns}_truth'], f'Truth: tiles of {g[ns]["size_km"]:.0f} km', None),
                  (zg[f'{ns}_noise'][:, 0], NAMES['gcm'] if k == 0 else '', g[ns]['contrast']),
                  (zd[f'{ns}_noise'][:, 0], NAMES['df_glc'] if k == 0 else '', d[ns]['contrast'])]
        for jj, (v, title, c) in enumerate(panels):
            with fig.shift_origin(xshift=f'{jj*(WG + 0.4)}c', yshift=f'{BASE + (len(nsides) - 1 - k)*(0.5*WG + 1.0)}c'):
                global_panel(fig, lon, lat, v, 'batlow', [0, 2], title,
                             'VIS (SI km)' if k == len(nsides) - 1 else None)
                if c is not None:
                    score(fig, f'{c:.2f}')
    fig.savefig(OUT / 'gallery_global.png', dpi=150)
    print('gallery_global.png')


if __name__ == '__main__':
    import sys
    todo = sys.argv[1:] or ['methods', 'aus', 'global']
    if 'methods' in todo:
        method_maps()
    if 'aus' in todo:
        gallery_aus()
    if 'global' in todo:
        gallery_global()
