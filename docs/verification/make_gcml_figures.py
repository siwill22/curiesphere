"""
Figures for the GCM-L part of the verification document (GCM with domain levels, and GCM-L against DF-GLC), from
benchmarks/verification/results/gcml_* (gcml_checks.py).

  gcml_l1_maps.png       GCM-L against the independent solver, provinces layout
  gcml_gallery.png       the four truths (rows, Africa) and what GCM-L and DF-GLC make of them; province-mean error in
                         the corner
  gcml_evidence.png      each layout's log evidence minus that of the provinces, per truth, both methods

Run from the repository root:
    PATH=~/anaconda3/envs/curiesphere/bin:$PATH PYTHONPATH=docs/verification:$PYTHONPATH \
        conda run -n curiesphere python docs/verification/make_gcml_figures.py
"""
import json

import numpy as np

from make_df_figures import BASE, OUT, RES
from make_gcm_figures import WG
from make_overview_figures import config, global_edges, global_panel, score

TRUTHS = {'A': 'A: levels per province,\nindependent cells', 'B': 'B: levels per province,\npattern breaks at edges',
          'C': 'C: one pattern,\nsteps at province edges', 'D': 'D: one pattern,\nno geology (null)'}
CONTROLS = ['one', 'groups'] + [f'voronoi_provinces:{i}' for i in range(3)] + [f'rotated_provinces:{i}' for i in range(3)]
SHORT = {'one': 'one level', 'groups': 'groups', 'voronoi_provinces:0': 'random 1', 'voronoi_provinces:1': 'random 2',
         'voronoi_provinces:2': 'random 3', 'rotated_provinces:0': 'rotated 1', 'rotated_provinces:1': 'rotated 2',
         'rotated_provinces:2': 'rotated 3'}


def rows():
    return {(r['method'], r['layout'], r['truth'], r['draw']): r
            for r in map(json.loads, open(RES / 'gcml_l2_fits.jsonl'))}


def l1_maps():
    import pygmt
    z = np.load(RES / 'gcml_l1.npz')
    lon, lat = z['clon'], z['clat']
    d = z['v'] - z['v_ref']
    r = z['sd']/z['sd_ref'] - 1
    lim = lambda a: float(f'{max(np.abs(a).max(), 1e-16):.0e}')
    edges = global_edges(lon, lat, z['label'])
    panels = [(z['truth'], 'batlow', [0, 2], 'Truth (GCM-L\'s model, provinces)', 'VIS (SI km)', edges),
              (z['v'], 'batlow', [0, 2], 'GCM-L', 'VIS (SI km)', edges),
              (d, 'vik', [-lim(d), lim(d)], f'GCM-L minus independent solver (max {np.abs(d).max():.0e})', 'SI km', None),
              (r, 'vik', [-lim(r), lim(r)], f'GCM-L s.d. / independent s.d. - 1 (max {np.abs(r).max():.0e})', '', None)]
    fig = pygmt.Figure()
    config()
    for j, (v, cpt, series, title, label, e) in enumerate(panels):
        with fig.shift_origin(xshift=f'{(j % 2)*(WG + 0.6)}c', yshift=f'{BASE + (1 - j//2)*(0.5*WG + 2.0)}c'):
            global_panel(fig, lon, lat, v, cpt, series, title, label or 'ratio - 1', e)
    fig.savefig(OUT / 'gcml_l1_maps.png', dpi=170)
    print('gcml_l1_maps.png')


AFRICA = [-20, 55, -37, 38]
WA = 5.2                                                    # width of an Africa panel (cm)


def africa_panel(fig, lon, lat, v, title, label, edges=None):
    """one map of Africa from the 0.5 deg nearest-cell grid of make_gcm_figures"""
    import pygmt
    from make_gcm_figures import global_grid
    g = global_grid(lon, lat, v).sel(lon=slice(AFRICA[0], AFRICA[1]), lat=slice(AFRICA[2], AFRICA[3]))
    pygmt.makecpt(cmap='batlow', series=[0, 2], background=True)
    fig.basemap(region=AFRICA, projection=f'M{WA}c', frame=['f'] + ([f'+t{title}'] if title else []))
    fig.grdimage(g, cmap=True, nan_transparent=True)
    if edges is not None:
        x, y = edges
        s = (x >= AFRICA[0]) & (x <= AFRICA[1]) & (y >= AFRICA[2]) & (y <= AFRICA[3])
        fig.plot(x=x[s], y=y[s], style='p0.012c', fill='white')
    fig.coast(shorelines='0.2p,gray40')
    if label:
        fig.colorbar(position=f'JBC+w{0.8*WA}c/0.13c+o0/0.3c+h+e', frame=[f'x+l{label}'])


def gallery():
    import pygmt
    z = np.load(RES / 'gcml_l2_maps.npz')
    R = rows()
    lon, lat = z['clon'], z['clat']
    edges = global_edges(lon, lat, z['provinces'])
    cols = [(None, 'Truth'), (('GCM-L', 'provinces'), 'GCM-L, provinces'), (('DF-GLC', 'provinces'), 'DF-GLC, provinces'),
            (('DF-GLC', 'one'), 'DF-GLC, one pattern')]
    fig = pygmt.Figure()
    config()
    for k, t in enumerate(TRUTHS):
        for j, (key, title) in enumerate(cols):
            v = z[f'truth_{t}'] if key is None else z[f'{key[0]}_{key[1]}_{t}']
            with fig.shift_origin(xshift=f'{j*(WA + 0.3)}c', yshift=f'{BASE + (len(TRUTHS) - 1 - k)*(1.07*WA + 1.0)}c'):
                ttl = f'Truth {t}' if j == 0 else (title if k == 0 else '')
                africa_panel(fig, lon, lat, v, ttl, 'VIS (SI km)' if k == len(TRUTHS) - 1 else None,
                             edges if j < 3 else None)
                if key is not None:
                    score(fig, f'{R[key[0], key[1], t, 0]["province_rms_error"]:.2f}')
    fig.savefig(OUT / 'gcml_gallery.png', dpi=170)
    print('gcml_gallery.png')


def evidence():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = rows()
    draws = sorted({k[3] for k in R})
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6), constrained_layout=True, sharey=False)
    x = np.arange(len(CONTROLS))
    for ax, t in zip(axs, TRUTHS):
        for m, colour, dx in (('GCM-L', '#d1495b', -0.15), ('DF-GLC', '#1b6ca8', 0.15)):
            for i in draws:
                d = [R[m, c, t, i]['ll'] - R[m, 'provinces', t, i]['ll'] for c in CONTROLS]
                ax.plot(x + dx, d, 'o', color=colour, ms=3.5, alpha=0.8, label=m if i == draws[0] else None)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[c] for c in CONTROLS], rotation=60, fontsize=7.5)
        ax.set_title(TRUTHS[t], fontsize=9, loc='left')
        ax.set_yscale('symlog', linthresh=100)
        ax.tick_params(axis='y', labelsize=7.5)
    axs[0].set_ylabel('log evidence minus provinces\n(above 0: beats the provinces)', fontsize=8.5)
    axs[0].legend(fontsize=8, frameon=False)
    fig.savefig(OUT / 'gcml_evidence.png', dpi=170)
    print('gcml_evidence.png')


if __name__ == '__main__':
    l1_maps()
    gallery()
    evidence()
