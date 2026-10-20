"""
Global maps (Robinson) for the GCM part of the verification document, from benchmarks/verification/results/gcm_*
(gcm_checks.py).

  gcm_g1_maps.png         GCM against the independent solver, at fixed damping
  gcm_g2_spread.png       GCM's errors in units of its reported s.d., settings at the truth and fitted
  gcm_shrink_maps.png     checkerboards of equal-area tiles: the truth, GCM without noise, with noise, and with noise
                          and a wrong ocean model
  gcm_shrink_contrast.png the recovered contrast against the tile size, GCM beside CM and DF for Australia
  gcm_ocean_maps.png      what the wrong ocean model changes in GCM's map

Run from the repository root:
    conda run -n curiesphere python docs/verification/make_gcm_figures.py
"""
import json
from pathlib import Path

import numpy as np
import pygmt
import xarray as xr
from scipy.spatial import cKDTree

RES = Path('benchmarks/verification/results')
OUT = Path('docs/verification/figures')
WG = 8.0                                    # map width (cm)
BASE = 2.0


def unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.c_[np.cos(la)*np.cos(lo), np.cos(la)*np.sin(lo), np.sin(la)]


def global_grid(lon, lat, v, step=0.5, max_km=70.):
    """cell values on a global 0.5 deg grid: each node takes its nearest cell centre (great-circle distance), blank
    farther than max_km from every cell (oceans)"""
    gx, gy = np.arange(-180., 180. + step/2, step), np.arange(-90., 90. + step/2, step)
    GX, GY = np.meshgrid(gx, gy)
    d, i = cKDTree(unit(lon, lat)).query(unit(GX.ravel(), GY.ravel()))
    g = np.where(d*6371. <= max_km, np.asarray(v)[i], np.nan).reshape(GX.shape)
    return xr.DataArray(g, coords=dict(lat=gy, lon=gx), dims=('lat', 'lon'))


def gmap(fig, lon, lat, v, cpt, series, title, sub, label):
    pygmt.makecpt(cmap=cpt, series=series, background=True)
    fig.basemap(region='g', projection=f'N0/{WG}c', frame=['f', f'+t{title}' + (f'+s{sub}' if sub else '')])
    fig.coast(land='gray92', water='white')
    fig.grdimage(global_grid(lon, lat, v), cmap=True, nan_transparent=True)
    fig.coast(shorelines='0.1p,gray55', area_thresh=20000)
    fig.colorbar(position=f'JBC+w{0.6*WG}c/0.13c+o0/0.25c+h+e', frame=[f'x+l{label}'])


def new_figure():
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='8p', FONT_SUBTITLE='6.5p', FONT_ANNOT_PRIMARY='5.5p', FONT_LABEL='6p',
                 MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE='plain')
    return fig


def place(fig, row, col, nrows):
    return fig.shift_origin(xshift=f'{col*(WG + 0.6)}c', yshift=f'{BASE + (nrows - 1 - row)*(0.5*WG + 2.0)}c')


def g1_maps():
    z = np.load(RES / 'gcm_g1.npz')
    lon, lat = z['clon'], z['clat']
    d = z['m'] - z['m_ref']
    r = z['sd']/z['sd_ref'] - 1
    lim = lambda a: float(f'{max(np.abs(a).max(), 1e-16):.0e}')
    panels = [(z['truth'], 'batlow', [0, 2], 'The truth (from GCM\'s model)', None, 'VIS (SI km)'),
              (z['m'], 'batlow', [0, 2], 'GCM', None, 'VIS (SI km)'),
              (d, 'vik', [-lim(d), lim(d)], 'GCM minus independent solver', f'max {np.abs(d).max():.1e} SI km',
               'SI km'),
              (r, 'vik', [-lim(r), lim(r)], 'GCM s.d. / independent s.d. - 1', f'max {np.abs(r).max():.1e}',
               'ratio - 1')]
    fig = new_figure()
    for j, (v, cpt, series, title, sub, label) in enumerate(panels):
        with place(fig, j//2, j % 2, 2):
            gmap(fig, lon, lat, v, cpt, series, title, sub, label)
    fig.savefig(OUT / 'gcm_g1_maps.png', dpi=170)
    print('gcm_g1_maps.png')


def g2_spread():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z = np.load(RES / 'gcm_g2.npz')
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True, sharey=True)
    x = np.linspace(-4.5, 4.5, 400)
    bell = np.exp(-x**2/2)/np.sqrt(2*np.pi)
    for ax, (key, what) in zip(axs, [('z_fixed', 'settings at the truth'), ('z_fit', 'settings fitted'),
                                     ('z_fit_full', 'settings fitted, s.d. with v0')]):
        v = z[key].ravel()
        ax.hist(v, bins=np.linspace(-4.5, 4.5, 61), density=True, color='0.7', label='all cells of all truths')
        ax.plot(x, bell, 'k', lw=1.2, label='what it should be')
        ax.set_title(f'GCM, {what}\nwithin 1 s.d.: {np.mean(np.abs(v) < 1):.0%} (should be 68%), '
                     f'within 2 s.d.: {np.mean(np.abs(v) < 2):.0%} (95%)', fontsize=8.5, loc='left')
        ax.set_xlabel('(GCM map minus truth) / reported s.d.')
    axs[0].legend(fontsize=7.5, frameon=False)
    fig.savefig(OUT / 'gcm_g2_spread.png', dpi=170)
    print('gcm_g2_spread.png')


def shrink_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = json.load(open(RES / 'gcm_shrink.json'))
    z = np.load(RES / 'gcm_shrink.npz')
    lon, lat = z['clon'], z['clat']
    nsides = sorted({r['nside'] for r in rows})
    noises = ['none', 'noise', 'noise and ocean error']
    fig = new_figure()
    for k, ns in enumerate(nsides):
        sel = {r['noise']: r for r in rows if r['nside'] == ns}
        size = sel['none']['size_km']
        panels = [(z[f'{ns}_truth'], f'Truth: tiles of {size:.0f} km', None)]
        for n in noises:
            panels.append((z[f'{ns}_{n}'][:, 0], f'GCM, {n if n != "none" else "no noise"}',
                           f'contrast {sel[n]["contrast"]:.2f}'))
        for j, (v, title, sub) in enumerate(panels):
            with place(fig, k, j, len(nsides)):
                gmap(fig, lon, lat, v, 'batlow', [0, 2], title, sub, 'VIS (SI km)')
    fig.savefig(OUT / 'gcm_shrink_maps.png', dpi=150)
    print('gcm_shrink_maps.png')

    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    for n, colour, name in (('none', '#2a2a2a', 'GCM, no noise'), ('noise', '#1b6ca8', 'GCM, noise'),
                            ('noise and ocean error', '#d1495b', 'GCM, noise and a wrong ocean model')):
        sel = sorted((r for r in rows if r['noise'] == n), key=lambda r: r['size_km'])
        ax.plot([r['size_km'] for r in sel], [r['contrast'] for r in sel], '-', color=colour, marker='o', ms=3.5,
                label=name)
    cm = sorted((r for r in json.load(open(RES / 'cm_shrink.json')) if r['noise'] == 'noise'), key=lambda r: r['size'])
    ax.plot([r['size']*111 for r in cm], [r['contrast'] for r in cm], '--', color='0.55', marker='s', ms=3,
            label='CM-AUS, noise (local fit, squares)')
    ax.axhline(1, color='0.6', lw=0.8)
    ax.set_xscale('log')
    ax.minorticks_off()
    ax.set_xticks([170, 204, 407, 815, 1630])
    ax.set_xticklabels(['170', '204', '407', '815', '1630'])
    ax.set_xlabel('checkerboard tile size (km)')
    ax.set_ylabel('recovered contrast / true contrast')
    ax.set_title('How much of the checkerboard GCM recovers', loc='left', fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(OUT / 'gcm_shrink_contrast.png', dpi=170)
    print('gcm_shrink_contrast.png')

    fig = new_figure()
    for k, ns in enumerate([8, 16]):
        d = z[f'{ns}_noise and ocean error'][:, 0] - z[f'{ns}_noise'][:, 0]
        size = next(r['size_km'] for r in rows if r['nside'] == ns)
        with place(fig, 0, k, 1):
            gmap(fig, lon, lat, d, 'vik', [-0.3, 0.3], f'Wrong ocean model: change in GCM\'s map',
                 f'tiles of {size:.0f} km; RMS {np.sqrt(np.mean(d**2)):.2f}, max {np.abs(d).max():.2f} SI km',
                 'SI km')
    fig.savefig(OUT / 'gcm_ocean_maps.png', dpi=170)
    print('gcm_ocean_maps.png')


if __name__ == '__main__':
    if (RES / 'gcm_g1.npz').exists():
        g1_maps()
    if (RES / 'gcm_g2.npz').exists():
        g2_spread()
    if (RES / 'gcm_shrink.json').exists():
        shrink_figures()
