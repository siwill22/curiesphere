"""Figures for section 7 of the 2026-10-11 report: the province polygons against 50 random Voronoi tilings and 50
spun copies of the provinces (each continent's polygons rotated about its centre), on real data and on the null.
Read from the ensemble runs gcm_glc_layouts_ens{0,1,2}_{lcs1_16-133,syn_s5}. Run from the repository root."""
import re

import matplotlib.pyplot as plt
import numpy as np
import pygmt
import xarray as xr

from remit.inversion import figures

R = 'inversions/results/'
OUT = 'docs/reports/2026-10-11-layouts/'
RUNS = {'real': 'lcs1_16-133', 'null': 'syn_s5'}


def evidence(tag):
    """log evidence of every layout in the three ensemble runs"""
    out = {}
    for i in range(3):
        for line in open(R + f'gcm_glc_layouts_ens{i}_{tag}.log'):
            m = re.search(r'\] (\S+) \(ocean.*log evidence ([-\d.]+)', line)
            if m:
                out[m.group(1)] = float(m.group(2))
    return out


def chart():
    rows = []
    for case, tag in RUNS.items():
        ev = evidence(tag)
        p = ev['provinces']
        for kind in ('voronoi', 'rotated'):
            rows.append((case, kind, np.array([v - p for k, v in ev.items() if k.startswith(kind)])))
    TILE, SPIN, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
    label = {('real', 'voronoi'): 'Random tilings, real data', ('real', 'rotated'): 'Spun provinces, real data',
             ('null', 'voronoi'): 'Random tilings, null', ('null', 'rotated'): 'Spun provinces, null'}
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=200)
    rng = np.random.default_rng(1)
    y = [3.2, 2.4, 1.0, 0.2]
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    ax.annotate('the real provinces', (0, 3.75), xytext=(4, 0), textcoords='offset points', fontsize=7, color=INK,
                va='center')
    ax.axhline(1.7, color='#e1e0d9', lw=0.8)
    for yi, (case, kind, d) in zip(y, rows):
        ax.scatter(d, yi + rng.uniform(-0.18, 0.18, len(d)), s=14, color=TILE if kind == 'voronoi' else SPIN,
                   marker='o' if kind == 'voronoi' else 'D', edgecolor='white', lw=0.5, zorder=3)
        n_above = int((d > 0).sum())
        ax.annotate(f'{n_above} of {len(d)} beat the provinces', (max(d.max(), 0), yi), xytext=(8, 0),
                    textcoords='offset points', fontsize=7, color=MUTED, va='center')
    ax.set_yticks(y, [label[r[:2]] for r in rows], fontsize=8, color=INK)
    ax.set_ylim(-0.3, 3.95)
    ax.set_xlabel('Log evidence minus that of the real provinces  (> 0: fits better than the provinces)', fontsize=8,
                  color=MUTED)
    ax.tick_params(axis='x', labelsize=7, colors=MUTED)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(MUTED)
    ax.grid(axis='x', color='#ecebe7', lw=0.6, zorder=0)
    ax.margins(x=0.25)
    ax.set_title('Province polygons against 100 layouts that ignore geology (global fit)', fontsize=10, loc='left',
                 color=INK)
    fig.tight_layout()
    fig.savefig(OUT + 'ensemble_evidence.png')
    for case, kind, d in rows:
        print(case, kind, f'best {d.max():+.0f}, median {np.median(d):+.0f}, worst {d.min():+.0f}, above {(d > 0).sum()}')


def best(kind, tag='lcs1_16-133'):
    ev = evidence(tag)
    k = max((k for k in ev if k.startswith(kind)), key=ev.get)
    return k, ev[k] - ev['provinces']


def layout_fields(name, tag='lcs1_16-133'):
    """VIS and domain labels of one layout, from whichever ensemble run fitted it"""
    v = name.replace(':', '_')
    for i in range(3):
        ds = xr.load_dataset(R + f'gcm_glc_layouts_ens{i}_{tag}.nc')
        if ds.attrs['layout'] == name:
            return ds, ds.vis.values, ds.domain.values
        if f'layout_vis_{v}' in ds:
            return ds, ds[f'layout_vis_{v}'].values, ds[f'layout_domain_{v}'].values
    raise KeyError(name)


def maps():
    tile, d_tile = best('voronoi')
    spin, d_spin = best('rotated')
    cols = [('provinces', 'The real provinces (724)', None),
            (tile, 'Best of 50 random tilings', d_tile),
            (spin, 'Best of 50 spun province sets', d_spin)]
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='9p', FONT_ANNOT_PRIMARY='6p', FONT_LABEL='7p', MAP_FRAME_TYPE='plain')
    rng = np.random.default_rng(3)
    for j, (name, title, d) in enumerate(cols):
        ds, vis, dom = layout_fields(name)
        cell = figures.cell_grid(ds)
        if j:
            fig.shift_origin(xshift='10.5c', yshift='6.2c')
        t = title + ('' if d is None else f': {d:+.0f} log units')
        fig.basemap(region=[-180, 180, -90, 85], projection='N0/10c', frame=['f', '+t' + t])
        colour = rng.permutation(int(dom.max()) + 1)[dom.astype(int)].astype(float)
        pygmt.makecpt(cmap='roma', series=[0, dom.max()])
        fig.grdimage(figures.cell_map(cell, colour), cmap=True, nan_transparent=True)
        fig.coast(shorelines='0.15p,gray40', resolution='c', area_thresh=5000)
        fig.shift_origin(yshift='-6.2c')
        fig.basemap(region=[-180, 180, -90, 85], projection='N0/10c', frame=['f', '+tFitted VIS averaged over each domain'])
        pygmt.makecpt(cmap=figures.VIS_CPT[0], series='0/1.5/0.05', background=True)
        lab, a = dom.astype(int), ds.area_km2.values
        avg = np.bincount(lab, a*vis)/np.bincount(lab, a)
        fig.grdimage(figures.cell_map(cell, avg[lab]), cmap=True, nan_transparent=True)
        fig.coast(shorelines='0.15p,gray40', resolution='c', area_thresh=5000)
    fig.colorbar(cmap=True, frame='xaf+lVIS (SI km)', position='JBC+w6c/0.25c+o-10.5c/0.5c')
    fig.savefig(OUT + 'ensemble_maps.png', dpi=200)


if __name__ == '__main__':
    chart()
    maps()
