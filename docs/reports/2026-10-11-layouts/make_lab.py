"""Does VIS follow lithospheric thickness? Province levels of the global layout fit against the depth of the
lithosphere-asthenosphere boundary (LAB) from seismic tomography, as converted by Hoggard et al. (2020, Nature
Geoscience): SL2013sv (Schaeffer & Lebedev 2013), SLNAAFSA (SL2013sv blended with regional models of North America,
Africa and South America) and CAM2016; with LithoRef18's Moho and Curie depths (make_thickness.py) for comparison.
All, per continent, and cratons + shields alone. The 0.5 deg gridline-registered LAB grids are interpolated bilinearly
to the 0.1 deg nodes and averaged over each HEALPix cell (cos(lat) weights), then over each province (area).
Run from the repository root."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr

import make_thickness as mt
import make_types as mty
from remit.inversion import covariates, figures, geometry

OUT = 'docs/reports/2026-10-11-layouts/'
LAB = os.path.expanduser('~/GIT/gpdata/lithosphere/lithospheric_thickness_maps/')
MODELS = ('SL2013sv', 'SLNAAFSA', 'CAM2016')
NAMES = dict(afr='Africa', ant='Antarctica', aus='Australia', eur='Eurasia', nam='North America', sam='South America',
             zea='Zealandia')


def lab_nodes(model, lon, lat):
    z = xr.open_dataset(LAB + model + '.nc').z
    return z.interp(lon=xr.DataArray(lon % 360, dims='i'), lat=xr.DataArray(lat, dims='i')).values


def province_table():
    g, cell, f = mt.cell_fields()
    grid = geometry.grid()
    i, j = np.nonzero(cell >= 0)
    c, lon, lat = cell[i, j], grid.lon[j], grid.lat[i]
    w = np.cos(np.radians(lat))
    for m in MODELS:
        f[m] = np.bincount(c, w*lab_nodes(m, lon, lat), len(g.cell))/np.bincount(c, w, len(g.cell))
    names = json.loads(g.attrs['domain_names'])['provinces']
    kind = mty.province_types()
    home = {}
    for k in mty.CONT:
        for p in pd.read_csv(mty.TABLE[k], keep_default_na=False).param:
            home.setdefault(p, k)
    lab, a = g.domain.values.astype(int), g.area_km2.values
    df = pd.DataFrame(dict(name=names, type=[kind.get(n, '') for n in names],
                           continent=[home.get(n, '') for n in names], area=np.bincount(lab, a),
                           level=np.bincount(lab, a*g.vis_domain.values)/np.bincount(lab, a)))
    df['cs'] = df.type.isin(['craton', 'shield'])
    for key in ('moho', 'curie', 'thick', *MODELS):
        df[key] = mt.by_province(g, f[key])
    return df[df.type != ''], g, cell, f


def rho(d, key):
    return spearmanr(d[key], d.level).statistic if len(d) > 5 else np.nan


if __name__ == '__main__':
    df, g, cell, f = province_table()
    rows = []
    for cont in ['all', *NAMES]:
        d = df if cont == 'all' else df[df.continent == cont]
        for subset, s in (('all types', d), ('cratons + shields', d[d.cs])):
            rows.append(dict(continent=NAMES.get(cont, 'All continents'), provinces=subset, n=len(s),
                             **{k: round(rho(s, k), 2) for k in (*MODELS, 'moho', 'curie', 'thick')}))
    table = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(table.to_string(index=False))
    table.to_csv(OUT + 'vis_vs_lab.csv', index=False)

    INK, MUTED, CS, OTHER = '#0b0b0b', '#52514e', '#eb6834', '#2a78d6'
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4), dpi=200, sharex=True, sharey=True)
    for ax, cont in zip(axes.ravel(), ['all', *NAMES]):
        d = df if cont == 'all' else df[df.continent == cont]
        for sel, col, lab_ in ((~d.cs, OTHER, 'other types'), (d.cs, CS, 'cratons + shields')):
            ax.scatter(d.SL2013sv[sel], d.level[sel], s=3 + 60*np.sqrt(d.area[sel]/1e6), color=col, alpha=0.55,
                       edgecolor='white', lw=0.3, label=lab_)
        r_cs = rho(d[d.cs], 'SL2013sv')
        ax.set_title(f'{NAMES.get(cont, "All continents")}: {rho(d, "SL2013sv"):+.2f}'
                     + ('' if np.isnan(r_cs) else f' (cratons + shields {r_cs:+.2f})'), fontsize=7.5, loc='left',
                     color=INK)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.grid(color='#ecebe7', lw=0.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel('LAB depth, SL2013sv (km)', fontsize=8, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel('Province level (SI km)', fontsize=8, color=MUTED)
    axes[0, 0].legend(fontsize=7, frameon=False, loc='upper left')
    fig.suptitle('Global fit province levels against lithospheric thickness (rank correlations)', fontsize=10, x=0.01,
                 ha='left', color=INK)
    fig.text(0.01, 0.005, 'LAB: Hoggard et al. (2020) from SL2013sv (Schaeffer & Lebedev 2013). Dots: provinces '
             '(size ~ area). Neighbouring provinces are not independent, so the correlations overstate how sure '
             'they are.', fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(OUT + 'vis_vs_lab.png')

    import pygmt
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='9p', FONT_ANNOT_PRIMARY='6p', FONT_LABEL='7p', MAP_FRAME_TYPE='plain')
    for k, (v, title, cmap, series) in enumerate([
            (g.vis_domain.values, 'Province levels, global fit (SI km)', figures.VIS_CPT[0], '0/1.5/0.05'),
            (f['SL2013sv'], 'LAB depth, SL2013sv (km)', 'devon', '50/300/5')]):
        if k:
            fig.shift_origin(xshift='13.5c')
        fig.basemap(region='d', projection='N0/12c', frame=['af', '+t' + title])
        pygmt.makecpt(cmap=cmap, series=series, background=True, reverse=cmap == 'devon')
        fig.grdimage(figures.cell_map(cell, v), cmap=True, nan_transparent=True)
        fig.coast(shorelines='0.2p,gray30', resolution='c', area_thresh=5000)
        fig.colorbar(cmap=True, frame='xaf', position='JBC+w6c/0.25c+o0c/0.5c')
    fig.savefig(OUT + 'lab_maps.png', dpi=200)
