"""Does VIS follow crustal thickness or Curie depth? VIS of the global layout fit (gcm_glc_layouts_oc_lcs1_16-133)
against LithoRef18 (Afonso et al. 2019, 2 deg): Moho depth, Curie depth (580 C isotherm, from LithoRef18's
temperatures) and the magnetic thickness they allow (the shallower of the two, less the depth of the top of basement).
LithoRef's cell-centred lon/lat grids are interpolated bilinearly (longitude wrapped) to the 0.1 deg nodes and averaged
over each HEALPix cell with cos(lat) weights. Province values: area-weighted means over each province's cells.
Run from the repository root."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pygmt
import xarray as xr
from scipy.stats import spearmanr

from remit import depth
from remit.inversion import figures, geometry

R = 'inversions/results/'
OUT = 'docs/reports/2026-10-11-layouts/'
LITHOREF = os.path.expanduser('~/Library/CloudStorage/OneDrive-UniversityofTasmania/Data/LithoRef_model/')
CURIE = os.path.expanduser('~/GIT/gpdata/lithosphere/LithoRef_model/LithoRef_CurieDepth.nc')


def to_nodes(path, lon, lat):
    """a 2 deg cell-centred grid (depth in m) at the nodes, in km"""
    z = xr.open_dataset(path).z
    z = xr.concat([z.isel(x=[-1]).assign_coords(x=[z.x[-1].item() - 360]), z,
                   z.isel(x=[0]).assign_coords(x=[z.x[0].item() + 360])], 'x')
    lon180 = ((lon + 180) % 360) - 180
    return z.interp(x=xr.DataArray(lon180, dims='i'), y=xr.DataArray(np.clip(lat, -89, 89), dims='i')).values/1e3


def cell_fields():
    g = xr.load_dataset(R + 'gcm_glc_layouts_oc_lcs1_16-133.nc')
    cell = figures.cell_grid(g)
    grid = geometry.grid()
    i, j = np.nonzero(cell >= 0)
    c, lon, lat = cell[i, j], grid.lon[j], grid.lat[i]
    w = np.cos(np.radians(lat))
    mean = lambda v: np.bincount(c, w*v, len(g.cell))/np.bincount(c, w, len(g.cell))
    basement = mean((depth.ellipsoid_radius(lat) - geometry.surface_radius('basement')[i, j])/1e3)
    moho = mean(to_nodes(LITHOREF + 'LithoRef_moho.nc', lon, lat))
    curie = mean(to_nodes(CURIE, lon, lat))
    thick = np.maximum(np.minimum(moho, curie) - basement, 0.)
    return g, cell, dict(moho=moho, curie=curie, thick=thick, basement=basement)


def by_province(g, x):
    lab, a = g.domain.values.astype(int), g.area_km2.values
    return np.bincount(lab, a*x)/np.bincount(lab, a)


if __name__ == '__main__':
    g, cell, f = cell_fields()
    vis, lab, a = g.vis.values, g.domain.values.astype(int), g.area_km2.values
    level = np.bincount(lab, a*g.vis_domain.values)/np.bincount(lab, a)
    parea = np.bincount(lab, a)
    cols = [('moho', 'Moho depth (km)'), ('curie', 'Curie depth, 580 °C (km)'),
            ('thick', 'Magnetic thickness (km):\nshallower of Moho and Curie, less basement depth')]
    INK, MUTED, COL = '#0b0b0b', '#52514e', '#2a78d6'
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), dpi=200, sharey='row')
    for k, (key, label) in enumerate(cols):
        x = f[key]
        ax = axes[0, k]
        ax.hexbin(x, vis, gridsize=45, cmap='Blues', mincnt=1, bins='log', extent=(0, 70, -0.5, 2.5))
        r = spearmanr(x, vis).statistic
        ax.set_title(f'Cells: rank correlation {r:+.2f}', fontsize=8, loc='left', color=INK)
        ax = axes[1, k]
        xp = by_province(g, x)
        ax.scatter(xp, level, s=4 + 80*np.sqrt(parea/1e6), color=COL, alpha=0.5, edgecolor='white', lw=0.3)
        r = spearmanr(xp, level).statistic
        rw = np.corrcoef(xp, level)[0, 1]
        ax.set_title(f'Provinces: rank correlation {r:+.2f} (linear {rw:+.2f})', fontsize=8, loc='left', color=INK)
        ax.set_xlabel(label, fontsize=8, color=MUTED)
        for ax in axes[:, k]:
            ax.set_xlim(0, 70)
            ax.tick_params(labelsize=7, colors=MUTED)
            ax.grid(color='#ecebe7', lw=0.6)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
        print(key, f'cells r_s {spearmanr(x, vis).statistic:+.3f}, provinces r_s {r:+.3f}, linear {rw:+.3f}')
    axes[0, 0].set_ylabel('VIS of each cell (SI km)', fontsize=8, color=MUTED)
    axes[1, 0].set_ylabel('Province level (SI km)', fontsize=8, color=MUTED)
    fig.suptitle('Global fit VIS against LithoRef18 crustal thickness and Curie depth', fontsize=10, x=0.01,
                 ha='left', color=INK)
    fig.text(0.01, 0.005, 'Top: every continental cell (shading: number of cells, log). Bottom: one dot per province '
             '(size ~ area). Neighbouring cells and provinces are not independent, so the correlations overstate '
             'how sure they are.', fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(OUT + 'vis_vs_thickness.png')

    panels = [(g.vis_domain.values, 'Province levels, global fit (SI km)', figures.VIS_CPT[0], '0/1.5/0.05'),
              (f['moho'], 'Moho depth, LithoRef18 (km)', 'devon', '10/60/1'),
              (f['curie'], 'Curie depth, 580 °C, LithoRef18 (km)', 'devon', '10/60/1'),
              (f['thick'], 'Magnetic thickness (km)', 'devon', '0/50/1')]
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='9p', FONT_ANNOT_PRIMARY='6p', FONT_LABEL='7p', MAP_FRAME_TYPE='plain')
    for k, (v, title, cmap, series) in enumerate(panels):
        if k:
            fig.shift_origin(xshift='13.5c' if k % 2 else '-13.5c', yshift='0c' if k % 2 else '-8.5c')
        fig.basemap(region='d', projection='N0/12c', frame=['af', '+t' + title])
        pygmt.makecpt(cmap=cmap, series=series, background=True, reverse=cmap == 'devon')
        fig.grdimage(figures.cell_map(cell, v), cmap=True, nan_transparent=True)
        fig.coast(shorelines='0.2p,gray30', resolution='c', area_thresh=5000)
        fig.colorbar(cmap=True, frame='xaf', position='JBC+w6c/0.25c+o0c/0.5c')
    fig.savefig(OUT + 'thickness_maps.png', dpi=200)
