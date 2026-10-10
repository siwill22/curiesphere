"""Maps for the 2026-10-11 report: global VIS with one level and with province levels (ocean-model error), and local
DF before and after the Gaussian prior on domain means (Africa, South America). Run from the repository root."""
import numpy as np
import pygmt
import xarray as xr

from remit.inversion import figures, geometry

R = 'inversions/results/'
OUT = 'docs/reports/2026-10-11-layouts/'


def global_maps():
    ds = xr.load_dataset(R + 'gcm_glc_layouts_oc_lcs1_16-133.nc')
    cell = figures.cell_grid(ds)
    one, prov, lev = ds.layout_vis_one.values, ds.vis.values, ds.vis_domain.values
    panels = [(one, 'VIS, one level for all continents (GCM-GLC)', *figures.VIS_CPT),
              (prov, 'VIS, a level per province (best layout)', *figures.VIS_CPT),
              (lev, 'The province levels alone', *figures.VIS_CPT),
              (prov - one, 'Difference: province levels minus one level', 'vik', '-0.8/0.8/0.02')]
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='9p', FONT_ANNOT_PRIMARY='6p', FONT_LABEL='7p', MAP_FRAME_TYPE='plain')
    for k, (v, title, cmap, series) in enumerate(panels):
        if k:
            fig.shift_origin(xshift='13.5c' if k % 2 else '-13.5c', yshift='0c' if k % 2 else '-8.5c')
        fig.basemap(region='d', projection='N0/12c', frame=['af', '+t' + title])
        pygmt.makecpt(cmap=cmap, series=series, background=True)
        fig.grdimage(figures.cell_map(cell, v), cmap=True, nan_transparent=True)
        fig.coast(shorelines='0.2p,gray30', resolution='c', area_thresh=5000)
        fig.colorbar(cmap=True, frame='xaf+lSI km', position='JBC+w6c/0.25c+o0c/0.5c')
    fig.savefig(OUT + 'global_vis.png', dpi=200)


def local_pair(c, title):
    a = xr.load_dataset(R + f'df_{c}_lcs1_16-133_gcmbg.nc')
    b = xr.load_dataset(R + f'df_{c}_lcs1_16-133_gcmbg_gm.nc')
    cell = figures.cell_grid(a)
    lon, lat = a.lon.values, a.lat.values
    xyz = np.stack([np.cos(np.radians(lat))*np.cos(np.radians(lon)), np.cos(np.radians(lat))*np.sin(np.radians(lon)),
                    np.sin(np.radians(lat))], 1)
    cc = xyz.mean(0)
    cc /= np.linalg.norm(cc)
    lon0, lat0 = np.degrees(np.arctan2(cc[1], cc[0])), np.degrees(np.arcsin(cc[2]))
    radius = np.degrees(np.arccos(np.clip(xyz @ cc, -1, 1)).max()) + 2
    proj = f'A{lon0:.2f}/{lat0:.2f}/{radius:.1f}/8c'
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='9p', FONT_ANNOT_PRIMARY='6p', FONT_LABEL='7p', MAP_FRAME_TYPE='plain')
    for k, (ds, t) in enumerate([(a, 'means anywhere (before)'), (b, 'means from one shared spread (after)')]):
        v = ds.vis.values
        if k:
            fig.shift_origin(xshift='9.5c')
        fig.basemap(region='g', projection=proj, frame=['+t' + f'{title}: {t}, {np.mean(v < 0):.0%} of cells < 0'])
        pygmt.makecpt(cmap=figures.VIS_CPT[0], series=figures.VIS_CPT[1], background=True)
        fig.grdimage(figures.cell_map(cell, v), cmap=True, nan_transparent=True)
        g = figures.cell_map(cell, (v < 0).astype(float))
        fig.grdcontour(g, levels=0.5, pen='0.4p,red')
        fig.coast(shorelines='0.2p,gray30', resolution='l', area_thresh=5000)
        fig.colorbar(cmap=True, frame='xaf+lVIS (SI km)', position='JBC+w5c/0.25c+o0c/0.5c')
    fig.savefig(OUT + f'df_{c}_means.png', dpi=200)


if __name__ == '__main__':
    global_maps()
    local_pair('afr', 'Africa')
    local_pair('sam', 'South America')
