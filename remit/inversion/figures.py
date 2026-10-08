"""
The standard figure set of a result (PyGMT): VIS, its uncertainty, the data Br, the model Br and the residual
(coefficient fits, degrees of the band, at the surface); for synthetic data, the truth and the error. Global maps
(Robinson); regional maps come with local fits. Explore-mode results are labelled as such in every title.
"""
import json

import numpy as np
import pyshtools
import xarray as xr

from remit import depth
from . import geometry

VIS_CPT = ('batlow', '-0.5/2.5/0.05')
BR_CPT = ('vik', '-40/40/1')
SD_CPT = ('devon', '0/1/0.02')
STEP = 3                                         # display grid: every third 0.1 deg node


def cell_grid(ds):
    """the grid of cell indices of a result, rebuilt from its config"""
    cfg = json.loads(ds.attrs['config'])
    cells = geometry.make_cells(geometry.target_mask(cfg['target']['area']), cfg['target']['nside'])
    for _ in range(cfg['compute'].get('coarsen', 0)):
        cells, _ = geometry.coarsen(cells)
    assert np.array_equal(cells.ids, ds.cell_id.values), 'cells of the result differ from its config'
    return cells.cell


def cell_map(cell, v):
    g = np.full(cell.shape, np.nan)
    g[cell >= 0] = v[cell[cell >= 0]]
    g = g[::STEP, :-1:STEP][::-1]                           # the 360 deg column repeats 0
    lat = np.linspace(90, -90, cell.shape[0])[::STEP][::-1]
    lon = np.linspace(0, 360, cell.shape[1])[:-1:STEP]
    east = lon >= 180
    return xr.DataArray(np.concatenate([g[:, east], g[:, ~east]], 1), dims=('lat', 'lon'),
                        coords=dict(lat=lat, lon=np.concatenate([lon[east] - 360, lon[~east]])))


def br_map(ds, values):
    lmax = int(ds.l.max())
    c = np.zeros((2, lmax + 1, lmax + 1))
    c[ds.gh.values, ds.l.values, ds.m.values] = values
    da = pyshtools.SHMagCoeffs.from_array(c, r0=depth.R0).expand(a=depth.R0, lmax=lmax, lmax_calc=lmax).rad.to_xarray()
    return da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby('lon').sortby('lat')


def standard(ds, stem):
    import pygmt
    cell = cell_grid(ds)
    tag = 'EXPLORE: ' if ds.attrs['mode'] == 'explore' else ''
    lmin, lmax = int(ds.l.min()), int(ds.l.max())
    panels = [(cell_map(cell, ds.vis.values), f'{tag}VIS (v0 = {ds.attrs["v0"]:.2f} SI km)', *VIS_CPT, 'SI km')]
    if 'vis_sd' in ds:
        panels.append((cell_map(cell, ds.vis_sd.values), f'{tag}Uncertainty of VIS (1 s.d.)', *SD_CPT, 'SI km'))
    if 'truth' in ds:
        panels += [(cell_map(cell, ds.truth.values), f'{tag}Synthetic truth', *VIS_CPT, 'SI km'),
                   (cell_map(cell, ds.vis.values - ds.truth.values), f'{tag}VIS minus truth', 'vik', '-0.6/0.6/0.02', 'SI km')]
    panels += [(br_map(ds, ds.data.values), f'{tag}Data less background, degrees {lmin}-{lmax}', *BR_CPT, 'Br (nT)'),
               (br_map(ds, ds.model.values), f'{tag}Model', *BR_CPT, 'Br (nT)'),
               (br_map(ds, ds.residual.values), f'{tag}Data minus model (RMS {ds.attrs["rms_br_misfit"]:.1f} nT)', *BR_CPT, 'Br (nT)')]
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='9p', FONT_ANNOT_PRIMARY='6p', FONT_LABEL='7p', MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE='plain')
    for k, (g, title, cmap, series, label) in enumerate(panels):
        if k:
            fig.shift_origin(xshift='13.5c' if k % 2 else '-13.5c', yshift='0c' if k % 2 else '-8.5c')
        fig.basemap(region='d', projection='N0/12c', frame=['af', '+t' + title])
        pygmt.makecpt(cmap=cmap, series=series, background=True, reverse=cmap == 'devon')
        fig.grdimage(g, cmap=True, nan_transparent=True)
        fig.coast(shorelines='0.2p,gray30', resolution='c', area_thresh=5000)
        fig.colorbar(cmap=True, frame=f'xaf+l{label}', position='JBC+w6c/0.25c+o0c/0.5c')
    out = f'{stem}_maps.png'
    fig.savefig(out, dpi=200)
    return [out]
