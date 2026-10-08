"""
The standard figure set of a result (PyGMT): VIS, its uncertainty, any further maps of the method, the data Br, the
model Br and the residual; for synthetic data, the truth and the error. Global fits: global maps (Robinson), Br from
the band's coefficients at the surface. Local fits: regional maps (Mercator) with the target outline, Br at the data
points. Explore-mode results are labelled as such in every title.
"""
import json

import numpy as np
import pyshtools
import xarray as xr

from remit import depth
from . import geometry
from .cache import Cache

VIS_CPT = ('batlow', '-0.5/2.5/0.05')
BR_CPT = ('vik', '-40/40/1')
SD_CPT = ('devon', '0/1/0.02')
STEP = 3                                         # display grid: every third 0.1 deg node


def cell_grid(ds):
    """the grid of cell indices of a result, rebuilt from its config"""
    cfg = json.loads(ds.attrs['config'])
    cells = geometry.make_cells(geometry.target_mask(cfg['target']['area'], Cache()), cfg['target']['nside'])
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


def _lon180(x):
    return ((np.asarray(x) + 180) % 360) - 180


def standard(ds, stem):
    import pygmt
    cell = cell_grid(ds)
    cfg = json.loads(ds.attrs['config'])
    local = 'point' in ds.dims
    tag = 'EXPLORE: ' if ds.attrs['mode'] == 'explore' else ''
    v0 = ds.attrs.get('v0')
    panels = [(cell_map(cell, ds.vis.values), f'{tag}VIS' + (f' (v0 = {v0:.2f} SI km)' if v0 is not None else ''),
               *VIS_CPT, 'SI km')]
    if 'vis_sd' in ds:
        panels.append((cell_map(cell, ds.vis_sd.values), f'{tag}Uncertainty of VIS (1 s.d.)', *SD_CPT, 'SI km'))
    maps = json.loads(ds.attrs.get('maps', '{}'))
    for k in [k for k in ds.data_vars if k.startswith('vis_') and k != 'vis_sd']:
        m = maps.get(k[4:])
        title = (f'VIS, {m["penalty"].upper()} map, length {m["length_km"]:.0f} km, bounded' if m else
                 'VIS, ' + k[4:].replace('_', ' ').replace('min', 'model of the smallest').replace('max', 'model of the largest'))
        panels.append((cell_map(cell, ds[k].values), tag + title, *VIS_CPT, 'SI km'))
    if 'truth' in ds:
        panels += [(cell_map(cell, ds.truth.values), f'{tag}Synthetic truth', *VIS_CPT, 'SI km'),
                   (cell_map(cell, ds.vis.values - ds.truth.values), f'{tag}VIS minus truth', 'vik', '-0.6/0.6/0.02', 'SI km')]
    if local:
        rms = float(np.sqrt(np.mean(ds.residual.values**2)))
        lo, hi = cfg['data']['band']
        panels += [(ds.data.values, f'{tag}Data less background, degrees {lo}-{hi}', *BR_CPT, 'Br (nT)'),
                   (ds.model.values, f'{tag}Model', *BR_CPT, 'Br (nT)'),
                   (ds.residual.values, f'{tag}Data minus model (RMS {rms:.1f} nT)', *BR_CPT, 'Br (nT)')]
        lon = _lon180(ds.plon.values)
        region = [float(lon.min()) - 1, float(lon.max()) + 1, float(ds.plat.min()) - 1, float(ds.plat.max()) + 1]
        proj, width, dx, dy = 'M11c', 11, '12.5c', None
    else:
        lmin, lmax = int(ds.l.min()), int(ds.l.max())
        panels += [(br_map(ds, ds.data.values), f'{tag}Data less background, degrees {lmin}-{lmax}', *BR_CPT, 'Br (nT)'),
                   (br_map(ds, ds.model.values), f'{tag}Model', *BR_CPT, 'Br (nT)'),
                   (br_map(ds, ds.residual.values), f'{tag}Data minus model (RMS {ds.attrs["rms_br_misfit"]:.1f} nT)', *BR_CPT, 'Br (nT)')]
        region, proj, width, dx, dy = 'd', 'N0/12c', 12, '13.5c', '-8.5c'
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='9p', FONT_ANNOT_PRIMARY='6p', FONT_LABEL='7p', MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE='plain')
    if dy is None:                                   # the height of a regional panel
        dy = f'-{width*(region[3] - region[2])/(region[1] - region[0])*1.15 + 3.0:.1f}c'
    for k, (g, title, cmap, series, label) in enumerate(panels):
        if k:
            fig.shift_origin(xshift=dx if k % 2 else f'-{dx}', yshift='0c' if k % 2 else dy)
        fig.basemap(region=region, projection=proj, frame=['af', '+t' + title])
        pygmt.makecpt(cmap=cmap, series=series, background=True, reverse=cmap == 'devon')
        if isinstance(g, xr.DataArray):
            fig.grdimage(g, cmap=True, nan_transparent=True)
        else:
            fig.plot(x=lon, y=ds.plat.values, fill=g, cmap=True, style='c0.07c')
        fig.coast(shorelines='0.2p,gray30', resolution='l' if local else 'c', area_thresh=5000)
        if local and cfg['target']['area'] != 'continental_crust':
            for poly in geometry.read_outline(cfg['target']['area'])[0]:
                la, lo_ = poly.to_lat_lon_array().T
                fig.plot(x=lo_, y=la, pen='0.3p,gray20')
        fig.colorbar(cmap=True, frame=f'xaf+l{label}', position=f'JBC+w{width/2:.0f}c/0.25c+o0c/0.5c')
    out = f'{stem}_maps.png'
    fig.savefig(out, dpi=200)
    return [out]
