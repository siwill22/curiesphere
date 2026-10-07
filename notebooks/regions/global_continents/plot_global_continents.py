"""Maps of run_global_continents.py (PyGMT, Robinson global and an Australia zoom).
  maps_<kind>.png      VIS (KIND=real_evid, default, or real_disc), its s.d., data Br, model Br, residual Br
                       (the paper's minimal damping is not shown: with a uniform prior it is noise, ~50% negative)
  maps_australia_<kind>.png   VIS and its s.d. over Australia
  maps_syn_crust_<kind>.png   the Crust1.0 synthetic: truth, recovered, difference
Run from notebooks/: env BAND=<name> PYTHONPATH=..:$PYTHONPATH python regions/global_continents/plot_global_continents.py"""
import os
import tomllib

import numpy as np
import pygmt
import pyshtools
import xarray as xr

import depth_models as dm
from domains.plotting import REGION_AUSTRALIA

REG = 'regions/global_continents'
cfg = tomllib.load(open(f'{REG}/config.toml', 'rb'))
BAND = os.environ.get('BAND', next(iter(cfg['bands'])))
r = dict(np.load(f'{REG}/cache/{BAND}/results.npz'))
cell = np.load(f'{REG}/cache/cells.npz')['cell']
lmin, lmax = int(r['lmin']), int(r['lmax'])
GLOBAL = 'N0/12c'
STEP = 3                                              # 0.3 deg display grid


def cell_map(v):
    g = np.full(cell.shape, np.nan)
    g[cell >= 0] = v[cell[cell >= 0]]
    g = g[::STEP, :-1:STEP][::-1]                       # the 360 deg column repeats 0
    lat = np.linspace(90, -90, cell.shape[0])[::STEP][::-1]
    lon = np.linspace(0, 360, cell.shape[1])[:-1:STEP]
    g, lon = np.concatenate([g[:, lon >= 180], g[:, lon < 180]], 1), np.concatenate([lon[lon >= 180] - 360, lon[lon < 180]])
    return xr.DataArray(g, coords=dict(lat=lat, lon=lon), dims=('lat', 'lon'))


def br_map(values):
    """Br (nT) at the surface from band coefficients (row order of the results)"""
    c = np.zeros((2, lmax + 1, lmax + 1))
    c[r['row_gh'], r['row_l'], r['row_m']] = values
    grid = pyshtools.SHMagCoeffs.from_array(c, r0=dm.R0).expand(a=dm.R0, lmax=lmax, lmax_calc=lmax).rad
    da = grid.to_xarray()
    da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby('lon').sortby('lat')
    return da


def panel(fig, grid, title, cmap, series, label, proj=GLOBAL, region='d', reverse=False):
    fig.basemap(region=region, projection=proj, frame=['af', '+t' + title] if region == 'd' else ['WSne+t' + title, 'xa10f5', 'ya10f5'])
    pygmt.makecpt(cmap=cmap, series=series, background=True, reverse=reverse)
    fig.grdimage(grid, cmap=True, nan_transparent=True)
    fig.coast(shorelines='0.2p,gray30', resolution='c' if region == 'd' else 'l', area_thresh=5000)
    fig.colorbar(cmap=True, frame=f'xaf+l{label}', position='JBC+w6c/0.25c+o0c/0.5c')


def setup():
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='9p', FONT_ANNOT_PRIMARY='6p', FONT_LABEL='7p', MAP_TITLE_OFFSET='3p',
                 MAP_FRAME_TYPE='plain')
    return fig


VIS = ('batlow', '-0.5/2.5/0.05')
BR = ('vik', '-40/40/1')

# real data
d = os.environ.get('KIND', 'real_evid')
SD = ('devon', '0/1/0.02')                            # small partial coastal cells go above 1
WHAT = {'real_evid': 'noise and prior fitted (evidence)', 'real_disc': 'damped to the noise level'}[d]
fig = setup()
panels = [
    (cell_map(r[f'{d}_m']), f'VIS, {WHAT} (v0 = {r[f"{d}_v0"]:.2f} SI km)', *VIS, 'SI km'),
    (cell_map(r[f'{d}_sd']), 'Uncertainty of that VIS (1 s.d., v0 fixed)', *SD, 'SI km'),
    (br_map(r['g_data'] + 0), f'Data: LCS-1 less the ocean model, degrees {lmin}-{lmax}', *BR, 'Br (nT)'),
    (br_map(r['g_data'] - r[f'{d}_resid']), f'Continents model ({WHAT})', *BR, 'Br (nT)'),
    (br_map(r[f'{d}_resid']), f'Data minus model (RMS {r[f"{d}_rms"]:.1f} nT)', *BR, 'Br (nT)'),
]
for k, (g, title, cmap, series, label) in enumerate(panels):
    if k:
        fig.shift_origin(xshift='13.5c' if k % 2 else '-13.5c', yshift='0c' if k % 2 else '-8.5c')
    panel(fig, g, title, cmap, series, label, reverse=(cmap == 'devon'))
fig.savefig(f'{REG}/maps_{d}_{BAND}.png', dpi=200)
print('wrote', f'{REG}/maps_{d}_{BAND}.png')

# Australia zoom
fig = setup()
AUS = 'M7c'
for k, (key, title) in enumerate([(f'{d}_m', f'VIS, {WHAT}'), (f'{d}_sd', 'Uncertainty (1 s.d.)')]):
    if k:
        fig.shift_origin(xshift='8.5c')
    g = cell_map(r[key])
    cmap, series = (SD if key.endswith('_sd') else VIS)
    panel(fig, g, title, cmap, series, 'SI km', proj=AUS, region=REGION_AUSTRALIA, reverse=key.endswith('_sd'))
fig.savefig(f'{REG}/maps_australia_{d}_{BAND}.png', dpi=200)
print('wrote', f'{REG}/maps_australia_{d}_{BAND}.png')

# Crust1.0 synthetic
s = 'syn_crust_' + d.split('_')[1]
fig = setup()
for k, (v, title, cmap, series) in enumerate([
        (r['syn_crust_truth'], f'Synthetic truth: {cfg["synthetic_chi"]:g} x Crust1.0 thickness', *VIS),
        (r[f'{s}_m'], f'Recovered (uniform prior, v0 = {r[f"{s}_v0"]:.2f})', *VIS),
        (r[f'{s}_m'] - r['syn_crust_truth'], 'Recovered minus truth', 'vik', '-0.6/0.6/0.02')]):
    if k:
        fig.shift_origin(xshift='13.5c' if k == 1 else '-13.5c', yshift='0c' if k == 1 else '-8.5c')
    panel(fig, cell_map(v), title, cmap, series, 'SI km')
fig.savefig(f'{REG}/maps_{s}_{BAND}.png', dpi=200)
print('wrote', f'{REG}/maps_{s}_{BAND}.png')
