"""Quick look at run_merge.py results for Australia: VIS of the merging models, refine against fine.
Run from notebooks/: PYTHONPATH=..:$PYTHONPATH python regions/australia/plot_merge_check.py"""
import numpy as np
import pygmt
import xarray as xr

C = 'regions/australia/cache'
cells = np.load(f'{C}/cells.npz')
cell = cells['cell']
m = np.load(f'{C}/merge.npz', allow_pickle=True)
glat = np.linspace(90, -90, cell.shape[0]); glon = np.linspace(0, 360, cell.shape[1])
REGION = [108, 158, -46, -3]


def grid(v):
    g = np.full(cell.shape, np.nan)
    g[cell >= 0] = v[cell[cell >= 0]]
    return xr.DataArray(g[::-1], coords=dict(lat=glat[::-1], lon=glon), dims=('lat', 'lon')).sel(
        lon=slice(REGION[0], REGION[1]), lat=slice(REGION[2], REGION[3]))


# edges between cells, as lon/lat midpoints of the cell centres, for drawing boundaries
clon, clat, edges = cells['clon'], cells['clat'], cells['edges']
ex, ey = 0.5*(clon[edges[:, 0]] + clon[edges[:, 1]]), 0.5*(clat[edges[:, 0]] + clat[edges[:, 1]])
settings = [tuple(s) for s in m['settings']]
panels = []
for lr, g in [(1., 0.), (1/3, 0.), (1/3, 0.01)]:
    i = [k for k, s in enumerate(settings) if np.isclose(s[0], lr) and np.isclose(s[1], g)][0]
    for mode in ('fine', 'refine'):
        panels.append((f'{mode}: lambda x{lr:.2g}, gamma {g:g}: K {m[f"{mode}_K"][i]}, chi2 {m[f"{mode}_chi2"][i]:.0f}',
                       m[f'{mode}_v'][i], m[f'{mode}_label'][i], None))
for k, mode in enumerate(('fine', 'refine')):
    panels.append((f'{mode}: bootstrap boundary probability (lambda x1, gamma 0)', None, None, m[f'{mode}_boot0_prob']))

fig = pygmt.Figure()
pygmt.config(FONT_TITLE='9p', FONT_ANNOT_PRIMARY='7p', MAP_TITLE_OFFSET='3p')
for n, (title, v, lab, prob) in enumerate(panels):
    if n:
        fig.shift_origin(xshift='9.5c' if n % 2 else '-9.5c', yshift='0c' if n % 2 else '-9.3c')
    fig.basemap(region=REGION, projection='M8.5c', frame=['xa10f5', 'ya10f5', f'WSne+t{title}'])
    if v is not None:
        pygmt.makecpt(cmap='viridis', series=[0, 1.6, 0.05], background=True)
        fig.grdimage(grid(v), cmap=True, nan_transparent=True)
        cut = lab[edges[:, 0]] != lab[edges[:, 1]]
        fig.plot(x=ex[cut], y=ey[cut], style='c0.05c', fill='white')
    else:
        pygmt.makecpt(cmap='hot', series=[0, 1], reverse=True)
        keep = prob > 0.05
        fig.plot(x=ex[keep], y=ey[keep], style='c0.07c', fill=prob[keep], cmap=True)
    fig.coast(shorelines='0.3p,gray30', resolution='l', area_thresh=2000)
fig.colorbar(cmap=True, frame='xa0.2+lboundary probability', position='JBC+w6c+o0c/0.8c')
fig.savefig('regions/australia/merge_check.png', dpi=150)
