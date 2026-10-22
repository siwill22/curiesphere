"""
Map helpers for cell values (PyGMT): cell values onto the 0.1 deg grid of the cells, and one
standard map panel (Mercator, coastlines) for multi-panel figures.
"""
import numpy as np
import xarray as xr

REGION_AUSTRALIA = [108, 158, -46, -3]


def cell_grid(v, cells, region=REGION_AUSTRALIA):
    """DataArray of the cell values v on the cells' 0.1 deg grid, cut to region (NaN outside)"""
    cell = cells['cell']
    glat = np.linspace(90, -90, cell.shape[0]); glon = np.linspace(0, 360, cell.shape[1])
    g = np.full(cell.shape, np.nan)
    g[cell >= 0] = v[cell[cell >= 0]]
    return xr.DataArray(g[::-1], coords=dict(lat=glat[::-1], lon=glon), dims=('lat', 'lon')).sel(
        lon=slice(region[0], region[1]), lat=slice(region[2], region[3]))


def point_grid(values, plon, plat, region=REGION_AUSTRALIA, spacing=0.5):
    """values at scattered points (e.g. the data points) gridded for display (nearest neighbour)"""
    import pygmt
    return pygmt.nearneighbor(x=np.mod(plon, 360), y=plat, z=values, region=region, spacing=spacing,
                              search_radius='1d')


def map_panel(fig, grid, title, cmap, series, region=REGION_AUSTRALIA, width='7c', label=None, reverse=False):
    """one map: grid on a Mercator frame with coastlines and a colour bar below"""
    import pygmt
    fig.basemap(region=region, projection=f'M{width}', frame=['xa10f5', 'ya10f5', f'WSne+t{title}'])
    pygmt.makecpt(cmap=cmap, series=series, background=True, reverse=reverse)
    fig.grdimage(grid, cmap=True, nan_transparent=True)
    fig.coast(shorelines='0.3p,gray30', resolution='l', area_thresh=2000)
    fig.colorbar(cmap=True, frame=f'xaf+l{label}' if label else 'xaf', position='JBC+w5c+o0c/0.7c')
