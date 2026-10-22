"""
Geological and geophysical covariates on the cells: grids averaged over each cell's 0.1 deg nodes
with cos(lat) weights (area on the sphere), and attributes of the Hasterok et al. (2022) province
containing each cell centre.

Grids (paths in [covariates] of the region's config.toml; '~' expanded)
  crust      crustal thickness (km) = AusMoho2012 Moho depth below sea level + elevation (earth relief,
             06m), CRUST1.0 crustal thickness where AusMoho2012 has no value
  lab        AuSREM LAB depth (km), LITHO1.0 lid bottom (m -> km) where AuSREM has no value
  heat_flow  Lucazeau (2019) HFgrid14 (mW/m2); alternatives in heat_flow_alt ("path:variable")
Province attributes
  class      0 cratons and shields, 1 orogenic belts, arcs and accretionary complexes,
             2 passive margins, rifts and oceanic crust
  age        oldest magmatic age of the province (mag_max_age, Ma); 'undated' marks provinces
             without one (their age is set to the median of the dated cells)
  group      prov_group (e.g. 'North Australian Craton')
The province polygons map surface geology; here they are covariates to compare with, not truth.
"""
import os

import numpy as np
import xarray as xr

CLASS_OF_TYPE = {'craton': 0, 'shield': 0,
                 'orogenic belt': 1, 'volcanic arc': 1, 'accretionary complex': 1,
                 'passive margin': 2, 'wide rift': 2, 'narrow rift': 2, 'oceanic crust': 2}
CLASS_NAMES = ('cratons+shields', 'orogens+arcs', 'margins+rifts+oceanic')
CRATON_GROUPS = ('North Australian Craton', 'West Australian Craton', 'South Australian Craton')


def nodes(cells):
    """lon, lat (deg) and cell index of the 0.1 deg nodes that belong to cells"""
    cell = cells['cell']
    glat = np.linspace(90, -90, cell.shape[0])
    glon = np.linspace(0, 360, cell.shape[1])
    i, j = np.nonzero(cell >= 0)
    return glon[j], glat[i], cell[i, j]


def cell_average(values, lat, idx, n):
    """area-weighted (cos lat) average per cell of node values; NaN nodes are left out"""
    w = np.cos(np.radians(lat))*np.isfinite(values)
    num = np.bincount(idx, w*np.nan_to_num(values), n)
    den = np.bincount(idx, w, n)
    return np.where(den > 0, num/np.where(den > 0, den, 1), np.nan)


def open_grid(spec):
    """'path' or 'path:variable' -> DataArray with lon in -180..180 and ascending lat"""
    path, _, var = spec.partition(':')
    ds = xr.open_dataset(os.path.expanduser(path))
    da = ds[var] if var else ds[list(ds.data_vars)[0]]
    da = da.rename({k: v for k, v in (('x', 'lon'), ('y', 'lat')) if k in da.dims})
    return da.sortby('lat').sortby('lon')


def xyz_grid(path, order=('lon', 'lat')):
    """regular xyz text grid -> DataArray"""
    a = np.loadtxt(os.path.expanduser(path))
    cols = {order[0]: a[:, 0], order[1]: a[:, 1]}
    lon, lat = np.unique(cols['lon']), np.unique(cols['lat'])
    g = np.full((len(lat), len(lon)), np.nan)
    g[np.searchsorted(lat, cols['lat']), np.searchsorted(lon, cols['lon'])] = a[:, 2]
    return xr.DataArray(g, coords=dict(lat=lat, lon=lon), dims=('lat', 'lon'))


def sample(da, lon, lat):
    """bilinear values of a lon/lat grid at points (lon in any convention)"""
    lon = np.asarray(lon, float)
    lon = np.where(lon > 180, lon - 360, lon) if float(da.lon.min()) < 0 else np.mod(lon, 360)
    return da.interp(lon=xr.DataArray(lon, dims='p'), lat=xr.DataArray(lat, dims='p')).values


def grid_covariates(cells, cfg):
    """crust, lab, heat_flow (+ heat_flow_<alt>) per cell"""
    import pygmt
    lon, lat, idx = nodes(cells)
    n = len(cells['cell_ids'])
    elev = sample(pygmt.datasets.load_earth_relief(resolution='06m', registration='gridline'), lon, lat)/1e3
    moho = sample(xyz_grid(cfg['moho'], order=('lat', 'lon')), lon, lat)
    crust = np.where(np.isfinite(moho), moho + elev, sample(open_grid(cfg['crust_fallback']), lon, lat))
    lab = sample(xyz_grid(cfg['lab']), lon, lat)
    lab = np.where(np.isfinite(lab), lab, sample(open_grid(cfg['lab_fallback']), lon, lat)/1e3)
    out = dict(crust=cell_average(crust, lat, idx, n), lab=cell_average(lab, lat, idx, n),
               heat_flow=cell_average(sample(open_grid(cfg['heat_flow']), lon, lat), lat, idx, n),
               crust_from_ausmoho=cell_average(np.isfinite(moho).astype(float), lat, idx, n))
    for name, spec in cfg.get('heat_flow_alt', {}).items():
        out[f'heat_flow_{name}'] = cell_average(sample(open_grid(spec), lon, lat), lat, idx, n)
    return out


def hm_cell_vis(cells, name='Hemant2005+slabs'):
    """the H&M (+ slabs) VIS averaged over each cell (for its statistics only, not as a map)"""
    from remit.data.models import load_vis_model
    hm = load_vis_model(name=name)
    da = xr.DataArray(np.asarray(hm.vis), coords=dict(lat=np.asarray(hm.lat), lon=np.asarray(hm.lon)),
                      dims=('lat', 'lon')).sortby('lat')
    lon, lat, idx = nodes(cells)
    return cell_average(sample(da, lon, lat), lat, idx, len(cells['cell_ids']))


def province_covariates(clon, clat):
    """class, age, undated, group and province key per cell (from the cell centre's province)"""
    from .region import cell_provinces
    key, prov = cell_provinces(clon, clat)
    assert (key >= 0).all(), 'cells outside every province'
    row = prov.loc[key]
    cls = np.array([CLASS_OF_TYPE[t] for t in row['prov_type']])
    age = row['mag_max_age'].to_numpy(float)
    undated = ~np.isfinite(age)
    age = np.where(undated, np.median(age[~undated]), age)
    return dict(cls=cls, age=age, undated=undated.astype(float), key=key,
                group=row['prov_group'].fillna('').to_numpy(str), prov_type=row['prov_type'].to_numpy(str))


def standardise(x, w=None):
    """(x - mean)/sd with area weights w"""
    w = np.ones_like(x) if w is None else w
    m = np.sum(w*x)/np.sum(w)
    return (x - m)/np.sqrt(np.sum(w*(x - m)**2)/np.sum(w))
