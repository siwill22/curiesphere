"""
Geological and geophysical covariates on cells (for CR and its kin): grids averaged over each cell's 0.1 deg nodes
with cos(lat) weights (area on the sphere), and attributes of the Hasterok et al. (2022) province holding each cell
centre (point in polygon on the sphere, pygplates). The province polygons map surface geology: covariates to compare
with, never truth.

Grids ([covariates] of the config; '~' expanded; 'path' or 'path:variable' for NetCDF)
  crust      crustal thickness (km): Moho depth below sea level (xyz: lat, lon, depth) + elevation (earth relief 06m);
             crust_fallback (a thickness grid, km) where the Moho grid has no value
  lab        LAB depth (km; xyz: lon, lat, depth); lab_fallback (a grid in m) where it has none
  heat_flow  a heat-flow grid (mW/m2); heat_flow_alt: {name: grid}, as heat_flow_<name>
Province attributes
  class      0 cratons and shields, 1 orogenic belts, arcs and accretionary complexes, 2 passive margins, rifts and
             oceanic crust
  age        oldest magmatic age of the province (Ma); 'undated' marks provinces without one (age set to the median of
             the dated cells)
  group      prov_group (e.g. 'North Australian Craton')
"""
import os

import numpy as np
import xarray as xr

from . import geometry

CLASS_OF_TYPE = {'craton': 0, 'shield': 0,
                 'orogenic belt': 1, 'volcanic arc': 1, 'accretionary complex': 1, 'ophiolite complex': 1,
                 'passive margin': 2, 'wide rift': 2, 'narrow rift': 2, 'oceanic crust': 2,
                 # types found outside Australia (global layouts): basins with the extended crust, large igneous
                 # provinces with it too (their crust was thinned or intruded, not stabilised)
                 'back-arc basin': 2, 'oceanic back-arc basin': 2, 'foredeep basin': 2, 'basin': 2,
                 'magmatic province': 2}
CLASS_NAMES = ('cratons+shields', 'orogens+arcs', 'margins+rifts+oceanic')
VERSION = 1


def _nodes(cells, g=None):
    g = g or geometry.grid()
    i, j = np.nonzero(cells.cell >= 0)
    return g.lon[j], g.lat[i], cells.cell[i, j]


def cell_average(values, lat, idx, n):
    """area-weighted (cos lat) average per cell of node values; NaN nodes are left out"""
    w = np.cos(np.radians(lat))*np.isfinite(values)
    num = np.bincount(idx, w*np.nan_to_num(values), n)
    den = np.bincount(idx, w, n)
    return np.where(den > 0, num/np.where(den > 0, den, 1), np.nan)


def open_grid(spec):
    """'path' or 'path:variable' -> DataArray with ascending lat and lon"""
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


def grid_covariates(cells, spec):
    import pygmt
    lon, lat, idx = _nodes(cells)
    n = len(cells)
    elev = sample(pygmt.datasets.load_earth_relief(resolution='06m', registration='gridline'), lon, lat)/1e3
    moho = sample(xyz_grid(spec['moho'], order=('lat', 'lon')), lon, lat)
    crust = np.where(np.isfinite(moho), moho + elev, sample(open_grid(spec['crust_fallback']), lon, lat))
    lab = sample(xyz_grid(spec['lab']), lon, lat)
    lab = np.where(np.isfinite(lab), lab, sample(open_grid(spec['lab_fallback']), lon, lat)/1e3)
    out = dict(crust=cell_average(crust, lat, idx, n), lab=cell_average(lab, lat, idx, n),
               heat_flow=cell_average(sample(open_grid(spec['heat_flow']), lon, lat), lat, idx, n),
               crust_from_moho_grid=cell_average(np.isfinite(moho).astype(float), lat, idx, n))
    for name, path in spec.get('heat_flow_alt', {}).items():
        out[f'heat_flow_{name}'] = cell_average(sample(open_grid(path), lon, lat), lat, idx, n)
    return out


def provinces():
    from gprm.datasets import Geology
    return Geology.fetch_GlobalTectonicMap()


def cell_provinces(clon, clat):
    """the Hasterok et al. (2022) province (index key of the gprm table) of each cell centre; -1 where none"""
    import pygplates
    from ptt.utils.points_in_polygons import find_polygons
    prov = provinces()
    polys, ids = [], []
    for k, geom in prov.geometry.items():
        for part in getattr(geom, 'geoms', [geom]):
            polys.append(pygplates.PolygonOnSphere([(y, x) for x, y in np.array(part.exterior.coords)]))
            ids.append(k)
    pts = [pygplates.PointOnSphere(la, lo if lo <= 180 else lo - 360) for lo, la in zip(clon, clat)]
    hit = find_polygons(pts, polys, ids)
    return np.array([-1 if h is None else h for h in hit]), prov


def province_covariates(clon, clat):
    key, prov = cell_provinces(clon, clat)
    if not (key >= 0).all():
        raise ValueError('cells outside every province')
    row = prov.loc[key]
    cls = np.array([CLASS_OF_TYPE[t] for t in row['prov_type']])
    age = row['mag_max_age'].to_numpy(float)
    undated = ~np.isfinite(age)
    age = np.where(undated, np.median(age[~undated]), age)
    return dict(cls=cls, age=age, undated=undated.astype(float), key=key,
                group=row['prov_group'].fillna('').to_numpy(str), prov_type=row['prov_type'].to_numpy(str))


def load(cells, spec, cache, area_name):
    """all covariates of the cells (dict of arrays), cached"""
    from .problem import _hash
    key = dict(area=area_name, nside=cells.nside, ids=_hash(cells.ids), covariates=spec, version=VERSION)
    d = cache.entry('covariates', key)
    f = os.path.join(d, 'covariates.npz')
    if not os.path.exists(f):
        out = {**grid_covariates(cells, spec), **province_covariates(cells.clon, cells.clat)}
        from .cache import save_atomic
        save_atomic(f, lambda p: np.savez(p, **out))
    return dict(np.load(f))


def standardise(x, w=None):
    """(x - mean)/sd with area weights w"""
    w = np.ones_like(x) if w is None else w
    m = np.sum(w*x)/np.sum(w)
    return (x - m)/np.sqrt(np.sum(w*(x - m)**2)/np.sum(w))
