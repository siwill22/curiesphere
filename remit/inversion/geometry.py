"""
Where sources live (CONTEXT.md): grid nodes, target areas, cells, layers and direction models.

  grid nodes   the 0.1 deg ocean-age grid (1801 x 3601, north pole first, 0-360 E): geometry only
  continental crust   every grid node without an ocean age, less the south-pole row and the 360 deg column
               (not in the Driscoll-Healy sums of the transform; the 360 column repeats 0)
  outline      a target area from a GeoPackage or GeoJSON of lon/lat (WGS84) polygons: the continental-crust
               nodes inside any polygon and outside its holes. Edges are read as straight lines in lon/lat, as
               GIS software draws them, densified to 0.1 deg before a great-circle point-in-polygon test
               (pygplates), so the test is on the sphere; an outline may cross 180 deg or contain a pole.
  cells        HEALPix pixels (ring numbering) holding the grid nodes of a mask; cells at the mask's edge are
               partial. Equal area on the sphere; areas are summed from the nodes' cos(latitude) weights.
  layer        a slab between a top and a base surface: 'basement' (ellipsoid - water - sediment) or a constant
               depth in km below the WGS84 ellipsoid
  direction    'induced_igrf13': magnetisation along the IGRF-13 field, per SI.km of VIS
"""
import functools
import hashlib
import os
from dataclasses import dataclass

import numpy as np
from astropy_healpix import healpy as hp

from remit import depth
from remit.data.models import load_ocean_age_model

AGE_GRID = 'Seton2020_Muller2019_Merdith2021'
KM2_PER_NODE = (np.radians(0.1)*depth.R0/1e3)**2          # a 0.1 deg node at the equator


@dataclass(frozen=True)
class Grid:
    ocean: object                 # remit SeafloorGrid (age, palaeolatitude, declination)
    lon: np.ndarray
    lat: np.ndarray
    age: np.ndarray


@functools.lru_cache(maxsize=1)
def grid():
    ocean = load_ocean_age_model(AGE_GRID)
    return Grid(ocean, np.asarray(ocean.lon), np.asarray(ocean.lat), np.asarray(ocean.age))


def continental_crust(g=None):
    """mask of the grid nodes of continental crust (no ocean age)"""
    g = g or grid()
    m = np.isnan(g.age)
    m[-1, :] = False
    m[:, -1] = False
    return m


DENSIFY_DEG = 0.1
OUTLINE_VERSION = 1


def _densified_polygon(ring):
    """PolygonOnSphere through a ring's vertices, with points every DENSIFY_DEG along each edge, so that its
    great-circle edges follow the straight lon/lat lines of the GIS source"""
    import pygplates
    xy = np.asarray(ring.coords)[:, :2]
    pts = []
    for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
        dx = (x1 - x0 + 180.) % 360. - 180.                        # shortest way round in longitude
        n = max(1, int(np.ceil(max(abs(dx), abs(y1 - y0))/DENSIFY_DEG)))
        t = np.arange(n)/n
        pts += list(zip(y0 + t*(y1 - y0), x0 + t*dx))
    return pygplates.PolygonOnSphere(pts)


def read_outline(path):
    """outline file -> (polygons, holes), lists of pygplates PolygonOnSphere"""
    import geopandas as gpd
    gdf = gpd.read_file(path)
    if not (gdf.crs is None or gdf.crs.to_epsg() == 4326):
        raise ValueError(f'{path}: outline must be lon/lat WGS84')
    polygons, holes = [], []
    for geom in gdf.geometry:
        for part in getattr(geom, 'geoms', [geom]):
            polygons.append(_densified_polygon(part.exterior))
            holes += [_densified_polygon(h) for h in part.interiors]
    return polygons, holes


def _inside(polygons, lon, lat):
    import pygplates
    from ptt.utils.points_in_polygons import find_polygons
    points = [pygplates.PointOnSphere(float(a), float(o)) for o, a in zip(lon, lat)]
    hit = find_polygons(points, polygons, list(range(len(polygons))), subdivision_depth=6)
    return np.array([h is not None for h in hit])


def outline_nodes(path, g=None):
    """boolean grid of the nodes inside an outline (only the latitude band it spans is tested, to a pole if a
    polygon contains it)"""
    import pygplates
    g = g or grid()
    polygons, holes = read_outline(path)
    lats = np.concatenate([[p.to_lat_lon()[0] for p in poly.get_points()] for poly in polygons])
    lat0, lat1 = lats.min() - 1., lats.max() + 1.
    if any(p.is_point_in_polygon(pygplates.PointOnSphere(90., 0.)) for p in polygons):
        lat1 = 90.
    if any(p.is_point_in_polygon(pygplates.PointOnSphere(-90., 0.)) for p in polygons):
        lat0 = -90.
    ilat = np.flatnonzero((g.lat >= lat0) & (g.lat <= lat1))
    II, JJ = np.meshgrid(ilat, np.arange(len(g.lon)), indexing='ij')
    lon, lat = g.lon[JJ.ravel()], g.lat[II.ravel()]
    hit = _inside(polygons, lon, lat)
    if holes:
        hit &= ~_inside(holes, lon, lat)
    out = np.zeros(g.age.shape, dtype=bool)
    out[II.ravel(), JJ.ravel()] = hit
    return out


def area_name(area):
    """the cache name of a target area: 'continental_crust', or 'outline:<hash of the file's bytes>'"""
    if area == 'continental_crust':
        return area
    with open(os.path.expanduser(area), 'rb') as f:
        return 'outline:' + hashlib.sha256(f.read()).hexdigest()[:16]


def target_mask(area, cache=None, g=None):
    """mask of a target area: 'continental_crust', or the path of an outline file (its continental-crust nodes;
    cached when a cache is given)"""
    if area == 'continental_crust':
        return continental_crust(g)
    compute = lambda: outline_nodes(os.path.expanduser(area), g) & continental_crust(g)
    if cache is None:
        return compute()
    spec = dict(grid=AGE_GRID, area=area_name(area), densify_deg=DENSIFY_DEG, version=OUTLINE_VERSION)
    return np.asarray(cache.array('target_mask', spec, 'mask', compute))


@dataclass(frozen=True)
class Cells:
    nside: int
    cell: np.ndarray              # grid of cell indices (-1 outside the mask)
    ids: np.ndarray               # HEALPix ids (ring) of the cells
    area_km2: np.ndarray
    clon: np.ndarray              # pixel centres
    clat: np.ndarray

    def __len__(self):
        return len(self.ids)


def make_cells(mask, nside, g=None):
    g = g or grid()
    LO, LA = np.meshgrid(g.lon, g.lat)
    pix = hp.ang2pix(nside, LO[mask], LA[mask], lonlat=True)
    ids = np.unique(pix)
    cell = np.full(mask.shape, -1, dtype=np.int32)
    cell[mask] = np.searchsorted(ids, pix)
    area = np.bincount(cell[mask], weights=np.cos(np.radians(LA[mask])), minlength=len(ids))*KM2_PER_NODE
    clon, clat = (np.asarray(x, float) for x in hp.pix2ang(nside, ids, lonlat=True))
    return Cells(nside, cell, ids, area, clon, clat)


@functools.lru_cache(maxsize=2)
def _basement(_key):
    return depth.top_of_basement_radius(grid().ocean)


def surface_radius(surface, g=None):
    """radius (m) of a layer surface on the grid: 'basement', 'basement+<km>' (that far below the basement), or a
    constant depth (km) below the ellipsoid"""
    g = g or grid()
    if surface == 'basement':
        return _basement(AGE_GRID)
    if isinstance(surface, str) and surface.startswith('basement+'):
        return _basement(AGE_GRID) - float(surface[9:])*1e3
    if isinstance(surface, (int, float)):
        return depth.ellipsoid_radius(g.lat)[:, None]*np.ones((len(g.lat), len(g.lon))) - surface*1e3
    raise NotImplementedError(f'layer surface {surface!r}')


def layer_radii(layer, mask=None, g=None):
    """(r_top, r_base) of a layer {'top': ..., 'base': ...} on the grid, checked to be the right way up"""
    r_top, r_base = surface_radius(layer['top'], g), surface_radius(layer['base'], g)
    sel = np.ones(r_top.shape, bool) if mask is None else mask
    if not (r_top[sel] > r_base[sel]).all():
        raise ValueError(f'layer {layer}: base above top at some nodes')
    return r_top, r_base


def direction_components(direction, g=None):
    """magnetisation (r, theta, phi) per SI.km of VIS on the grid, in A. 'background:<remanence>:<vis model>[:nr]'
    is the background's own vertically integrated magnetisation per node instead (remanence of the ocean model, with
    the near-ridge enhancement for 'nr', plus the VIS model induced by IGRF-13; zero on continental crust), so a
    column is the field of the background in one cell (the ocean-model error of ADR 0009)"""
    g = g or grid()
    if direction == 'induced_igrf13':
        return depth.induced_components(g.lat)
    if direction.startswith('background:'):
        return _background_vim(direction)
    raise NotImplementedError(f'direction model {direction!r}')


@functools.lru_cache(maxsize=1)
def _background_vim(direction):
    from remit.data.models import OCEAN_MODEL_LIST, load_vis_model
    from remit.utils.grid import agearray2magnetisation, paleoIncDec2field
    _, remanence, vis_model, *nr = direction.split(':')
    g = grid()
    ocean, cont = g.ocean, continental_crust(g)
    params = OCEAN_MODEL_LIST[remanence]
    params = depth.with_near_ridge(params) if nr and nr[0] == 'nr' else params
    prof, _, rm = depth.profile_slices(params)
    M = agearray2magnetisation(ocean.age, prof.age, rm.sum(0))         # RVIM of the whole column (A)
    Br, Bt, Bp = paleoIncDec2field(ocean.paleolatitude, ocean.declination)
    C = np.sqrt(1 + 3*np.sin(np.radians(ocean.paleolatitude))**2)
    vis = load_vis_model(name=vis_model, match=(ocean.lon, ocean.lat, ocean.age)).vis
    vis = np.where(cont, 0., np.nan_to_num(np.asarray(vis)))
    ind = depth.induced_components(g.lat)
    return [np.where(cont, 0., np.nan_to_num(np.asarray(M*C*B))) + vis*i for B, i in zip((Br, Bt, Bp), ind)]


def ocean_mask(g=None):
    """the grid nodes outside continental crust (the background's sources in a global fit)"""
    g = g or grid()
    m = ~continental_crust(g)
    m[0, :] = False                 # the north-pole row: zero weight in the quadrature, and no Legendre derivative
    m[-1, :] = False
    m[:, -1] = False
    return m


def coarsen(cells):
    """the cells at NSIDE/2 made of whole groups of four nested children (no new forward calculation: a coarse
    column is the sum of its children's); returns the coarse cells and each fine cell's parent index"""
    parent_nest = hp.ring2nest(cells.nside, cells.ids)//4
    nside = cells.nside//2
    ids, parent = np.unique(hp.nest2ring(nside, parent_nest), return_inverse=True)
    cell = np.where(cells.cell >= 0, parent[np.maximum(cells.cell, 0)], -1).astype(np.int32)
    area = np.bincount(parent, weights=cells.area_km2, minlength=len(ids))
    clon, clat = (np.asarray(x, float) for x in hp.pix2ang(nside, ids, lonlat=True))
    return Cells(nside, cell, ids, area, clon, clat), parent
