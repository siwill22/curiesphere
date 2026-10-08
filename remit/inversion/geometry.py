"""
Where sources live (CONTEXT.md): grid nodes, target areas, cells, layers and direction models.

  grid nodes   the 0.1 deg ocean-age grid (1801 x 3601, north pole first, 0-360 E): geometry only
  continental crust   every grid node without an ocean age, less the south-pole row and the 360 deg column
               (not in the Driscoll-Healy sums of the transform; the 360 column repeats 0)
  cells        HEALPix pixels (ring numbering) holding the grid nodes of a mask; cells at the mask's edge are
               partial. Equal area on the sphere; areas are summed from the nodes' cos(latitude) weights.
  layer        a slab between a top and a base surface: 'basement' (ellipsoid - water - sediment) or a constant
               depth in km below the WGS84 ellipsoid
  direction    'induced_igrf13': magnetisation along the IGRF-13 field, per SI.km of VIS
"""
import functools
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


def target_mask(area, g=None):
    """mask of a target area: 'continental_crust' (outline files come with the local fits)"""
    if area == 'continental_crust':
        return continental_crust(g)
    raise NotImplementedError(f'target area {area!r}')


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
    """radius (m) of a layer surface on the grid: 'basement', or a constant depth (km) below the ellipsoid"""
    g = g or grid()
    if surface == 'basement':
        return _basement(AGE_GRID)
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
    """magnetisation (r, theta, phi) per SI.km of VIS on the grid, in A"""
    g = g or grid()
    if direction == 'induced_igrf13':
        return depth.induced_components(g.lat)
    raise NotImplementedError(f'direction model {direction!r}')


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
