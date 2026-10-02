"""
Set-up for the free-domain inversion of another continent (North America): the outer extent,
fixed background, data points, data and noise, written in the form that cell_basis.py and
merge_domains.py read. Same choices as invert_domains.py (Australia), sections 1-5.

Extent: Hasterok et al. (2022) provinces (gprm) with continent == 'North America' and continental
or transitional crust, west of 180 deg (Chukotka and the far Aleutians are cut off), with the units
in EXCLUDE left out: Central America and the Caribbean arcs, the Aleutian arc, the Arctic Ocean
ridges and the Siberian units. Greenland, the Arctic islands and Mexico are in. Cells with ocean age
keep the ocean model. inversion_<region>/domains.csv lists every polygon with include = 1/0; it is
written once and then read, so it can be edited by hand. Only the outer extent is used: the free
domains form from the data, with no polygon boundaries inside.

VIS floor, as for Australia: the H&M (+ slabs) area-mean and 10th percentile over the region's
Phanerozoic continental provinces (Hasterok lastorogen Cordilleran, Caledonian or Variscan-Hercynian).

Run from notebooks/ (conda env pygmt17), then cell_basis.py, merge_domains.py and merge_compact.py
with the same REGION:
    REGION=north_america PYTHONPATH=..:$PYTHONPATH python region_setup.py
"""
import os
import time

import numpy as np
import pandas as pd
import pygplates
import pyshtools
from scipy.spatial import cKDTree
from gprm.datasets import Geology
from ptt.utils.points_in_polygons import find_polygons

import depth_models as dm
from remit.data.models import load_ocean_age_model, load_vis_model, load_lcs, DATA_DIR
from remit.utils.region import generate_healpix_points

REGION = os.environ.get('REGION', 'north_america')
OUT = f'inversion_{REGION}'
os.makedirs(OUT, exist_ok=True)
TABLE = f'{OUT}/domains.csv'

R0 = dm.R0
LMIN, LMAX_FIT, LMAX = 16, 133, dm.LMAX
BASE_DEPTH = 30e3
BUFFER_KM = 300.
NSIDE = 128
DENSIFY_DEG = 0.1
N_NOISE = 100
SEED = 20261001
CONTINENT = 'North America'
BBOX = (180., 360., 5., 90.)          # lon 0-360: the western hemisphere
EXCLUDE = [
    # Central America and the Caribbean
    'Chortis Block', 'Mosquito Lowlands', 'Central American Arc', 'Central American Forearc',
    'Chorotega Volcanic Arc', 'Chorotega Backarc', 'Choco Block', 'Nicaragua Plateau', 'Cayman Arc',
    'Greater Antilles Arc', 'Greater Antilles Accretionary Complex', 'South Hispaniola Wedge',
    # Aleutian arc
    'East Aleutian Arc', 'East Aleutian Forearc',
    # Arctic Ocean ridges and basins, Siberian units
    'Alpha Mendeleev Ridge', 'Lomonosov Ridge', 'Podvodnikov Basin', 'Unnamed Fold Belt',
    'Novosibirsk-Chukhi Fold Belt', 'Bennett-Borrovia Block', 'North Chuchi Basin']
PHANEROZOIC = ['Cordilleran', 'Caledonian', 'Variscan-Hercynian']


# ============================================================ 1. extent table
provinces = Geology.fetch_GlobalTectonicMap()
reg = provinces[provinces.continent == CONTINENT].copy()
reg['key'] = reg.index
if not os.path.exists(TABLE):
    table = reg[['key', 'id', 'prov_name', 'prov_type', 'crust_type', 'prov_group', 'lastorogen']].copy()
    table['include'] = (reg.crust_type.isin(['continental', 'transitional'])
                        & ~reg.prov_name.isin(EXCLUDE)).astype(int)
    table.to_csv(TABLE, index=False)
    print(f'wrote draft {TABLE}')
table = pd.read_csv(TABLE, keep_default_na=False)
table = table[table.include == 1]
print(f'{len(table)} polygons in the extent')


# ============================================================ 2. extent on the 0.1 deg grid
ocean = load_ocean_age_model()
glon, glat = np.asarray(ocean.lon), np.asarray(ocean.lat)
age = np.asarray(ocean.age)
LO, LA = np.meshgrid(glon, glat)
cache = f'{OUT}/param_grid.npz'
if os.path.exists(cache):
    z = np.load(cache)
    pgrid, phan = z['pgrid'], z['phanerozoic']
else:
    def densified_polygon(ring):
        xy = np.asarray(ring.coords)
        pts = []
        for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
            n = max(1, int(np.ceil(max(abs(x1-x0), abs(y1-y0))/DENSIFY_DEG)))
            t = np.arange(n)/n
            pts += list(zip(y0 + t*(y1-y0), x0 + t*(x1-x0)))
        return pygplates.PolygonOnSphere(pts)

    # proxy 0: Phanerozoic continental province, 1: any other province in the extent; holes of a
    # polygon are tested separately and removed
    polygons, proxies, holes = [], [], []
    for _, row in table.iterrows():
        geom = reg.loc[row.key].geometry
        proxy = 0 if (row.crust_type == 'continental' and row.lastorogen in PHANEROZOIC) else 1
        for part in getattr(geom, 'geoms', [geom]):
            polygons.append(densified_polygon(part.exterior))
            proxies.append(proxy)
            holes += [densified_polygon(h) for h in part.interiors]
    ilon = np.where((glon >= BBOX[0]) & (glon <= BBOX[1]))[0]
    ilat = np.where((glat >= BBOX[2]) & (glat <= BBOX[3]))[0]
    II, JJ = np.meshgrid(ilat, ilon, indexing='ij')
    points = [pygplates.PointOnSphere(float(glat[i]), float(glon[j])) for i, j in zip(II.ravel(), JJ.ravel())]
    t = time.time()
    hit = np.array([-1 if h is None else h for h in find_polygons(points, polygons, proxies, subdivision_depth=6)])
    if holes:
        in_hole = np.array([h is not None for h in find_polygons(points, holes, list(range(len(holes))),
                                                                 subdivision_depth=6)])
        hit[in_hole] = -1
    print(f'point-in-polygon on {len(points)} cells ({len(polygons)} polygons, {len(holes)} holes): '
          f'{time.time()-t:.0f} s')
    grid = np.full(age.shape, -1, dtype=np.int16)
    grid[II.ravel(), JJ.ravel()] = hit
    phan = grid == 0
    pgrid = np.where(grid >= 0, 0, -1).astype(np.int16)
    pgrid[~np.isnan(age)] = -1                                        # age grid wins
    phan &= pgrid == 0
    np.savez(cache, pgrid=pgrid, params=np.array(['extent']), phanerozoic=phan)
free = pgrid >= 0
km2 = (np.radians(0.1)*R0/1e3)**2
w = np.cos(np.radians(LA))
print(f'extent: {free.sum()} grid cells, {w[free].sum()*km2/1e6:.2f} M km2')

vis = load_vis_model(name='Hemant2005+slabs', match=(ocean.lon, ocean.lat, ocean.age))
hm = np.asarray(vis.vis)
x = hm[phan]
order = np.argsort(x)
cum = np.cumsum(w[phan][order])/w[phan].sum()
floor_mean, floor_p10 = np.average(x, weights=w[phan]), x[order][np.searchsorted(cum, 0.1)]
np.savez(f'{OUT}/floor.npz', v_mins=[floor_p10, floor_mean])
print(f'VIS floor: H&M over Phanerozoic continental provinces ({w[phan].sum()*km2/1e6:.2f} M km2): '
      f'area-mean {floor_mean:.3f}, 10th percentile {floor_p10:.3f} SI.km')


# ============================================================ 3. data points
hlon, hlat = generate_healpix_points(NSIDE)
hlon = np.mod(hlon, 360.)

def unit(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], -1)

chord, _ = cKDTree(unit(LO[free], LA[free])).query(unit(hlon, hlat))
keep = 2*np.arcsin(np.clip(chord/2, 0, 1))*R0/1e3 <= BUFFER_KM
plon, plat = hlon[keep], hlat[keep]
print(f'points: {keep.sum()}')


# ============================================================ 4. extent column, background, data
l = np.arange(LMAX+1)
F = dm.lcs1_filter(l, dm.fit_lh())

def br_at_points(c):
    c = np.array(c)[:, :LMAX_FIT+1, :LMAX_FIT+1]*F[None, :LMAX_FIT+1, None]
    c[:, :LMIN] = 0
    return np.asarray(pyshtools.SHMagCoeffs.from_array(c, r0=R0).expand(
        a=R0, lat=plat, lon=plon, lmax_calc=LMAX_FIT))[:, 0]

def load_cof(name):
    clm, _ = pyshtools.shio.shread(f'{DATA_DIR}/shc/{name}', lmax=LMAX_FIT)
    return clm

cache = f'{OUT}/columns.npz'
if not os.path.exists(cache):
    t = time.time()
    r_top = dm.top_of_basement_radius(ocean)
    r_base = dm.ellipsoid_radius(glat)[:, None]*np.ones_like(r_top) - BASE_DEPTH
    assert (r_top[free] > r_base[free]).all(), 'basement below the slab base'
    to_gmm = dm.induced_to_gmm(glon, glat)
    C_cols = np.array([dm.slab_vis_coeffs(np.where(free, 1., 0.), r_top, r_base, to_gmm).coeffs])
    vis.vis = np.where(free, 0., vis.vis)
    C_bg = dm.vis_coeffs(ocean, vis, dm.top_of_crust_radius(ocean)).coeffs
    C_bg += np.load(f'{dm.CACHE}/GK07_nr_{LMAX}.npy')
    np.savez(f'{OUT}/column_coeffs.npz', params=np.array(['extent']), C_cols=C_cols, C_bg=C_bg)
    G = br_at_points(C_cols[0])[:, None]
    b = br_at_points(C_bg)
    d_lcs = br_at_points(load_lcs(lmin=LMIN, lmax=LMAX).coeffs)
    d_mf7 = br_at_points(load_cof('MF7_mod.cof'))
    np.savez(cache, params=np.array(['extent']), plon=plon, plat=plat, G=G, b=b, d_lcs=d_lcs, d_mf7=d_mf7)
    print(f'columns: {time.time()-t:.0f} s')
z = np.load(cache)
assert np.array_equal(z['plon'], plon)
d_lcs, d_mf7 = z['d_lcs'], z['d_mf7']


# ============================================================ 5. error model
sigma = np.sqrt(np.mean((d_lcs - d_mf7)**2))
cache = f'{OUT}/noise.npz'
if not os.path.exists(cache):
    diff = load_lcs(lmin=LMIN, lmax=LMAX_FIT).coeffs - load_cof('MF7_mod.cof')
    diff[:, :LMIN] = 0
    ll = l[:LMAX_FIT+1]
    R_l = (ll+1)*(diff**2).sum((0, 2))
    var = np.where(ll >= LMIN, R_l/((ll+1)*(2*ll+1)), 0.)
    rng = np.random.default_rng(SEED)
    noise = []
    for _ in range(N_NOISE):
        c = rng.standard_normal((2, LMAX_FIT+1, LMAX_FIT+1))*np.sqrt(var)[None, :, None]
        c[1, :, 0] = 0
        c *= np.arange(LMAX_FIT+1)[None, None, :] <= ll[None, :, None]
        noise.append(np.asarray(pyshtools.SHMagCoeffs.from_array(c, r0=R0).expand(
            a=R0, lat=plat, lon=plon, lmax_calc=LMAX_FIT))[:, 0])
    np.savez(cache, noise=np.array(noise))
print(f'sigma = RMS(LCS-1 - MF7) at the points = {sigma:.2f} nT; LCS-1 RMS = {d_lcs.std():.2f} nT')
