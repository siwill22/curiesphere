"""
Region: the area inverted, from an outline file.

An outline is a GeoPackage or GeoJSON of polygons in lon/lat (WGS84). Its edges are read as
straight lines in lon/lat, as GIS software draws them, and densified before the great-circle
point-in-polygon test (pygplates), so the test is done on the sphere. Longitudes are unwrapped
along each edge, so an outline may cross 180 deg; a polygon may contain a pole.

The extent is the set of 0.1 deg grid nodes inside any outline polygon (and outside its holes),
minus the nodes with an ocean age: the ocean model is kept there.
"""
import time

import geopandas as gpd
import numpy as np
import pygplates
from ptt.utils.points_in_polygons import find_polygons

DENSIFY_DEG = 0.1


def densified_polygon(ring):
    """PolygonOnSphere through the ring's vertices, with points every DENSIFY_DEG along each
    edge, so that its great-circle edges follow the straight lon/lat lines of the GIS source"""
    xy = np.asarray(ring.coords)[:, :2]
    pts = []
    for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:]):
        dx = (x1 - x0 + 180.) % 360. - 180.                        # shortest way round in longitude
        n = max(1, int(np.ceil(max(abs(dx), abs(y1 - y0))/DENSIFY_DEG)))
        t = np.arange(n)/n
        pts += list(zip(y0 + t*(y1 - y0), x0 + t*dx))
    return pygplates.PolygonOnSphere(pts)


def read_outline(path):
    """outline file -> (list of PolygonOnSphere, list of hole PolygonOnSphere)"""
    gdf = gpd.read_file(path)
    assert gdf.crs is None or gdf.crs.to_epsg() == 4326, f'{path}: outline must be lon/lat WGS84'
    polygons, holes = [], []
    for geom in gdf.geometry:
        for part in getattr(geom, 'geoms', [geom]):
            polygons.append(densified_polygon(part.exterior))
            holes += [densified_polygon(h) for h in part.interiors]
    return polygons, holes


def inside(polygons, lon, lat):
    """boolean: which points (lon, lat arrays) fall in any of the polygons"""
    points = [pygplates.PointOnSphere(float(a), float(o)) for o, a in zip(lon, lat)]
    hit = find_polygons(points, polygons, list(range(len(polygons))), subdivision_depth=6)
    return np.array([h is not None for h in hit])


def extent_grid(outline_path, glon, glat, age):
    """boolean grid (lat, lon) of the 0.1 deg nodes in the outline and without an ocean age.
    Only the latitude band spanned by the outline is tested (to the pole if a polygon contains it)."""
    polygons, holes = read_outline(outline_path)
    lats = np.concatenate([[p.to_lat_lon()[0] for p in poly.get_points()] for poly in polygons])
    lat0, lat1 = lats.min() - 1., lats.max() + 1.
    if any(p.is_point_in_polygon(pygplates.PointOnSphere(90., 0.)) for p in polygons):
        lat1 = 90.
    if any(p.is_point_in_polygon(pygplates.PointOnSphere(-90., 0.)) for p in polygons):
        lat0 = -90.
    ilat = np.where((glat >= lat0) & (glat <= lat1))[0]
    II, JJ = np.meshgrid(ilat, np.arange(len(glon)), indexing='ij')
    lon, lat = glon[JJ.ravel()], glat[II.ravel()]
    t = time.time()
    hit = inside(polygons, lon, lat)
    if holes:
        hit &= ~inside(holes, lon, lat)
    print(f'point-in-polygon: {len(lon)} nodes, {len(polygons)} polygons, {len(holes)} holes, {time.time()-t:.0f} s')
    ext = np.zeros(age.shape, dtype=bool)
    ext[II.ravel(), JJ.ravel()] = hit
    return ext & np.isnan(age)


def hasterok_outline(path, continent=None, keys=None, crust_types=('continental', 'transitional'), exclude=()):
    """write an outline from the Hasterok et al. (2022) provinces (gprm): either the polygons with
    the given index keys, or those of a continent with the given crust types, minus the names in
    exclude. The province attributes are kept for reference."""
    from gprm.datasets import Geology
    prov = Geology.fetch_GlobalTectonicMap()
    if keys is not None:
        sel = prov.loc[list(keys)]
    else:
        sel = prov[(prov.continent == continent) & prov.crust_type.isin(crust_types)
                   & ~prov.prov_name.isin(list(exclude))]
    sel = sel[['id', 'prov_name', 'prov_type', 'crust_type', 'prov_group', 'lastorogen', 'geometry']].copy()
    sel['key'] = sel.index
    sel.set_crs(epsg=4326, allow_override=True).to_file(path, driver='GPKG')
    print(f'wrote {path}: {len(sel)} polygons')
    return sel


def cell_provinces(clon, clat):
    """Hasterok et al. (2022) province (index key of the gprm table) containing each cell centre,
    by point-in-polygon on the sphere (pygplates); -1 where none. Returns (keys, province table)."""
    from gprm.datasets import Geology
    prov = Geology.fetch_GlobalTectonicMap()
    polys, ids = [], []
    for k, geom in prov.geometry.items():
        for part in getattr(geom, 'geoms', [geom]):
            polys.append(pygplates.PolygonOnSphere([(y, x) for x, y in np.array(part.exterior.coords)]))
            ids.append(k)
    pts = [pygplates.PointOnSphere(la, lo if lo <= 180 else lo - 360) for lo, la in zip(clon, clat)]
    hit = find_polygons(pts, polys, ids)
    return np.array([-1 if h is None else h for h in hit]), prov
