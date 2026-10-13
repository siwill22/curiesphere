"""
Candidate geology maps for the global layout test (docs/reports/2026-10-13-candidate-sets): other maps that claim to
divide the continents into meaningful units, turned into domains of cells the way the Hasterok provinces are, so the
evidence can rank them and spun copies of each can serve as its own null. Maps to compare, never truth.

  sources      polygons, one unit per polygon (point in polygon on the sphere, pygplates):
                 merdith      Merdith et al. (2021, v1.1) static polygons less the oceanic ones (gpml:Basin)
               lon/lat class grids, the unit of a node being the class of the grid cell holding it:
                 tc1          tectono-thermal age bins (Artemieva 2006; grid by C. Heine)
                 ecm1         ECM1 crustal type (Mooney et al. 2023), 1 deg
               a clustering, the unit of a node being the cluster of the nearest knot (great circle):
                 sl2013_k<k>  k-means on SL2013sv absolute Vs profiles over SL2013_DEPTHS km at the knots on
                              continental crust (Schaeffer & Lebedev 2013)
               ('hasterok' needs no source here: parameterise.layouts takes its provinces from the tables.)
  cell units   each cell takes the unit holding most of its area (cos(lat) weights); -1 where no unit holds a node
  patches      every connected piece of equal label is its own domain (the cell graph without its bridges, so an
               island is its own piece); pieces of fewer than MIN_CELLS cells are merged, smallest first, into the
               neighbouring piece they share most cell edges with (an island through its bridge). Labels marked as
               rest are kept whole, as in the province layouts.
"""
import os
import re

import numpy as np

from . import geometry, graph

VERSION = 1
MIN_CELLS = 3
DATA = os.path.expanduser('~/Data')
SL2013 = f'{DATA}/SeismicTomography/SL2013SV/SL2013sv_tri-grd_v2/SL2013sv_25k-k.mod'
SL2013_DEPTHS = (50., 300.)
TC1 = f'{DATA}/GMTdata/TectonoThermalAge/TC1_TectThermAge_RevEng_AgeBins.nc'
ECM1 = os.path.expanduser('~/Downloads/ECM1/ECM1.txt')
MERDITH = 'Merdith2021/SM2-Merdith_et_al_1_Ga_reconstruction_v1.1/shapes_static_polygons_Merdith_et_al.gpml'
SOURCES = ('tc1', 'ecm1', 'merdith')                       # plus sl2013_k<k>


def _west(lon):
    return (np.asarray(lon) + 180.) % 360. - 180.


def _class_grid(values, lon_c, lat_c, lon, lat):
    """class of the grid cell (centres lon_c, lat_c, both ascending, regular) holding each point; -1 for NaN"""
    d_lon, d_lat = lon_c[1] - lon_c[0], lat_c[1] - lat_c[0]
    i = np.clip(np.floor((lat - lat_c[0] + d_lat/2)/d_lat).astype(int), 0, len(lat_c) - 1)
    j = np.floor((_west(lon) - lon_c[0] + d_lon/2)/d_lon).astype(int) % len(lon_c)
    v = values[i, j]
    return np.where(np.isfinite(v), v, -1).astype(int)


def _tc1(lon, lat):
    import xarray as xr
    g = xr.open_dataarray(TC1)                                # dims (lat, lon) under any names, both ascending
    return _class_grid(g.values, g[g.dims[1]].values, g[g.dims[0]].values, lon, lat)


def _ecm1(lon, lat):
    import pandas as pd
    t = pd.read_csv(ECM1, sep=r'\s+', usecols=['Lon', 'Lat', 'Type'])
    names = sorted(t.Type.unique())
    t['code'] = t.Type.map({n: i for i, n in enumerate(names)}).astype(float)
    g = t.set_index(['Lat', 'Lon']).code.to_xarray()
    return _class_grid(g.values, g.Lon.values, g.Lat.values, lon, lat)


def sl2013_knots(k, g=None):
    """(lon, lat, cluster) of the SL2013sv knots on continental crust, clusters numbered slowest (0) to fastest"""
    import pandas as pd
    from sklearn.cluster import KMeans
    g = g or geometry.grid()
    m = pd.read_csv(SL2013, sep=r'\s+', comment='#', header=None,
                    names=['depth', 'lon', 'lat', 'dvs', 'vsref', 'dvs_pc', 'vsabs', 'vsmean', 'dvsm_pc'])
    m = m[(m.depth >= SL2013_DEPTHS[0]) & (m.depth <= SL2013_DEPTHS[1])]
    prof = m.pivot_table(index=['lon', 'lat'], columns='depth', values='vsabs')
    klon, klat = (prof.index.get_level_values(n).values for n in ('lon', 'lat'))
    # a knot is continental when the grid node nearest to it is (grid: north pole first, 0-360 E, 0.1 deg)
    i = np.clip(np.rint((g.lat[0] - klat)/(g.lat[0] - g.lat[1])).astype(int), 0, len(g.lat) - 1)
    j = np.rint((klon % 360.)/(g.lon[1] - g.lon[0])).astype(int) % (len(g.lon) - 1)
    keep = geometry.continental_crust(g)[i, j]
    lab = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(prof.values[keep])
    order = np.argsort(np.argsort([prof.values[keep][lab == c].mean() for c in range(k)]))
    return klon[keep], klat[keep], order[lab]


def _sl2013(k, lon, lat):
    from scipy.spatial import cKDTree
    klon, klat, lab = sl2013_knots(k)
    near = cKDTree(graph.unit(klon, klat)).query(graph.unit(lon, lat))[1]    # chord length: great-circle order
    return lab[near]


def _merdith(lon, lat):
    import pooch
    import pygplates
    from ptt.utils.points_in_polygons import find_polygons
    polygons, proxies = [], []
    for n, feature in enumerate(pygplates.FeatureCollection(os.path.join(pooch.os_cache('gprm'), MERDITH))):
        if str(feature.get_feature_type()) == 'gpml:Basin':
            continue
        for geom in feature.get_geometries():
            if isinstance(geom, pygplates.PolygonOnSphere):
                polygons.append(geom)
                proxies.append(n)
    points = [pygplates.PointOnSphere(float(a), float(o)) for o, a in zip(lon, lat)]
    hit = find_polygons(points, polygons, proxies, subdivision_depth=6)
    return np.array([-1 if h is None else h for h in hit])


def _source(name):
    if name.startswith('sl2013_k'):
        return lambda lon, lat: _sl2013(int(name[8:]), lon, lat)
    return {'tc1': _tc1, 'ecm1': _ecm1, 'merdith': _merdith}[name]


def node_units(name, cells, cache, area_name, g=None):
    """unit of every node of the cells (in the order of np.nonzero(cells.cell >= 0)); -1 where none (cached)"""
    g = g or geometry.grid()
    spec = dict(area=area_name, source=name, version=VERSION)
    if name.startswith('sl2013'):
        spec['depths'] = list(SL2013_DEPTHS)

    def compute():
        i, j = np.nonzero(cells.cell >= 0)
        return _source(name)(g.lon[j], g.lat[i]).astype(np.int32)
    return np.asarray(cache.array('candidate_nodes', spec, 'units', compute))


def cell_units(name, cells, cache, area_name, g=None):
    """unit of each cell: the one holding most of its area (cos(lat) weights); -1 where no unit holds a node"""
    g = g or geometry.grid()
    units = node_units(name, cells, cache, area_name, g)
    i, j = np.nonzero(cells.cell >= 0)
    c, w = cells.cell[i, j], np.cos(np.radians(g.lat[i]))
    sel = units >= 0
    key, inv = np.unique(np.stack([c[sel], units[sel]], 1), axis=0, return_inverse=True)
    tot = np.bincount(inv.ravel(), w[sel])
    order = np.lexsort((-tot, key[:, 0]))                     # per cell, the heaviest unit first
    first = order[np.r_[True, key[order[1:], 0] != key[order[:-1], 0]]]
    out = np.full(len(cells), -1)
    out[key[first, 0]] = key[first, 1]
    return out


def is_rest(label):
    """a rest domain, also once spun ('aus rest aus')"""
    return re.search(r'(^| )rest ', label) is not None


def patches(labels, edges, bridges, min_cells=MIN_CELLS, keep_whole=is_rest):
    """label of each cell after splitting labels into connected pieces (cell edges without the bridges) and merging
    pieces of fewer than min_cells cells, smallest first, into the neighbouring piece they share most edges with
    (bridges count for islands); labels for which keep_whole is true are not split"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    labels = np.asarray(labels).astype(str)
    n = len(labels)
    lab, inv = np.unique(labels, return_inverse=True)
    whole = np.array([keep_whole(s) for s in lab])[inv]
    is_bridge = np.zeros(len(edges), bool)
    if len(bridges):
        b = {tuple(e) for e in bridges}
        is_bridge = np.array([tuple(e) in b for e in edges])
    a, b_ = edges[:, 0], edges[:, 1]
    join = (inv[a] == inv[b_]) & (~is_bridge | whole[a])
    g = coo_matrix((np.ones(join.sum()), (a[join], b_[join])), shape=(n, n))
    piece = connected_components(g, directed=False)[1]
    for k in np.unique(inv[whole]):                           # a rest label stays one piece, however scattered
        piece[inv == k] = piece[np.flatnonzero(inv == k)[0]]
    while True:
        size = np.bincount(piece, minlength=n)
        whole_piece = np.bincount(piece, whole, minlength=n) > 0
        small = np.flatnonzero((size > 0) & (size < min_cells) & ~whole_piece)
        if not len(small):
            break
        p = small[np.argmin(size[small])]
        pa, pb = piece[a], piece[b_]
        out = np.concatenate([pb[(pa == p) & (pb != p)], pa[(pb == p) & (pa != p)]])
        if not len(out):
            break                                             # a piece alone in the graph: nothing to merge into
        cnt = np.bincount(out, minlength=n)
        q = max(np.flatnonzero(cnt == cnt.max()), key=lambda r: size[r])
        piece[piece == p] = q
    _, piece = np.unique(piece, return_inverse=True)
    # each piece is named after the label holding most of its cells
    key, cnt = np.unique(np.stack([piece, inv], 1), axis=0, return_counts=True)
    order = np.lexsort((-cnt, key[:, 0]))
    main = key[order[np.r_[True, key[order[1:], 0] != key[order[:-1], 0]]], 1]
    names = np.array([lab[m] if keep_whole(lab[m]) else f'{lab[m]} #{p}' for p, m in enumerate(main)], object)
    return names[piece]
