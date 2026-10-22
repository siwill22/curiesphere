"""Candidate partitions of the continents to test against the magnetic field, one global map each.

Run from the repo root:  python docs/reports/2026-10-13-candidate-sets/make_candidates.py
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pooch
import pygmt
import pygplates
import xarray as xr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from sklearn.cluster import KMeans

from remit.inversion.covariates import provinces

OUT = 'docs/reports/2026-10-13-candidate-sets/'
CACHE = Path(pooch.os_cache('gprm'))
SL2013 = '/Users/simon/Data/SeismicTomography/SL2013SV/SL2013sv_tri-grd_v2/SL2013sv_25k-k.mod'
ECM1 = '/Users/simon/Downloads/ECM1/ECM1.txt'
CGMW = '/Users/simon/Data/Geology/CGMW_GeologicalUnits.shp'
GLIM = '/Users/simon/Data/Geology/GLiM/glim_wgs84_0point5deg.txt.asc'    # PANGAEA doi:10.1594/PANGAEA.788537
# GLiM classes (Hartmann & Moosdorf 2012); water bodies, no data and ice are not rock types and are left blank
GLIM_CLASSES = {1: 'unconsolidated sed.', 2: 'basic volcanic', 3: 'siliciclastic sed.', 4: 'basic plutonic',
                5: 'mixed sed.', 6: 'carbonate sed.', 7: 'acid volcanic', 8: 'metamorphic', 9: 'acid plutonic',
                10: 'intermediate volcanic', 12: 'pyroclastic', 13: 'intermediate plutonic', 14: 'evaporite'}
K = 6                    # SL2013sv clusters
DEPTHS = (50, 300)       # km, depth range of the velocity profiles that are clustered
AGE_CPT = dict(cmap='batlow', series=[0, 4000, 250], background=True)
# 20 distinguishable colours (matplotlib tab20) for class maps
PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
           '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7',
           '#dbdb8d', '#9edae5']


# ---------- helpers ----------

def unit_vectors(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])


def count_patches(cls):
    """connected patches of equal class on a global lon/lat grid (4-neighbour), joined across the dateline"""
    ny, nx = cls.shape
    idx = np.arange(ny*nx).reshape(ny, nx)
    ok = np.isfinite(cls)
    pairs = [(idx[:, :-1], idx[:, 1:], cls[:, :-1], cls[:, 1:]),     # east-west
             (idx[:, -1], idx[:, 0], cls[:, -1], cls[:, 0]),          # across the dateline
             (idx[:-1], idx[1:], cls[:-1], cls[1:])]                  # north-south
    rows, cols = [], []
    for a, b, ca, cb in pairs:
        same = (ca == cb) & np.isfinite(ca)
        rows.append(a[same]), cols.append(b[same])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(rows.size), (rows, cols)), shape=(ny*nx, ny*nx))
    _, lab = connected_components(graph, directed=False)
    return np.unique(lab[ok.ravel()]).size


def gpml_polygons(path, skip_type=None):
    """polygons of a GPlates feature file, cut at the dateline (pygplates) so GMT draws them correctly"""
    wrapper = pygplates.DateLineWrapper()
    rows = []
    for feature in pygplates.FeatureCollection(str(path)):
        if skip_type and str(feature.get_feature_type()) == skip_type:
            continue
        for geom in feature.get_geometries():
            if not isinstance(geom, pygplates.PolygonOnSphere):
                continue
            for piece in wrapper.wrap(geom):
                pts = [(p.get_longitude(), p.get_latitude()) for p in piece.get_exterior_points()]
                if len(pts) > 2:
                    rows.append(dict(name=feature.get_name(), geometry=Polygon(pts)))
    return gpd.GeoDataFrame(rows, crs='EPSG:4326')


def random_codes(labels, seed=1):
    """a random colour index per unit, so neighbours rarely share a colour"""
    codes = pd.factorize(labels)[0]
    return np.random.default_rng(seed).permutation(codes.max() + 1)[codes]


# ---------- the candidates ----------

def sl2013_clusters():
    """k-means on absolute Vs profiles at the SL2013sv knots, then nearest knot (great circle) for each 0.5° node"""
    m = pd.read_csv(SL2013, sep=r'\s+', comment='#', header=None,
                    names=['depth', 'lon', 'lat', 'dvs', 'vsref', 'dvs_pc', 'vsabs', 'vsmean', 'dvsm_pc'])
    m = m[(m.depth >= DEPTHS[0]) & (m.depth <= DEPTHS[1])]
    prof = m.pivot_table(index=['lon', 'lat'], columns='depth', values='vsabs')
    lab = KMeans(n_clusters=K, n_init=20, random_state=0).fit_predict(prof.values)
    # number clusters from slowest (0) to fastest (K-1), by mean Vs over the profile
    order = np.argsort(np.argsort([prof.values[lab == c].mean() for c in range(K)]))
    lab = order[lab]
    klon, klat = prof.index.get_level_values(0).values, prof.index.get_level_values(1).values
    lon, lat = np.arange(-180, 180, 0.5) + 0.25, np.arange(-90, 90, 0.5) + 0.25
    glon, glat = np.meshgrid(lon, lat)
    _, near = cKDTree(unit_vectors(klon, klat)).query(unit_vectors(glon.ravel(), glat.ravel()))
    return xr.DataArray(lab[near].reshape(glon.shape).astype(float), coords=dict(lat=lat, lon=lon), dims=('lat', 'lon'))


def ecm1_types():
    t = pd.read_csv(ECM1, sep=r'\s+', usecols=['Lon', 'Lat', 'Type'])
    names = sorted(t.Type.unique())
    t['code'] = t.Type.map({n: i for i, n in enumerate(names)}).astype(float)
    grid = t.set_index(['Lat', 'Lon']).code.to_xarray().rename(Lat='lat', Lon='lon')
    return grid, names


def surface_geology():
    from gprm.datasets import Geology
    return Geology.fetch_SurfaceGeology().to_crs('EPSG:4326')     # supplied in WGS72


def cgmw_units():
    """CGMW Geological Map of the World units, with the map's own colours (CMYK in %) as RGB"""
    g = gpd.read_file(CGMW)
    k = 1 - g.NOIR/100
    rgb = [(255*(1 - g[c]/100)*k).round().astype(int) for c in ('CYAN', 'MAGENTA', 'JAUNE')]
    g['colour'] = rgb[0].astype(str) + '/' + rgb[1].astype(str) + '/' + rgb[2].astype(str)
    return g


def glim_classes():
    """GLiM 0.5° grid, classes renumbered 0..n-1 in the order of GLIM_CLASSES"""
    raw = np.loadtxt(GLIM, skiprows=6)
    code = np.full(raw.shape, np.nan)
    for i, value in enumerate(GLIM_CLASSES):
        code[raw == value] = i
    lat, lon = 89.75 - 0.5*np.arange(raw.shape[0]), -179.75 + 0.5*np.arange(raw.shape[1])
    return xr.DataArray(code[::-1], coords=dict(lat=lat[::-1], lon=lon), dims=('lat', 'lon'))


def tc1_ages():
    from gprm.datasets import Geology
    return Geology.fetch_TC1().age


# ---------- the figure ----------

def pixel(grid):
    grid.gmt.registration, grid.gmt.gtype = 1, 1
    return grid


def panel(fig, title):
    fig.basemap(region='d', projection='N0/12c', frame=['f', '+t' + title])


def coast(fig):
    fig.coast(shorelines='0.15p,gray40', resolution='c', area_thresh=5000)


def polygons(fig, gdf, codes, cpt, pen='0.1p,gray15', cpt_file=None):
    gdf = gdf[['geometry']].assign(z=np.asarray(codes, float)).explode(index_parts=False)
    # holes confuse the per-polygon colours in GMT, so keep outer rings only, draw large polygons first
    # and let the small ones (including those that sat in holes) cover them
    gdf['geometry'] = [Polygon(g.exterior) for g in gdf.geometry]
    gdf = gdf.iloc[np.argsort(-gdf.geometry.to_crs('+proj=cea').area.values)]
    if cpt_file is None:
        pygmt.makecpt(**cpt)
    fig.plot(data=gdf, fill='+z', cmap=cpt_file or True, aspatial='Z=z', pen=pen, close=True)


def class_cpt(labels, colours=PALETTE):
    """a categorical CPT file with one key per integer class code 0..n-1"""
    path = OUT + '_classes.cpt'
    with open(path, 'w') as f:
        f.write('# COLOR_MODEL = RGB\n')
        for i, (colour, label) in enumerate(zip(colours, labels)):
            f.write(f'{i}\t{colour}\t;{label}\n')
        f.write('N\t255/255/255\n')
    return path


def class_legend(fig, labels, columns=5):
    spec = OUT + '_legend.txt'
    with open(spec, 'w') as f:
        f.write(f'N {columns}\n')
        for colour, label in zip(PALETTE, labels):
            f.write(f'S 0.12c s 0.2c {colour} 0.1p,gray30 0.3c {label}\n')
    fig.legend(spec=spec, position='JBC+o0/0.15c+w12c')
    Path(spec).unlink()


def main():
    pygmt.config(FONT_TITLE='9p', FONT_ANNOT_PRIMARY='5p', FONT_LABEL='6p', MAP_FRAME_TYPE='plain',
                 MAP_TITLE_OFFSET='4p')
    fig = pygmt.Figure()
    k = 0

    def next_panel(title):
        nonlocal k
        if k:
            fig.shift_origin(xshift='13c' if k % 2 else '-13c', yshift='0c' if k % 2 else '-8.6c')
        k += 1
        panel(fig, title)

    # 1 Hasterok provinces, coloured by province type
    prov = provinces()
    types = sorted(prov.prov_type.dropna().unique())
    next_panel(f'Hasterok et al. (2022) provinces: {len(prov)} polygons')
    polygons(fig, prov, prov.prov_type.map({t: i for i, t in enumerate(types)}).fillna(-1),
             None, cpt_file=class_cpt(types))
    coast(fig)
    class_legend(fig, types)

    # 2 CGMW geological map, in its own colours
    cg = cgmw_units()
    colours = pd.factorize(cg.colour)
    next_panel(f'CGMW Geological Map of the World: {len(cg)} polygons')
    polygons(fig, cg, colours[0], None, pen=None, cpt_file=class_cpt([str(i) for i in range(len(colours[1]))], list(colours[1])))
    coast(fig)

    # 3 NRCan surface geology, coloured by its older age limit
    geo = surface_geology()
    next_panel(f'NRCan surface geology: {len(geo)} polygons')
    polygons(fig, geo, geo.FROMAGE.where(geo.FROMAGE > 0), AGE_CPT, pen=None)
    coast(fig)
    fig.colorbar(cmap=True, position='JBC+w8c/0.2c+o0/0.3c+h', frame='xa500+lOlder age limit of the unit (Ma)')

    # 4 GLiM lithology
    gl = glim_classes()
    next_panel(f'GLiM lithology (Hartmann & Moosdorf 2012), 0.5\\260: {count_patches(gl.values)} patches')
    cpt = class_cpt(list(GLIM_CLASSES.values()))
    fig.grdimage(pixel(gl), cmap=cpt, nan_transparent=True, interpolation='n')
    coast(fig)
    class_legend(fig, list(GLIM_CLASSES.values()))

    # 5 SL2013sv clusters
    sl = sl2013_clusters()
    next_panel(f'SL2013sv k-means, k = {K}, {DEPTHS[0]}-{DEPTHS[1]} km: {count_patches(sl.values)} patches')
    pygmt.makecpt(cmap='roma', series=[0, K, 1], output=OUT + '_roma.cpt')
    rgb = [line.split()[1] for line in open(OUT + '_roma.cpt') if line[0].isdigit()][:K]
    Path(OUT + '_roma.cpt').unlink()
    cpt = class_cpt(['slowest'] + [str(i) for i in range(2, K)] + ['fastest'], rgb)
    fig.grdimage(pixel(sl), cmap=cpt, nan_transparent=True, interpolation='n')
    coast(fig)
    fig.colorbar(cmap=cpt, position='JBC+w8c/0.2c+o0/0.3c', equalsize=0.4)

    # 6 ECM1 crustal type
    ecm, names = ecm1_types()
    next_panel(f'ECM1 crustal type (Mooney et al. 2023): {count_patches(ecm.values)} patches')
    cpt = class_cpt(names)
    fig.grdimage(pixel(ecm), cmap=cpt, nan_transparent=True, interpolation='n')
    coast(fig)
    class_legend(fig, names, columns=6)

    # 7 TC1 tectono-thermal age
    tc = tc1_ages()
    next_panel(f'TC1 tectono-thermal age (Artemieva 2006): {count_patches(tc.values)} patches')
    pygmt.makecpt(**AGE_CPT)
    fig.grdimage(pixel(tc), cmap=True, nan_transparent=True)
    coast(fig)
    fig.colorbar(cmap=True, position='JBC+w8c/0.2c+o0/0.3c+h', frame='xa500+lMid-point of age bin (Ma)')

    # 8 Continental static polygons of a plate model ('Basin' features are the oceanic ones)
    static = gpml_polygons(CACHE/'Merdith2021/SM2-Merdith_et_al_1_Ga_reconstruction_v1.1/shapes_static_polygons_Merdith_et_al.gpml',
                           skip_type='gpml:Basin')
    next_panel(f'Merdith et al. (2021) static polygons, non-oceanic: {static.name.nunique()} units')
    polygons(fig, static, random_codes(static.name), dict(cmap='roma', series=[0, static.name.nunique()]))
    coast(fig)

    fig.savefig(OUT + 'candidates.png', dpi=200)
    Path(OUT + '_classes.cpt').unlink()


if __name__ == '__main__':
    main()
