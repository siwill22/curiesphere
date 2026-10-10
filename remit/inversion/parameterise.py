"""
Parameterisations (ADR 0001): every one is a grouping of cells, G_param = G_cells S.

  domains   one value per domain: a table maps polygons of a source (the Hasterok et al. 2022 provinces) to named
            parameters (several polygons may share one; an empty name leaves the polygon out). Each grid node of the
            target area takes the parameter of the polygon holding it (point in polygon on the sphere), and each
            cell the parameter holding most of its area, so domain edges follow cell edges.
  design    columns for the regressions (CR), with the box of the uniform prior on each coefficient:
              intercept        one mean level (LEVEL)
              class            one mean per province class, in place of the intercept (LEVEL each)
              crust, lab, heat_flow, heat_flow_<alt>, age   standardised covariates (SLOPE per s.d.)
              undated, craton, group:<prov_group>           indicator offsets (SLOPE)
  indicator one column per label of a partition
  layouts   for a global fit (GCM, ADR 0008): the provinces of several continent tables grouped four ways, plus one
            level: 'continents', 'groups' (prov_group, else the province), 'provinces' (the tables' parameters) and
            'class' (covariates.CLASS_NAMES). Cells holding no node of any polygon form one rest domain per continent
            (the continent of the nearest cell in a polygon) in every layout.
            Controls blind to geology (CONTROLS): 'tiles<nside>' (the HEALPix pixels of that NSIDE) and
            'voronoi_groups', 'voronoi_provinces' (as many random Voronoi tiles on the sphere as the geological layout
            has domains; seeds drawn uniformly by area, seeded so they repeat).
"""
import hashlib
import os

import numpy as np

from . import covariates, geometry, graph

LEVEL, SLOPE = (0., 4.), (-2., 2.)
GROUPS = {'class': ['class'], 'crust': ['crust'], 'age': ['age', 'undated'], 'heat_flow': ['heat_flow'],
          'lab': ['lab']}
CRATON_GROUPS = ('North Australian Craton', 'West Australian Craton', 'South Australian Craton')
DOMAINS_VERSION = 1


def indicator(label):
    lab = np.unique(label, return_inverse=True)[1]
    return np.stack([lab == k for k in range(lab.max() + 1)], 1).astype(float)


def _domain_nodes(table, mask, g=None):
    """parameter index of each node of the mask (-1: none), and the parameter names"""
    import pandas as pd
    import pygplates
    from ptt.utils.points_in_polygons import find_polygons
    g = g or geometry.grid()
    t = table if isinstance(table, pd.DataFrame) else pd.read_csv(os.path.expanduser(table), keep_default_na=False)
    t = t[t.param != '']
    params = sorted(set(t.param), key=lambda p: (p.startswith('margin'), p))
    prov = covariates.provinces()
    polygons, proxies = [], []
    for _, row in t.iterrows():
        geom = prov.loc[row.key].geometry
        for part in getattr(geom, 'geoms', [geom]):
            polygons.append(geometry._densified_polygon(part.exterior))
            proxies.append(params.index(row.param))
    i, j = np.nonzero(mask)
    hit = find_polygons([pygplates.PointOnSphere(float(g.lat[a]), float(g.lon[b])) for a, b in zip(i, j)],
                        polygons, proxies, subdivision_depth=6)
    out = np.full(mask.shape, -1, dtype=np.int32)
    out[i, j] = [-1 if h is None else h for h in hit]
    return out, params


def domains(cells, table, cache, area_name, g=None):
    """(parameter index of each cell, parameter names, node coverage) for a polygon table"""
    g = g or geometry.grid()
    with open(os.path.expanduser(table), 'rb') as f:
        table_hash = hashlib.sha256(f.read()).hexdigest()[:16]
    spec = dict(area=area_name, table=table_hash, source='hasterok2022', version=DOMAINS_VERSION)
    d = cache.entry('domain_nodes', spec)
    f = os.path.join(d, 'nodes.npz')
    if not os.path.exists(f):
        nodes, params = _domain_nodes(table, cells.cell >= 0, g)
        from .cache import save_atomic
        save_atomic(f, lambda p: np.savez(p, nodes=nodes, params=np.array(params)))
    z = np.load(f)
    nodes, params = z['nodes'], [str(p) for p in z['params']]
    sel = (cells.cell >= 0) & (nodes >= 0)
    w = np.cos(np.radians(np.broadcast_to(g.lat[:, None], nodes.shape)[sel]))
    A = np.zeros((len(cells), len(params)))
    np.add.at(A, (cells.cell[sel], nodes[sel]), w)
    if (A.sum(1) == 0).any():
        raise ValueError(f'{(A.sum(1) == 0).sum()} cells hold no node of any domain')
    covered = (cells.cell >= 0).sum()
    return A.argmax(1), params, float(sel.sum()/covered)


def _continent(path):
    stem = os.path.basename(path).split('_')[0]
    return {'australia': 'aus'}.get(stem, stem)


LAYOUTS = ('one', 'continents', 'groups', 'provinces', 'class')
CONTROLS = ('tiles8', 'tiles16', 'voronoi_groups', 'voronoi_provinces')
VORONOI_SEED = 20261010


def _control(name, cells, xyz, by):
    """label of each cell for a control layout"""
    if name.startswith('tiles'):
        from astropy_healpix import healpy as hp
        return hp.ang2pix(int(name[5:]), cells.clon, cells.clat, lonlat=True)
    from scipy.spatial import cKDTree
    k = len(np.unique(by[name.split('_')[1]].astype(str)))
    rng = np.random.default_rng(VORONOI_SEED + k)
    seeds = rng.choice(len(cells), k, replace=False, p=cells.area_km2/cells.area_km2.sum())
    return cKDTree(xyz[seeds]).query(xyz)[1]             # nearest seed by chord length = by great-circle distance


def layouts(cells, tables, cache, area_name, names=LAYOUTS, g=None):
    """(dict name -> (label of each cell 0..K-1, domain names), share of the target nodes in a polygon) for the
    provinces of several tables (each with its continent from the file name)"""
    import pandas as pd
    from scipy.spatial import cKDTree
    g = g or geometry.grid()
    frames, h = [], hashlib.sha256()
    for path in tables:
        path = os.path.expanduser(path)
        with open(path, 'rb') as f:
            h.update(f.read())
        frames.append(pd.read_csv(path, keep_default_na=False).assign(continent=_continent(path)))
    t = pd.concat(frames, ignore_index=True)
    t = t[t.param != '']
    spec = dict(area=area_name, tables=h.hexdigest()[:16], source='hasterok2022', version=DOMAINS_VERSION)
    d = cache.entry('domain_nodes', spec)
    f = os.path.join(d, 'nodes.npz')
    if not os.path.exists(f):
        nodes, params = _domain_nodes(t, cells.cell >= 0, g)
        from .cache import save_atomic
        save_atomic(f, lambda p: np.savez(p, nodes=nodes, params=np.array(params)))
    z = np.load(f)
    nodes, params = z['nodes'], [str(p) for p in z['params']]
    sel = (cells.cell >= 0) & (nodes >= 0)
    w = np.cos(np.radians(np.broadcast_to(g.lat[:, None], nodes.shape)[sel]))
    A = np.zeros((len(cells), len(params)))
    np.add.at(A, (cells.cell[sel], nodes[sel]), w)
    inside = A.sum(1) > 0
    prov = np.where(inside, A.argmax(1), -1)
    first = t.drop_duplicates('param').set_index('param')
    cont_of = np.array([first.continent[p] for p in params])
    group_of = np.array([first.prov_group[p] or p for p in params])
    class_of = np.array([covariates.CLASS_NAMES[covariates.CLASS_OF_TYPE[first.prov_type[p]]] for p in params])
    xyz = graph.unit(cells.clon, cells.clat)
    near = np.flatnonzero(inside)[cKDTree(xyz[inside]).query(xyz[~inside])[1]]   # chord distance: great circle order
    cont = np.empty(len(cells), object)
    cont[inside] = cont_of[prov[inside]]
    cont[~inside] = cont_of[prov[near]]
    rest = np.array([f'rest {c}' for c in cont], object)
    by = {'continents': cont,
          'groups': np.where(inside, group_of[np.maximum(prov, 0)], rest),
          'provinces': np.where(inside, np.array(params, object)[np.maximum(prov, 0)], rest),
          'class': np.where(inside, class_of[np.maximum(prov, 0)], rest)}
    out = {}
    for name in names:
        if name == 'one':
            out[name] = (np.zeros(len(cells), int), ['all'])
        elif name in CONTROLS:
            lab = np.unique(_control(name, cells, xyz, by), return_inverse=True)[1]
            out[name] = (lab, [f'{name} {j}' for j in range(lab.max() + 1)])
        else:
            dom, lab = np.unique(by[name].astype(str), return_inverse=True)
            out[name] = (lab, [str(x) for x in dom])
    return out, float(sel.sum()/(cells.cell >= 0).sum())


def columns(groups):
    """design columns for a set of covariate groups (the class means replace the intercept)"""
    cols = [] if 'class' in groups else ['intercept']
    return cols + [c for g in GROUPS if g in groups for c in GROUPS[g]]


def design(cols, cov, area):
    """design matrix (cells x p), coefficient boxes (lo, hi) and names for a list of columns"""
    n = len(area)
    B, lo, hi, names = [], [], [], []

    def add(x, bounds, name):
        B.append(np.asarray(x, float)); lo.append(bounds[0]); hi.append(bounds[1]); names.append(name)
    for c in cols:
        if c == 'intercept':
            add(np.ones(n), LEVEL, 'intercept')
        elif c == 'class':
            for k, cname in enumerate(covariates.CLASS_NAMES):
                add(cov['cls'] == k, LEVEL, f'mean {cname}')
        elif c == 'undated':
            add(cov['undated'], SLOPE, 'undated')
        elif c == 'craton':
            add(np.isin(cov['group'], CRATON_GROUPS), SLOPE, 'craton')
        elif c.startswith('group:'):
            add(cov['group'] == c[6:], SLOPE, c)
        else:
            add(covariates.standardise(cov[c], area), SLOPE, c)
    return np.stack(B, 1), np.array(lo), np.array(hi), names


def quantities(cov, area):
    """the standard summaries c'v: the area mean, the mean of each Australian craton (NAC, WAC, SAC) and the contrast of
    the cratons with the rest (cells grouped by the province of their centre)"""
    def mean(members):
        w = area*members
        return w/np.sum(w)
    craton = np.isin(cov['group'], CRATON_GROUPS)
    out = {'area mean': mean(np.ones(len(area), bool))}
    out.update({g.split()[0][0] + 'AC mean': mean(cov['group'] == g) for g in CRATON_GROUPS if (cov['group'] == g).any()})
    if craton.any():
        out['craton - rest'] = mean(craton) - mean(~craton)
    return out
