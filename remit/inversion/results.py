"""
The result type (one for all methods): an xarray Dataset saved as NetCDF.

  per cell     vis, vis_sd (when the method gives one), truth (synthetic data); coordinates cell_id, lon, lat, area_km2
  per row      for coefficient fits: l, m, gh (0 = g, 1 = h), data, model, residual (nT, band rows)
  per point    for local fits: plon, plat, data, model, residual (Br, nT, background removed)
  attributes   method, mode, the resolved config (JSON), the fitted quantities (v0, damping, sigma, evidence...),
               the git commit, the creation time
A result is self-describing: its config reproduces it.
"""
import datetime
import json
import os
import subprocess

import numpy as np
import xarray as xr


def _commit():
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        out = subprocess.run(['git', '-C', root, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
        dirty = subprocess.run(['git', '-C', root, 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ('+dirty' if dirty else '')
    except OSError:
        return 'unknown'


def make(cfg, cells, cell_vars, row_vars=None, band=None, points=None, point_vars=None, **attrs):
    ds = xr.Dataset(coords=dict(cell=np.arange(len(cells)), cell_id=('cell', cells.ids), lon=('cell', cells.clon),
                                lat=('cell', cells.clat), area_km2=('cell', cells.area_km2)))
    for k, v in cell_vars.items():
        ds[k] = ('cell', np.asarray(v, float))
    if row_vars:
        ds = ds.assign_coords(row=np.arange(len(band)), l=('row', band.l), m=('row', band.m), gh=('row', band.gh))
        for k, v in row_vars.items():
            ds[k] = ('row', np.asarray(v, float))
    if point_vars:
        ds = ds.assign_coords(point=np.arange(len(points[0])), plon=('point', points[0]), plat=('point', points[1]))
        for k, v in point_vars.items():
            ds[k] = ('point', np.asarray(v, float))
    ds.attrs.update(method=cfg['method']['name'], mode=cfg['mode'], nside=cells.nside, config=json.dumps(cfg),
                    commit=_commit(), created=datetime.datetime.now().isoformat(timespec='seconds'))
    for k, v in attrs.items():
        ds.attrs[k] = v if isinstance(v, (str, int, float)) else json.dumps(v)
    return ds


def save(ds, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ds.to_netcdf(path)
    return path


def load(path):
    return xr.load_dataset(path)
