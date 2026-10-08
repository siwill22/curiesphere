"""
Truths for synthetic data (CONTEXT.md): cell VIS (SI.km) from a spec.
  {'kind': 'uniform', 'value': 1.0}
  {'kind': 'crust_thickness', 'chi': 0.04, 'grid': path}   chi x crustal thickness (km) at the cell centres
                                                            (a lon/lat grid such as CRUST1.0, nearest node)
"""
import os

import numpy as np
import xarray as xr


def truth(spec, cells):
    if spec['kind'] == 'uniform':
        return np.full(len(cells), float(spec['value']))
    if spec['kind'] == 'crust_thickness':
        h = xr.open_dataarray(os.path.expanduser(spec['grid']))
        lon = h['lon'] if 'lon' in h.coords else h['x']
        lon180 = float(lon.min()) < 0
        x = ((cells.clon + 180) % 360) - 180 if lon180 else cells.clon % 360
        names = ('lon', 'lat') if 'lon' in h.coords else ('x', 'y')
        v = h.sel({names[0]: xr.DataArray(x), names[1]: xr.DataArray(cells.clat)}, method='nearest').values
        if not np.isfinite(v).all():
            raise ValueError('crust thickness grid has gaps at some cells')
        return spec['chi']*v
    raise ValueError(f'truth {spec["kind"]!r}')
