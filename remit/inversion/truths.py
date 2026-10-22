"""
Truths for synthetic data (CONTEXT.md): cell VIS (SI.km) from a spec.
  {'kind': 'uniform', 'value': 1.0}
  {'kind': 'crust_thickness', 'chi': 0.04, 'grid': path}   chi x crustal thickness (km) at the cell centres
                                                            (a lon/lat grid such as CRUST1.0, nearest node)
  {'kind': 'layout_levels', 'layout': 'groups', 'mean': 0.7, 'sd': 0.2, 'seed': 1}
                                                            one level per domain of a layout (parameterise.layouts,
                                                            passed in by the method), drawn from N(mean, sd^2)
  {'kind': 'layout_fractal', 'layout': 'groups', 'mean': 0.7, 'sd': 0.2, 's': 0.4, 'beta': 2.5, 'ell_km': 500,
   'seed': 1}                                               a level per domain from N(mean, sd^2) plus DF's fractal
                                                            pattern of strength s inside each domain, drawn afresh in
                                                            each (breaks at the domain boundaries; check S3)
  {'kind': 'random_field', 'mean': 0.75, 'sd': 0.4, 'beta': 2.5, 'ell_km': 500, 'seed': 1}
                                                            a Gaussian random field on the sphere, blind to geology,
                                                            with DF's spectrum (l(l+1) + (R/ell)^2)^(-beta/2) to degree
                                                            400, at the cell centres, scaled to mean and sd over them
"""
import os

import numpy as np
import xarray as xr


def truth(spec, cells, layouts=None):
    if spec['kind'] == 'layout_fractal':
        from . import graph
        from .methods import df
        label, names = layouts[spec['layout']]
        rng = np.random.default_rng(spec['seed'])
        v = rng.normal(spec['mean'], spec['sd'], len(names))[label]
        xyz = graph.unit(cells.clon, cells.clat)
        window = df.cap_window(4*np.pi*df.R_KM**2/(12*cells.nside**2))
        for k in range(len(names)):
            b = np.flatnonzero(label == k)
            psi = np.arccos(np.clip(xyz[b] @ xyz[b].T, -1., 1.))
            R = df.correlation(spec['beta'], spec['ell_km'], psi.ravel(), window).reshape(psi.shape)
            L = np.linalg.cholesky(R + 1e-8*np.eye(len(b)))
            v[b] += spec['s']*(L @ rng.standard_normal(len(b)))
        return v
    if spec['kind'] == 'random_field':
        import pyshtools
        lmax, R = 400, 6371.2
        l = np.arange(lmax + 1)
        S = (l*(l + 1) + (R/spec['ell_km'])**2)**(-spec['beta']/2)
        S[0] = 0.
        c = pyshtools.SHCoeffs.from_random(S, seed=spec['seed'])
        v = c.expand(lat=cells.clat, lon=cells.clon)
        return spec['mean'] + spec['sd']*(v - v.mean())/v.std()
    if spec['kind'] == 'layout_levels':
        label, names = layouts[spec['layout']]
        levels = np.random.default_rng(spec['seed']).normal(spec['mean'], spec['sd'], len(names))
        return levels[label]
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
