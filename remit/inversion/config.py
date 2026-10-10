"""
Inversion configs: one TOML file fully specifies an inversion. A file may name a parent with `inherit`
(a path relative to the file); the child's values override the parent's, tables merging key by key and
lists being replaced. Defaults come from DEFAULTS and from the defaults of the config's mode (ADR 0006);
values given in the files always win. The resolved config is what results carry.

File paths in a config (an outline as the target area, a domain table, hypothesis files) are relative to the file that sets them, and are made
absolute when it is read.
"""
import copy
import hashlib
import json
import os
import tomllib

MODES = ('explore', 'final')

DEFAULTS = {
    'mode': 'final',
    'target': {'area': 'continental_crust', 'nside': 64},
    'geometry': {'layers': [{'top': 'basement', 'base': 30.0}],          # base: km below the ellipsoid
                 'direction': 'induced_igrf13', 'lmax': 185},
}

MODE_DEFAULTS = {
    'explore': {'compute': {'columns_backend': 'mps', 'products_backend': 'mps', 'solver': 'cholesky', 'coarsen': 1}},
    'final': {'compute': {'columns_backend': 'numpy', 'products_backend': 'numpy', 'solver': 'eigh', 'coarsen': 0}},
}


def merge(base, over):
    """deep merge: tables merge key by key, anything else in `over` replaces the value in `base`"""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


PATH_KEYS = [('target', 'area'), ('parameterisation', 'table')]
NESTED_PATH_KEYS = [('fit', 'background', 'outside'), ('method', 'hypotheses', 'hand'), ('method', 'hypotheses', 'labels'), ('method', 'layouts', 'merge')]
LIST_PATH_KEYS = [('method', 'start')]          # [path, ...]: the first item is a path


def _absolute_paths(cfg, base):
    for table, key in PATH_KEYS:
        v = cfg.get(table, {}).get(key)
        if isinstance(v, str) and os.path.splitext(v)[1] in ('.gpkg', '.geojson', '.json', '.csv'):
            cfg[table][key] = os.path.normpath(os.path.join(base, os.path.expanduser(v)))
    for table, key in LIST_PATH_KEYS:
        v = cfg.get(table, {}).get(key)
        if isinstance(v, list) and v and isinstance(v[0], str):
            v[0] = os.path.normpath(os.path.join(base, os.path.expanduser(v[0])))
    for table, sub, key in NESTED_PATH_KEYS:
        t = cfg.get(table, {}).get(sub)
        if isinstance(t, dict) and isinstance(t.get(key), str):
            t[key] = os.path.normpath(os.path.join(base, os.path.expanduser(t[key])))


def _read(path):
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, 'rb') as f:
        cfg = tomllib.load(f)
    _absolute_paths(cfg, os.path.dirname(path))
    parent = cfg.pop('inherit', None)
    if parent is None:
        return cfg
    return merge(_read(os.path.join(os.path.dirname(path), parent)), cfg)


# methods whose quick setting keeps the 100 km cells (ADR 0006, amendment): for domain methods coarser cells change the
# domains themselves, so explore saves time with fewer repeat runs instead
METHOD_MODE_DEFAULTS = {
    'pd': {'explore': {'compute': {'coarsen': 0}, 'method': {'noise_draws': 30}},
           'final': {'method': {'noise_draws': 100}}},
    'merge': {'explore': {'compute': {'coarsen': 0}, 'method': {'null_runs': 10, 'bootstrap_runs': 10}},
              'final': {'method': {'null_runs': 20, 'bootstrap_runs': 30}}},
    'voronoi': {'explore': {'compute': {'coarsen': 0}, 'method': {'sweeps': 500}}, 'final': {'method': {'sweeps': 2500}}},
    'lvoronoi': {'explore': {'compute': {'coarsen': 0}, 'method': {'sweeps': 500}}, 'final': {'method': {'sweeps': 2500}}},
    'partition': {'explore': {'compute': {'coarsen': 0}, 'method': {'sweeps': 400}}, 'final': {'method': {'sweeps': 2000}}},
    'df': {'explore': {'compute': {'coarsen': 0}, 'method': {'hyper_sd': False}}, 'final': {'method': {'hyper_sd': True}}},
}


def resolve(cfg):
    """the config with all defaults filled in: DEFAULTS, then the mode's defaults, then the method's defaults for the
    mode, then cfg"""
    mode = cfg.get('mode', DEFAULTS['mode'])
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}, not {mode!r}')
    method = cfg.get('method', {}).get('name')
    out = merge(DEFAULTS, MODE_DEFAULTS[mode])
    out = merge(out, METHOD_MODE_DEFAULTS.get(method, {}).get(mode, {}))
    return merge(out, cfg)


def load(path):
    """the resolved config of a TOML file (with its inherit chain)"""
    out = resolve(_read(path))
    out['source'] = os.path.abspath(os.path.expanduser(path))
    return out


def spec_hash(spec):
    """stable short hash of a JSON-like spec (key order does not matter)"""
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]
