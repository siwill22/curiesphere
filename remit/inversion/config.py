"""
Inversion configs: one TOML file fully specifies an inversion. A file may name a parent with `inherit`
(a path relative to the file); the child's values override the parent's, tables merging key by key and
lists being replaced. Defaults come from DEFAULTS and from the defaults of the config's mode (ADR 0006);
values given in the files always win. The resolved config is what results carry.
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


def _read(path):
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, 'rb') as f:
        cfg = tomllib.load(f)
    parent = cfg.pop('inherit', None)
    if parent is None:
        return cfg
    return merge(_read(os.path.join(os.path.dirname(path), parent)), cfg)


def resolve(cfg):
    """the config with all defaults filled in: DEFAULTS, then the mode's defaults, then cfg"""
    mode = cfg.get('mode', DEFAULTS['mode'])
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}, not {mode!r}')
    return merge(merge(DEFAULTS, MODE_DEFAULTS[mode]), cfg)


def load(path):
    """the resolved config of a TOML file (with its inherit chain)"""
    out = resolve(_read(path))
    out['source'] = os.path.abspath(os.path.expanduser(path))
    return out


def spec_hash(spec):
    """stable short hash of a JSON-like spec (key order does not matter)"""
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]
