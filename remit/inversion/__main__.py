"""
Run an inversion from its config:
    python -m remit.inversion run <config.toml> [--out result.nc] [--no-figures]
The result (NetCDF) goes to results/<config name>.nc next to the config unless --out is given, with the standard
figures beside it. Single-threaded BLAS and low priority are the caller's choice (OMP/VECLIB_MAXIMUM_THREADS, nice).
"""
import argparse
import os

from . import config, results

METHODS = {'gcm': 'remit.inversion.methods.gcm', 'cm': 'remit.inversion.methods.cm', 'pd': 'remit.inversion.methods.pd',
           'cr': 'remit.inversion.methods.cr', 'bounds': 'remit.inversion.methods.bounds'}


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m remit.inversion')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='run an inversion config')
    r.add_argument('config')
    r.add_argument('--out')
    r.add_argument('--no-figures', action='store_true')
    a = ap.parse_args(argv)
    cfg = config.load(a.config)
    name = cfg['method']['name']
    if name not in METHODS:
        raise SystemExit(f'unknown method {name!r}; known: {sorted(METHODS)}')
    import importlib
    ds = importlib.import_module(METHODS[name]).run(cfg)
    stem = os.path.splitext(os.path.basename(a.config))[0]
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.config)), 'results', f'{stem}.nc')
    print('wrote', results.save(ds, out))
    if not a.no_figures:
        from . import figures
        for f in figures.standard(ds, os.path.splitext(out)[0]):
            print('wrote', f)


if __name__ == '__main__':
    main()
