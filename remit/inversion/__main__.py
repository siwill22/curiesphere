"""
Run an inversion from its config:
    python -m remit.inversion run <config.toml> [--out result.nc] [--no-figures]
    python -m remit.inversion gallery [--open]
The result (NetCDF) goes to results/<config name>.nc next to the config unless --out is given; the standard figures go
to a folder per region beside it (results/global/, results/aus/, ...), and the gallery of all figures
(results/index.html, gallery.py) is rewritten. Single-threaded BLAS and low priority are the caller's choice (OMP/VECLIB_MAXIMUM_THREADS, nice).
"""
import argparse
import os

from . import config, results

METHODS = {'gcm': 'remit.inversion.methods.gcm', 'cm': 'remit.inversion.methods.cm', 'pd': 'remit.inversion.methods.pd',
           'cr': 'remit.inversion.methods.cr', 'bounds': 'remit.inversion.methods.bounds',
           'merge': 'remit.inversion.methods.merge',
           'voronoi': 'remit.inversion.methods.voronoi', 'lvoronoi': 'remit.inversion.methods.lvoronoi',
           'partition': 'remit.inversion.methods.partition',
           'df': 'remit.inversion.methods.df'}


def region(cfg):
    """the figure folder of a config: 'global' for all continental crust, else the outline's continent tag"""
    area = cfg['target']['area']
    if area == 'continental_crust':
        return 'global'
    stem = os.path.splitext(os.path.basename(area))[0]
    return {'australia': 'aus'}.get(stem, stem)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m remit.inversion')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='run an inversion config')
    r.add_argument('config')
    r.add_argument('--out')
    r.add_argument('--no-figures', action='store_true')
    g = sub.add_parser('gallery', help='rewrite results/index.html, the page of all figures')
    g.add_argument('--open', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'gallery':
        from . import gallery
        out = gallery.write()
        print('wrote', out)
        if a.open:
            import webbrowser
            webbrowser.open('file://' + out)
        return
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
        fig_dir = os.path.join(os.path.dirname(out), region(cfg))         # one folder of figures per region
        os.makedirs(fig_dir, exist_ok=True)
        for f in figures.standard(ds, os.path.join(fig_dir, os.path.splitext(os.path.basename(out))[0])):
            print('wrote', f)
        from . import gallery
        print('wrote', gallery.write())


if __name__ == '__main__':
    main()
