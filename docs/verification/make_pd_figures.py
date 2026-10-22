"""
Maps for the PD part of the verification document, from benchmarks/verification/results/pd_* (pd_checks.py).

  pd_p1_maps.png       a truth with one level per domain, PD's map without noise, and their difference
  pd_p2_coverage.png   how often the domain means fall within their reported s.d., for three kinds of data and three
                       kinds of s.d.
  pd_p2_maps.png       one truth with a pattern inside the domains, and per domain over all truths: the RMS error,
                       PD's s.d. from refits, and their ratio
  pd_p2_area.png       the RMS error and the s.d. of each domain against its area

Run from the repository root:
    conda run -n curiesphere python docs/verification/make_pd_figures.py
"""
import json

import numpy as np
import pygmt

from make_df_figures import BASE, OUT, REGION, RES, W, boundaries, cell_grid


def panel(fig, lon, lat, v, cpt, series, title, sub, label, edges=None):
    pygmt.makecpt(cmap=cpt, series=series, background=True)
    fig.basemap(region=REGION, projection=f'M{W}c', frame=['af', f'+t{title}' + (f'+s{sub}' if sub else '')])
    fig.grdimage(cell_grid(lon, lat, v), cmap=True, nan_transparent=True)
    if edges is not None and len(edges[0]):
        fig.plot(x=edges[0], y=edges[1], style='p0.02c', fill='white')
    fig.coast(shorelines='0.2p,gray30')
    fig.colorbar(position=f'JBC+w{0.8*W}c/0.13c+o0/0.3c+h+e', frame=[f'x+l{label}'])


def config(fig):
    pygmt.config(FONT_TITLE='7.5p', FONT_SUBTITLE='6.5p', FONT_ANNOT_PRIMARY='5.5p', FONT_LABEL='6p',
                 MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE='plain')


def p1_maps():
    z = np.load(RES / 'pd_p1.npz')
    lon, lat = z['clon'], z['clat']
    e = boundaries(lon, lat, z['label'])
    d = z['v'] - z['truth']
    fig = pygmt.Figure()
    config(fig)
    for j, (v, cpt, series, title, sub, label) in enumerate([
            (z['truth'], 'batlow', [0, 1.6], 'Truth: one level per domain', None, 'VIS (SI km)'),
            (z['v'], 'batlow', [0, 1.6], 'PD, no noise', None, 'VIS (SI km)'),
            (d, 'vik', [-2e-14, 2e-14], 'PD minus truth', f'max {np.abs(d).max():.1e} SI km', 'SI km')]):
        with fig.shift_origin(xshift=f'{j*(W + 0.5)}c', yshift=f'{BASE}c'):
            panel(fig, lon, lat, v, cpt, series, title, sub, label, e)
    fig.savefig(OUT / 'pd_p1_maps.png', dpi=170)
    print('pd_p1_maps.png')


def p2_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = json.load(open(RES / 'pd_p2.json'))
    z = np.load(RES / 'pd_p2.npz')
    cases = [('white', 'errors as PD assumes'), ('noise', 'field-model noise'),
             ('pattern', 'noise and a pattern\ninside the domains')]
    kinds = [('z_formal', 'formal s.d.', '#9aa5b1'), ('z_scaled', 'formal s.d. scaled by the misfit', '#1b6ca8'),
             ('z_refit', 's.d. from refits (final mode)', '#d1495b')]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
    x = np.arange(len(cases))
    for j, (k, name, colour) in enumerate(kinds):
        vals = [s[c][k]['within_1sd'] for c, _ in cases]
        bars = ax.bar(x + (j - 1)*0.26, vals, 0.24, color=colour, label=name)
        ax.bar_label(bars, [f'{v:.0%}' for v in vals], fontsize=7, padding=1)
    ax.axhline(0.683, color='k', lw=0.8, ls='--')
    ax.text(len(cases) - 0.45, 0.70, 'should be 68%', fontsize=7.5, ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in cases], fontsize=8)
    ax.set_ylabel('domain means within 1 s.d.')
    ax.set_ylim(0, 1.05)
    ax.set_title('How often PD\'s s.d. covers the error of a domain mean (200 truths, 39 domains)', loc='left',
                 fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc='upper right')
    fig.savefig(OUT / 'pd_p2_coverage.png', dpi=170)
    print('pd_p2_coverage.png')

    lon, lat, label = z['clon'], z['clat'], z['label']
    e = boundaries(lon, lat, label)
    rms = np.sqrt(np.mean(z['pattern_err']**2, 0))
    sdr = np.median(z['pattern_sd_refit'], 0)
    fig = pygmt.Figure()
    config(fig)
    top = [(z['pattern_map_truth'], 'batlow', [0, 1.6], 'Truth: levels and a pattern', None, 'VIS (SI km)'),
           (z['pattern_map_target'], 'batlow', [0, 1.6], 'Its domain means', None, 'VIS (SI km)'),
           (z['pattern_map_v'], 'batlow', [0, 1.6], 'PD', None, 'VIS (SI km)'),
           (z['pattern_map_v'] - z['pattern_map_target'], 'vik', [-0.5, 0.5], 'PD minus domain means', None,
            'SI km')]
    bottom = [(rms[label], 'lajolla', [0, 0.5], 'RMS error, 200 truths', None, 'SI km'),
              (sdr[label], 'lajolla', [0, 0.5], 'PD s.d. (refits)', None, 'SI km'),
              ((rms/sdr)[label], 'lajolla', [1, 12], 'RMS error / PD s.d.', None, 'ratio')]
    for row, panels in enumerate((top, bottom)):
        for j, (v, cpt, series, title, sub, lab) in enumerate(panels):
            with fig.shift_origin(xshift=f'{j*(W + 0.5)}c', yshift=f'{BASE + (1 - row)*(0.9*W + 1.8)}c'):
                panel(fig, lon, lat, v, cpt, series, title, sub, lab, e)
    fig.savefig(OUT / 'pd_p2_maps.png', dpi=170)
    print('pd_p2_maps.png')

    area = z['area']
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    for case, colour, name in (('noise', '#1b6ca8', 'RMS error, noise only'),
                               ('pattern', '#d1495b', 'RMS error, noise and a pattern inside')):
        ax.scatter(area, np.sqrt(np.mean(z[f'{case}_err']**2, 0)), s=14, color=colour, label=name)
    ax.scatter(area, np.median(z['pattern_sd_refit'], 0), s=14, marker='x', color='0.3', label='PD s.d. (refits)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('domain area (km²)')
    ax.set_ylabel('SI km')
    ax.set_title('Error of each domain mean against the size of the domain', loc='left', fontsize=9.5)
    ax.legend(fontsize=7.5, frameon=False)
    fig.savefig(OUT / 'pd_p2_area.png', dpi=170)
    print('pd_p2_area.png')


if __name__ == '__main__':
    if (RES / 'pd_p1.npz').exists():
        p1_maps()
    if (RES / 'pd_p2.npz').exists():
        p2_figures()
