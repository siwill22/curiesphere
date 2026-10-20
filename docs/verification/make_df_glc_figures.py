"""
Global maps (Robinson) for the DF-GLC part of the verification document, from benchmarks/verification/results/df_glc_*
(df_glc_checks.py).

  df_glc_d1_maps.png         DF-GLC against the independent solver, at fixed settings
  df_glc_d2_spread.png       DF-GLC's errors in units of its reported s.d.: settings at the truth, fitted at the true
                             shape, and fitted with the shape searched
  df_glc_shrink_maps.png     checkerboards of equal-area tiles: the truth, DF-GLC (one domain) without and with noise
  df_glc_shrink_contrast.png the recovered contrast against the tile size, DF-GLC beside GCM (same tiles, same noise)

Run from the repository root:
    conda run -n curiesphere python docs/verification/make_df_glc_figures.py
"""
import json

import numpy as np

from make_gcm_figures import OUT, RES, gmap, new_figure, place


def d1_maps():
    z = np.load(RES / 'df_glc_d1.npz')
    lon, lat = z['clon'], z['clat']
    d = z['v'] - z['v_ref']
    r = z['sd']/z['sd_ref'] - 1
    lim = lambda a: float(f'{max(np.abs(a).max(), 1e-16):.0e}')
    panels = [(z['truth'], 'batlow', [0, 2], 'The truth (from DF-GLC\'s model)', None, 'VIS (SI km)'),
              (z['v'], 'batlow', [0, 2], 'DF-GLC', None, 'VIS (SI km)'),
              (d, 'vik', [-lim(d), lim(d)], 'DF-GLC minus independent solver', f'max {np.abs(d).max():.1e} SI km',
               'SI km'),
              (r, 'vik', [-lim(r), lim(r)], 'DF-GLC s.d. / independent s.d. - 1', f'max {np.abs(r).max():.1e}',
               'ratio - 1')]
    fig = new_figure()
    for j, (v, cpt, series, title, sub, label) in enumerate(panels):
        with place(fig, j//2, j % 2, 2):
            gmap(fig, lon, lat, v, cpt, series, title, sub, label)
    fig.savefig(OUT / 'df_glc_d1_maps.png', dpi=170)
    print('df_glc_d1_maps.png')


def d2_spread():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z = np.load(RES / 'df_glc_d2.npz')
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True, sharey=True)
    x = np.linspace(-4.5, 4.5, 400)
    bell = np.exp(-x**2/2)/np.sqrt(2*np.pi)
    for ax, (key, what) in zip(axs, [('z_fixed', 'settings at the truth'),
                                     ('z_fit_shape', 'fitted, shape at the truth'),
                                     ('z_fit', 'fitted, shape searched (as run)')]):
        v = z[key].ravel()
        ax.hist(v, bins=np.linspace(-4.5, 4.5, 61), density=True, color='0.7', label='all cells of all truths')
        ax.plot(x, bell, 'k', lw=1.2, label='what it should be')
        ax.set_title(f'DF-GLC, {what}\nwithin 1 s.d.: {np.mean(np.abs(v) < 1):.0%} (should be 68%), '
                     f'within 2 s.d.: {np.mean(np.abs(v) < 2):.0%} (95%)', fontsize=8.5, loc='left')
        ax.set_xlabel('(DF-GLC map minus truth) / reported s.d.')
    axs[0].legend(fontsize=7.5, frameon=False)
    fig.savefig(OUT / 'df_glc_d2_spread.png', dpi=170)
    print('df_glc_d2_spread.png')


def shrink_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = json.load(open(RES / 'df_glc_shrink.json'))
    z = np.load(RES / 'df_glc_shrink.npz')
    zg = np.load(RES / 'gcm_shrink.npz')
    lon, lat = z['clon'], z['clat']
    nsides = sorted({r['nside'] for r in rows})
    fig = new_figure()
    for k, ns in enumerate(nsides):
        sel = {r['noise']: r for r in rows if r['nside'] == ns}
        panels = [(zg[f'{ns}_truth'], f'Truth: tiles of {sel["none"]["size_km"]:.0f} km', None)]
        for n in ('none', 'noise'):
            r = sel[n]
            panels.append((z[f'{ns}_{n}'][:, 0], f'DF-GLC, {"no noise" if n == "none" else "noise"}',
                           f'contrast {r["contrast"]:.2f}; beta {r["beta"]:g}, ell {r["ell_km"]:g} km'))
        for j, (v, title, sub) in enumerate(panels):
            with place(fig, k, j, len(nsides)):
                gmap(fig, lon, lat, v, 'batlow', [0, 2], title, sub, 'VIS (SI km)')
    fig.savefig(OUT / 'df_glc_shrink_maps.png', dpi=150)
    print('df_glc_shrink_maps.png')

    g = json.load(open(RES / 'gcm_shrink.json'))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    for src, n, colour, style, name in ((rows, 'none', '#2a2a2a', '-o', 'DF-GLC, no noise'),
                                        (rows, 'noise', '#1b6ca8', '-o', 'DF-GLC, noise'),
                                        (g, 'none', '0.55', '--s', 'GCM, no noise'),
                                        (g, 'noise', '#d1495b', '--s', 'GCM, noise')):
        sel = sorted((r for r in src if r['noise'] == n), key=lambda r: r['size_km'])
        ax.plot([r['size_km'] for r in sel], [r['contrast'] for r in sel], style, color=colour, ms=3.5, label=name)
    ax.axhline(1, color='0.6', lw=0.8)
    ax.set_xscale('log')
    ax.minorticks_off()
    ax.set_xticks([204, 407, 815, 1630])
    ax.set_xticklabels(['204', '407', '815', '1630'])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('checkerboard tile size (km, equal-area tiles)')
    ax.set_ylabel('recovered contrast / true contrast')
    ax.set_title('DF-GLC and GCM on the same truths and noise', loc='left', fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(OUT / 'df_glc_shrink_contrast.png', dpi=170)
    print('df_glc_shrink_contrast.png')


if __name__ == '__main__':
    if (RES / 'df_glc_d1.npz').exists():
        d1_maps()
    if (RES / 'df_glc_d2.npz').exists():
        d2_spread()
    if (RES / 'df_glc_shrink.json').exists():
        shrink_figures()
