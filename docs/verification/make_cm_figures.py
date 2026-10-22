"""
Maps for the CM part of the verification document, from benchmarks/verification/results/cm_* (cm_checks.py).

  cm_c1_maps.png        CM against the independent solver at fixed settings
  cm_c2_spread.png      CM's errors in units of its reported s.d.: settings at the truth, fitted, and fitted with
                        errors from the error model
  cm_c2_maps.png        three of the truths: the truth, CM's maps and s.d. with both kinds of errors
  cm_shrink_contrast.png  the recovered contrast of shrinking checkerboards, CM beside DF
  cm_shrink_maps.png    CM's maps of the checkerboards at four sizes

Run from the repository root:
    conda run -n curiesphere python docs/verification/make_cm_figures.py
"""
import json

import numpy as np
import pygmt

from make_df_figures import BASE, OUT, REGION, RES, W, cell_grid, cells_map


def c1_maps():
    z = np.load(RES / 'cm_c1.npz')
    lon, lat = z['clon'], z['clat']
    d = z['v_cm'] - z['v_ref']
    r = z['sd_cm']/z['sd_ref'] - 1
    lim = lambda a: float(f'{np.abs(a).max():.0e}') or 1e-15
    panels = [(z['v_true'], 'batlow', [0, 1.6], 'The truth (from CM\'s model)', None, 'VIS (SI km)'),
              (z['v_cm'], 'batlow', [0, 1.6], 'CM', None, 'VIS (SI km)'),
              (z['v_ref'], 'batlow', [0, 1.6], 'Independent solver', None, 'VIS (SI km)'),
              (d, 'vik', [-lim(d), lim(d)], 'CM minus independent', f'max {np.abs(d).max():.1e} SI km', 'SI km'),
              (z['sd_cm'], 'lajolla', [0, 0.3], 'CM: s.d.', None, 'SI km'),
              (r, 'vik', [-lim(r), lim(r)], 'CM s.d. / independent s.d. - 1', f'max {np.abs(r).max():.1e}',
               'ratio - 1')]
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='7.5p', FONT_SUBTITLE='6.5p', FONT_ANNOT_PRIMARY='5.5p', FONT_LABEL='6p',
                 MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE='plain')
    for j, (v, cpt, series, title, sub, label) in enumerate(panels):
        with fig.shift_origin(xshift=f'{(j % 3)*(W + 0.5)}c', yshift=f'{BASE + (1 - j//3)*(0.9*W + 1.8)}c'):
            pygmt.makecpt(cmap=cpt, series=series, background=True)
            fig.basemap(region=REGION, projection=f'M{W}c', frame=['af', f'+t{title}' + (f'+s{sub}' if sub else '')])
            fig.grdimage(cell_grid(lon, lat, v), cmap=True, nan_transparent=True)
            fig.coast(shorelines='0.2p,gray30')
            fig.colorbar(position=f'JBC+w{0.8*W}c/0.13c+o0/0.3c+h+e', frame=[f'x+l{label}'])
    fig.savefig(OUT / 'cm_c1_maps.png', dpi=170)
    print('cm_c1_maps.png')


def c2_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z = np.load(RES / 'cm_c2.npz')
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True, sharey=True)
    x = np.linspace(-4.5, 4.5, 400)
    bell = np.exp(-x**2/2)/np.sqrt(2*np.pi)
    for ax, (key, what) in zip(axs, [('z_fixed', 'settings at the truth'), ('z_fit', 'settings fitted'),
                                     ('z_real', 'errors from the error model')]):
        v = z[key].ravel()
        ax.hist(v, bins=np.linspace(-4.5, 4.5, 61), density=True, color='0.7', label='all cells of all truths')
        ax.plot(x, bell, 'k', lw=1.2, label='what it should be')
        ax.set_title(f'CM, {what}\nwithin 1 s.d.: {np.mean(np.abs(v) < 1):.0%} (should be 68%), '
                     f'within 2 s.d.: {np.mean(np.abs(v) < 2):.0%} (95%)', fontsize=8.5, loc='left')
        ax.set_xlabel('(CM map minus truth) / reported s.d.')
    axs[0].legend(fontsize=7.5, frameon=False)
    fig.savefig(OUT / 'cm_c2_spread.png', dpi=170)
    print('cm_c2_spread.png')

    lon, lat = z['clon'], z['clat']
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='7.5p', FONT_ANNOT_PRIMARY='5.5p', FONT_LABEL='6p', MAP_TITLE_OFFSET='3p',
                 MAP_FRAME_TYPE='plain')
    n = len(z['v_true'])
    for k in range(n):
        vt = z['v_true'][k]
        panels = [(vt, 'batlow', [0, 1.6], f'Truth {k + 1}', 'VIS (SI km)'),
                  (z['v_fit'][k], 'batlow', [0, 1.6], 'CM, errors as assumed', 'VIS (SI km)'),
                  (z['v_real'][k], 'batlow', [0, 1.6], 'CM, error-model errors', 'VIS (SI km)'),
                  (z['sd_fit'][k], 'lajolla', [0, 0.4], 's.d., errors as assumed', 'SI km'),
                  (z['sd_real'][k], 'lajolla', [0, 0.4], 's.d., error-model errors', 'SI km'),
                  ((z['v_real'][k] - vt)/z['sd_real'][k], 'vik', [-3, 3], 'Error / s.d., error model',
                   'number of s.d.')]
        for j, (v, cpt, series, title, label) in enumerate(panels):
            with fig.shift_origin(xshift=f'{j*(W + 0.4)}c', yshift=f'{BASE + (n - 1 - k)*(0.9*W + 1.6)}c'):
                cells_map(fig, lon, lat, v, cpt, series, title, label)
    fig.savefig(OUT / 'cm_c2_maps.png', dpi=170)
    print('cm_c2_maps.png')


def shrink_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = json.load(open(RES / 'cm_shrink.json'))
    df = [r for r in json.load(open(RES / 'df_shrink.json')) if 'failed' not in r]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    for noise, colour, name in (('none', '#2a2a2a', 'CM, no noise'), ('noise', '#1b6ca8', 'CM, noise (as DF)'),
                                ('noise and model error', '#d1495b', 'CM, noise and model error (as real data)')):
        sel = sorted((r for r in rows if r['noise'] == noise), key=lambda r: r['size'])
        ax.plot([r['size']*111 for r in sel], [r['contrast'] for r in sel], '-', color=colour, marker='o', ms=3.5,
                label=name)
    sel = sorted((r for r in df if r['layout'] == 'one' and r['noise']), key=lambda r: r['size'])
    ax.plot([r['size']*111 for r in sel], [r['contrast'] for r in sel], '--', color='0.55', marker='s', ms=3,
            label='DF, one domain, noise')
    ax.axhline(1, color='0.6', lw=0.8)
    ax.set_xscale('log')
    ax.minorticks_off()
    ax.set_xticks([170, 250, 330, 450, 550, 670, 900])
    ax.set_xticklabels(['170', '250', '330', '450', '550', '670', '900'])
    ax.set_xlabel('checkerboard square size (km, north-south)')
    ax.set_ylabel('recovered contrast / true contrast')
    ax.set_title('How much of the checkerboard CM recovers', loc='left', fontsize=9.5)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(OUT / 'cm_shrink_contrast.png', dpi=170)
    print('cm_shrink_contrast.png')

    z = np.load(RES / 'cm_shrink.npz')
    lon, lat = z['clon'], z['clat']
    show = [8., 4., 2.5, 1.5]
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='7.5p', FONT_SUBTITLE='6.5p', FONT_ANNOT_PRIMARY='5.5p', FONT_LABEL='6p',
                 MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE='plain')
    for k, size in enumerate(show):
        i, j = np.floor(lat/size).astype(int), np.floor(np.mod(lon, 360)/size).astype(int)
        panels = [(np.where((i + j) % 2 == 0, 1.5, 0.5), f'Truth: {size:g}° squares', None)]
        for noise, name in (('none', 'No noise'), ('noise', 'Noise (as DF)'),
                            ('noise and model error', 'Noise and model error')):
            r = next(r for r in rows if r['size'] == size and r['noise'] == noise)
            panels.append((z[f'{size:g}_{noise}'][:, 0], f'CM, {name[0].lower() + name[1:]}',
                           f'contrast {r["contrast"]:.2f}'))
        for jj, (v, title, sub) in enumerate(panels):
            with fig.shift_origin(xshift=f'{jj*(W + 0.5)}c', yshift=f'{BASE + (len(show) - 1 - k)*(0.9*W + 1.8)}c'):
                pygmt.makecpt(cmap='batlow', series=[0, 2], background=True)
                fig.basemap(region=REGION, projection=f'M{W}c',
                            frame=['af', f'+t{title}' + (f'+s{sub}' if sub else '')])
                fig.grdimage(cell_grid(lon, lat, v), cmap=True, nan_transparent=True)
                fig.coast(shorelines='0.2p,gray30')
                fig.colorbar(position=f'JBC+w{0.8*W}c/0.13c+o0/0.3c+h+e', frame=['x+lVIS (SI km)'])
    fig.savefig(OUT / 'cm_shrink_maps.png', dpi=170)
    print('cm_shrink_maps.png')


if __name__ == '__main__':
    if (RES / 'cm_c1.npz').exists():
        c1_maps()
    if (RES / 'cm_c2.npz').exists():
        c2_figures()
    if (RES / 'cm_shrink.json').exists():
        shrink_figures()
