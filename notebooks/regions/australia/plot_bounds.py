"""Charts and maps of run_bounds.py for Australia.
  bounds.png                 per case: the interval of each quantity with positivity only and with
                             roughness limits (k x the L2 map), the GP posterior +-2 sd, the truth
  bounds_extremal_real.png   the two extremal VIS maps on the real data (WAC mean and craton - rest,
                             positivity only and roughness <= 1 x L2)
Run from notebooks/: PYTHONPATH=..:$PYTHONPATH python regions/australia/plot_bounds.py"""
import numpy as np
import pygmt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from domains import problem
from domains.plotting import cell_grid, map_panel

OUT = 'regions/australia'
pb = problem.load(f'{OUT}/config.toml')
cells = pb['cells']
b = dict(np.load(f'{pb["cache"]}/bounds.npz'))
lm = dict(np.load(f'{pb["cache"]}/linear_models.npz'))
cases, quants = [str(c) for c in b['cases']], [str(q) for q in b['quantities']]
settings = ['pos'] + [f'r{k:g}' for k in b['rough']]
label = {'pos': 'positivity only', **{f'r{k:g}': f'roughness ≤ {k:g}× L2' for k in b['rough']}}
SEQ = ['#b7d3f3', '#77aee9', '#3a86d9', '#1b5aa6']       # one hue, light -> dark: weaker -> stronger smoothness
INK, MUTED = '#0b0b0b', '#52514e'

fig, axes = plt.subplots(1, len(cases), figsize=(3.6*len(cases), 3.6), sharey=True, squeeze=False)
for j, case in enumerate(cases):
    ax = axes[0, j]
    names = [str(x) for x in lm[f'{case}_gp_fun_names']]
    for qi, q in enumerate(quants):
        for si, st in enumerate(settings):
            key = f'{case}_{q}_{st}'
            if f'{key}_min' not in b:
                continue
            y = qi + (si - (len(settings) - 1)/2)*0.15
            ax.plot([b[f'{key}_min'], b[f'{key}_max']], [y, y], lw=3.5, color=SEQ[si], solid_capstyle='round',
                    label=label[st] if qi == 0 else None)
        m, sd = lm[f'{case}_gp_fun'][names.index(q)]
        ax.errorbar(m, qi + 0.42, xerr=2*sd, fmt='o', ms=3, color=INK, lw=1, label='GP ±2 sd' if qi == 0 else None)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(quants)), quants)
    ax.set_title(case, fontsize=9)
    ax.set_xlabel('VIS (SI.km)')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
axes[0, 0].invert_yaxis()
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, frameon=False, loc='lower center', ncol=len(l))
fig.suptitle('Every value in a bar fits the data; bars shrink as smoother models are required', fontsize=9)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(f'{OUT}/bounds.png', dpi=150)
print('wrote', f'{OUT}/bounds.png')

if 'real' in cases:
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='8p', FONT_ANNOT_PRIMARY='6p', FONT_LABEL='7p', MAP_TITLE_OFFSET='2p')
    panels = [(q, st, e) for q in ('WAC mean', 'craton - rest') for st in ('pos', f'r{min(b["rough"]):g}') for e in ('min', 'max')]
    for k, (q, st, e) in enumerate(panels):
        if k:
            fig.shift_origin(xshift='6.6c') if k % 4 else fig.shift_origin(xshift='-19.8c', yshift='-7.8c')
        key = f'real_{q}_{st}'
        map_panel(fig, cell_grid(b[f'{key}_v{e}'], cells), f'{q}, {label[st]}: {e} {float(b[f"{key}_{e}"]):.2f}',
                  'viridis', [0, 4, 0.05], width='5.6c', label='SI.km')
    fig.savefig(f'{OUT}/bounds_extremal_real.png', dpi=150)
    print('wrote', f'{OUT}/bounds_extremal_real.png')
