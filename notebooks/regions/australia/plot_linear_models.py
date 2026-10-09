"""Maps and charts of run_linear_models.py (and run_covariates.py) for Australia.
  covariates.png               the covariates on the cells
  linear_models_<case>.png     VIS maps (truth or merging, GP mean, GP MAP, L2, TV, GP sd) and Br
                               (true, model from the GP MAP, residual) at the data points
  linear_models_charts.png     regression coefficients, class statistics, hypothesis ranking
Run from notebooks/: PYTHONPATH=..:$PYTHONPATH python regions/australia/plot_linear_models.py"""
import numpy as np
import pygmt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from domains import covariates, problem
from domains.plotting import cell_grid, point_grid, map_panel

OUT = 'regions/australia'
pb = problem.load(f'{OUT}/config.toml')
cells, G, z = pb['cells'], pb['G'], pb['z']
cov = np.load(f'{pb["cache"]}/covariates.npz')
lm = np.load(f'{pb["cache"]}/linear_models.npz')
rs = np.load(f'{pb["cache"]}/realistic_synthetic.npz', allow_pickle=True)
ms = np.load(f'{pb["cache"]}/merge_points_sweep.npz', allow_pickle=True)
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']           # categorical slots 1-3 (dataviz reference palette)
INK, MUTED = '#0b0b0b', '#52514e'


def grid_of_panels(panels, path, ncol=3, width=5.6):
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='8p', FONT_ANNOT_PRIMARY='6p', FONT_LABEL='7p', MAP_TITLE_OFFSET='2p')
    for k, p in enumerate(panels):
        if k:
            if k % ncol:
                fig.shift_origin(xshift=f'{width + 1.0}c')
            else:
                fig.shift_origin(xshift=f'-{(ncol - 1)*(width + 1.0)}c', yshift=f'-{width + 2.2}c')
        g, title, cmap, series, label = p
        map_panel(fig, g, title, cmap, series, width=f'{width}c', label=label)
    fig.savefig(path, dpi=150)
    print('wrote', path)


# ------------------------------------------------------------------ covariates
pan = [(cell_grid(cov['crust'], cells), 'Crustal thickness (AusMoho2012 / CRUST1)', 'batlow', [10, 50, 1], 'km'),
       (cell_grid(cov['lab'], cells), 'LAB depth (AuSREM / LITHO1)', 'batlow', [50, 250, 5], 'km'),
       (cell_grid(cov['heat_flow'], cells), 'Heat flow (Lucazeau 2019)', 'lajolla', [30, 110, 2], 'mW/m@+2@+'),
       (cell_grid(cov['heat_flow_seisheat'], cells), 'Heat flow (SeisHeat, tomography)', 'lajolla', [30, 110, 2], 'mW/m@+2@+'),
       (cell_grid(np.where(cov['undated'] > 0, np.nan, cov['age']), cells), 'Oldest magmatic age (blank: undated)', 'batlow', [0, 3500, 100], 'Ma'),
       (cell_grid(cov['cls'].astype(float), cells), 'Province class (0 craton/shield, 1 orogen, 2 margin)', 'categorical', [-0.5, 2.5, 1], 'class')]
grid_of_panels(pan, f'{OUT}/covariates.png')

# ------------------------------------------------------------------ maps per case
for case in [str(c) for c in lm['cases']]:
    k = lambda s: f'{case}_{s}'
    if k('gp_map') not in lm:
        continue
    truth = lm[k('truth')] if k('truth') in lm else None
    if case == 'real':
        i = [j for j, s in enumerate(ms['settings']) if np.isclose(s[0], 1.) and np.isclose(s[1], 0.01)][0]
        first = (ms['fine_v'][i], f'Merging (K {ms["fine_K"][i]}), for comparison')
        br_true = z['d_lcs'] - z['b']
    else:
        first = (truth, 'True VIS')
        br_true = G @ truth
    br_model = G @ lm[k('gp_map')]
    vmax = float(np.percentile(np.r_[lm[k('gp_map')], first[0]], 98))
    vis = dict(cmap='viridis', series=[0, round(vmax, 1), 0.05], label='SI.km')
    bmax = float(np.percentile(np.abs(br_true), 98))
    br = dict(cmap='vik', series=[-round(bmax), round(bmax), 1], label='nT')
    h, s, ell = float(lm[k('gp_h')]), float(lm[k('gp_s')]), float(lm[k('gp_ell')])
    panels = [(cell_grid(first[0], cells), first[1]),
              (cell_grid(lm[k('gp_v')], cells), f'GP mean (length {ell:.0f} km)'),
              (cell_grid(lm[k('gp_map')], cells), 'GP, positive VIS (MAP)'),
              (cell_grid(lm[k('l2_v')], cells), 'L2 (smooth)'),
              (cell_grid(lm[k('tv_v')], cells), 'TV (blocky)')]
    if case in ('hast', 'hast_var'):
        panels.append((cell_grid(rs[f'{case}_n1_l1.00_v'], cells), f'Merging (K {int(rs[f"{case}_n1_l1.00_K"])})'))
    pan = [(g, t, vis['cmap'], vis['series'], vis['label']) for g, t in panels]
    pan.append((cell_grid(lm[k('gp_sd')], cells), 'GP uncertainty (1 sd)', 'oslo', [0, round(s, 2), 0.01], 'SI.km'))
    pan += [(point_grid(br_true, z['plon'], z['plat']), 'Br: data' if case == 'real' else 'Br: true (no noise)',
             br['cmap'], br['series'], br['label']),
            (point_grid(br_model, z['plon'], z['plat']), 'Br: GP model', br['cmap'], br['series'], br['label']),
            (point_grid(br_true - br_model, z['plon'], z['plat']), 'Br: residual', br['cmap'], br['series'], br['label'])]
    grid_of_panels(pan, f'{OUT}/linear_models_{case}.png')

# ------------------------------------------------------------------ charts
cases = [str(c) for c in lm['cases'] if f'{c}_reg_main_theta' in lm]
fig, axes = plt.subplots(3, len(cases), figsize=(4.2*len(cases), 11), squeeze=False)
for j, case in enumerate(cases):
    k = lambda s: f'{case}_{s}'
    # regression coefficients (slopes only; class means are on their own panel)
    ax = axes[0, j]
    names = [str(x) for x in lm[k('reg_main_names')]]
    slope = [i for i, nm in enumerate(names) if not nm.startswith('mean')]
    variants = [v for v in ('main', 'heat_flow_seisheat', 'heat_flow_heatr') if k(f'reg_{v}_theta') in lm]
    for vi, v in enumerate(variants):
        th, sd = lm[k(f'reg_{v}_theta')][slope], lm[k(f'reg_{v}_sd')][slope]
        y = np.arange(len(slope)) + (vi - 1)*0.22
        ax.errorbar(th, y, xerr=2*sd, fmt='o', ms=4, lw=1.5, color=SERIES[vi], label=v.replace('_', ' '))
    if case == 'cov_truth':
        tn = {'crust': lm['theta_true'][3], 'heat_flow': lm['theta_true'][4]}
        for i, si in enumerate(slope):
            ax.plot(tn.get(names[si], 0.), i, marker='|', ms=14, color=INK)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(len(slope)), [names[i].replace('_', ' ') for i in slope])
    ax.set_xlabel('VIS change per standard deviation (SI.km), ±2 sd')
    ax.set_title(f'{case}: covariate effects' + (' (| = truth)' if case == 'cov_truth' else ''), fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    # class statistics
    ax = axes[1, j]
    if k('cls_mean') in lm:
        m, msd, s, ell = lm[k('cls_mean')], lm[k('cls_mean_sd')], lm[k('cls_s')], lm[k('cls_ell')]
        x = np.arange(3)
        ax.errorbar(x - 0.1, m, yerr=2*msd, fmt='o', color=SERIES[0], ms=5, lw=1.5, label='mean ±2 sd')
        ax.plot(x + 0.1, s, 's', color=SERIES[1], ms=5, label='spread (sd)')
        ax.set_xticks(x, [n.replace('+', '\n') for n in covariates.CLASS_NAMES], fontsize=7)
        ax.set_ylabel('SI.km')
        for xi in x:
            ax.annotate(f'{ell[xi]:.0f} km', (xi + 0.1, s[xi]), xytext=(4, 0), textcoords='offset points',
                        fontsize=7, color=MUTED, va='center')
        dll = float(lm[k('cls_ll')] - lm[k('cls_shared_ll')])
        ax.set_title(f'{case}: classes (separate vs shared stats: Δlog ev {dll:+.1f})', fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    # hypotheses: top 12 by evidence with u
    ax = axes[2, j]
    if k('hyp_ll') in lm:
        ll, names = lm[k('hyp_ll')], [str(x) for x in lm[k('hyp_names')]]
        o = np.argsort(-ll)[:12]
        ax.barh(np.arange(len(o)), ll[o] - ll[o[0]], color=SERIES[0], height=0.6)
        ax.set_yticks(np.arange(len(o)), [names[i].replace('auto: ', '') for i in o], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel('log evidence relative to the best')
        ax.set_title(f'{case}: hypotheses (top 12)', fontsize=9)
for ax in axes.ravel():
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=7)
fig.tight_layout()
fig.savefig(f'{OUT}/linear_models_charts.png', dpi=150)
print('wrote', f'{OUT}/linear_models_charts.png')
