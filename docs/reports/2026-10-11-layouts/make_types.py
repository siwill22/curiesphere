"""Province VIS by province type (Hasterok et al. 2022 prov_type): the province averages of the seven continent fits
(DF, global layout fit as background) and the province levels of the global layout fit, as violins, boxes and one
dot per province (dot area ~ province area). Run from the repository root."""
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from remit.inversion import covariates

R = 'inversions/results/'
OUT = 'docs/reports/2026-10-11-layouts/'
CONT = ('afr', 'ant', 'aus', 'eur', 'nam', 'sam', 'zea')
TABLE = {c: f'inversions/domains/{"australia" if c == "aus" else c}_domains.csv' for c in CONT}


def province_types():
    """param -> the most common prov_type among its table rows"""
    rows = pd.concat([pd.read_csv(TABLE[c], keep_default_na=False) for c in CONT])
    rows = rows[rows.param != '']
    return {p: Counter(g.prov_type).most_common(1)[0][0] for p, g in rows.groupby('param')}


def values():
    kind = province_types()
    out = []
    for c in CONT:
        d = json.loads(xr.open_dataset(R + f'df_{c}_lcs1_16-133_laybg_gm.nc').attrs['domains'])
        out += [dict(fit='continent', name=k, type=kind[k], vis=v['mean'], area=v['area_km2'])
                for k, v in d.items() if k in kind]
    d = json.loads(xr.open_dataset(R + 'gcm_glc_layouts_oc_lcs1_16-133.nc').attrs['domains'])
    out += [dict(fit='global', name=k, type=kind[k], vis=v['level'], area=v['area_km2'])
            for k, v in d.items() if k in kind]
    return pd.DataFrame(out)


def plot(df, group, order, path, title):
    fits = [('continent', 'Continent fits: VIS averaged over each province', '#2a78d6'),
            ('global', 'Global fit: province level', '#eb6834')]
    INK, MUTED = '#0b0b0b', '#52514e'
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42*len(order) + 1.6), dpi=200, sharey=True)
    rng = np.random.default_rng(0)
    y = np.arange(len(order))[::-1]
    for ax, (fit, head, col) in zip(axes, fits):
        sub = df[df.fit == fit]
        for yi, t in zip(y, order):
            s = sub[sub[group] == t]
            if len(s) > 2:
                vp = ax.violinplot(s.vis, positions=[yi], orientation='horizontal', widths=0.8, showextrema=False)
                for b in vp['bodies']:
                    b.set_facecolor(col)
                    b.set_alpha(0.25)
                    b.set_edgecolor('none')
            ax.boxplot(s.vis, positions=[yi], orientation='horizontal', widths=0.28, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor='white', edgecolor=INK, lw=0.7), medianprops=dict(color=col, lw=2),
                       whiskerprops=dict(color=INK, lw=0.7), capprops=dict(color=INK, lw=0.7))
            ax.scatter(s.vis, yi + rng.uniform(-0.3, 0.3, len(s)), s=3 + 60*np.sqrt(s.area/1e6), color=col,
                       alpha=0.55, edgecolor='white', lw=0.3, zorder=3)
            w = np.average(s.vis, weights=s.area)
            ax.plot(w, yi, marker='|', ms=12, mew=2, color=INK, zorder=4)
        allw = np.average(sub.vis, weights=sub.area)
        ax.axvline(allw, color=MUTED, lw=0.8, ls='--', zorder=0)
        ax.set_title(head, fontsize=9, loc='left', color=INK)
        ax.set_xlim(-0.6, 2.2)
        ax.set_xlabel('VIS (SI km)', fontsize=8, color=MUTED)
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.grid(axis='x', color='#ecebe7', lw=0.6, zorder=0)
        for s_ in ('top', 'right', 'left'):
            ax.spines[s_].set_visible(False)
        ax.spines['bottom'].set_color(MUTED)
    n = df[df.fit == 'global'].groupby(group).size()
    axes[0].set_yticks(y, [f'{t}  ({n.get(t, 0)})' for t in order], fontsize=8, color=INK)
    fig.suptitle(title, fontsize=10, x=0.01, ha='left', color=INK)
    fig.text(0.01, 0.005, 'Violin: spread of the provinces; box: middle half and median (coloured); dots: provinces '
             '(size ~ area); black tick: area-weighted mean; dashed: area-weighted mean of all. (n): provinces.',
             fontsize=6.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path)


if __name__ == '__main__':
    df = values()
    df['class'] = [covariates.CLASS_NAMES[covariates.CLASS_OF_TYPE[t]] for t in df.type]
    g = df[df.fit == 'global']
    order = g.groupby('type').vis.median().sort_values(ascending=False).index.tolist()
    plot(df, 'type', order, OUT + 'vis_by_type.png', 'Province VIS by province type')
    plot(df, 'class', list(covariates.CLASS_NAMES), OUT + 'vis_by_class.png', 'Province VIS by class of province type')
    print(df.groupby(['fit', 'type']).vis.agg(['size', 'median']).unstack(0).round(2))
