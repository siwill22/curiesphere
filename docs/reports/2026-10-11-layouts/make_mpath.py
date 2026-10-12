"""Chart for section 8 of the 2026-10-11 report: Australia, DF scoring FD-merge's partitions (K domains), the 39
polygons and random tiles, on real data and on 20 null data sets without boundaries (acceptance/df_merge_path_null.py).
Run from the repository root."""
import json
import re

import matplotlib.pyplot as plt
import numpy as np

R = 'inversions/results/'
d = json.load(open(R + 'df_aus_mpath_null.json'))
real = {}
for line in open(R + 'df_aus_lcs1_16-133_gcmbg_gm_mpath.log'):
    m = re.search(r'\] (polygons|merge path K \d+|one domain|random tiles \(K \d+\)): K \d+, log evidence ([-\d.]+)', line)
    if m:
        real[m.group(1)] = float(m.group(2))
gain = lambda k: real[k] - real['one domain']
null = lambda k: np.array([x[k] - x['one domain'] for x in d['draws']])
Ks = np.array(d['K'])
mp = [f'merge path K {k}' for k in Ks]
tiles = [k for k in real if k.startswith('random tiles')][0]

REAL, NULL, POLY, TILE, INK, MUTED = '#2a78d6', '#898781', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e'
fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=200)
lo, hi = np.array([np.percentile(null(k), 2.5) for k in mp]), np.array([np.percentile(null(k), 97.5) for k in mp])
ax.fill_between(Ks, lo, hi, color='#e1e0d9', lw=0, zorder=1)
ax.plot(Ks, [null(k).mean() for k in mp], color=NULL, lw=2, zorder=2)
ax.plot(Ks, [gain(k) for k in mp], color=REAL, lw=2, marker='o', ms=4, zorder=3)
for k, c, mk in (('polygons', POLY, 'D'), (tiles, TILE, 's')):
    ax.errorbar(39, null(k).mean(), yerr=2*null(k).std(ddof=1), color=NULL, fmt=mk, ms=5, capsize=2, zorder=3)
    ax.scatter(39, gain(k), color=c, marker=mk, s=40, edgecolor='white', lw=1, zorder=4)
ax.annotate('merge domains, real data', (100, gain(mp[-1])), xytext=(-4, 8), textcoords='offset points', ha='right',
            fontsize=7, color=REAL)
ax.annotate('merge domains, fake data without boundaries\n(mean and 95% range of 20 sets)', (100, lo[-1]),
            xytext=(-4, -70), textcoords='offset points', ha='right', va='top', fontsize=7, color=MUTED)
ax.annotate('39 polygons', (39, gain('polygons')), xytext=(7, 0), textcoords='offset points', va='center', fontsize=7,
            color=INK)
ax.annotate('39 random tiles', (39, gain(tiles)), xytext=(7, -2), textcoords='offset points', va='top', fontsize=7,
            color=INK)
ax.annotate('the same two on fake data', (39, null('polygons').mean()), xytext=(7, -6), textcoords='offset points',
            va='top', fontsize=7, color=MUTED)
ax.set_xscale('log')
ax.set_xticks(Ks, [str(k) for k in Ks])
ax.minorticks_off()
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xlabel('Number of domains', fontsize=8, color=MUTED)
ax.set_ylabel('Log evidence gained over one domain', fontsize=8, color=MUTED)
ax.tick_params(labelsize=7, colors=MUTED)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color(MUTED)
ax.grid(axis='y', color='#ecebe7', lw=0.6, zorder=0)
ax.set_title('Australia: domains made from the data against the polygons', fontsize=10, loc='left', color=INK)
fig.tight_layout()
fig.savefig('docs/reports/2026-10-11-layouts/mpath_aus.png')


def all_continents():
    """the same comparison for every continent with a null calibration (small multiples)"""
    conts = [('aus', 'Australia', 'df_aus_mpath_null.json', 'df_aus_lcs1_16-133_gcmbg_gm_mpath.log'),
             *[(c, n, f'df_{c}_lcs1_16-133_gcmbg_gm_mpath_null.json', f'df_{c}_lcs1_16-133_gcmbg_gm_mpath.log')
               for c, n in (('afr', 'Africa'), ('nam', 'North America'), ('ant', 'Antarctica'),
                            ('sam', 'South America'), ('zea', 'Zealandia'))]]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), dpi=200)
    for ax, (c, title, nf, lf) in zip(axes.ravel(), conts):
        dd = json.load(open(R + nf))
        rl = {}
        for line in open(R + lf):
            m = re.search(r'\] (polygons|merge path K \d+|one domain|random tiles \(K \d+\)): K \d+, log evidence '
                          r'([-\d.]+)', line)
            if m:
                rl[m.group(1)] = float(m.group(2))
        gn = lambda k: rl[k] - rl['one domain']
        nl = lambda k: np.array([x[k] - x['one domain'] for x in dd['draws']])
        ks = np.array(dd['K'])
        mk = [f'merge path K {k}' for k in ks]
        tl = [k for k in rl if k.startswith('random tiles')][0]
        kpoly = int(re.search(r'K (\d+)', tl).group(1))
        ax.fill_between(ks, [nl(k).min() for k in mk], [nl(k).max() for k in mk], color='#e1e0d9', lw=0, zorder=1)
        ax.plot(ks, [nl(k).mean() for k in mk], color=NULL, lw=2, zorder=2)
        ax.plot(ks, [gn(k) for k in mk], color=REAL, lw=2, marker='o', ms=3.5, zorder=3)
        for k, col, mkr in (('polygons', POLY, 'D'), (tl, TILE, 's')):
            ax.errorbar(kpoly, nl(k).mean(), yerr=[[nl(k).mean() - nl(k).min()], [nl(k).max() - nl(k).mean()]],
                        color=NULL, fmt=mkr, ms=4, capsize=2, zorder=3)
            ax.scatter(kpoly, gn(k), color=col, marker=mkr, s=36, edgecolor='white', lw=1, zorder=4)
        best = max(gn(k) - nl(k).mean() for k in mk)
        ax.set_title(f'{title} ({kpoly} polygons)\nbeyond fake data: polygons '
                     f'{gn("polygons") - nl("polygons").mean():+.0f}, merge {best:+.0f}, '
                     f'tiles {gn(tl) - nl(tl).mean():+.0f}', fontsize=7.5, loc='left', color=INK)
        ax.set_xscale('log')
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.tick_params(labelsize=7, colors=MUTED)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(MUTED)
        ax.grid(axis='y', color='#ecebe7', lw=0.6, zorder=0)
    for ax in axes[1]:
        ax.set_xlabel('Number of domains', fontsize=8, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel('Log evidence gained over one domain', fontsize=8, color=MUTED)
    fig.suptitle('Domains made from the data (blue: merge path) against the polygons (orange diamond) and random tiles '
                 '(green square)\ngrey: the same on fake data without boundaries (mean and range)', fontsize=9, x=0.01,
                 ha='left', color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig('docs/reports/2026-10-11-layouts/mpath_all.png')


all_continents()
