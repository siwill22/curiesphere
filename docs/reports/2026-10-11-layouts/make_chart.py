"""Chart for the 2026-10-11 report: how far the polygons beat random tiles of the same number (log evidence), on
real data and on a smooth random truth that ignores geology (the null). Read from the run logs; run from the
repository root."""
import re

import matplotlib.pyplot as plt

R = 'inversions/results/'


def local(log):
    try:
        t = open(R + log).read()
    except FileNotFoundError:
        return None
    p = re.search(r'\] polygons: K \d+, log evidence ([-\d.]+)', t)
    r = re.search(r'\] random tiles \(K \d+\): K \d+, log evidence ([-\d.]+)', t)
    return float(p.group(1)) - float(r.group(1)) if p and r else None


def local_eur():
    try:
        p = re.search(r'\] polygons: K \d+, log evidence ([-\d.]+)', open(R + 'df_eur_lcs1_16-133_gcmbg_gm.log').read())
        r = re.search(r'\] random tiles \(K \d+\): K \d+, log evidence ([-\d.]+)',
                      open(R + 'df_eur_lcs1_16-133_gcmbg_gm_ctrl.log').read())
        return float(p.group(1)) - float(r.group(1))
    except (FileNotFoundError, AttributeError):
        return None


def glob(log, a, b):
    t = open(R + log).read()
    rank = t[t.rindex('ranking:'):]
    v = {k: float(x) for k, x in re.findall(r'(\w+) ([-+][\d.]+) \(ocean', rank)}
    return v[a] - v[b]


rows = [('Global, provinces', glob('gcm_glc_layouts_oc_lcs1_16-133_ctrl.log', 'provinces', 'voronoi_provinces'),
         glob('gcm_glc_layouts_oc_syn_s5_ctrl.log', 'provinces', 'voronoi_provinces')),
        ('Global, groups', glob('gcm_glc_layouts_oc_lcs1_16-133_ctrl.log', 'groups', 'voronoi_groups'),
         glob('gcm_glc_layouts_oc_syn_s5_ctrl.log', 'groups', 'voronoi_groups'))]
for c, name in [('nam', 'North America'), ('ant', 'Antarctica'), ('aus', 'Australia'), ('sam', 'South America'),
                ('afr', 'Africa'), ('zea', 'Zealandia'), ('eur', 'Eurasia')]:
    real = local_eur() if c == 'eur' else local(f'df_{c}_lcs1_16-133_gcmbg_gm.log')
    rows.append((name, real, local(f'df_{c}_syn_rf_gm.log')))
rows = [r for r in rows if r[1] is not None]

fig, ax = plt.subplots(figsize=(7.5, 0.42*len(rows) + 1.4), dpi=200)
REAL, NULL, INK, MUTED = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e'
y = list(range(len(rows)))[::-1]
ax.axvline(0, color=MUTED, lw=0.8)
for yi, (name, real, null) in zip(y, rows):
    ax.plot([min(real, null if null is not None else real), max(real, null if null is not None else real)], [yi, yi],
            color='#d8d7d2', lw=2, zorder=1)
    ax.scatter(real, yi, s=60, color=REAL, edgecolor='white', lw=1.5, zorder=3)
    ax.annotate(f'{real:+.0f}', (real, yi), xytext=(6, 0), textcoords='offset points', va='center', fontsize=7,
                color=INK)
    if null is not None:
        ax.scatter(null, yi, s=60, color=NULL, edgecolor='white', lw=1.5, zorder=3, marker='D')
ax.set_yticks(y, [r[0] for r in rows], fontsize=8, color=INK)
ax.set_xlabel('Log evidence of the polygons minus random tiles of the same number  (> 0: polygons fit better)',
              fontsize=8, color=MUTED)
ax.tick_params(axis='x', labelsize=7, colors=MUTED)
for s in ('top', 'right', 'left'):
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(MUTED)
ax.grid(axis='x', color='#ecebe7', lw=0.6, zorder=0)
ax.scatter([], [], s=50, color=REAL, label='Real data (LCS-1)')
ax.scatter([], [], s=50, color=NULL, marker='D', label='Null: smooth random truth, blind to geology')
ax.legend(fontsize=7, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
ax.margins(x=0.08)
ax.set_title('Do the geological polygons beat random patches?', fontsize=10, loc='left', color=INK, pad=22)
fig.tight_layout()
fig.savefig('docs/reports/2026-10-11-layouts/polygons_vs_random.png')
for r in rows:
    print(r)
