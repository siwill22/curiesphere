"""Per-continent comparison: each candidate map swapped in for the Hasterok provinces on one continent at a time
(Hasterok elsewhere); the change of the global evidence is that continent's share of the difference between the maps.
Read from gcm_glc_swap_<map>_lcs1_16-133. Run from the repository root."""
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, 'docs/reports/2026-10-13-candidate-sets')
import make_ranking as mr          # noqa: E402

R, OUT = mr.R, mr.OUT
CONTINENTS = {'afr': 'Africa', 'eur': 'Eurasia', 'nam': 'North\nAmerica', 'sam': 'South\nAmerica',
              'aus': 'Australia', 'ant': 'Antarctica', 'zea': 'Zealandia'}
INK, MUTED = '#0b0b0b', '#52514e'
# diverging: red (worse than Hasterok) - neutral grey - blue (better)
CMAP = LinearSegmentedColormap.from_list('div', ['#c0392b', '#e34948', '#f0efec', '#2a78d6', '#1c5cab'])


def swaps():
    """(evidence change, change in domain count) against cand:hasterok, maps x continents"""
    d, n = {}, {}
    for f in glob.glob(R + 'gcm_glc_swap_*_lcs1_16-133.log'):
        e = mr.evidence(f)
        v0, n0 = e['cand:hasterok']
        for k, (v, m) in e.items():
            if k.startswith('swap:'):
                _, c, cont = k.split(':')
                d.setdefault(mr.NAMES[f'cand:{c}'], {})[CONTINENTS[cont]] = v - v0
                n.setdefault(mr.NAMES[f'cand:{c}'], {})[CONTINENTS[cont]] = m - n0
    order = list(CONTINENTS.values())
    d, n = pd.DataFrame(d).T[order], pd.DataFrame(n).T[order]
    rows = d.sum(1).sort_values(ascending=False).index
    return d.loc[rows], n.loc[rows]


def main():
    d, n = swaps()
    fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=200)
    lim = np.abs(d.values).max()
    ax.imshow(d.values, cmap=CMAP, norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            dark = abs(d.values[i, j]) > 0.55*lim                # light ink on the strongest cells
            ax.text(j, i - 0.12, f'{d.values[i, j]:+.0f}', ha='center', va='center', fontsize=8,
                    color='white' if dark else INK)
            ax.text(j, i + 0.25, f'{n.values[i, j]:+d} domains', ha='center', va='center', fontsize=5.5,
                    color='#f0efec' if dark else MUTED)
    ax.set_xticks(range(d.shape[1]), d.columns, fontsize=8, color=INK)
    ax.set_yticks(range(d.shape[0]), [f'{r}  (total {d.loc[r].sum():+.0f})' for r in d.index], fontsize=8, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, d.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, d.shape[0]), minor=True)
    ax.grid(which='minor', color='white', lw=2)
    ax.tick_params(which='minor', length=0)
    ax.set_title('Each map in place of the Hasterok provinces on one continent at a time\n'
                 'Change of log evidence (blue: fits better than Hasterok there); small: change in number of domains',
                 fontsize=8.5, loc='left', color=INK)
    fig.tight_layout()
    fig.savefig(OUT + 'swaps.png')
    d.round(1).rename(columns=lambda c: c.replace('\n', ' ')).to_csv(OUT + 'swaps.csv')
    print(d.round(0).assign(total=d.sum(1).round(0)).to_string())


if __name__ == '__main__':
    main()
