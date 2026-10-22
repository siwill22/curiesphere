"""Does a map fit better only because it has more domains? Each candidate map's evidence (against the Hasterok
provinces) at its number of domains, its spun copies at theirs, and random Voronoi tilings of the continents at a
range of tile counts: the score that domain count alone reaches.
Read from the candidate runs (make_ranking.py), gcm_glc_vork<k>_lcs1_16-133 and the 724-tile tilings of
gcm_glc_layouts_ens{0,1,2}_lcs1_16-133. Run from the repository root."""
import glob
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, 'docs/reports/2026-10-13-candidate-sets')
import make_ranking as mr          # noqa: E402
from make_sizes import PASS, FAIL  # noqa: E402

R, OUT = mr.R, mr.OUT
INK, MUTED, GRID, BAND = '#0b0b0b', '#52514e', '#ecebe7', '#d9d8d2'


def tilings():
    """{tile count: evidence of each random tiling}"""
    out = {}
    for f in glob.glob(R + 'gcm_glc_vork*_lcs1_16-133.log'):
        for name, (v, n) in mr.evidence(f).items():
            if name.startswith('voronoi_k'):
                out.setdefault(n, []).append(v)
    for f in glob.glob(R + 'gcm_glc_layouts_ens[012]_lcs1_16-133.log'):
        for name, (v, n) in mr.evidence(f).items():
            if name.startswith('voronoi_provinces'):
                out.setdefault(n, []).append(v)
    return {k: np.array(v) for k, v in sorted(out.items())}


def spun_with_counts(cand):
    f = [g for b in mr.BATCHES for g in glob.glob(R + f'gcm_glc_{b}_null_{cand[5:]}_lcs1_16-133.log')]
    return np.array(list(mr.evidence(f[-1]).values())) if f else np.empty((0, 2))


def main():
    real = mr.runs('lcs1_16-133')
    ref = real[mr.REF][0]
    tiles = tilings()
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=200)
    k = np.array(list(tiles))
    # 10th-90th percentile, so counts with more draws (50 at 724 tiles) do not get a wider band
    lo, med, hi = (np.array([np.percentile(v - ref, q) for v in tiles.values()]) for q in (10, 50, 90))
    ax.fill_between(k, lo, hi, color=BAND, lw=0, zorder=1, label='Random tilings: 10th–90th percentile')
    ax.plot(k, med, color=MUTED, lw=1.6, zorder=2, label='Random tilings: median')
    for kk, v in tiles.items():
        ax.scatter(np.full(len(v), kk), v - ref, s=6, color=MUTED, alpha=0.5, lw=0, zorder=2)
    for name, label, colour in FAIL + PASS[::-1]:
        if name not in real:
            continue
        s = spun_with_counts(name)
        if len(s):
            ax.scatter(s[:, 1], s[:, 0] - ref, s=9, color=colour, alpha=0.35, lw=0, zorder=3)
        v, n = real[name]
        ax.scatter([n], [v - ref], s=46, color=colour, edgecolor='white', lw=1, zorder=5)
        ax.annotate(label, (n, v - ref), xytext=(6, 0), textcoords='offset points', fontsize=6.5, color=INK,
                    va='center', zorder=6)
    ax.axhline(0, color=INK, lw=0.8, zorder=1)
    ax.set_xlabel('Number of domains', fontsize=8, color=MUTED)
    ax.set_ylabel('Log evidence minus that of the Hasterok provinces', fontsize=8, color=MUTED)
    ax.set_title('Each map (large dot) and its spun copies (faint) against random tilings with the same number of '
                 'domains', fontsize=9, loc='left', color=INK)
    ax.tick_params(labelsize=7, colors=MUTED)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.grid(color=GRID, lw=0.6, zorder=0)
    ax.legend(fontsize=7, frameon=False, loc='lower right')
    ax.set_xlim(0, max(k.max(), max(n for _, n in real.values())) * 1.12)
    fig.tight_layout()
    fig.savefig(OUT + 'domain_count.png')
    print('tile counts:', {kk: len(v) for kk, v in tiles.items()})
    for kk, v in tiles.items():
        print(f'{kk:5d} tiles: median {np.median(v) - ref:+7.1f}, best {v.max() - ref:+7.1f}')


if __name__ == '__main__':
    main()
