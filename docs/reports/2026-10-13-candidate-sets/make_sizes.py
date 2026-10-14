"""Size-frequency of the candidate maps' domains, as Bird (2003) plots plates: each domain's area (as fitted: the
patches on the ~100 km cells, rest domains left out) against the number of domains at least that large.
Read from gcm_glc_cand1/2_lcs1_16-133.nc (TC1 from the Keller-grid rerun gcm_glc_cand_tc1k). Run from the repository root."""
import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

R = 'inversions/results/'
OUT = 'docs/reports/2026-10-13-candidate-sets/'
# the six maps that beat their spun copies, in the fixed categorical order; SL2013sv (fails) in muted greys
PASS = [('cand:hasterok', 'Hasterok provinces', '#2a78d6'), ('cand:nrcan', 'NRCan surface geology', '#eb6834'),
        ('cand:ecm1', 'ECM1 crustal type', '#1baf7a'), ('cand:merdith', 'Merdith static polygons', '#eda100'),
        ('cand:cgmw', 'CGMW geological map', '#e87ba4'), ('cand:tc1', 'TC1 tectono-thermal age', '#008300')]
FAIL = [(f'cand:sl2013_k{k}', f'SL2013sv k = {k}', g) for k, g in
        [(4, '#c9c8c2'), (6, '#b0afa8'), (8, '#97968f'), (12, '#7e7d77')]]
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#ecebe7'


def areas(name, sets):
    """area (km2, on the sphere) of every non-rest domain of a layout"""
    v = name.replace(':', '_')
    ds = next(d for d in sets if f'layout_domain_{v}' in d)
    dom = ds[f'layout_domain_{v}'].values.astype(int)
    names = json.loads(ds.attrs['domain_names'])[name]
    a = np.bincount(dom, ds.area_km2.values, minlength=len(names))
    a = a[[i for i, n in enumerate(names) if 'rest ' not in n]]
    return np.sort(a[a > 0])[::-1]


def main():
    # the TC1 rerun with the Keller grid first, so its TC1 domains are the ones used
    sets = [xr.load_dataset(f) for f in glob.glob(R + 'gcm_glc_cand_tc1k_lcs1_16-133.nc')
            + sorted(glob.glob(R + 'gcm_glc_cand[12]_lcs1_16-133.nc'))]
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
    for name, label, colour in FAIL + PASS[::-1]:
        a = areas(name, sets)
        n = np.arange(1, len(a) + 1)
        lw = 2.4 if name == 'cand:hasterok' else 1.6
        ax.step(a, n, where='post', color=colour, lw=lw, label=f'{label} ({len(a)})', zorder=2 if name.startswith('cand:sl2013') else 3)
    cell = sets[0].area_km2.median()
    ax.axvline(3*cell, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.annotate('3 whole cells, the smallest domain\n(coastal cells can be partial)', (3*cell, 1.2), xytext=(4, 0), textcoords='offset points',
                fontsize=7, color=MUTED, va='bottom')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Domain area (km²)', fontsize=8, color=MUTED)
    ax.set_ylabel('Number of domains at least this large', fontsize=8, color=MUTED)
    ax.set_title('Size–frequency of the candidate maps\' domains (as fitted, rest domains left out)', fontsize=9,
                 loc='left', color=INK)
    ax.tick_params(labelsize=7, colors=MUTED)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.grid(color=GRID, lw=0.6, which='major', zorder=0)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], fontsize=7, frameon=False, loc='upper right', title='Map (number of domains)',
              title_fontsize=7, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT + 'size_frequency.png')
    for name, label, _ in PASS + FAIL:
        a = areas(name, sets)
        print(f'{label:28s} {len(a):4d} domains, largest {a[0]:9.3g} km2, median {np.median(a):9.3g} km2')


if __name__ == '__main__':
    main()
