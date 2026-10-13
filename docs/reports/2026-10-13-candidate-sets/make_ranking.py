"""The candidate-map test (batches 1 and 2): each map's evidence against the Hasterok provinces, against 20 spun copies of
itself, and on the S5 synthetic data; and maps of each map's domains and fitted VIS.
Read from the runs gcm_glc_cand1_* and gcm_glc_cand2_*. Run from the repository root."""
import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import pygmt
import xarray as xr

from remit.inversion import figures

R = 'inversions/results/'
OUT = 'docs/reports/2026-10-13-candidate-sets/'
REF = 'cand:hasterok'
NAMES = {'cand:hasterok': 'Hasterok provinces', 'cand:sl2013_k4': 'SL2013sv clusters, k = 4',
         'cand:sl2013_k6': 'SL2013sv clusters, k = 6', 'cand:sl2013_k8': 'SL2013sv clusters, k = 8',
         'cand:sl2013_k12': 'SL2013sv clusters, k = 12', 'cand:tc1': 'TC1 tectono-thermal age',
         'cand:ecm1': 'ECM1 crustal type', 'cand:merdith': 'Merdith (2021) static polygons',
         'cand:nrcan': 'NRCan surface geology', 'cand:cgmw': 'CGMW geological map'}
BATCHES = ('cand1', 'cand2')


def evidence(path):
    """(log evidence, number of domains) of every layout in a run log"""
    out = {}
    for line in open(path):
        m = re.search(r'\] (\S+) \(ocean.*: (\d+) domains, log evidence ([-\d.]+)', line)
        if m:
            out[m.group(1)] = (float(m.group(3)), int(m.group(2)))
    return out


def runs(tag):
    """log evidence of every layout of the batches run so far, on real data (tag 'lcs1_16-133') or S5 ('syn_s5')"""
    out = {}
    for b in BATCHES:
        for f in glob.glob(R + f'gcm_glc_{b}_{tag}.log'):
            out = {**evidence(f), **out}                     # the first batch's value of a layout run twice
    return out


def spun(cand):
    """log evidence of the spun copies of a candidate, if its null run has finished"""
    f = glob.glob(R + f'gcm_glc_cand*_null_{cand[5:]}_lcs1_16-133.log')
    return np.array([v for v, _ in evidence(f[0]).values()]) if f else np.array([])


def chart():
    real, syn = runs('lcs1_16-133'), runs('syn_s5')
    cands = sorted((c for c in NAMES if c in real), key=lambda c: real[c][0])
    ref = real[REF][0]
    INK, MUTED, REAL, SPIN, SYN = '#0b0b0b', '#52514e', '#2a78d6', '#eb6834', '#8a8984'
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 1.2 + 0.4*len(cands)), dpi=200, gridspec_kw=dict(width_ratios=[3, 1.2]))
    rng = np.random.default_rng(1)
    for y, c in enumerate(cands):
        s = spun(c) - ref
        if len(s):
            ax.scatter(s, y + rng.uniform(-0.18, 0.18, len(s)), s=10, color=SPIN, marker='D', edgecolor='white',
                       lw=0.4, zorder=3)
        d = real[c][0] - ref
        ax.scatter([d], [y], s=40, color=REAL, zorder=4)
        note = f'{(s > d).sum()} of {len(s)} spun copies beat it' if len(s) else 'spin test pending'
        ax.annotate(note, (max([d] + list(s)), y), xytext=(8, 0), textcoords='offset points', fontsize=7,
                    color=MUTED, va='center')
        if c in syn:
            bx.scatter([syn[c][0] - syn[REF][0]], [y], s=30, color=SYN, zorder=3)
    labels = [f'{NAMES[c]}  ({real[c][1]} domains)' for c in cands]
    for a in (ax, bx):
        a.axvline(0, color=INK, lw=1, zorder=2)
        a.set_yticks(range(len(cands)), labels if a is ax else [], fontsize=8, color=INK)
        a.tick_params(axis='x', labelsize=7, colors=MUTED)
        for side in ('top', 'right', 'left'):
            a.spines[side].set_visible(False)
        a.spines['bottom'].set_color(MUTED)
        a.grid(axis='x', color='#ecebe7', lw=0.6, zorder=0)
    ax.margins(x=0.3)
    ax.set_xlabel('Log evidence minus that of the Hasterok provinces  (> 0: fits better)', fontsize=8, color=MUTED)
    ax.set_title('Real data: each map (blue) and 20 spun copies of it (orange)', fontsize=9, loc='left', color=INK)
    bx.set_xlabel('Same, synthetic data', fontsize=8, color=MUTED)
    bx.set_title('Synthetic check (S5)', fontsize=9, loc='left', color=INK)
    fig.tight_layout()
    fig.savefig(OUT + 'ranking.png')
    for c in cands[::-1]:
        s = spun(c) - ref
        print(f'{NAMES[c]:32s} {real[c][1]:4d} domains  {real[c][0] - ref:+8.1f}'
              + (f'  spun best {s.max():+.1f} median {np.median(s):+.1f}' if len(s) else '')
              + (f'  S5 {syn[c][0] - syn[REF][0]:+.1f}' if c in syn else ''))


def maps():
    """for each map: its domains (random colours) and the fitted VIS averaged over each domain"""
    sets = [xr.load_dataset(f) for b in BATCHES for f in glob.glob(R + f'gcm_glc_{b}_lcs1_16-133.nc')]
    ds = sets[0]
    cell = figures.cell_grid(ds)
    a = ds.area_km2.values
    real = runs('lcs1_16-133')
    cands = sorted((c for c in NAMES if c in real), key=lambda c: -real[c][0])
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='8p', FONT_ANNOT_PRIMARY='6p', FONT_LABEL='7p', MAP_FRAME_TYPE='plain',
                 MAP_TITLE_OFFSET='3p')
    rng = np.random.default_rng(3)
    for k, c in enumerate(cands):
        v = c.replace(':', '_')
        d = next(x for x in sets if f'layout_vis_{v}' in x)
        vis, dom = d[f'layout_vis_{v}'].values, d[f'layout_domain_{v}'].values.astype(int)
        if k:
            fig.shift_origin(xshift='-10.5c', yshift='-6.0c')
        fig.basemap(region=[-180, 180, -90, 85], projection='N0/10c', frame=['f', f'+t{NAMES[c]}: domains'])
        colour = rng.permutation(dom.max() + 1)[dom].astype(float)
        pygmt.makecpt(cmap='roma', series=[0, dom.max()])
        fig.grdimage(figures.cell_map(cell, colour), cmap=True, nan_transparent=True)
        fig.coast(shorelines='0.15p,gray40', resolution='c', area_thresh=5000)
        fig.shift_origin(xshift='10.5c')
        fig.basemap(region=[-180, 180, -90, 85], projection='N0/10c', frame=['f', '+tFitted VIS averaged over each domain'])
        pygmt.makecpt(cmap=figures.VIS_CPT[0], series='0/1.5/0.05', background=True)
        avg = np.bincount(dom, a*vis)/np.bincount(dom, a)
        fig.grdimage(figures.cell_map(cell, avg[dom]), cmap=True, nan_transparent=True)
        fig.coast(shorelines='0.15p,gray40', resolution='c', area_thresh=5000)
    fig.colorbar(cmap=True, frame='xaf+lVIS (SI km)', position='JBC+w6c/0.25c+o0/0.5c')
    fig.savefig(OUT + 'candidate_maps.png', dpi=200)


if __name__ == '__main__':
    chart()
    maps()
