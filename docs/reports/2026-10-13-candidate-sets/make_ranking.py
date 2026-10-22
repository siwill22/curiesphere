"""The candidate-map test (batches 1 and 2): each map's evidence against the Hasterok provinces, against 20 spun copies of
itself, and on 11 draws of S5 synthetic data (geology-blind), with the score corrected for geometry (real minus the
S5 mean); and maps of each map's domains and fitted VIS.
Read from the runs gcm_glc_cand1_*, gcm_glc_cand2_*, gcm_glc_cand_syn_s5_d* and gcm_glc_cand_tc1k_* (TC1 rerun with the
Keller grid, which supersedes the cand1 TC1 rows). Run from the repository root."""
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
         'cand:nrcan': 'NRCan surface geology', 'cand:cgmw': 'CGMW geological map',
         'cand:glim': 'GLiM lithology', 'cand:ee2025_plate': 'Eglington & Evans (2025), per plate',
         'cand:ee2025_poly': 'Eglington & Evans (2025), per polygon'}
# runs in order: where a layout was run again later (TC1 with the Keller grid; NRCan without ice), the later run wins
BATCHES = ('cand1', 'cand2', 'cand_tc1k', 'cand3')


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
            out.update(evidence(f))
    return out


def s5_draws():
    """evidence of every layout in each S5 draw (0-10), the later run of a layout winning as in runs()"""
    out = []
    for i in range(11):
        first = ['cand1_syn_s5', 'cand2_syn_s5', 'cand_tc1k_syn_s5'] if i == 0 else [f'cand_syn_s5_d{i}']
        d = {}
        for stem in first + [f'cand3_syn_s5_d{i}']:
            for f in glob.glob(R + f'gcm_glc_{stem}.log'):
                d.update(evidence(f))
        if REF in d:
            out.append(d)
    return out


def s5_relative(c, draws):
    """a map's S5 evidence minus Hasterok's, in each draw that has it"""
    return np.array([d[c][0] - d[REF][0] for d in draws if c in d])


def spun(cand):
    """log evidence of the spun copies of a candidate, if its null run has finished"""
    f = [g for b in BATCHES for g in glob.glob(R + f'gcm_glc_{b}_null_{cand[5:]}_lcs1_16-133.log')]
    return np.array([v for v, _ in evidence(f[-1]).values()]) if f else np.array([])        # the latest batch


def chart():
    real, draws = runs('lcs1_16-133'), s5_draws()
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
        z = s5_relative(c, draws)
        if len(z):
            bx.scatter(z, y + rng.uniform(-0.15, 0.15, len(z)), s=10, color=SYN, alpha=0.7, lw=0, zorder=3)
            bx.plot([z.mean()]*2, [y - 0.3, y + 0.3], color=INK, lw=1.6, zorder=4)
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
    bx.set_xlabel('Same, synthetic data (bar: mean)', fontsize=8, color=MUTED)
    bx.set_title(f'Synthetic check: {len(draws)} S5 draws', fontsize=9, loc='left', color=INK)
    fig.tight_layout()
    fig.savefig(OUT + 'ranking.png')
    rows = []
    for c in cands[::-1]:
        s, z = spun(c), s5_relative(c, draws)
        d = real[c][0] - ref
        rows.append(dict(map=NAMES[c], domains=real[c][1], real_vs_hasterok=round(d, 1),
                         real_minus_median_spun=round(real[c][0] - np.median(s), 1) if len(s) else np.nan,
                         spun_beating_it=int((s > real[c][0]).sum()) if len(s) else np.nan,
                         s5_draws=len(z), s5_mean=round(z.mean(), 1), s5_sd=round(z.std(ddof=1), 1) if len(z) > 1 else np.nan,
                         corrected=round(d - z.mean(), 1)))
    import pandas as pd
    t = pd.DataFrame(rows).sort_values('corrected', ascending=False)
    t.to_csv(OUT + 'corrected.csv', index=False)
    print(t.to_string(index=False))


def maps():
    """for each map: its domains (random colours) and the fitted VIS averaged over each domain"""
    sets = [xr.load_dataset(f) for b in BATCHES[::-1] for f in glob.glob(R + f'gcm_glc_{b}_lcs1_16-133.nc')]
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
