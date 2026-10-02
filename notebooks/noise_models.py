"""
How large is the LCS-1 error? Compare four lithospheric field models over the Australian data
points of invert_domains.py, and split the differences into per-model errors.

Models (Br at r0 = 6371.0 km; degree 16 up to the band limit):
  LCS-1   CHAMP + Swarm (remit/data/shc/LCS_mod.cof, r0 taken as 6371.0 km as in the notebooks)
  MF7     CHAMP only (remit/data/shc/MF7_mod.cof, same convention)
  MLI_2D  Swarm dedicated lithospheric inversion, Thebault (ESA MLI_SHA_2D 0501, degrees 16-133)
  MLI_2C  Swarm comprehensive inversion, Sabaka (ESA MLI_SHA_2C 1101, degrees 16-120)
The two MLI files are in the .shc format with the conventional reference radius 6371.2 km; their
coefficients are converted to 6371.0 km.

Three-cornered hat: if three models A, B, C have independent errors, then
    sigma_A^2 = (D_AB^2 + D_AC^2 - D_BC^2) / 2,   D_XY = RMS(X - Y)
LCS-1 shares CHAMP data with MF7 and Swarm data with the MLI models, so its errors are positively
correlated with both. Shared errors cancel in the differences, so the hat underestimates the
LCS-1 error; MF7 - MLI uses independent data (CHAMP against Swarm).

Run from notebooks/ (conda env pygmt17):
    PYTHONPATH=..:$PYTHONPATH python noise_models.py
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd
import pyshtools
import pygmt
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from remit.data.models import DATA_DIR

OUT = 'inversion'
MLI_DIR = os.path.expanduser('~/Data/Swarm/MLI')
R0 = 6371.0e3
LMIN = 16


def read_shc(path, r_file=6371.2e3):
    """.shc file (one epoch, lines 'n m value', m < 0 for h) -> array (2, L+1, L+1) at r0 = R0"""
    rows = [l.split() for l in open(path) if not l.startswith('#')]
    nmin, nmax = int(rows[0][0]), int(rows[0][1])
    c = np.zeros((2, nmax+1, nmax+1))
    for n, m, val in rows[2:]:
        n, m = int(n), int(m)
        c[0 if m >= 0 else 1, n, abs(m)] = float(val)
    l = np.arange(nmax+1)
    return c*((r_file/R0)**(l+2))[None, :, None]           # same potential, referenced to R0


def read_cof(name, lmax):
    clm, _ = pyshtools.shio.shread(f'{DATA_DIR}/shc/{name}', lmax=lmax)
    return clm


def pad(c, L):
    out = np.zeros((2, L+1, L+1))
    k = min(L, c.shape[1]-1)
    out[:, :k+1, :k+1] = c[:, :k+1, :k+1]
    out[:, :LMIN] = 0
    return out


models = {'LCS-1': read_cof('LCS_mod.cof', 185), 'MF7': read_cof('MF7_mod.cof', 133),
          'MLI_2D': read_shc(f'{MLI_DIR}/SW_OPER_MLI_SHA_2D_00000000T000000_99999999T999999_0501.shc'),
          'MLI_2C': read_shc(f'{MLI_DIR}/SW_OPER_MLI_SHA_2C_00000000T000000_99999999T999999_1101.shc')}
print({k: v.shape[1]-1 for k, v in models.items()}, '(max degree)')

z = np.load(f'{OUT}/cell_columns.npz')
plon, plat = z['plon'], z['plat']


def br_points(c, L):
    return np.asarray(pyshtools.SHMagCoeffs.from_array(pad(c, L), r0=R0).expand(
        a=R0, lat=plat, lon=plon, lmax_calc=L))[:, 0]


def lowes(c):
    l = np.arange(c.shape[1])
    return (l+1)*(c**2).sum((0, 2))


rows = []
for L, names in [(133, ['LCS-1', 'MF7', 'MLI_2D']), (120, ['LCS-1', 'MF7', 'MLI_2D', 'MLI_2C'])]:
    br = {k: br_points(models[k], L) for k in names}
    D = {(a, b): np.sqrt(np.mean((br[a] - br[b])**2)) for a, b in combinations(names, 2)}
    print(f'\nDegrees {LMIN}-{L}, Australian data points ({len(plon)}): RMS Br')
    for k in names:
        print(f'  {k:7s} {np.sqrt(np.mean(br[k]**2)):6.2f} nT')
    for (a, b), d in D.items():
        print(f'  {a} - {b}: {d:6.2f} nT')
        rows.append(dict(band=f'{LMIN}-{L}', kind='difference', models=f'{a} - {b}', rms_nT=d))
    for trio in combinations(names, 3):
        for x in trio:
            y, w = [t for t in trio if t != x]
            key = lambda p, q: D[(p, q)] if (p, q) in D else D[(q, p)]
            s2 = (key(x, y)**2 + key(x, w)**2 - key(y, w)**2)/2
            rows.append(dict(band=f'{LMIN}-{L}', kind='three-cornered hat', models=f'{x} in {trio}',
                             rms_nT=np.sqrt(s2) if s2 > 0 else np.nan, sigma2=s2))
        print('  hat ' + str(trio) + ': ' + ', '.join(
            f'{r["models"].split(" in ")[0]} {r["rms_nT"]:.2f}' for r in rows[-3:]))
tab = pd.DataFrame(rows)
tab.to_csv(f'{OUT}/noise_models.csv', index=False)

# per-degree (global) Lowes spectra of the differences
fig, ax = plt.subplots(figsize=(8, 4.5))
ref = lowes(pad(models['LCS-1'], 133))
ax.semilogy(np.arange(134), ref, 'k', lw=2, label='LCS-1 itself')
for a, b in [('LCS-1', 'MF7'), ('LCS-1', 'MLI_2D'), ('MF7', 'MLI_2D'), ('LCS-1', 'MLI_2C'), ('MLI_2D', 'MLI_2C')]:
    L = 120 if 'MLI_2C' in (a, b) else 133
    ax.semilogy(np.arange(L+1)[LMIN:], lowes(pad(models[a], L) - pad(models[b], L))[LMIN:], label=f'{a} - {b}')
ax.set(xlim=(LMIN, 133), xlabel='degree l', ylabel='Lowes power at r = 6371 km (nT$^2$)',
       title='Global spectra of model differences')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f'{OUT}/noise_spectra.png', dpi=150)
print('\nwrote', f'{OUT}/noise_models.csv', f'{OUT}/noise_spectra.png')

# maps of the pairwise differences over the Australian data points, degrees 16-133
br = {k: br_points(models[k], 133) for k in ['LCS-1', 'MF7', 'MLI_2D']}
fig = pygmt.Figure()
pygmt.config(FONT_TITLE='10p', FONT_ANNOT_PRIMARY='7p')
pygmt.makecpt(cmap='polar', series=[-60, 60], background=True)
for n, (a, b) in enumerate([('LCS-1', 'MF7'), ('LCS-1', 'MLI_2D'), ('MF7', 'MLI_2D')]):
    if n:
        fig.shift_origin(xshift='12c')
    d = br[a] - br[b]
    fig.plot(x=plon, y=plat, style='c0.075c', fill=d, cmap=True, region=[95, 160, -50, -5], projection='M11c',
             frame=['xa10f5', 'ya10f5', f'WSne+t{a} - {b}: RMS {np.sqrt(np.mean(d**2)):.1f} nT'])
    fig.coast(shorelines='0.3p,gray40', resolution='l', area_thresh=2000)
fig.colorbar(frame='xa20f10+lBr difference (nT), r = 6371 km, degrees 16-133', position='JBC+w10c+o0c/1c')
fig.savefig(f'{OUT}/noise_difference_maps.png', dpi=200)
print('wrote', f'{OUT}/noise_difference_maps.png')
