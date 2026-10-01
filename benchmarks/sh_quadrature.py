"""
Benchmark 2: remit's Gauss coefficients against an independent quadrature of the
potential integral (benchmarks/sh_reference.py), coefficient by coefficient, on the
full 0.1 deg Driscoll-Healy grid.

Both sides evaluate the same discrete quadrature on the same samples, so they should
agree to rounding at every degree and order, for real (not band-limited) inputs.

Cases
  1  closed forms at high degree: radial, poloidal (grad_h S_lm) and toroidal
     (r x grad_h S_lm) VIM for a few (l, m); remit and the reference against the
     analytic values
  2  real inputs on the r0 sphere: GK07 remanent (P = 5) and induced VIS, lmax 400.
     The reference builds the induced VIM independently (fix B).
  3  realistic depth: GK07_NR remanent and induced in the 500 m slices of
     benchmarks/common.py, lmax 185. The reference applies the exact factor
     (r_s/r0)^(l+1) node by node; remit uses its depth series (depth_weighted_coeffs).

Metrics: per degree, sqrt(sum_m dg^2 + dh^2) / sqrt(sum_m g^2 + h^2); and per (l, m),
|difference| / RMS coefficient of that degree.

Run from the repo root (case 3 takes ~15-20 min; results are cached):
    PYTHONPATH=.:notebooks:benchmarks:$PYTHONPATH conda run -n pygmt17 python benchmarks/sh_quadrature.py
Outputs: benchmarks/results/sh_quadrature.csv, benchmarks/results/sh_quadrature.png
"""
import os

import numpy as np
import pandas as pd

from common import (R0, RESULTS, log, cached, LAT, LON, COLAT, VIS, induced_vim, gv, rem_gmm,
                    depth_setup)
from sh_reference import gauss_coeffs, gauss_coeffs_at_radius, _legendre, MU0
from remit.vhtools import GlobalMagnetizationModel

LMAX_SPHERE = 400
LMAX_DEPTH = 185
CLOSED_FORMS = [(150, 77), (185, 185), (400, 250)]
M0 = 1000.


def remit(mr, mt, mp, lmax):
    _, c = GlobalMagnetizationModel(LON, LAT, mr, mt, mp, R0).transform(lmax=lmax)
    return c.coeffs


def reference(mr, mt, mp, lmax):
    return gauss_coeffs(LAT, LON, mr, mt, mp, R0, lmax)


def per_degree(test, ref):
    d = ((test - ref)**2).sum(axis=(0, 2))
    r = (ref**2).sum(axis=(0, 2))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sqrt(d/r)


def per_lm(test, ref):
    rms_l = np.sqrt((ref**2).sum(axis=(0, 2))/(2*np.arange(ref.shape[1]) + 1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.abs(test - ref).max(axis=0)/rms_l[:, None]


rows = []

# ================================================================ case 1: closed forms

log('case 1: closed forms')
PHI = np.radians(LON)[None, :]
for l, m in CLOSED_FORMS:
    S, dth, dph = (np.zeros_like(COLAT) for _ in range(3))
    k = l*(l + 1)//2 + m
    for i in range(1, COLAT.shape[0]):
        p, dp = _legendre(l, COLAT[i, 0])
        S[i] = p[k]*np.cos(m*PHI[0])
        dth[i] = dp[k]*np.cos(m*PHI[0])
        dph[i] = -m*p[k]*np.sin(m*PHI[0])/np.sin(COLAT[i, 0])
    Z = np.zeros_like(S)
    forms = {'radial': ((M0*S, Z, Z), l),
             'poloidal': ((Z, M0*dth, M0*dph), l*(l + 1)),
             'toroidal': ((Z, -M0*dph, M0*dth), 0)}
    for form, (vim, factor) in forms.items():
        expected = MU0*M0*factor/((2*l + 1)*R0)*1e9
        scale = MU0*M0*l/((2*l + 1)*R0)*1e9          # radial value, sets the scale for the toroidal case
        for method, fn in (('remit', remit), ('reference', reference)):
            c = fn(*vim, l)
            err_lm = abs(c[0, l, m] - expected)/scale
            c[0, l, m] = 0
            rows.append(dict(case='1 closed form', part=f'{form} l={l} m={m}', method=method,
                             rel_err_target=err_lm, rel_max_other=np.abs(c).max()/scale))
    log(f'  l={l} m={m} done')

# ================================================================ case 2: real inputs on the sphere

log('case 2: real inputs on the r0 sphere')
ref2 = {'remanent': cached('sh_case2_reference_remanent_400',
                           lambda: reference(rem_gmm.mrad, rem_gmm.mtheta, rem_gmm.mphi, LMAX_SPHERE)),
        'induced': cached('sh_case2_reference_induced_400', lambda: reference(*induced_vim(VIS), LMAX_SPHERE))}
rem2 = {'remanent': cached('case2_remit_remanent_400', lambda: rem_gmm.transform(lmax=400)[1].coeffs),
        'induced': cached('case2_remit_induced_400', lambda: gv.transform(lmax=400)[1].coeffs)}
ref2['combined'] = ref2['remanent'] + ref2['induced']
rem2['combined'] = rem2['remanent'] + rem2['induced']

# ================================================================ case 3: realistic depth

log('case 3: realistic depth, GK07_NR, exact radius factor')
d = depth_setup()


def ref_depth_remanent():
    total = 0.
    for j, (zi, w) in enumerate(d.rem_slices):
        g = d.to_gmm_rem(w)
        total = total + cached(f'sh_case3_reference_remanent_slice{j}_{LMAX_DEPTH}',
                               lambda: gauss_coeffs_at_radius(LAT, LON, g.mrad, g.mtheta, g.mphi,
                                                              (d.r_top - zi)/R0, R0, LMAX_DEPTH))
    return total


def ref_depth_induced():
    total = 0.
    for k in range(len(d.z_ind)):
        vim = induced_vim(d.ind_weight(k)[:-1, :-1])
        rho = d.ind_radius(k)[:-1, :-1]/R0
        total = total + cached(f'sh_case3_reference_induced_slice{k}_{LMAX_DEPTH}',
                               lambda: gauss_coeffs_at_radius(LAT, LON, *vim, rho, R0, LMAX_DEPTH))
    return total


ref3 = {'remanent': ref_depth_remanent(), 'induced': ref_depth_induced()}
L1 = LMAX_DEPTH + 1
rem3 = {part: cached(f'case3_remit_{part}_400', lambda part=part: d.remit_coeffs(part, 400))[:, :L1, :L1]
        for part in ('remanent', 'induced')}
ref3['combined'] = ref3['remanent'] + ref3['induced']
rem3['combined'] = rem3['remanent'] + rem3['induced']

# ================================================================ report

spectra = []
for case, ref, rem in (('2 real, sphere', ref2, rem2), ('3 real, depth', ref3, rem3)):
    for part in ('remanent', 'induced', 'combined'):
        pdg = per_degree(rem[part], ref[part])
        plm = per_lm(rem[part], ref[part])
        rows.append(dict(case=case, part=part, method='remit vs reference',
                         rel_max_per_degree=np.nanmax(pdg[1:]), rel_median_per_degree=np.nanmedian(pdg[1:]),
                         rel_max_per_lm=np.nanmax(plm[1:])))
        spectra += [dict(case=case, part=part, l=l, rel_rms=v) for l, v in enumerate(pdg) if l > 0]
        log(f'  {case}, {part}: per-degree max {np.nanmax(pdg[1:]):.1e}, per-(l,m) max {np.nanmax(plm[1:]):.1e}')

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, 'sh_quadrature.csv'), index=False, float_format='%.3e')
pd.DataFrame(spectra).to_csv(os.path.join(RESULTS, 'sh_quadrature_per_degree.csv'), index=False,
                             float_format='%.3e')
with pd.option_context('display.width', 200, 'display.max_rows', 200):
    print(df.to_string(index=False, float_format='%.2e'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
for ax, (case, ref, rem) in zip(axs[:2], (('2 real, sphere', ref2, rem2), ('3 real, depth', ref3, rem3))):
    for part, colour in zip(('remanent', 'induced', 'combined'), ('tab:blue', 'tab:orange', 'k')):
        pdg = per_degree(rem[part], ref[part])
        ax.semilogy(np.arange(1, len(pdg)), pdg[1:], color=colour, lw=1, label=part)
    ax.set(xlabel='degree l', ylabel='RMS |remit - reference| / RMS reference', ylim=(1e-17, 1e-10),
           title=f'{case}: per degree')
    ax.grid(alpha=0.4, which='both')
    ax.legend(fontsize=8)
plm = per_lm(rem2['combined'], ref2['combined'])
im = axs[2].imshow(np.log10(np.where(np.tril(np.ones_like(plm)) > 0, plm, np.nan) + 1e-20).T, origin='lower',
                   aspect='auto', cmap='viridis', vmin=-17, vmax=-11)
axs[2].set(xlabel='degree l', ylabel='order m', title='2 real, sphere, combined: per (l, m)')
fig.colorbar(im, ax=axs[2], label='log10 |difference| / RMS coefficient of degree l')
fig.savefig(os.path.join(RESULTS, 'sh_quadrature.png'), dpi=150)
log('done')
