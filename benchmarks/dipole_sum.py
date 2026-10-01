"""
Benchmark 1: remit's vector-spherical-harmonic forward model against an
independent equivalent-source calculation (a direct sum of point dipoles in
spherical coordinates).

The dipole side shares no code with remit's transform:
    B(x) = mu0/(4 pi) sum_j [3 (m_j . R) R / |R|^5 - m_j / |R|^3],  R = x - s_j
with one dipole per grid node, moment m_j = VIM_j r_s^2 w_j dlon, where w_j are the
Driscoll & Healy (1994) latitude weights (computed here from their formula). Plain
sin(colat) dcolat weights are only accurate to about dcolat^2 and leave a floor of
~1e-5 in the comparison; case 1 is also run with them to show this.
The remanent VIM (remit's model definition) is taken from remit. The induced VIM
is built here from VIS x IGRF, independently of remit.GlobalVIS.vim.

Cases
  0  induced-VIM units: remit's VIM grid vs the one built here (fix B)
  1  exact test: GK07 + VIS on the r0 sphere, magnetisation low-passed to degree
     149 in Cartesian components, so the external field is band-limited to degree
     150 and remit at lmax 150 should match the dipole sum to rounding (with DH
     weights; also run with plain sin(colat) weights)
  2  real inputs on the r0 sphere: GK07 remanent, induced (VIS), combined;
     remit at lmax 150, 300, 400
  3  realistic depth (notebooks/depth_models.py): GK07_NR remanent and induced,
     with 500 m slices identical for both methods; remit's depth_weighted_coeffs at
     lmax 150, 300, 400

Observation points: HEALPix NSIDE 16 (3072), at 100 and 300 km above r0 = 6371 km,
components Br, Btheta, Bphi, and every degree from 1 up (no lmin). For cases 2 and 3
the remit-minus-dipole difference at lmax L is set against the part of remit's own
field in degrees L+1..400 (the truncation, as far as remit can see it).

Run from the repo root (takes about 20-30 min; results are cached):
    PYTHONPATH=.:notebooks:$PYTHONPATH conda run -n pygmt17 python benchmarks/dipole_sum.py
BENCH_NSIDE (default 16) and BENCH_CASES (default 0123) select a denser point set or
a subset of cases, e.g. for maps; the dipole sum costs ~1.5e9 source-point pairs/s.
Outputs: benchmarks/results/dipole_sum.csv, benchmarks/results/dipole_sum.png
"""
import os

import numba
import numpy as np
import pandas as pd
import pyshtools

from common import (R0, MU0, RESULTS, log, cached, unit_vectors, LAT, LON, COLAT, RH, TH, PH, W_DH,
                    VIS, induced_vim, gv, rem_gmm, depth_setup)
from remit.utils.region import generate_healpix_points
from remit.vhtools import GlobalMagnetizationModel

NSIDE = int(os.environ.get('BENCH_NSIDE', '16'))    # HEALPix resolution of the observation points
CASES = os.environ.get('BENCH_CASES', '0123')       # which cases to run
TAG = '' if NSIDE == 16 else f'_n{NSIDE}'           # dipole caches depend on the point set
ALTITUDES = [100e3, 300e3]
LMAX_EXACT = 150
LMAX_SERIES = [150, 300, 400]
COMPONENTS = ['Br', 'Btheta', 'Bphi']


# ================================================================ 1. dipole sum

@numba.njit(parallel=True, cache=True)
def _dipole_sum(obs, src, mom, out):
    """out[i] += sum_j field of dipole mom[j] at src[j], observed at obs[i]
    (Cartesian, SI, without the mu0/4pi factor). Observers are processed in
    blocks so each pass over the sources serves 16 of them."""
    nobs, nsrc, nb = obs.shape[0], src.shape[0], 16
    for b in numba.prange((nobs + nb - 1)//nb):
        i0 = b*nb
        i1 = min(i0 + nb, nobs)
        acc = np.zeros((nb, 3))
        for j in range(nsrc):
            sx, sy, sz = src[j, 0], src[j, 1], src[j, 2]
            mx, my, mz = mom[j, 0], mom[j, 1], mom[j, 2]
            for i in range(i0, i1):
                rx, ry, rz = obs[i, 0] - sx, obs[i, 1] - sy, obs[i, 2] - sz
                inv2 = 1.0/(rx*rx + ry*ry + rz*rz)
                inv3 = inv2*np.sqrt(inv2)
                t = 3.0*(mx*rx + my*ry + mz*rz)*inv2
                acc[i - i0, 0] += (t*rx - mx)*inv3
                acc[i - i0, 1] += (t*ry - my)*inv3
                acc[i - i0, 2] += (t*rz - mz)*inv3
        for i in range(i0, i1):
            out[i, 0] += acc[i - i0, 0]
            out[i, 1] += acc[i - i0, 1]
            out[i, 2] += acc[i - i0, 2]


class DipoleSum:
    """Accumulates the field of gridded VIM layers at the observation points"""

    def __init__(self, obs_xyz, cell=None):
        self.obs = obs_xyz
        self.cell = CELL if cell is None else cell
        self.B = np.zeros_like(obs_xyz)

    def add(self, mr, mt, mp, r_s):
        """VIM components (A) on the DH2 grid, at source radius r_s (m, scalar or grid)"""
        r_s = np.broadcast_to(r_s, mr.shape)
        m = mr[..., None]*RH + mt[..., None]*TH + mp[..., None]*PH
        m *= (r_s**2*self.cell)[..., None]
        keep = np.any(m != 0, axis=-1)
        src = (r_s[..., None]*RH)[keep]
        _dipole_sum(self.obs, np.ascontiguousarray(src), np.ascontiguousarray(m[keep]), self.B)

    def components(self):
        """(r, theta, phi) components in nT at each observation point"""
        B = self.B*MU0/(4*np.pi)*1e9
        return np.stack([(B*OBS_RH).sum(1), (B*OBS_TH).sum(1), (B*OBS_PH).sum(1)], 1)


def remit_components(coeffs, lmax):
    """remit Gauss coefficients (r0 = R0), all degrees 1..lmax, at the observation points"""
    c = pyshtools.SHMagCoeffs.from_array(np.array(coeffs[:, :lmax+1, :lmax+1]), r0=R0)
    return np.concatenate([np.asarray(c.expand(a=R0+h, lat=OBS_LAT, lon=OBS_LON)) for h in ALTITUDES])


# ================================================================ 2. grids and points

# solid angle per node: Driscoll & Healy (1994) latitude weights (common.dh_weights)
CELL_DH = W_DH[:, None]*np.radians(360/3600)*np.ones((1, 3600))
CELL_SIN = np.sin(COLAT)*np.radians(180/1800)*np.radians(360/3600)
CELL = CELL_DH

olon, olat = generate_healpix_points(NSIDE)
OBS_LAT, OBS_LON = olat, np.mod(olon, 360)
ocol, olonr = np.radians(90 - OBS_LAT), np.radians(OBS_LON)
o_rh, o_th, o_ph = unit_vectors(ocol, olonr)
OBS_XYZ = np.concatenate([(R0 + h)*o_rh for h in ALTITUDES])
OBS_RH, OBS_TH, OBS_PH = [np.concatenate([u]*len(ALTITUDES)) for u in (o_rh, o_th, o_ph)]
NOBS = len(OBS_LAT)
log(f'{NOBS} HEALPix points x {len(ALTITUDES)} altitudes; source grid {COLAT.shape}')

rows = []


def compare(case, part, lmax, dip, rem, tail=None):
    for ia, h in enumerate(ALTITUDES):
        s = slice(ia*NOBS, (ia+1)*NOBS)
        for ic, comp in enumerate(COMPONENTS):
            d, r = dip[s, ic], rem[s, ic]
            row = dict(case=case, part=part, altitude_km=int(h/1e3), lmax=lmax, component=comp,
                       rms_dipole_nT=np.sqrt(np.mean(d**2)),
                       rel_rms_diff=np.sqrt(np.mean((r-d)**2))/np.sqrt(np.mean(d**2)),
                       rel_max_diff=np.abs(r-d).max()/np.sqrt(np.mean(d**2)))
            if tail is not None:
                row['rel_rms_truncation'] = np.sqrt(np.mean(tail[s, ic]**2))/np.sqrt(np.mean(d**2))
            rows.append(row)


# ================================================================ case 0: induced units

mine = induced_vim(VIS)
rel = max(np.abs(a - b).max()/np.abs(b).max() for a, b in zip((gv.mrad, gv.mtheta, gv.mphi), mine))
log(f'case 0: induced VIM, remit vs independent: max relative difference {rel:.1e}')
rows.append(dict(case='0 induced units', part='VIM grid', component='all', rel_max_diff=rel))

# ================================================================ case 1: exact band-limited test

log('case 1: band-limited GK07 + VIS on the r0 sphere')
mr, mt, mp = rem_gmm.mrad + gv.mrad, rem_gmm.mtheta + gv.mtheta, rem_gmm.mphi + gv.mphi
if '1' in CASES:
    mxyz = [mr*RH[..., k] + mt*TH[..., k] + mp*PH[..., k] for k in range(3)]
    low = [pyshtools.expand.MakeGridDH(pyshtools.expand.SHExpandDH(m, sampling=2, lmax_calc=LMAX_EXACT-1),
                                       lmax=899, sampling=2) for m in mxyz]
    bl = [sum(low[k]*U[..., k] for k in range(3)) for U in (RH, TH, PH)]


    def dip_case1(cell):
        ds = DipoleSum(OBS_XYZ, cell)
        ds.add(*bl, R0)
        return ds.components()


    def rem_case1():
        _, c = GlobalMagnetizationModel(LON, LAT, *bl, R0).transform(lmax=LMAX_EXACT)
        return c.coeffs


    rem1 = remit_components(cached('case1_remit', rem_case1), LMAX_EXACT)
    compare('1 exact (band-limited)', 'GK07+VIS', LMAX_EXACT,
            cached('case1_dipole_dhweights' + TAG, lambda: dip_case1(CELL_DH)), rem1)
    compare('1 exact, sin(colat) weights', 'GK07+VIS', LMAX_EXACT,
            cached('case1_dipole_sinweights' + TAG, lambda: dip_case1(CELL_SIN)), rem1)


# ================================================================ case 2: real inputs on the sphere

log('case 2: real inputs on the r0 sphere')


def dip_case2_rem():
    ds = DipoleSum(OBS_XYZ)
    ds.add(rem_gmm.mrad, rem_gmm.mtheta, rem_gmm.mphi, R0)
    return ds.components()


def dip_case2_ind():
    ds = DipoleSum(OBS_XYZ)
    ds.add(*induced_vim(VIS), R0)
    return ds.components()


if '2' in CASES:
    dip2 = {'remanent': cached('case2_dipole_remanent' + TAG, dip_case2_rem),
            'induced': cached('case2_dipole_induced' + TAG, dip_case2_ind)}
    dip2['combined'] = dip2['remanent'] + dip2['induced']
    rem2 = {'remanent': cached('case2_remit_remanent_400', lambda: rem_gmm.transform(lmax=400)[1].coeffs),
            'induced': cached('case2_remit_induced_400', lambda: gv.transform(lmax=400)[1].coeffs)}
    rem2['combined'] = rem2['remanent'] + rem2['induced']
    for part in dip2:
        full = remit_components(rem2[part], 400)
        for L in LMAX_SERIES:
            at_L = remit_components(rem2[part], L)
            compare('2 real, sphere', part, L, dip2[part], at_L, tail=full - at_L)


# ================================================================ case 3: realistic depth

if '3' in CASES:
    log('case 3: realistic depth, GK07_NR, 500 m slices')
    d = depth_setup()


    def dip_case3_rem():
        ds = DipoleSum(OBS_XYZ)
        for zi, w in d.rem_slices:
            g = d.to_gmm_rem(w)
            ds.add(g.mrad, g.mtheta, g.mphi, d.r_top - zi)
        return ds.components()


    def dip_case3_ind():
        ds = DipoleSum(OBS_XYZ)
        for k in range(len(d.z_ind)):
            ds.add(*induced_vim(d.ind_weight(k)[:-1, :-1]), d.ind_radius(k)[:-1, :-1])
        return ds.components()


    dip3 = {'remanent': cached('case3_dipole_remanent' + TAG, dip_case3_rem),
            'induced': cached('case3_dipole_induced' + TAG, dip_case3_ind)}
    dip3['combined'] = dip3['remanent'] + dip3['induced']
    rem3 = {part: cached(f'case3_remit_{part}_400', lambda part=part: d.remit_coeffs(part, 400))
            for part in ('remanent', 'induced')}
    rem3['combined'] = rem3['remanent'] + rem3['induced']
    # the depth series is summed per lmax, so check that truncating the lmax-400 result
    # equals a separate run at lower lmax
    rem3_150 = cached('case3_remit_remanent_150', lambda: d.remit_coeffs('remanent', 150))
    series_check = np.abs(rem3_150 - rem3['remanent'][:, :151, :151]).max()/np.abs(rem3_150).max()
    log(f'  depth series: lmax-150 run vs truncated lmax-400 run, max relative difference {series_check:.1e}')
    for part in dip3:
        full = remit_components(rem3[part], 400)
        for L in LMAX_SERIES:
            at_L = remit_components(rem3[part], L)
            compare('3 real, depth', part, L, dip3[part], at_L, tail=full - at_L)


# ================================================================ report

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS, f'dipole_sum{TAG}.csv'), index=False, float_format='%.3e')
with pd.option_context('display.width', 200, 'display.max_rows', 200):
    print(df.to_string(index=False, float_format='%.2e'))

if NSIDE != 16 or CASES != '0123':
    log('done (no summary figure for a partial run)')
    raise SystemExit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, constrained_layout=True)
for i, case in enumerate(['2 real, sphere', '3 real, depth']):
    for j, h in enumerate(ALTITUDES):
        ax = axs[j, i]
        sub = df[(df.case == case) & (df.altitude_km == int(h/1e3)) & (df.component == 'Br')]
        for part, colour in zip(['remanent', 'induced', 'combined'], ['tab:blue', 'tab:orange', 'k']):
            s = sub[sub.part == part]
            ax.semilogy(s.lmax, s.rel_rms_diff, 'o-', color=colour, label=f'{part}: remit - dipole sum')
            t = s[s.lmax < max(LMAX_SERIES)]           # zero by construction at the top lmax
            ax.semilogy(t.lmax, t.rel_rms_truncation, 's--', color=colour, mfc='none', ms=9,
                        label=f'{part}: remit degrees lmax+1..400')
        exact = df[(df.case == '1 exact (band-limited)') & (df.altitude_km == int(h/1e3)) & (df.component == 'Br')]
        if i == 0:
            ax.semilogy(exact.lmax, exact.rel_rms_diff, 'r*', ms=12, label='band-limited exact test')
        ax.set_title(f'{case}, {int(h/1e3)} km, Br', loc='left')
        ax.grid(alpha=0.4, which='both')
        ax.set_xlabel('remit lmax')
        ax.set_ylabel('RMS difference / RMS dipole-sum field')
axs[0, 0].legend(fontsize=7)
fig.savefig(os.path.join(RESULTS, 'dipole_sum.png'), dpi=150)
log('done')
