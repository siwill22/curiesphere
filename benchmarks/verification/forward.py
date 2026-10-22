"""
Verification of remit's global forward transform (remit.vhtools, and the depth series of remit.depth) against the
direct dipole sum of reference.py, for the test models of models.py and for real geology.

Why the two must agree. Both methods see the same sources: one point dipole per grid node, of moment
VIM r^2 w dlon. The dipole sum adds their fields directly. remit computes the exact spherical-harmonic (Gauss)
coefficients of the same dipoles and stops at degree lmax. So

    remit(lmax) - dipole sum = - (the field of the same sources above degree lmax)

which remit can compute itself by going to a higher lmax (899, the most the 0.1 deg grid samples exactly). Two
checks follow for every model:
  1. remit(lmax) - dipole sum equals remit(lmax) - remit(899), i.e. remit(899) - dipole sum is small (what is
     left above degree 899);
  2. for band-limited models (no field above degree 150) remit(150 or more) - dipole sum is rounding error.

Cases (all on the 0.1 deg grid, sources on the R0 sphere unless stated):
  dipoles, dipoles_bl            seven planted point dipoles; the same low-passed to degree 150
  checker, checker_bl            20 deg lon/lat checkerboard of VIS +-1 SI.km, induced by IGRF-13
  checker_radial                 the same pattern as radial magnetisation of +-40 A
  healpix, healpix_bl            HEALPix NSIDE 4 checkerboard of VIS +-1, induced
  uniform                        VIS 1 everywhere, induced: zero field outside (Runcorn's theorem)
  geology                        GK07 ocean remanence + Hemant & Maus VIS induced (benchmarks/common.py)
  geology_depth                  GK07_NR and H&M with realistic source depth, 1 km slices (remit.depth series,
                                 GK07 and H&M from benchmarks/common.py);
                                 a 2 deg map grid, because every slice is a full set of dipoles

Field maps: Br on a 1 deg grid (0.2 deg for dipoles, 2 deg for geology_depth) at 100 and 300 km above
R0 = 6371 km, all degrees from 1.
Run from the repository root (about 1.5 h at 3 threads; every dipole sum is cached):
    PYTHONPATH=benchmarks/verification:benchmarks:notebooks:$PYTHONPATH OMP_NUM_THREADS=3 \
        nice -n 15 conda run -n curiesphere python benchmarks/verification/forward.py [case ...]
Outputs: benchmarks/verification/results/forward_<case>.nc and forward_summary.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

import models
from reference import R0, DipoleSum, dh_grid, obs_grid, remit_br
from remit.vhtools import GlobalMagnetizationModel

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, RESULTS = os.path.join(HERE, 'cache'), os.path.join(HERE, 'results')
ALTITUDES = [100e3, 300e3]
L_BAND = 150                         # band limit of the *_bl models
L_LIST = [150, 185, 400, 899]        # remit lmax compared; 185 is the production value
L_DEPTH = [150, 185, 400]            # the depth series to 400 (as benchmark 1)
VIM_RADIAL = 50000*models.VIM_PER_VIS   # A: 1 SI.km in a 50 000 nT field
OBS_STEP = dict(dipoles=0.2)         # map grid (deg); a point dipole's field is only a degree or two across


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def cached(name, fn):
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, name + '.npy')
    if not os.path.exists(f):
        t = time.time()
        np.save(f, fn())
        log(f'  {name}: {time.time() - t:.0f} s')
    return np.load(f)


# ---------------------------------------------------------------- the models

G = dh_grid(1800)


def case_model(case):
    """VIM (A) on G for sources on the R0 sphere, and the map that shows the model (name, units, grid)"""
    base = case.removesuffix('_bl')
    extra = {}
    if base == 'dipoles':
        vim, table = models.dipoles(G)
        extra['dipoles'] = pd.DataFrame(table)
        show = ('dipoles', '', np.zeros(G.colat.shape))
    elif base in ('checker', 'healpix', 'uniform'):
        vis = dict(checker=models.checkerboard, healpix=models.healpix_tiles,
                   uniform=lambda g: np.ones(g.colat.shape))[base](G)
        vim = models.induced(G, vis)
        show = ('VIS', 'SI·km', vis)
    elif base == 'checker_radial':
        vis = models.checkerboard(G)
        vim = [VIM_RADIAL*vis, 0*vis, 0*vis]
        show = ('radial VIM', 'A', vim[0])
    elif base == 'geology':
        import common
        ind = common.induced_vim(common.VIS)                         # built independently of remit
        vim = [common.rem_gmm.mrad + ind[0], common.rem_gmm.mtheta + ind[1], common.rem_gmm.mphi + ind[2]]
        extra['remit_gmm'] = common.rem_gmm.add(common.gv)           # remit's own induced VIM
        show = ('radial VIM', 'A', vim[0])
    else:
        raise ValueError(case)
    if case.endswith('_bl'):
        vim = models.band_limited(G, vim, L_BAND)
        show = (f'radial VIM, low-passed to degree {L_BAND - 1}', 'A', vim[0])
    return vim, show, extra


def remit_coeffs(vim, lmax, gmm=None):
    gmm = gmm or GlobalMagnetizationModel(G.lon, G.lat, *vim, R0)
    return gmm.transform(lmax=lmax)[1].coeffs


# ---------------------------------------------------------------- comparisons

rows = []


def summarise(case, dip, rem, L_list, altitudes=ALTITUDES):
    """RMS and maximum of remit(L) - dipole sum, and of the field above L as remit sees it, relative to the RMS of
    the dipole-sum field (absolute values in nT as well)"""
    top = rem[max(L_list)]
    for ia, h in enumerate(altitudes):
        d = dip[ia]
        rms_d = np.sqrt(np.mean(d**2))
        for L in L_list:
            diff = rem[L][ia] - d
            rows.append(dict(case=case, altitude_km=int(h/1e3), lmax=L, rms_dipole_nT=rms_d,
                             max_abs_diff_nT=np.abs(diff).max(), rms_diff_nT=np.sqrt(np.mean(diff**2)),
                             rel_rms_diff=np.sqrt(np.mean(diff**2))/rms_d,
                             rel_rms_above_L=np.sqrt(np.mean((top[ia] - rem[L][ia])**2))/rms_d
                             if L < max(L_list) else np.nan))


def save(case, obs, dip, rem, show, extra, altitudes=ALTITUDES, prefix='forward'):
    os.makedirs(RESULTS, exist_ok=True)
    Ls = sorted(rem)
    coords = dict(altitude_km=[int(h/1e3) for h in altitudes], lat=obs.lat, lon=obs.lon)
    ds = xr.Dataset(dict(
        dipole_br=(('altitude_km', 'lat', 'lon'), dip),
        remit_br=(('lmax', 'altitude_km', 'lat', 'lon'), np.array([rem[L] for L in Ls]))),
        coords=dict(coords, lmax=Ls))
    k = G.n//360                                      # the model map at 0.5 deg
    ds['model'] = (('mlat', 'mlon'), show[2][::k, ::k])
    ds = ds.assign_coords(mlat=G.lat[::k], mlon=G.lon[::k])
    ds['model'].attrs.update(name=show[0], units=show[1])
    if 'dipoles' in extra:
        for c in ('lat', 'lon'):
            ds[f'dipole_{c}'] = ('dipole', extra['dipoles'][c].values)
        ds['dipole_label'] = ('dipole', extra['dipoles'].label.values.astype(str))
        ds['dipole_direction'] = ('dipole', extra['dipoles'].direction.values.astype(str))
    ds.attrs.update(case=case, created=time.strftime('%Y-%m-%d'))
    ds.to_netcdf(os.path.join(RESULTS, f'{prefix}_{case}.nc'))


def run(case):
    log(f'{case}')
    if case == 'geology_depth':
        return run_depth()
    step = OBS_STEP.get(case, 1.0)
    obs = obs_grid(step)
    tag = '' if step == 1.0 else f'_step{step}'
    vim, show, extra = case_model(case)

    def dipole():
        ds = DipoleSum(obs, ALTITUDES)
        ds.add(G, *vim, R0)
        return ds.br()

    dip = cached(f'forward_{case}_dipole{tag}', dipole)
    c = cached(f'forward_{case}_remit_{max(L_LIST)}', lambda: remit_coeffs(vim, max(L_LIST), extra.get('remit_gmm')))
    rem = {L: remit_br(c, obs, ALTITUDES, L) for L in L_LIST}
    summarise(case, dip, rem, L_LIST)
    save(case, obs, dip, rem, show, extra)


def run_depth():
    """GK07_NR remanence and H&M induced VIS with realistic source depth (benchmark 1, case 3), in 1 km slices: the
    same slices for both methods, so the comparison is exact whatever the slicing"""
    import common
    common.SLICE_BENCH = 1000.
    d = common.depth_setup()
    obs = obs_grid(2.0)

    def dipole():
        ds = DipoleSum(obs, ALTITUDES)
        for zi, w in d.rem_slices:
            g = d.to_gmm_rem(w)
            ds.add(G, g.mrad, g.mtheta, g.mphi, d.r_top - zi)
        for k in range(len(d.z_ind)):
            ds.add(G, *common.induced_vim(d.ind_weight(k)[:-1, :-1]), d.ind_radius(k)[:-1, :-1])
        return ds.br()

    def remit_depth(lmax):
        """remit.depth's series (the package, not the copy in notebooks/depth_models.py) over the same slices"""
        from remit import depth
        rem = [(w, np.log((d.r_top_full - zi)/R0)) for zi, w in d.rem_slices]
        ind = [(d.ind_weight(k), np.log(d.ind_radius(k)/R0)) for k in range(len(d.z_ind))]
        out = 0.
        for slices, to_gmm in ((rem, d.to_gmm_rem), (ind, depth.induced_to_gmm(common.vis.lon, common.vis.lat))):
            umax = max(np.nanmax(np.abs(u)) for _, u in slices)
            out = out + depth.depth_weighted_coeffs(slices, to_gmm, lmax, depth.n_terms(umax, lmax)).coeffs
        return out

    dip = cached('forward_geology_depth_dipole', dipole)
    c = cached(f'forward_geology_depth_remit_{max(L_DEPTH)}', lambda: remit_depth(max(L_DEPTH)))
    rem = {L: remit_br(c, obs, ALTITUDES, L) for L in L_DEPTH}
    summarise('geology_depth', dip, rem, L_DEPTH)
    show = ('top of the sources below the ellipsoid (water and sediment)', 'km',
            (d.r_ell_full[:-1, :-1] - d.r_top)/1e3)
    save('geology_depth', obs, dip, rem, show, {})


CASES = ['dipoles', 'dipoles_bl', 'checker', 'checker_bl', 'checker_radial', 'healpix', 'healpix_bl', 'uniform',
         'geology', 'geology_depth']

if __name__ == '__main__':
    todo = sys.argv[1:] or CASES
    for case in todo:
        run(case)
        df = pd.DataFrame(rows)
        f = os.path.join(RESULTS, 'forward_summary.csv')
        if os.path.exists(f):                         # keep the rows of cases not rerun
            old = pd.read_csv(f)
            df = pd.concat([old[~old.case.isin(df.case.unique())], df])
        df.to_csv(f, index=False, float_format='%.3e')
        rows.clear()
    with pd.option_context('display.width', 200, 'display.max_rows', 200):
        print(pd.read_csv(os.path.join(RESULTS, 'forward_summary.csv')).to_string(index=False))
    log('done')
