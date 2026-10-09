"""
Error model for a region (docs/domain-inversion-framework.md, step 2).

1. Reference fit: the old 8-domain free-domain model of merge_domains.py (Australia), with its
   partition carried over to the region's cells (cells new to the region take the label of a
   touching cell). Its residual r = LCS-1 - background - G v (v >= 0 refitted) is used only for
   the *shape* of the model-error covariance, never as a map.
2. Noise-shaped term Cn: shape from the Lowes spectrum of LCS-1 - MF7. For the shape comparison
   sigma_n = RMS(LCS-1 - MF7) at the points (config 'sigma_n_nT' overrides); the saved reference
   amplitudes (sigma_n^2, s^2) are both estimated from the whitened residual, because the
   short-wavelength error is larger than the data noise alone (fine-scale model error).
3. Model error Cm: candidate spectral shapes are compared with the empirical covariance of r
   against distance, with r's covariance modelled as sigma_n^2 Cn(theta) + s^2 Cm(theta):
     'lcs1'   the spectrum of the data model itself (LCS-1; unmodelled magnetisation looks like the field)
     'noise'  the noise spectrum (LCS-1 - MF7)
     'power'  R_l proportional to (l+1)^p, p fitted
4. Compression and whitening: the data are projected onto the leading eigenvectors of
   Cn + Cm (99.9% of the trace), then whitened in that subspace for any s (errors.whitening).

build() does all four for any data model, band and noise spectrum (build_bands.py uses it); without a
reference fit (synthetic-only bands) the model error is left out (s = 0) and only Cn compresses.

Outputs: <region>/cache/errors.npz and <region>/error_model.png.

Run from notebooks/ (conda env pygmt17):
    PYTHONPATH=..:$PYTHONPATH python error_model.py regions/australia/config.toml
"""
import os
import sys
import tomllib

import numpy as np
from scipy.optimize import lsq_linear, minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy_healpix import healpy as hp

from domains import errors, basis
from remit.data.models import load_lcs

KEEP = 0.999


def reference_residual(G, d, b, cells):
    """residual of the old 8-domain model (merge_real.npz) refitted with v >= 0 on these cells"""
    old_c = np.load('inversion/cells.npz')
    old_lab = np.load('inversion/merge_real.npz')['label'][0]
    lab_of = dict(zip(old_c['cell_ids'], old_lab))
    ids, nside = cells['cell_ids'], int(cells['nside'])
    while nside > int(old_c['nside']):                                      # finer cells: their NSIDE 64 parents
        ids = hp.nest2ring(nside//2, hp.ring2nest(nside, ids)//4)
        nside //= 2
    lab =np.array([lab_of.get(cid, -1) for cid in ids])
    assert (lab >= 0).any(), 'no cell of the old partition found'
    edges = cells['edges']
    while (lab < 0).any():
        for i, j in edges:
            if lab[i] < 0 <= lab[j]:
                lab[i] = lab[j]
            elif lab[j] < 0 <= lab[i]:
                lab[j] = lab[i]
    reps, idx = np.unique(lab, return_inverse=True)
    A = np.zeros((len(lab), len(reps)))
    A[np.arange(len(lab)), idx] = 1.
    v = lsq_linear(G @ A, d - b, bounds=(0, np.inf)).x
    return d - b - G @ A @ v, len(reps)


def build(cache, plon, plat, lmin, lmax_fit, R_noise, R_field, sigma_n, noise, r=None, choice=None, F=None,
          plot=None, label='LCS-1 - MF7', s2_compress=None):
    """the error model of one data set; saves <cache>/errors.npz and returns its contents.
    R_noise, R_field: Lowes spectra of the noise and of the data model; r: the reference residual
    (None: no model-error term); F: per-degree filter (default LCS-1); noise: realisations (check a);
    s2_compress: variance (nT^2) of the field-shaped term used for the compression only (default: the
    model error s^2). Synthetic runs with low noise at some degrees need it (the signal's mean square),
    or the compression keeps only the noisy directions and drops the informative ones."""
    l = np.arange(lmax_fit+1)
    c_noise = errors.cov_coeffs(R_noise, lmin, lmax_fit, F)
    shapes = {'lcs1': errors.cov_coeffs(R_field, lmin, lmax_fit, F), 'noise': c_noise}
    Cn = errors.cov_matrix(c_noise, plon, plat)
    if r is not None:
        print(f'reference fit: residual RMS {np.sqrt(np.mean(r**2)):.1f} nT; sigma_n {sigma_n:.1f} nT')
        # 2-3. spectra and the empirical covariance of the residual
        edges_km = np.r_[0., np.arange(12.5, 2000., 25.)]
        mid, emp, n = errors.empirical_cov(r, plon, plat, edges_km)
        theta = mid/6371.
        # the zero-lag bin (variances) carries the mean square; noise term fixed, s^2 fitted by weighted LSQ
        cn_t = errors.cov_function(c_noise, theta)*sigma_n**2

        def fit_s2(cm):
            m = errors.cov_function(cm, theta)
            w = np.sqrt(n)
            s2 = max(np.sum(w**2*m*(emp - cn_t))/np.sum(w**2*m*m), 0.)
            return s2, np.sqrt(np.sum(n*(emp - cn_t - s2*m)**2)/np.sum(n))

        def power_shape(p):
            return errors.cov_coeffs(np.where(l >= 1, (l+1.)**p, 0.), lmin, lmax_fit, F)

        res = minimize_scalar(lambda p: fit_s2(power_shape(p))[1], bounds=(-6, 4), method='bounded')
        shapes['power'] = power_shape(res.x)
        fits = {k: fit_s2(c) for k, c in shapes.items()}
        for k, (s2, rms) in fits.items():
            print(f'  model-error shape {k:6s}' + (f' (p = {res.x:.2f})' if k == 'power' else '') +
                  f': s = {np.sqrt(s2):.1f} nT, misfit of the covariance curve {rms:.1f} nT^2')
        best = min(fits, key=lambda k: fits[k][1])
        choice = choice or best
        print(f'model-error shape used: {choice}')
        s2_fit = fits[choice][0]
        if plot:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(mid[n > 0], emp[n > 0], 'k.', label='residual of the reference fit (binned)')
            ax.plot(mid, cn_t, 'k:', label=f'data noise alone (sigma_n {sigma_n:.1f} nT)')
            for k, c in shapes.items():
                s2, rms = fits[k]
                ax.plot(mid, cn_t + s2*errors.cov_function(c, theta), label=f'noise + model error "{k}" (s {np.sqrt(s2):.0f} nT)')
            ax.axhline(0, color='gray', lw=0.5)
            ax.set(xlabel='distance (km)', ylabel='covariance (nT$^2$)', xlim=(0, 2000),
                   title='Error covariance against distance: reference-fit residual and models')
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plot, dpi=150)
    else:
        choice, s2_fit = choice or 'lcs1', 0.
        print(f'no reference fit: noise only (sigma_n {sigma_n:.1f} nT)')

    # 4. compression and whitening (Cn and Cm normalised to mean variance 1)
    Cm = errors.cov_matrix(shapes[choice], plon, plat)
    s2_c = s2_fit if s2_compress is None else s2_compress
    w, V = np.linalg.eigh(Cn*sigma_n**2 + Cm*s2_c)
    k = np.searchsorted(np.cumsum(w[::-1])/w.sum(), KEEP) + 1
    P = V[:, ::-1][:, :k]                                   # points x k
    Tk, lam = errors.whitening(P.T @ Cn @ P, P.T @ Cm @ P, nugget=1e-6)
    T = Tk @ P.T                                            # k x points
    print(f'compressed to {k} of {len(plon)} dimensions ({KEEP:.1%} of the trace of sigma_n^2 Cn + s^2 Cm, '
          f's {np.sqrt(s2_c):.1f} nT)')

    # checks: (a) whitened noise realisations (random-phase synthesis, independent of the Legendre
    # sums) should have unit variance in every direction for s = 0 (scaled to sigma_n);
    # (b) maximum-likelihood s from the whitened reference residual
    noise = noise*sigma_n/np.sqrt(np.mean(noise**2))
    wn = (T @ noise.T)/np.sqrt(sigma_n**2*(1 + 1e-6))
    var = wn.var(axis=1)
    print(f'check (a): whitened noise variance per direction: median {np.median(var):.2f}, '
          f'5-95% {np.percentile(var, 5):.2f}-{np.percentile(var, 95):.2f} (expect 1, from {noise.shape[0]} realisations)')
    if r is not None:
        s2_ml = errors.ml_model_error(T @ r, lam, sigma_n**2)
        print(f'check (b): model error from the whitened residual (maximum likelihood): s = {np.sqrt(s2_ml):.1f} nT '
              f'(covariance-curve fit {np.sqrt(s2_fit):.1f} nT); chi2/k at that s = '
              f'{np.mean((T @ r)**2/(sigma_n**2 + s2_ml*lam)):.2f}')

        # (c) the same by scale: whitened residual chi2 in groups of directions ordered by lam (small lam:
        # directions where the noise-shaped term dominates, large lam: where model error dominates), with
        # sigma_n fixed and with both amplitudes estimated from the data
        rw = T @ r
        q = np.array_split(np.argsort(lam), 5)
        by_scale = lambda sn2, s2: ', '.join(f'{np.mean(rw[g]**2/(sn2 + s2*lam[g])):.2f}' for g in q)
        print(f'check (c): chi2 per direction by fifths of lam (short -> field-shaped error dominating):')
        print(f'  sigma_n fixed at {sigma_n:.1f} nT, s {np.sqrt(s2_ml):.1f} nT: {by_scale(sigma_n**2, s2_ml)}')
        sn2_free, s2_free = errors.ml_amplitudes(rw, lam, start=(sigma_n, np.sqrt(s2_ml)))
        print(f'  both estimated: sigma_n {np.sqrt(sn2_free):.1f} nT, s {np.sqrt(s2_free):.1f} nT: {by_scale(sn2_free, s2_free)}')
    else:
        sn2_free, s2_free = sigma_n**2, 0.
    out = dict(T=T, lam=lam, sigma_n_ref=sigma_n, sn2=sn2_free, s2=s2_free, shape=choice,
               c_noise=c_noise, c_model=shapes[choice], keep=KEEP, noise_label=label)
    np.savez(f'{cache}/errors.npz', **out)
    print(f'saved: {cache}/errors.npz (T: {T.shape[0]} x {T.shape[1]}; reference amplitudes sigma_n^2, s^2 '
          f'{"both estimated" if r is not None else "noise only"})')
    return out


if __name__ == '__main__':
    cfg_path = sys.argv[1]
    with open(cfg_path, 'rb') as f:
        cfg = tomllib.load(f)
    REG = os.path.dirname(os.path.abspath(cfg_path))
    CACHE = f'{REG}/cache'
    lmin, lmax_fit = cfg['lmin'], cfg['lmax_fit']
    z = np.load(f'{CACHE}/data.npz')
    plon, plat, b, d, d_mf7 = z['plon'], z['plat'], z['b'], z['d_lcs'], z['d_mf7']
    G = np.load(f'{CACHE}/G.npz')['G']
    cells = np.load(f'{CACHE}/cells.npz')
    sigma_n = cfg.get('sigma_n_nT', float(np.sqrt(np.mean((d - d_mf7)**2))))
    r, K = reference_residual(G, d, b, cells)
    print(f'reference fit: {K} domains')
    diff = load_lcs(lmin=lmin, lmax=lmax_fit).coeffs - basis.load_cof('MF7_mod.cof', lmax_fit)
    diff[:, :lmin] = 0
    lcs = load_lcs(lmin=lmin, lmax=lmax_fit).coeffs
    build(CACHE, plon, plat, lmin, lmax_fit, errors.lowes(diff), errors.lowes(lcs), sigma_n,
          np.load(f'{CACHE}/noise.npz')['noise'], r=r, choice=cfg.get('model_error_shape'),
          plot=f'{REG}/error_model.png')
