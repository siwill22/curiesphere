"""
Verification of PD (remit.inversion.methods.pd: one VIS per polygon domain by bounded least squares), Australia local
fit, as df_checks.py for DF. PD-AUS weights every data point by 1/(noise RMS) ('diagonal'), ignoring the noise's
correlations; its s.d. is the formal one or, in final mode, the spread of 100 refits to error-model draws.

P1  correctness: a truth with one level per domain and no noise must come back exactly; the bounded least squares
    against scipy's lsq_linear with some levels at the bound
P2  calibration and model error: 200 truths with a level per domain (N(0.75, 0.3^2), at least 0.05) and three kinds
    of data:
      'white'    errors as PD assumes them (white, unit variance after weighting)
      'noise'    the field model's noise, with its correlations (as on real data)
      'pattern'  a fractal pattern inside each domain as well (s 0.3, beta 2.25, ell 1000 km, as DF's truths), with
                 the noise: the domain mean is then the target
    the errors of the domain means against the formal s.d., the formal s.d. scaled by the misfit (PD's noise =
    'evidence'), and the s.d. from refits (for the first 40 truths, 50 refits each)

Run from the repository root (~15 min, one process):
    PYTHONPATH=benchmarks/verification:benchmarks:notebooks:$PYTHONPATH OMP_NUM_THREADS=1 \
        nice -n 15 conda run -n curiesphere python benchmarks/verification/pd_checks.py p1|p2
Outputs: benchmarks/verification/results/pd_*.json, pd_*.npz
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import lsq_linear

from df_checks import RESULTS, TRUE, log
from remit.inversion import config, parameterise, problem
from remit.inversion.cache import Cache
from remit.inversion.geometry import area_name
from remit.inversion.methods import df
from remit.inversion.methods.pd import bounded_lsq

CONFIG = 'inversions/pd_aus_lcs1_16-133.toml'
N_TRUTHS, N_REFIT_TRUTHS, N_REFITS = 200, 40, 50


def setup():
    cfg = config.load(CONFIG)
    pb = problem.local(cfg, Cache(), log)
    cells = pb.cells
    label, params, _ = parameterise.domains(cells, cfg['parameterisation']['table'], Cache(),
                                            area_name(cfg['target']['area']))
    used = np.unique(label)
    label = np.searchsorted(used, label)
    S = parameterise.indicator(label)
    A = np.asarray(pb.Gw @ S)
    lo, hi = cfg['method'].get('bounds', [0., np.inf])
    c = np.asarray(S.todense() if hasattr(S, 'todense') else S)*cells.area_km2[:, None]
    c /= c.sum(0)
    return dict(cfg=cfg, pb=pb, label=label, names=[params[k] for k in used], S=S, A=A, lo=lo, hi=hi, avg=c,
                area=np.bincount(label, cells.area_km2))


def formal_sd(A, sigma=1.):
    return sigma*np.sqrt(np.diag(np.linalg.inv(A.T @ A)))


def p1():
    P = setup()
    A, lo, hi = P['A'], P['lo'], P['hi']
    rng = np.random.default_rng(20261022)
    out = {}
    theta = TRUE['v0'] + TRUE['tau']*rng.standard_normal(A.shape[1])
    est = bounded_lsq(A, A @ theta, lo, hi)
    out['exact'] = dict(max_abs_error=float(np.abs(est - theta).max()))
    log(f'levels per domain, no noise: max |error| {out["exact"]["max_abs_error"]:.1e} SI km')

    theta_low = 0.1 + 0.3*rng.standard_normal(A.shape[1])          # some levels below zero: the bound is active
    y = A @ theta_low + np.asarray(P['pb'].W @ P['pb'].draw(rng))
    est = bounded_lsq(A, y, lo, hi)
    ref = lsq_linear(A, y, bounds=(lo, hi), method='bvls', tol=1e-14).x
    out['bounded'] = dict(at_bound=int(np.sum(est <= lo + 1e-9)), max_abs_diff=float(np.abs(est - ref).max()),
                          misfit=float(np.sum((y - A @ est)**2)), misfit_ref=float(np.sum((y - A @ ref)**2)))
    log(f'bounded least squares against scipy (bvls): {out["bounded"]["at_bound"]} levels at the bound, max |diff| '
        f'{out["bounded"]["max_abs_diff"]:.1e} SI km, misfit {out["bounded"]["misfit"]:.6f} vs '
        f'{out["bounded"]["misfit_ref"]:.6f}')
    with open(os.path.join(RESULTS, 'pd_p1.json'), 'w') as f:
        json.dump(out, f, indent=1)
    v = P['S'] @ theta
    np.savez(os.path.join(RESULTS, 'pd_p1.npz'), clon=P['pb'].cells.clon, clat=P['pb'].cells.clat, label=P['label'],
             truth=v, v=P['S'] @ bounded_lsq(A, A @ theta, lo, hi))
    log('P1 done')


def p2():
    P = setup()
    pb, A, lo, hi = P['pb'], P['A'], P['lo'], P['hi']
    N, K = A.shape
    W = pb.W
    noise = lambda rng: np.asarray(W @ pb.draw(rng))
    m = df.DomainFractal(pb.Gw, pb.yw, P['label'], pb.cells.clon, pb.cells.clat,
                         4*np.pi*df.R_KM**2/(12*pb.cells.nside**2), lo, 4.)
    _, Ls = m.factors(TRUE['beta'], TRUE['ell'])
    sd_f = formal_sd(A)
    res = {k: dict(err=[], z_formal=[], z_scaled=[], z_refit=[], chi2=[], sd_refit=[]) for k in ('white', 'noise',
                                                                                              'pattern')}
    maps = {}
    for i in range(N_TRUTHS):
        rng = np.random.default_rng(20261023 + i)
        theta = np.maximum(TRUE['v0'] + TRUE['tau']*rng.standard_normal(K), 0.05)   # no level below PD's bound
        pattern = np.zeros(len(P['label']))
        for b, L in zip(m.blocks, Ls):
            pattern[b] = TRUE['s']*(L @ rng.standard_normal(len(b)))
        v_pat = P['S'] @ theta + pattern
        for case in res:
            if case == 'white':
                y, target = A @ theta + rng.standard_normal(N), theta
            elif case == 'noise':
                y, target = A @ theta + noise(rng), theta
            else:
                y, target = np.asarray(pb.Gw @ v_pat) + noise(rng), P['avg'].T @ v_pat
            est = bounded_lsq(A, y, lo, hi)
            chi2 = float(np.sum((y - A @ est)**2))
            r = res[case]
            r['err'].append(est - target)
            r['z_formal'].append((est - target)/sd_f)
            r['z_scaled'].append((est - target)/(sd_f*np.sqrt(chi2/(N - K))))
            r['chi2'].append(chi2/N)
            if i < N_REFIT_TRUTHS:                     # PD's final-mode s.d.: refits to its prediction plus noise
                refits = np.array([bounded_lsq(A, A @ est + noise(rng), lo, hi) for _ in range(N_REFITS)])
                sd_r = refits.std(0)
                r['sd_refit'].append(sd_r)
                r['z_refit'].append((est - target)/sd_r)
            if i == 0:
                maps[case] = dict(truth=v_pat if case == 'pattern' else P['S'] @ theta, v=P['S'] @ est,
                                  target=P['S'] @ target)
        if i % 20 == 0:
            log(f'  truth {i}: ' + ', '.join(f'{k} chi2/N {res[k]["chi2"][-1]:.2f}' for k in res))
    summary = {}
    arrays = dict(clon=pb.cells.clon, clat=pb.cells.clat, label=P['label'], area=P['area'], sd_formal=sd_f)
    for case, r in res.items():
        s = {}
        for k in ('z_formal', 'z_scaled', 'z_refit'):
            z = np.array(r[k])
            arrays[f'{case}_{k}'] = z
            s[k] = dict(sd=float(z.std()), within_1sd=float(np.mean(np.abs(z) < 1)),
                        within_2sd=float(np.mean(np.abs(z) < 2)))
        err = np.array(r['err'])
        arrays[f'{case}_err'] = err
        arrays[f'{case}_sd_refit'] = np.array(r['sd_refit'])
        s['rms_error'] = float(np.sqrt(np.mean(err**2)))
        s['chi2_per_datum'] = float(np.median(r['chi2']))
        s['sd_formal_median'] = float(np.median(sd_f))
        s['sd_refit_median'] = float(np.median(r['sd_refit']))
        summary[case] = s
        for k, v in maps[case].items():
            arrays[f'{case}_map_{k}'] = v
        log(f'{case}: RMS error {s["rms_error"]:.3f} SI km, chi2/N {s["chi2_per_datum"]:.2f}; within 1 s.d.: formal '
            f'{s["z_formal"]["within_1sd"]:.2f}, scaled by the misfit {s["z_scaled"]["within_1sd"]:.2f}, refits '
            f'{s["z_refit"]["within_1sd"]:.2f}; s.d. median formal {s["sd_formal_median"]:.3f}, refits '
            f'{s["sd_refit_median"]:.3f}')
    with open(os.path.join(RESULTS, 'pd_p2.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    np.savez(os.path.join(RESULTS, 'pd_p2.npz'), **arrays)
    log('P2 done')


if __name__ == '__main__':
    {'p1': p1, 'p2': p2}[sys.argv[1]]()
