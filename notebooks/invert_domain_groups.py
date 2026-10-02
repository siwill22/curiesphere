"""
Merged domains: does coarser grouping make the domain values reliable?

The field is linear in the domain values, so the column of a group is the sum of the columns of
its member domains: no new forward runs. Reuses the caches of invert_domains.py (run it first):
columns, noise, and the T3 truths (lognormal internal variation). For a group, the "true" value
is the area-weighted mean of the truth over the group, so the differences between the member
domains count as internal variation of the group.

Groupings are in inversion/australia_groups.csv (one column per scheme, one row per parameter of
invert_domains.py).

Run from notebooks/ (conda env pygmt17):
    PYTHONPATH=..:$PYTHONPATH python invert_domain_groups.py
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

OUT = 'inversion'
TABLE = f'{OUT}/australia_groups.csv'
T3_SIGMAS, T3_BETAS, N_EPS, N_NOISE_T3 = [0.25, 0.5, 1.0], [2., 3.], 5, 20   # as in invert_domains.py
SIGMA_LN_REAL = {2.: 0.55, 3.: 0.84}            # implied by the 39-domain misfit to LCS-1

z = np.load(f'{OUT}/columns.npz')
params = list(z['params'])
G, b, d_lcs, d_mf7 = z['G'], z['b'], z['d_lcs'], z['d_mf7']
sigma = np.sqrt(np.mean((d_lcs - d_mf7)**2))
noise = np.load(f'{OUT}/noise.npz')['noise']
noise *= sigma/np.sqrt(np.mean(noise**2))
t3 = np.load(f'{OUT}/T3_truths.npz')
y_t3, m_t3 = t3['y'], t3['m']                   # [sigma, beta, eps, point], [sigma, beta, eps, par]
pgrid = np.load(f'{OUT}/param_grid.npz')['pgrid']
glat = np.linspace(90, -90, pgrid.shape[0])
area = np.array([np.cos(np.radians(glat))[np.where(pgrid == k)[0]].sum() for k in range(len(params))])

if not os.path.exists(TABLE):
    groups13 = {
        'NAC': ['Kimberley_Craton', 'Pine_Creek_Inlier', 'Tennant_Creek_Terrane', 'Mount_Isa_Province',
                'MacArthur-Carpentaria_Basin', 'Cape_York_Belt'],
        'WAC': ['Pilbara_Craton', 'Western_Yilgarn_Craton', 'Eastern_Yilgarn_Craton', 'Capricorn_Orogen',
                'Albany-Fraser_Orogen', 'Darling_Orogen'],
        'SAC': ['Gawler_Craton', 'Curnamona_Province', 'Coompana_Province', 'Madura_Province',
                'Adelaide_Rift_Complex'],
        'Central': ['Arunta_Block', 'Muscgrave_Ranges'],
        'Delamerian': ['Delamerian_Orogen', 'North_Delamerian_Orogen', 'Georgetown_and_Coen_Inliers'],
        'Thomson': ['East_Thompson_Orogen', 'West_Thompson_Orogen'],
        'Lachlan': ['Lachlan_Fold_Belt', 'Stalwell_Zone', 'Darling_Basin', 'Wagga-Omeo_Metamorphic_Complex',
                    'Macquarie_Arc', 'Hay-Booligal_Zone'],
        'East': ['New_England_Orogen', 'Sydney-Surat_Basin', 'Mossman_Orogen'],
        'Tasmania': ['Eastern_Tasmania', 'Western_Tasmania']}
    precambrian = ['NAC', 'WAC', 'SAC', 'Central']
    rows = []
    for p in params:
        g13 = p if p.startswith('margin') else next(g for g, members in groups13.items() if p in members)
        g6 = p if p.startswith('margin') else ('Precambrian' if g13 in precambrian else 'Phanerozoic')
        rows.append(dict(param=p, domains39=p, groups13=g13, groups6=g6))
    pd.DataFrame(rows).to_csv(TABLE, index=False)
    print(f'wrote {TABLE}')
table = pd.read_csv(TABLE).set_index('param').loc[params]


def run(scheme):
    names = sorted(set(table[scheme]), key=lambda g: (g.startswith('margin'), g))
    A = np.array([[table[scheme].iloc[k] == g for g in names] for k in range(len(params))], float)
    Gg = G @ A
    Gw = Gg/sigma
    solve = lambda dw: lsq_linear(Gw, dw, bounds=(0., np.inf)).x
    gmean = lambda m: (m*area) @ A/(area @ A)                # area-weighted group mean of domain means

    # T3: bias against the true group mean, per sigma_ln and beta
    rel = np.zeros((len(T3_SIGMAS), len(T3_BETAS), len(names)))
    chi = np.zeros((len(T3_SIGMAS), len(T3_BETAS)))
    for i_s in range(len(T3_SIGMAS)):
        for ib in range(len(T3_BETAS)):
            bias, c = [], []
            for ie in range(N_EPS):
                truth = gmean(m_t3[i_s, ib, ie])
                est = np.array([solve((y_t3[i_s, ib, ie] + noise[j])/sigma) for j in range(N_NOISE_T3)])
                bias.append((est.mean(0) - truth)/truth)
                c += [np.sqrt(np.mean((Gw @ e - (y_t3[i_s, ib, ie] + noise[j])/sigma)**2)) for j, e in enumerate(est)]
            rel[i_s, ib] = np.sqrt(np.mean(np.array(bias)**2, 0))
            chi[i_s, ib] = np.mean(c)

    # real data
    dw = (d_lcs - b)/sigma
    v = solve(dw)
    chi_real = np.sqrt(np.mean((Gw @ v - dw)**2))
    noise_std = np.array([solve((Gg @ v + noise[j])/sigma) for j in range(len(noise))]).std(0)
    model_rel = np.max([[np.interp(SIGMA_LN_REAL[beta], [0.] + T3_SIGMAS, np.r_[0., rel[:, ib, k]])
                         for k in range(len(names))] for ib, beta in enumerate(T3_BETAS)], axis=0)
    err = np.sqrt(noise_std**2 + (model_rel*v)**2)
    out = pd.DataFrame(dict(group=names, n_domains=A.sum(0).astype(int), v=v, noise_std=noise_std,
                            model_rel_err=model_rel, total_err=err, rel_err=err/np.maximum(v, 1e-6)))
    out.to_csv(f'{OUT}/groups_{scheme}_real.csv', index=False)

    print(f'\n=== {scheme}: {len(names)} parameters ===')
    print('  T3 RMS relative bias (median over groups):')
    for i_s, sg in enumerate(T3_SIGMAS):
        print('    sigma_ln {:.2f}: '.format(sg) + ', '.join(
            f'beta {beta:.0f} {np.median(rel[i_s, ib]):.2f} (misfit/sigma {chi[i_s, ib]:.2f})'
            for ib, beta in enumerate(T3_BETAS)))
    print(f'  LCS-1: misfit/sigma {chi_real:.2f}')
    print(out.round(3).to_string(index=False))
    return out


for scheme in ['domains39', 'groups13', 'groups6']:
    run(scheme)
