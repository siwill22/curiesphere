"""
Stage 4 acceptance (docs/inversion-toolbox-spec.md): the samplers FD-vor, FD-lvor and FD-part against the first
build's (notebooks/domains/voronoi.py, lvoronoi.py, partition.py, imported read-only).

The first build's sampler results (voronoi_synthetic.npz, lvoronoi_synthetic*.npz, partition_synthetic.npz,
partition_diagnostics.npz) cannot be reproduced draw for draw: run_voronoi_synthetic.py seeded its chains with Python's
hash() of a string, which changes with every interpreter, and the chains had not converged (two chains on the same
data disagree on K). So the port is checked in two ways:
  same steps    on the first build's whitened data, the old and the new samplers run from the same seed for short
                chains and must pass through identical states (labels, K, log likelihood) and identical annealing
  prior tests   with the likelihood switched off, P(K) must be uniform (FD-vor), P(M) and P(U | M) uniform (FD-lvor),
                and the visit frequencies of all 203 partitions of a 2 x 3 grid must match the exact target (FD-part)
The synthetic suites are rerun separately for comparison (acceptance/stage4_synthetic.py).

Run from the repository root (conda env curiesphere):
    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 15 python acceptance/stage4.py
"""
import sys

import numpy as np

sys.path.insert(0, 'notebooks')
from domains import lvoronoi as old_lv, partition as old_pa, voronoi as old_vo      # noqa: E402

from remit.inversion.methods import lvoronoi as lv, merge, partition as pa, voronoi as vo   # noqa: E402

OLD = 'notebooks/regions/australia/cache'
ok = True


def check(label, value, expected, tol):
    global ok
    good = abs(value - expected) <= tol
    ok &= good
    print(f'  {label:62s} {value:11.5g}  expected {expected:9.4g} +- {tol:g}  {"ok" if good else "FAIL"}', flush=True)


oc, od, E = np.load(f'{OLD}/cells.npz'), np.load(f'{OLD}/data.npz'), np.load(f'{OLD}/errors.npz')
P0 = np.linalg.svd(E['T'].T, full_matrices=False)[0]
sigma = float(np.sqrt(np.mean((od['d_lcs'] - od['d_mf7'])**2)))
Gw = P0.T @ np.load(f'{OLD}/G.npz')['G']/sigma
dw = P0.T @ (od['d_lcs'] - od['b'])/sigma
edges = oc['edges']
temps = list(np.geomspace(1., 200., 3))


def same(a, b):
    return all(np.array_equal(np.asarray(x['label']), np.asarray(y['label'])) for x, y in zip(a, b)) and len(a) == len(b)


print('FD-vor: same steps')
Pn, Po = vo.setup(Gw, oc['clon'], oc['clat'], dw), old_vo.setup(Gw, oc['clon'], oc['clat'], dw)
a, acc_a = vo.sample(Pn, 40, temps, np.random.default_rng(5), k_max=40, k_start=3, thin=1, sweep=10)
b, acc_b = old_vo.sample(Po, 40, temps, np.random.default_rng(5), k_max=40, k_start=3, thin=1, sweep=10)
check('samples with different labels (of 40)', sum(not np.array_equal(x['label'], y['label']) for x, y in zip(a, b)), 0, 0)
check('max |log likelihood difference|', max(abs(x['ll'] - y['ll']) for x, y in zip(a, b)), 0, 1e-8)
ca = vo.anneal(Pn, 6, 2000, np.random.default_rng(6))
cb = old_vo.anneal(Po, 6, 2000, np.random.default_rng(6))
check('annealing: chi2 difference', ca[0] - cb[0], 0, 1e-8)
check('annealing: cells labelled differently', np.sum(ca[2] != cb[2]), 0, 0)

print('FD-lvor: same steps (guided births, boundary penalty)')
Pn = dict(vo.setup(Gw, oc['clon'], oc['clat'], dw), edges=edges, gamma=3.)
Po = dict(old_vo.setup(Gw, oc['clon'], oc['clat'], dw), edges=edges, gamma=3.)
lv.setup_births(Pn), old_lv.setup_births(Po)
mp = np.load(f'{OLD}/merge_points.npz', allow_pickle=True)
sn, un = lv.seeds_from_partition(Pn, mp['fine_label'][3])
so, uo = old_lv.seeds_from_partition(Po, mp['fine_label'][3])
check('seeds from a partition: seeds differing', int(not (np.array_equal(sn, so) and np.array_equal(un, uo))), 0, 0)
a, _ = lv.sample(Pn, 30, temps, np.random.default_rng(7), thin=1, sweep=10, edges=edges, start=(sn, un))
b, _ = old_lv.sample(Po, 30, temps, np.random.default_rng(7), thin=1, sweep=10, edges=edges, start=(so, uo))
check('samples with different labels (of 30)', sum(not np.array_equal(x['label'], y['label']) for x, y in zip(a, b)), 0, 0)
check('max |log likelihood difference|', max(abs(x['ll'] - y['ll']) for x, y in zip(a, b)), 0, 1e-8)

print('FD-part: same steps')
st = old_pa.state(old_pa.setup(Gw, dw, edges), mp['fine_label'][3])
beta, kappa, _ = old_pa.merging_prior(418., 12.5, st['chi2'], len(dw), int(mp['fine_K'][3]))
Pn, Po = pa.setup(Gw, dw, edges, beta, kappa), old_pa.setup(Gw, dw, edges, beta, kappa)
a, _ = pa.sample(Pn, mp['fine_label'][3], 30, temps, np.random.default_rng(8), thin=1, sweep=20)
b, _ = old_pa.sample(Po, mp['fine_label'][3], 30, temps, np.random.default_rng(8), thin=1, sweep=20)
check('samples with different labels (of 30)', sum(not np.array_equal(x['label'], y['label']) for x, y in zip(a, b)), 0, 0)
check('max |log target difference|', max(abs(x['ll'] - y['ll']) for x, y in zip(a, b)), 0, 1e-8)

print('prior tests (likelihood switched off)')
P = vo.setup(Gw, oc['clon'], oc['clat'], dw)
flat = lambda P, s, h=None: (0., 0., np.zeros(len(s)), vo.assign(P, s))
out, _ = vo.sample(P, 4000, [1.], np.random.default_rng(9), k_max=10, k_start=3, thin=2, evaluate_fn=flat)
Ks = np.array([o['K'] for o in out])[500:]
hist = np.bincount(Ks, minlength=11)[1:]/len(Ks)
print(f'  FD-vor P(K = 1..10): {np.round(hist, 3)}')
check('FD-vor: largest |P(K) - 0.1|', np.abs(hist - 0.1).max(), 0, 0.03)
P = dict(vo.setup(Gw, oc['clon'], oc['clat'], dw), edges=edges)
flat = lambda P, s, u, h=None: (0., 0., None, u[vo.assign(P, s)])
out, _ = lv.sample(P, 3000, [1.], np.random.default_rng(10), m_max=8, m_start=3, thin=1, evaluate_fn=flat)
M = np.array([o['M'] for o in out])[500:]
U = np.array([o['U'] for o in out])[500:]
hm = np.bincount(M, minlength=9)[1:]/len(M)
print(f'  FD-lvor P(M = 1..8): {np.round(hm, 3)}')
check('FD-lvor: largest |P(M) - 1/8|', np.abs(hm - 1/8).max(), 0, 0.04)
dev = max(abs(np.mean(U[M == m] == u) - 1/m) for m in range(2, 9) for u in range(1, m + 1) if (M == m).sum() > 200)
check('FD-lvor: largest |P(U | M) - 1/M|', dev, 0, 0.08)
g = np.array([[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]])
exact, freq = pa.partition_prior_test(g, 6, lambda U, L: -0.7*L - 0.5*U, 300000, np.random.default_rng(1))
check('FD-part: total variation to the exact target (first build 0.0163)', 0.5*np.abs(exact - freq).sum(), 0.0163, 5e-5)
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
