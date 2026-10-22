"""remit.inversion.gmrf: explicit and profiled evidence agree; the posterior against a dense Gaussian calculation."""
import numpy as np

from remit.inversion import core, gmrf, graph


def chain(n=30, m=20, seed=0):
    """cells on a line 50 km apart, a random forward operator and data from a smooth truth"""
    rng = np.random.default_rng(seed)
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], 1)
    Lw = graph.laplacian(edges, np.full(n - 1, 1/50.**2), n)
    Gw = rng.standard_normal((m, n))
    v = 1. + 0.3*np.sin(np.arange(n)/5)
    return Gw, Gw @ v + 0.1*rng.standard_normal(m), np.ones(n), Lw


def test_profiled_evidence_is_the_explicit_maximum():
    Gw, y, a, Lw = chain()
    g = gmrf.GMRF(Gw, y, a, Lw)
    Bd = np.ones((len(a), 1))
    ll, h, s0, alpha2, lr, L = g.best(Bd, 300., box=([0.], [4.]))
    assert np.isclose(g.evidence(Bd, h, s0, 300., ([0.], [4.])), ll, atol=1e-8)
    for dh, ds in ((1.05, 1.), (1., 1.05), (0.95, 0.95)):
        assert g.evidence(Bd, h*dh, s0*ds, 300., ([0.], [4.])) < ll


def test_posterior_against_dense():
    Gw, y, a, Lw = chain(seed=1)
    g = gmrf.GMRF(Gw, y, a, Lw)
    Bd = np.stack([np.ones(len(a)), np.linspace(-1, 1, len(a))], 1)
    fit = g.fit(Bd, length=200.)
    p = g.posterior(Bd, fit, {'mean': np.full(len(a), 1/len(a))})
    h, s = fit['h'], fit['s'][0]
    R = fit['L'] @ fit['L'].T
    C = h**2*np.eye(len(y)) + s**2*Gw @ R @ Gw.T
    Ci = np.linalg.inv(C)
    U = Gw @ Bd
    Ct = np.linalg.inv(U.T @ Ci @ U)
    th = Ct @ U.T @ Ci @ y
    K = s**2*R @ Gw.T
    v = Bd @ th + K @ Ci @ (y - U @ th)
    M = Bd - K @ Ci @ U
    var = s**2*np.diag(R) - np.sum((K @ Ci)*K, 1) + np.sum((M @ Ct)*M, 1)
    assert np.allclose(p['theta'], th, atol=1e-8) and np.allclose(p['v'], v, atol=1e-8)
    assert np.allclose(p['v_sd'], np.sqrt(var), rtol=1e-6)
    assert np.isclose(p['functionals']['mean'][1], np.sqrt(np.full(len(a), 1/len(a)) @ (s**2*R - K @ Ci @ K.T + M @ Ct @ M.T)
                                                            @ np.full(len(a), 1/len(a))), rtol=1e-6)


def test_regression_evidence_box():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((40, 2))
    y = A @ np.array([1., -1.]) + 0.1*rng.standard_normal(40)
    assert np.isfinite(core.regression_evidence(A, y, [-2, -2], [2, 2]))
    assert core.regression_evidence(A, y, [0, 0], [2, 2]) == -np.inf
