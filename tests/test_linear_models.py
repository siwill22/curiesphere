"""
Tests for the shared linear-model code of the inversions (notebooks/domains: linear, graph, gmrf, bounds).

Run with:  conda run -n pygmt17 python -m pytest tests
"""
import os
import sys

import numpy as np
from scipy.stats import multivariate_normal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))
from domains import graph, gmrf, linear

# a 2 x 3 grid of cells about 100 km apart
CLON = np.array([130., 131., 132., 130., 131., 132.])
CLAT = np.array([-20., -20., -20., -21., -21., -21.])
EDGES = np.array([(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)])
A = np.ones(6)


def toy(N=9, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, 6)), rng


def test_difference_operator():
    D = graph.difference_operator(EDGES, CLON, CLAT)
    assert np.allclose(D @ np.ones(6), 0)
    d = graph.edge_lengths_km(EDGES, CLON, CLAT)
    assert np.allclose((D.T @ D).toarray(), graph.laplacian(EDGES, 1/d**2))


def test_evidence_against_monte_carlo():
    """log p(dw) of v = 1 theta + u, theta ~ U[-10, 14] (wide, so the box cuts off no posterior mass),
    u ~ GMRF, by Monte Carlo over theta and u"""
    G, rng = toy()
    d = graph.edge_lengths_km(EDGES, CLON, CLAT)
    m = gmrf.Model(G, np.zeros(9), A, graph.laplacian(EDGES, 1/d**2))
    h, s, ell = 2., 0.8, 150.          # broad likelihood, so plain Monte Carlo converges
    Su = m.prior_cov(s, ell)
    dw = G @ (1.7 + rng.multivariate_normal(np.zeros(6), Su)) + h*rng.standard_normal(9)
    ll = m.evidence(np.ones((6, 1)), -10., 14., h, s, ell, dw=dw)[0]
    n = 1000000
    theta = rng.uniform(-10, 14, n)
    u = rng.multivariate_normal(np.zeros(6), Su, n)
    r = dw[None, :] - (theta[:, None] + u) @ G.T
    logl = -0.5*np.sum(r**2, 1)/h**2 - 9*np.log(np.sqrt(2*np.pi)*h)
    mc = np.log(np.mean(np.exp(logl - logl.max()))) + logl.max()
    assert abs(ll - mc) < 0.02


def test_evidence_matches_integral_with_C():
    """with u integrated in C, the theta integral is a 1-D Gaussian one: check with quadrature"""
    G, rng = toy(seed=1)
    d = graph.edge_lengths_km(EDGES, CLON, CLAT)
    m = gmrf.Model(G, np.zeros(9), A, graph.laplacian(EDGES, 1/d**2))
    C = m.cov(0.7, 0.3, 200.)
    dw = G @ np.full(6, 1.2) + rng.multivariate_normal(np.zeros(9), C)
    ll = linear.log_evidence(G @ np.ones((6, 1)), dw, 0., 4., h=1., C=C)[0]
    th = np.linspace(-20, 20, 40001)
    p = np.array([multivariate_normal(G @ np.full(6, t), C).logpdf(dw) for t in th[::40]])
    q = np.log((getattr(np, 'trapezoid', None) or np.trapz)(np.exp(p - p.max()), th[::40])) + p.max() - np.log(4.)
    assert abs(ll - q) < 1e-6


def test_map_is_least_squares_without_penalty():
    G, rng = toy(N=12)
    dw = rng.standard_normal(12)
    v = gmrf.map_estimate(G, dw, np.zeros((0, 6)), np.zeros(6), lo=-np.inf)
    assert np.allclose(v, np.linalg.lstsq(G, dw, rcond=None)[0], atol=1e-8)


def test_posterior_mean_is_map_at_theta_hat():
    G, rng = toy(seed=2)
    d = graph.edge_lengths_km(EDGES, CLON, CLAT)
    m = gmrf.Model(G, rng.standard_normal(9) + 3, A, graph.laplacian(EDGES, 1/d**2))
    h, s, ell = 0.6, 0.5, 120.
    B = np.ones((6, 1))
    post = m.posterior(B, h, s, ell)
    v = gmrf.map_estimate(G, m.dw, m.prior_rows(s, ell), B @ post['theta'], h=h, lo=-np.inf)
    assert np.allclose(v, post['v'], atol=1e-8)
    assert np.allclose(np.linalg.inv(m.prior_rows(s, ell).T @ m.prior_rows(s, ell)), m.prior_cov(s, ell))


def test_extremes_on_identity():
    """G = I: min/max of v_0 with |v - d|^2 <= t are d_0 -/+ sqrt(t) (inside the box)"""
    from domains import bounds
    d = np.array([1.5, 2.0, 0.7, 3.1])
    c = np.array([1., 0, 0, 0])
    out = bounds.extremes(np.eye(4), d, c, 0.25, lo=0., hi=4.)
    assert abs(out['min']['value'] - 1.0) < 1e-4 and abs(out['max']['value'] - 2.0) < 1e-4
    # with the lower bound active
    out = bounds.extremes(np.eye(4), d, c, 4.0, lo=0., hi=4.)
    assert abs(out['min']['value']) < 1e-6 and abs(out['max']['value'] - 3.5) < 1e-4


def test_class_covariance():
    G, rng = toy(seed=3)
    d = graph.edge_lengths_km(EDGES, CLON, CLAT)
    cls = np.array([0, 0, 1, 1, 0, 1])
    m = gmrf.Model(G, np.zeros(9), A, graph.laplacian(EDGES, 1/d**2), cls)
    s, ell = np.array([0.3, 0.7]), np.array([100., 400.])
    Su = m.prior_cov(s, ell)
    assert np.allclose(m.cov(0.5, s, ell), G @ Su @ G.T + 0.25*np.eye(9))
    assert np.allclose(np.sqrt(np.diag(Su)), s[cls])


def test_box_qp_against_bvls():
    from scipy.optimize import lsq_linear
    rng = np.random.default_rng(4)
    A = rng.standard_normal((40, 30)) @ np.diag(np.geomspace(1, 1e-3, 30))   # ill-conditioned
    b = rng.standard_normal(40) + 2
    ref = lsq_linear(A, b, bounds=(0., 3.), method='bvls').x
    v = linear.box_qp(A.T @ A, A.T @ b, 0., 3.)
    assert ((ref < 1e-9) | (ref > 3 - 1e-9)).sum() >= 3              # some bounds active
    assert np.allclose(v, ref, atol=1e-6)


def test_posterior_functional_sd():
    """sd of c'v from posterior() equals sqrt(c' Cov(v) c) built from the joint Gaussian (wide theta prior)"""
    G, rng = toy(seed=5)
    d = graph.edge_lengths_km(EDGES, CLON, CLAT)
    m = gmrf.Model(G, rng.standard_normal(9) + 2, A, graph.laplacian(EDGES, 1/d**2))
    h, s, ell = 0.8, 0.4, 150.
    B = np.ones((6, 1))
    c = np.array([1., 0, 0, 1, 0, 0])/2
    post = m.posterior(B, h, s, ell, functionals={'c': c})
    tau = 1e4                                                    # theta ~ N(0, tau^2): flat in the limit
    Pv = B @ B.T*tau**2 + m.prior_cov(s, ell)
    Cv = Pv - Pv @ G.T @ np.linalg.solve(G @ Pv @ G.T + h**2*np.eye(9), G @ Pv)
    assert np.isclose(post['functionals']['c'][1], np.sqrt(c @ Cv @ c), rtol=1e-4)
    assert np.allclose(post['v_sd'], np.sqrt(np.diag(Cv)), rtol=1e-4)


def test_sparse_plus_low_rank_box_qp():
    """box_qp and map_estimate with H = sparse + low rank (Woodbury) equal the dense versions, bounds active"""
    from scipy.sparse import random as sprandom, identity, vstack
    rng = np.random.default_rng(3)
    n, k = 60, 8
    R = vstack([sprandom(30, n, density=0.05, random_state=4), identity(n)]).tocsr()
    A = rng.standard_normal((k, n))
    dw = rng.standard_normal(k)*3
    v0 = np.full(n, 0.5)
    H = linear.SparsePlusLowRank(R.T @ R, A)
    Hd = (R.T @ R).toarray() + A.T @ A
    g = A.T @ dw + (R.T @ R) @ v0
    assert np.allclose(H @ v0, Hd @ v0) and np.allclose(H.diag, np.diag(Hd))
    F = rng.random(n) < 0.7
    r = rng.standard_normal(F.sum())
    assert np.allclose(H.solver(F)(r), np.linalg.solve(Hd[np.ix_(F, F)], r))
    v_s, v_d = linear.box_qp(H, g, 0., 1.), linear.box_qp(Hd, g, 0., 1.)
    assert ((v_d <= 1e-9) | (v_d >= 1 - 1e-9)).sum() > 3          # some bounds active
    assert np.allclose(v_s, v_d, atol=1e-8)
    # map_estimate: sparse rows (structured path, k < n/3) against the same rows dense
    assert np.allclose(gmrf.map_estimate(A, dw, R, v0, hi=1.), gmrf.map_estimate(A, dw, R.toarray(), v0, hi=1.), atol=1e-8)


def test_two_scale_posterior():
    """two-scale GP: the posterior equals the joint Gaussian with Cov(u) = s1^2 R1 + s2^2 R2; its broad and
    detail parts add up to the mean; with s2 -> 0 the evidence is the one-scale evidence"""
    G, rng = toy(seed=7)
    d = graph.edge_lengths_km(EDGES, CLON, CLAT)
    m = gmrf.Model(G, rng.standard_normal(9) + 2, A, graph.laplacian(EDGES, 1/d**2))
    B, lo, hi = np.ones((6, 1)), np.array([-50.]), np.array([50.])
    fit = dict(h=0.7, s1=0.5, ell1=400., s2=0.3, ell2=60.)
    post = m.posterior2(B, fit)
    tau = 1e4
    Pv = B @ B.T*tau**2 + m.prior_cov2(0.5, 400., 0.3, 60.)
    mean = Pv @ G.T @ np.linalg.solve(G @ Pv @ G.T + 0.49*np.eye(9), m.dw)
    Cv = Pv - Pv @ G.T @ np.linalg.solve(G @ Pv @ G.T + 0.49*np.eye(9), G @ Pv)
    assert np.allclose(post['v'], mean, atol=1e-4)
    assert np.allclose(post['v_sd'], np.sqrt(np.diag(Cv)), rtol=1e-4)
    assert np.allclose(post['v_broad'] + post['v_detail'], post['v'])
    one = m.evidence(B, lo, hi, 0.7, 0.5, 400.)[0]
    assert np.isclose(m.evidence2(B, lo, hi, 0.7, 0.5, 400., 1e-6, 60.)[0], one, atol=1e-8)
