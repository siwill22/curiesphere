"""DF-GLC: the evidence and the posterior of one shape against the dense Gaussian formulas"""
import numpy as np

from remit.inversion import graph, parameterise
from remit.inversion.methods import df, df_global


def test_shape_against_dense():
    rng = np.random.default_rng(3)
    N, n, K = 45, 30, 3
    lon, lat = 130 + rng.uniform(-10, 10, n), -25 + rng.uniform(-10, 10, n)
    label = np.repeat(np.arange(K), n//K)
    G = rng.standard_normal((N, n))
    u = G @ np.ones(n)
    y = G @ (0.8 + 0.3*rng.standard_normal(n)) + rng.standard_normal(N)
    xyz = graph.unit(lon, lat)
    window = df.cap_window(1e4)
    beta, ell, s, tau, h = 2.4, 400., 0.35, 0.2, 1.1
    sh = df_global.Shape(G, u, y, label, xyz, window, beta, ell, 0., 4., 0.)
    R = np.zeros((n, n))
    for b in sh.blocks:
        R[np.ix_(b, b)] = sh.R(b)
    S = parameterise.indicator(label)
    Sig = s**2*R + tau**2*S @ S.T
    C = h**2*np.eye(N) + G @ Sig @ G.T
    Ci = np.linalg.inv(C)
    Au = u @ Ci @ u
    v0 = u @ Ci @ y/Au
    r = y - u*v0
    ll = (-0.5*np.linalg.slogdet(C)[1] - 0.5*np.log(Au) - 0.5*r @ Ci @ r - 0.5*(N - 1)*np.log(2*np.pi) - np.log(4.))
    got, v0_got = sh.evidence(s, tau, h)
    assert abs(got - ll) < 1e-8 and abs(v0_got - v0) < 1e-10
    # posterior with v0 integrated out under a flat prior: the GLS limit of v0 ~ N(0, big^2)
    big = 1e4
    Sv = Sig + big**2*np.ones((n, n))
    Cy = C + big**2*np.outer(u, u)
    mean = Sv @ G.T @ np.linalg.solve(Cy, y)
    sd = np.sqrt(np.diag(Sv - Sv @ G.T @ np.linalg.solve(Cy, G @ Sv)))
    post = sh.posterior(dict(s=s, tau=tau, h=h), None, np.ones(N), np.ones(n))
    assert np.allclose(post['v'], mean, atol=1e-6) and np.allclose(post['sd'], sd, atol=1e-5)
