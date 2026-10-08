"""remit.inversion.core: the eigh and Cholesky routes agree, and match a direct solve of the normal equations."""
import numpy as np
import pytest

from remit.inversion import core


@pytest.mark.parametrize('shape', [(40, 60), (60, 40)])          # data side and cell side
def test_routes_agree(shape):
    rng = np.random.default_rng(3)
    B = rng.standard_normal(shape)
    u = B @ np.ones(shape[1]) + rng.standard_normal(shape[0])
    y = 0.7*u + B @ rng.standard_normal(shape[1]) + rng.standard_normal(shape[0])
    e, c = core.LevelRidge(B, u, 'eigh'), core.LevelRidge(B, u, 'cholesky')
    for alpha2 in (0.1, 3.):
        se, sc = e.solve(y, alpha2), c.solve(y, alpha2)
        assert abs(se['v0'] - sc['v0']) < 1e-10 and np.allclose(se['x'], sc['x'], atol=1e-10)
        assert np.isclose(se['J'], sc['J']) and np.isclose(se['prec_v0'], sc['prec_v0'])
        assert np.isclose(e.logdet(alpha2), c.logdet(alpha2))
        # direct: minimise |y - v0 u - B x|^2 + alpha2 |x|^2 over (v0, x)
        A = np.hstack([u[:, None], B])
        P = np.diag(np.r_[0., np.full(shape[1], alpha2)])
        z = np.linalg.solve(A.T @ A + P, A.T @ y)
        assert abs(z[0] - se['v0']) < 1e-9 and np.allclose(z[1:], se['x'], atol=1e-9)
        if c.side == 'cells':
            assert np.allclose(e.posterior_sd_x(alpha2), c.posterior_sd_x(alpha2))


def test_evidence_peak_recovers_prior_scale():
    rng = np.random.default_rng(4)
    N, K, tau = 400, 300, 0.5
    B = rng.standard_normal((N, K))
    u = rng.standard_normal(N)
    y = 2.*u + B @ (tau*rng.standard_normal(K)) + rng.standard_normal(N)
    s = core.LevelRidge(B, u, 'eigh')
    alpha2 = s.by_evidence(y, fit_sigma=False)
    assert 0.5 < np.sqrt(1/alpha2)/tau < 2.


@pytest.mark.parametrize('shape', [(40, 60), (60, 40)])
def test_marginal_level_and_posterior_map(shape):
    """marginal evidence against the Gaussian integral, and the s.d. of v = v0 + M x against the dense posterior"""
    rng = np.random.default_rng(5)
    N, K = shape
    B = rng.standard_normal(shape)
    u = B @ np.ones(K)                                       # the level in the range of B, as in the cell maps
    y = 0.7*u + B @ (0.3*rng.standard_normal(K)) + rng.standard_normal(N)
    M = rng.standard_normal((7, K))
    alpha2, sigma = 2.5, 1.3
    for route in ('eigh', 'cholesky'):
        s = core.LevelRidge(B, u, route, level='marginal')
        # y ~ N(v0 u, sigma^2 (I + BB'/alpha^2)), v0 flat: log p = -1/2 (log|C| + log u'C^-1u + chi2_C) + const
        C = sigma**2*(np.eye(N) + B @ B.T/alpha2)
        Ci = np.linalg.inv(C)
        v0 = (u @ Ci @ y)/(u @ Ci @ u)
        r = y - v0*u
        direct = -0.5*(np.linalg.slogdet(C)[1] + np.log(u @ Ci @ u) + r @ Ci @ r)
        mine = s.log_evidence(y/sigma, alpha2, fit_sigma=False) - (N - 1)*np.log(sigma)
        assert np.isclose(mine, direct, atol=1e-8)
        assert np.isclose(s.solve(y, alpha2)['v0'], v0, atol=1e-10)
        A = np.hstack([u[:, None], B])
        H = A.T @ A + np.diag(np.r_[0., np.full(K, alpha2)])
        cov = sigma**2*np.linalg.inv(H)
        Mv = np.hstack([np.ones((7, 1)), M])
        assert np.allclose(s.posterior_sd_map(alpha2, sigma, M), np.sqrt(np.diag(Mv @ cov @ Mv.T)), rtol=1e-8)


def test_box_qp_matches_active_set():
    rng = np.random.default_rng(6)
    A = rng.standard_normal((30, 20))
    H, g = A.T @ A + 0.1*np.eye(20), rng.standard_normal(20)
    v = core.box_qp(H, g, 0., np.inf)
    grad = H @ v - g
    assert (v >= 0).all() and np.all(np.abs(grad[v > 1e-9]) < 1e-6) and np.all(grad[v <= 1e-9] > -1e-6)
