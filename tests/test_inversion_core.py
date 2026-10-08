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
