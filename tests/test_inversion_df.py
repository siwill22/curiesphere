"""DF (domain fractal): the pattern's correlation, the evidence, its gradient and the posterior against dense formulas"""
import numpy as np
from scipy.special import eval_legendre

from remit.inversion.methods import df


def small(seed=0, N=40, K=3, n_per=10):
    rng = np.random.default_rng(seed)
    n = K*n_per
    clon, clat = 130 + rng.uniform(-8, 8, n), -25 + rng.uniform(-8, 8, n)
    label = np.repeat(np.arange(K), n_per)
    Gw = rng.standard_normal((N, n))
    yw = Gw @ (1 + 0.3*rng.standard_normal(n)) + rng.standard_normal(N)
    return df.DomainFractal(Gw, yw, label, clon, clat, 1e4, 0., 4.), rng


def test_correlation_is_a_legendre_series_and_positive_definite():
    w = df.cap_window(1e4, 300)
    psi = np.array([0., 0.01, 0.05, 0.2])
    l = np.arange(301)
    c = (2*l + 1)*df.spectrum(2.5, 400., 300)*w**2
    direct = np.array([np.sum(c*eval_legendre(l, np.cos(p))) for p in psi])/np.sum(c)
    assert np.allclose(df.correlation(2.5, 400., psi, w), direct, atol=1e-12)
    m, _ = small()
    for beta in (2., 2.5, 3.):
        M, Ls = m.factors(beta, 300.)
        assert all(np.all(np.isfinite(L)) for L in Ls)
        R = Ls[0] @ Ls[0].T
        assert np.allclose(np.diag(R), 1., atol=1e-6) and np.linalg.eigvalsh(R).min() > 0


def dense(m, s, beta, ell, h, tau=1e3):
    """log evidence with theta ~ N(2, tau^2) (tends to the flat box prior times the box volume) and the posterior"""
    M, Ls = m.factors(beta, ell)
    n = len(m.label)
    L = np.zeros((n, n))
    for b, Lk in zip(m.blocks, Ls):
        L[np.ix_(b, b)] = Lk
    Sigma = L @ np.diag(s[m.label]**2) @ L.T
    Cv = Sigma + tau**2*m.S @ m.S.T
    Cy = m.Gw @ Cv @ m.Gw.T + h**2*np.eye(len(m.yw))
    mu = m.Gw @ (m.S @ np.full(m.K, 2.))
    r = m.yw - mu
    logpdf = -0.5*(np.linalg.slogdet(2*np.pi*Cy)[1] + r @ np.linalg.solve(Cy, r))
    ll = logpdf + m.K*np.log(np.sqrt(2*np.pi)*tau) - np.sum(np.log(m.hi - m.lo))
    gain = Cv @ m.Gw.T @ np.linalg.inv(Cy)
    v = m.S @ np.full(m.K, 2.) + gain @ (m.yw - mu)
    return ll, v, np.sqrt(np.diag(np.linalg.inv(np.linalg.inv(Cv) + m.Gw.T @ m.Gw/h**2)))


def test_evidence_and_posterior_against_dense_formulas():
    m, _ = small()
    s, beta, ell, h = np.array([0.3, 0.6, 0.2]), 2.4, 350., 1.3
    x = np.r_[np.log(s), beta, np.log(ell), np.log(h)]
    ll, v, sd = dense(m, s, beta, ell, h)
    assert abs(m.evidence(x, fit_h=True) - ll) < 1e-4
    post = m.posterior(dict(s=s, beta=beta, ell=ell, h=h), np.ones(len(m.label)))
    assert np.allclose(post['v'], v, atol=1e-5) and np.allclose(post['v_sd'], sd, atol=1e-4)


def test_gradient_against_finite_differences():
    m, _ = small(1)
    x = np.r_[np.log([0.3, 0.6, 0.2]), 2.4, np.log(350.), np.log(1.2)]
    _, g = m.gradient(x, fit_h=True)
    for j in range(len(x)):
        e = np.zeros(len(x))
        e[j] = 1e-5
        fd = (m.evidence(x + e, True) - m.evidence(x - e, True))/2e-5
        assert abs(g[j] - fd) < 1e-4*max(1, abs(fd)), (j, g[j], fd)


def test_no_variation_is_the_limit_of_small_strengths():
    m, _ = small(2)
    x = np.r_[np.full(m.K, np.log(1e-7)), 2.5, np.log(300.)]
    assert abs(m.evidence(x) - m.no_variation()[0]) < 1e-6


def test_gradient_with_an_extra_error_shape():
    m0, rng = small(3)
    B = rng.standard_normal((len(m0.yw), 5))
    m = df.DomainFractal(m0.Gw, m0.yw, m0.label, *np.zeros((2, len(m0.label))), 1e4, 0., 4., extra=[B @ B.T/5])
    m.psi, m.psi_max = m0.psi, m0.psi_max
    x = np.r_[np.log([0.3, 0.6, 0.2]), 2.4, np.log(350.), np.log(1.2), np.log(0.7)]
    _, g = m.gradient(x, fit_h=True)
    for j in range(len(x)):
        e = np.zeros(len(x))
        e[j] = 1e-5
        fd = (m.evidence(x + e, True) - m.evidence(x - e, True))/2e-5
        assert abs(g[j] - fd) < 1e-4*max(1, abs(fd)), (j, g[j], fd)


def test_bounded_map():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((6, 6))
    Cov = A @ A.T + np.eye(6)
    v = np.array([1., 2., 0.5, 1.5, 3., 1.])
    assert np.allclose(df.bounded_map(v, Cov), v, atol=1e-6)
    v[2] = -1.
    vb = df.bounded_map(v, Cov)
    assert vb.min() >= -1e-9
    H = np.linalg.inv(Cov)
    g = H @ (vb - v)                                 # KKT: zero gradient on free cells, >= 0 at the bound
    assert np.all(abs(g[vb > 1e-6]) < 1e-5) and np.all(g[vb <= 1e-6] >= -1e-5)


def test_chain_runs_and_stays_in_bounds():
    m, _ = small(5)
    f = m.fit()
    bnds = m.bounds((1e-3, 10.), (2., 3.), (50., 5000.), False)
    X, LL, acc = m.chain(f, bnds, 300, 1, np.eye(len(f['x']))*0.01)
    lo, hi = np.array(bnds).T
    assert 0.05 < acc < 0.95 and np.all((X >= lo - 1e-12) & (X <= hi + 1e-12))
    assert abs(df.rhat(np.stack([X[100:200], X[200:300]]))).max() < 10
