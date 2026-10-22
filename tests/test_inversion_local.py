"""Local fits: Br synthesis at points, the point error model's whitening, the cell graph."""
import numpy as np
import pyshtools

from remit import depth
from remit.inversion import data, errors, geometry, graph


def test_synthesis_matches_pyshtools():
    rng = np.random.default_rng(1)
    band = data.Band(5, 30)
    c = rng.standard_normal((2, 31, 31))
    c[1, :, 0] = 0
    c *= np.tril(np.ones((31, 31)))[None]
    lon, lat = rng.uniform(0, 360, 50), rng.uniform(-89, 89, 50)
    c_band = band.coeffs(band.rows(c))
    ref = np.asarray(pyshtools.SHMagCoeffs.from_array(c_band, r0=depth.R0).expand(a=depth.R0, lat=lat, lon=lon,
                                                                                    lmax_calc=30))[:, 0]
    assert np.allclose(data.synthesis(lon, lat, band) @ band.rows(c), ref, rtol=0, atol=1e-10*np.abs(ref).max())


def test_point_whitening_diagonalises_any_amplitudes():
    rng = np.random.default_rng(2)
    band = data.Band(3, 40)
    lon, lat = rng.uniform(100, 160, 120), rng.uniform(-45, -10, 120)
    l = np.arange(41.)
    Cn = errors.point_shape(np.ones(41), band, lon, lat)
    Cm = errors.point_shape((l + 1)**-1.5, band, lon, lat)
    P, T, lam = errors.point_whitening(Cn, Cm, 2., 3., 1.0 - 1e-12, 1e-6)
    for sn, s in ((2., 3.), (1., 10.)):
        C = sn**2*(P.T @ Cn @ P + 1e-6*np.eye(P.shape[1])) + s**2*(P.T @ Cm @ P)
        assert np.allclose(T @ C @ T.T, np.diag(sn**2*(1 + 1e-6) + s**2*lam), atol=1e-6*np.abs(C).max())


def test_cell_graph_connects_islands():
    cell = np.full((1801, 3601), -1, np.int32)
    cell[900, 100:102] = 0                       # two separate pieces: 10.0-10.1 E and 20.0 E on the equator
    cell[900, 200] = 1
    cells = geometry.Cells(64, cell, np.array([0, 1]), np.ones(2), np.array([10., 20.]), np.array([0., 0.]))
    edges, bridges = graph.cell_edges(cells)
    assert edges.tolist() == [[0, 1]] and bridges.tolist() == [[0, 1]]
    assert np.isclose(graph.edge_lengths_km(edges, cells.clon, cells.clat)[0], np.radians(10)*graph.R_KM)
