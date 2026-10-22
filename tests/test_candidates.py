"""Candidate maps as layouts (remit/inversion/candidates.py): the patch rule and the class-grid lookup."""
import numpy as np

from remit.inversion import candidates


def chain(n):
    """edges of n cells in a row"""
    return np.stack([np.arange(n - 1), np.arange(1, n)], 1)


def test_two_pieces_of_one_unit_become_two_domains():
    labels = ['a']*4 + ['b']*4 + ['a']*4
    out = candidates.patches(labels, chain(12), np.empty((0, 2), int))
    assert len(set(out)) == 3
    assert out[0] != out[-1]


def test_small_piece_merges_into_the_neighbour_it_shares_most_edges_with():
    # cells 0-3 'a', 4 'x', 5-8 'b'; cell 4 touches 'b' twice (an extra edge 4-6), 'a' once
    labels = ['a']*4 + ['x'] + ['b']*4
    edges = np.vstack([chain(9), [[4, 6]]])
    out = candidates.patches(labels, edges, np.empty((0, 2), int))
    assert len(set(out)) == 2
    assert out[4] == out[5]


def test_rest_stays_whole_and_islands_split_across_bridges():
    # two islands of 'a' joined by a bridge, plus a scattered rest domain
    labels = ['a']*3 + ['rest aus'] + ['a']*3 + ['rest aus']
    edges = np.array([[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [3, 4]])
    bridges = np.array([[3, 4]])
    out = candidates.patches(labels, edges, bridges)
    assert out[3] == out[7] == 'rest aus'
    assert out[0] != out[4]


def test_spun_rest_is_recognised():
    assert candidates.is_rest('aus rest aus') and candidates.is_rest('rest nam')
    assert not candidates.is_rest('Forest Hill') and not candidates.is_rest('tc1 3')


def test_class_grid_wraps_at_the_dateline():
    lon_c, lat_c = np.arange(-179.5, 180), np.arange(-89.5, 90)
    values = np.tile(np.arange(360.), (180, 1))
    got = candidates._class_grid(values, lon_c, lat_c, np.array([179.9, 180.1, 359.9, 0.2]), np.zeros(4))
    assert list(got) == [359, 0, 179, 180]


def test_map_units_keep_holes_and_take_the_smallest_polygon():
    import geopandas as gpd
    from shapely.geometry import Polygon
    outer = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(4, 4), (6, 4), (6, 6), (4, 6)]])
    lake = Polygon([(4.5, 4.5), (5.5, 4.5), (5.5, 5.5), (4.5, 5.5)])      # sits in part of the hole
    big = Polygon([(-1, -1), (11, -1), (11, 11), (-1, 11)])               # overlaps everything
    g = gpd.GeoDataFrame(dict(u=['outer', 'lake', 'big']), geometry=[outer, lake, big], crs='EPSG:4326')
    got = candidates._map_units(g, 'u', np.array([2., 5., 4.2, 10.5]), np.array([2., 5., 4.2, 10.5]))
    assert list(got) == [0, 1, 2, 2]                 # outer; lake; in the hole only 'big'; outside outer: 'big'
