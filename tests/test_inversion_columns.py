"""remit.inversion stage 1: configs, cache, cell columns by node sum against the global transform."""
import numpy as np
import pytest

from remit.inversion import cache, columns, config, geometry


def test_config_inherit_and_mode_defaults(tmp_path):
    (tmp_path / 'base.toml').write_text('mode = "explore"\n[target]\nnside = 32\n[geometry]\nlmax = 100\n')
    (tmp_path / 'child.toml').write_text('inherit = "base.toml"\n[geometry]\nlmax = 133\n'
                                         '[compute]\nsolver = "eigh"\n')
    cfg = config.load(tmp_path / 'child.toml')
    assert cfg['target'] == {'area': 'continental_crust', 'nside': 32}       # default merged with the parent
    assert cfg['geometry']['lmax'] == 133                                     # child overrides parent
    assert cfg['geometry']['layers'] == [{'top': 'basement', 'base': 30.0}]   # default
    assert cfg['compute']['columns_backend'] == 'mps'                         # explore-mode default
    assert cfg['compute']['solver'] == 'eigh'                                 # file overrides the mode default
    with pytest.raises(ValueError):
        config.resolve({'mode': 'quick'})


def test_spec_hash_ignores_key_order():
    assert config.spec_hash({'a': 1, 'b': [1, 2]}) == config.spec_hash({'b': [1, 2], 'a': 1})
    assert config.spec_hash({'a': 1}) != config.spec_hash({'a': 2})


def test_cache_computes_once(tmp_path):
    c = cache.Cache(tmp_path)
    calls = []
    f = lambda: calls.append(1) or np.arange(3.)
    a = c.array('thing', {'x': 1}, 'values', f)
    b = c.array('thing', {'x': 1}, 'values', f)
    assert np.array_equal(a, b) and len(calls) == 1
    c.array('thing', {'x': 2}, 'values', f)
    assert len(calls) == 2


@pytest.fixture(scope='module')
def cells64():
    return geometry.make_cells(geometry.continental_crust(), 64)


def _pick(cells):
    """a mid-latitude cell and the cell nearest the south pole"""
    return np.array([int(np.argmin(np.abs(cells.clat + 25))), int(np.argmin(cells.clat))])


def test_node_sum_matches_transform(cells64):
    S = columns.Setup({'top': 'basement', 'base': 30.0}, 'induced_igrf13', 40, cells64.cell >= 0)
    idx = _pick(cells64)
    a = columns.node_sum(cells64.cell, idx, S, 'numpy')
    ref = columns.transform_columns(cells64.cell, idx, S)
    assert np.abs(a - ref).max() < 1e-10*np.abs(ref).max()


def test_backends_agree(cells64):
    torch = pytest.importorskip('torch')
    S = columns.Setup({'top': 'basement', 'base': 30.0}, 'induced_igrf13', 80, cells64.cell >= 0)
    idx = np.r_[_pick(cells64), np.arange(0, len(cells64), len(cells64)//6)]
    a = columns.node_sum(cells64.cell, idx, S, 'numpy')
    b = columns.node_sum(cells64.cell, idx, S, 'torch-cpu')
    assert np.abs(a - b).max() < 1e-12*np.abs(a).max()
    if torch.backends.mps.is_available():
        c = columns.node_sum(cells64.cell, idx, S, 'mps')
        assert np.abs(a - c).max() < 1e-5*np.abs(a).max()


def test_layer_must_be_right_way_up():
    with pytest.raises(ValueError):
        geometry.layer_radii({'top': 30.0, 'base': 10.0})
