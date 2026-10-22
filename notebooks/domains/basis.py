"""
Cell basis, data and noise for a region: the shared engine of the three methods.

  cells        HEALPix NSIDE cells (ring numbering) over the extent; a cell holds the 0.1 deg grid
               nodes of the extent that fall in its pixel, so cells at the edge are partial
  columns      Gauss coefficients of 1 SI.km of VIS in one cell, spread uniformly in radius from
               the top of basement to base_depth below the ellipsoid (IGRF-13 inducing field);
               computed in parallel, one file per cell, resumable
  data points  HEALPix NSIDE_DATA points within buffer_km (great circle) of an extent node
  G            Br (nT per SI.km) of each cell at the points, degrees lmin-lmax_fit, LCS-1 filter F(l)
  background   the depth model outside the extent: H&M + slabs VIS (continental VIS on the
               ellipsoid, oceanic VIS over the HM05 oceanic crust) with the extent removed,
               plus GK07_NR remanence
  data         LCS-1 and MF7 Br at the points, same band and filter
  noise        random-phase realisations with the global per-degree spectrum of LCS-1 - MF7
  coarse       the NSIDE/2 basis: each coarse cell is the sum of its four nested children

Same choices as notebooks/invert_domains.py and cell_basis.py (Australia), which this reproduces.
"""
import os
import time
import multiprocessing

import numpy as np
import pyshtools
from astropy_healpix import healpy as hp
from scipy.spatial import cKDTree

import depth_models as dm
from remit.data.models import load_ocean_age_model, load_vis_model, load_lcs, DATA_DIR
from remit.utils.region import generate_healpix_points

R0 = dm.R0
KM2_PER_NODE = (np.radians(0.1)*R0/1e3)**2          # 0.1 deg node at the equator


def unit(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], -1)


def ocean_grid():
    ocean = load_ocean_age_model()
    return ocean, np.asarray(ocean.lon), np.asarray(ocean.lat), np.asarray(ocean.age)


def make_cells(ext, glon, glat, nside):
    """cell index grid (-1 outside the extent), HEALPix ids (ring) and areas (km2) of the cells"""
    LO, LA = np.meshgrid(glon, glat)
    pix = hp.ang2pix(nside, LO[ext], LA[ext], lonlat=True)
    cell_ids = np.unique(pix)
    cell = np.full(ext.shape, -1, dtype=np.int32)
    cell[ext] = np.searchsorted(cell_ids, pix)
    area = np.bincount(cell[ext], weights=np.cos(np.radians(LA[ext])), minlength=len(cell_ids))*KM2_PER_NODE
    return cell, cell_ids, area


def data_points(ext, glon, glat, nside_data, buffer_km):
    LO, LA = np.meshgrid(glon, glat)
    hlon, hlat = generate_healpix_points(nside_data)
    hlon = np.mod(hlon, 360.)
    chord, _ = cKDTree(unit(LO[ext], LA[ext])).query(unit(hlon, hlat))
    keep = 2*np.arcsin(np.clip(chord/2, 0, 1))*R0/1e3 <= buffer_km
    return hlon[keep], hlat[keep]


# ------------------------------------------------------------ cell columns (parallel, resumable)
_job = {}


def _compute_one(i):
    j = _job
    f = f'{j["dir"]}/{j["cell_ids"][i]}.npy'
    if not os.path.exists(f):
        c = dm.slab_vis_coeffs(np.where(j['cell'] == i, 1., 0.), j['r_top'], j['r_base'], j['to_gmm'],
                               lmax=j['lmax']).coeffs
        np.save(f + '.tmp.npy', c)
        os.replace(f + '.tmp.npy', f)
    return i


def compute_cell_columns(cell, cell_ids, ocean, cell_dir, base_depth_km, lmax_cell, n_workers):
    """one Gauss-coefficient file per cell in cell_dir; existing files are kept. Run with
    single-threaded BLAS (OMP/OPENBLAS/VECLIB_..._THREADS=1): the workers fill the cores."""
    os.makedirs(cell_dir, exist_ok=True)
    glat = np.asarray(ocean.lat)
    r_top = dm.top_of_basement_radius(ocean)
    r_base = dm.ellipsoid_radius(glat)[:, None]*np.ones_like(r_top) - base_depth_km*1e3
    assert (r_top[cell >= 0] > r_base[cell >= 0]).all(), 'basement below the slab base'
    _job.update(dir=cell_dir, cell_ids=cell_ids, cell=cell, r_top=r_top, r_base=r_base, lmax=lmax_cell,
                to_gmm=dm.induced_to_gmm(np.asarray(ocean.lon), glat))
    todo = [i for i in range(len(cell_ids)) if not os.path.exists(f'{cell_dir}/{cell_ids[i]}.npy')]
    print(f'{len(todo)} of {len(cell_ids)} cells to compute with {n_workers} workers', flush=True)
    t = time.time()
    with multiprocessing.get_context('fork').Pool(n_workers) as pool:       # workers share the grids
        for n, _ in enumerate(pool.imap_unordered(_compute_one, todo), 1):
            if n % 100 == 0 or n == len(todo):
                el = time.time() - t
                print(f'  {n}/{len(todo)} cells, {el/60:.0f} min, ~{el/n*(len(todo)-n)/60:.0f} min left', flush=True)


# ------------------------------------------------------------ Br at the data points
def band_filter(lmax_fit):
    return dm.lcs1_filter(np.arange(lmax_fit+1), dm.fit_lh())


def br_points(c, plon, plat, lmin, lmax_fit, F=None):
    """Br (nT) at the points from Gauss coefficients at r0 = R0: degrees lmin-lmax_fit, F(l) applied
    (F: per-degree filter, default the LCS-1 filter)"""
    F = band_filter(lmax_fit) if F is None else F[:lmax_fit+1]
    c = np.array(c)[:, :lmax_fit+1, :lmax_fit+1]*F[None, :, None]
    c[:, :lmin] = 0
    return np.asarray(pyshtools.SHMagCoeffs.from_array(c, r0=R0).expand(
        a=R0, lat=plat, lon=plon, lmax_calc=lmax_fit))[:, 0]


def packed_lm(lmax):
    """degree and order of each packed (l, m) index, as pyshtools.legendre (k = l(l+1)/2 + m)"""
    return (np.concatenate([np.full(l+1, l) for l in range(lmax+1)]),
            np.concatenate([np.arange(l+1) for l in range(lmax+1)]))


def cell_matrix(cell_dir, cell_ids, plon, plat, lmin, lmax_fit, F=None):
    """G (points x cells): Br of each cell column, by an explicit synthesis matrix (Schmidt
    harmonics, as pyshtools), checked against pyshtools expand for three cells"""
    C = np.array([np.load(f'{cell_dir}/{cid}.npy')[:, :lmax_fit+1, :lmax_fit+1] for cid in cell_ids])
    L, M = packed_lm(lmax_fit)
    Pc, Ps = synthesis_parts(plon, plat, lmin, lmax_fit, F)
    G = Pc @ C[:, 0, L, M].T + Ps @ C[:, 1, L, M].T
    for i in [0, len(cell_ids)//2, len(cell_ids)-1]:
        ref = br_points(C[i], plon, plat, lmin, lmax_fit, F)
        assert np.abs(G[:, i] - ref).max() < 1e-8*np.abs(ref).max(), 'synthesis mismatch'
    return G


# ------------------------------------------------------------ cell columns by direct summation (fine cells, high degree)
def direct_setup(ocean, base_depth_km, lmax, n_gl=8):
    """what direct_columns needs, computed once: magnetisation per SI.km, radii, quadrature, Legendre cache"""
    from remit.vhtools import _dh_weights
    glat = np.asarray(ocean.lat)
    r_top = dm.top_of_basement_radius(ocean)
    r_base = dm.ellipsoid_radius(glat)[:, None]*np.ones_like(r_top) - base_depth_km*1e3
    nth = len(glat) - 1                                             # DH2 rows: north pole to one row short of the south pole
    w, wsin = _dh_weights(np.arange(nth)*np.pi/nth)
    x, wq = np.polynomial.legendre.leggauss(n_gl)
    L, M = packed_lm(lmax)
    return dict(comps=dm.induced_components(glat), r_top=r_top, r_base=r_base, w=w, wsin=wsin, dphi=2*np.pi/(2*nth),
                nth=nth, x=x, wq=wq, lmax=lmax, L=L, M=M, legendre={})


def _legendre_row(S, i):
    """Schmidt P_lm and dP_lm/dtheta (packed) at grid row i, cached"""
    if i not in S['legendre']:
        th = i*np.pi/S['nth']
        p, dp = pyshtools.legendre.PlmSchmidt_d1(S['lmax'], np.cos(th))
        S['legendre'][i] = (p, -dp*np.sin(th))
    return S['legendre'][i]


def direct_columns(cell, idx, S):
    """packed Gauss coefficients (g, h: len(idx) x (lmax+1)(lmax+2)/2, r0 = R0) of 1 SI.km of VIS in each
    cell of idx, spread uniformly in radius from the top of basement to the slab base, induced by IGRF-13.
    The same finite sum as slab_vis_coeffs (the global transform of remit.vhtools.forward_transform,
    Gubbins et al. 2011 eq. 32-34, with Driscoll-Healy weights), restricted to the cell's own 0.1 deg
    nodes, so it is exact to rounding; the radial factor (r/R0)^(l+1) is applied exactly instead of
    by its series. Cost ~ nodes x lmax^2 per cell instead of a global transform."""
    L, M, lmax = S['L'], S['M'], S['lmax']
    K = 1e9*pyshtools.constants.mu0.value/(4*np.pi*R0)
    out = np.zeros((2, len(idx), len(L)))
    for c, ci in enumerate(idx):
        rows, cols = np.nonzero(cell == ci)
        assert rows.max() < S['nth'] and cols.max() < 2*S['nth'], 'cell touches the dropped pole row or 360 column'
        g, h = np.zeros(len(L)), np.zeros(len(L))
        for i, j in zip(rows, cols):
            p, dp = _legendre_row(S, i)
            mr, mt, mp = (comp[i, j] for comp in S['comps'])
            rt, rb = S['r_top'][i, j], S['r_base'][i, j]
            r = rt + (rb - rt)*(S['x'] + 1)/2                         # Gauss-Legendre radii
            rad = (S['wq']/2) @ (r[:, None]/R0)**(np.arange(lmax+1)[None, :] + 1.)
            A = (L*p*mr + dp*mt)*S['w'][i]*rad[L]
            Bm = M*p*mp*S['wsin'][i]*rad[L]
            ph = M*j*S['dphi']
            cs, sn = np.cos(ph), np.sin(ph)
            g += A*cs - Bm*sn
            h += A*sn + Bm*cs
        out[0, c], out[1, c] = g*K*S['dphi'], h*K*S['dphi']
    out[:, :, L == 0] = 0.
    return out


def synthesis_parts(plon, plat, lmin, lmax_fit, F=None):
    """the two halves of the Br synthesis at the points (points x packed (l, m)): Br = Pc @ g + Ps @ h,
    degrees lmin-lmax_fit, filter F (default LCS-1)"""
    F = band_filter(lmax_fit) if F is None else F[:lmax_fit+1]
    P = np.array([pyshtools.legendre.PlmSchmidt(lmax_fit, z) for z in np.sin(np.radians(plat))])
    L, M = packed_lm(lmax_fit)
    w = (L+1)*F[L]*(L >= lmin)
    phi = np.radians(plon)[:, None]*M[None, :]
    return P*w*np.cos(phi), P*w*np.sin(phi)


_direct_job = {}


def _direct_block(idx):
    j = _direct_job
    gh = direct_columns(j['cell'], idx, j['S'])
    n = j['Pc'].shape[1]
    return idx, j['Pc'] @ gh[0][:, :n].T + j['Ps'] @ gh[1][:, :n].T


def cell_matrix_direct(cell, n_cells, ocean, base_depth_km, plon, plat, lmin, lmax_fit, F=None, n_workers=3, block=100):
    """G (points x cells) as cell_matrix, from direct_columns (no column files), in parallel blocks of cells"""
    S = direct_setup(ocean, base_depth_km, lmax_fit)
    Pc, Ps = synthesis_parts(plon, plat, lmin, lmax_fit, F)
    _direct_job.update(cell=cell, S=S, Pc=Pc, Ps=Ps)
    G = np.zeros((len(plon), n_cells))
    blocks = [np.arange(k, min(k + block, n_cells)) for k in range(0, n_cells, block)]
    t = time.time()
    with multiprocessing.get_context('fork').Pool(n_workers) as pool:
        for n, (idx, Gb) in enumerate(pool.imap_unordered(_direct_block, blocks), 1):
            G[:, idx] = Gb
            if n % 5 == 0 or n == len(blocks):
                el = time.time() - t
                print(f'  G: {n}/{len(blocks)} blocks of {block} cells, {el/60:.0f} min, ~{el/n*(len(blocks)-n)/60:.0f} min left', flush=True)
    return G


def load_cof(name, lmax):
    clm, _ = pyshtools.shio.shread(f'{DATA_DIR}/shc/{name}', lmax=lmax)
    return clm                                          # r0 taken as 6371.0 km, as for LCS-1


def read_shc(path, lmax):
    """Gauss coefficients (2, lmax+1, lmax+1) of a Swarm-format .shc file (n, m, value; m < 0 for h),
    moved from its reference radius (header comment or 6371.2 km) to r0 = R0 by (a/R0)^(n+2)"""
    c = np.zeros((2, lmax+1, lmax+1))
    a = 6371.2e3
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) != 3:
                continue
            n, m = int(p[0]), int(p[1])
            if n > lmax:
                break
            c[0 if m >= 0 else 1, n, abs(m)] = float(p[2])
    n = np.arange(lmax+1)
    return c*((a/R0)**(n+2))[None, :, None]


def field_model(spec, lmax):
    """Gauss coefficients (r0 = R0, zero-padded to lmax) of a lithospheric field model:
    'lcs1', 'mf7' or the path of a Swarm .shc file (e.g. MLI_SHA_2E)"""
    if spec == 'lcs1':
        c = load_lcs(lmin=0, lmax=dm.LMAX).coeffs
    elif spec == 'mf7':
        c = load_cof('MF7_mod.cof', 133)
    else:
        return read_shc(spec, lmax)
    out = np.zeros((2, lmax+1, lmax+1))
    k = min(lmax, c.shape[1]-1) + 1
    out[:, :k, :k] = c[:, :k, :k]
    return out


def background_coeffs(ext, ocean, lmax=dm.LMAX):
    """Gauss coefficients of the background: H&M + slabs VIS outside the extent, plus GK07_NR ocean
    remanence (cached to degree 185 only: above it the background is the induced part alone)"""
    vis = load_vis_model(name='Hemant2005+slabs', match=(ocean.lon, ocean.lat, ocean.age))
    vis.vis = np.where(ext, 0., vis.vis)
    C_bg = dm.vis_coeffs(ocean, vis, dm.top_of_crust_radius(ocean), lmax=lmax).coeffs
    k = min(lmax, dm.LMAX) + 1
    C_bg[:, :k, :k] += np.load(f'{dm.CACHE}/GK07_nr_{dm.LMAX}.npy')[:, :k, :k]
    return C_bg


def background_and_data(ext, ocean, plon, plat, lmin, lmax_fit):
    """background Gauss coefficients and Br at the points: background, LCS-1, MF7"""
    C_bg = background_coeffs(ext, ocean)
    b = br_points(C_bg, plon, plat, lmin, lmax_fit)
    d_lcs = br_points(load_lcs(lmin=lmin, lmax=dm.LMAX).coeffs, plon, plat, lmin, lmax_fit)
    d_mf7 = br_points(load_cof('MF7_mod.cof', lmax_fit), plon, plat, lmin, lmax_fit)
    return C_bg, b, d_lcs, d_mf7


def noise_realisations(plon, plat, lmin, lmax_fit, n, seed, R_l=None, F=None):
    """random-phase fields at the points (unscaled) with a global Lowes spectrum R_l, by default that
    of LCS-1 - MF7, times F(l)^2 if a filter F is given (none by default, as in the first build)"""
    ll = np.arange(lmax_fit+1)
    if R_l is None:
        diff = load_lcs(lmin=lmin, lmax=lmax_fit).coeffs - load_cof('MF7_mod.cof', lmax_fit)
        diff[:, :lmin] = 0
        R_l = (ll+1)*(diff**2).sum((0, 2))
    R_l = np.asarray(R_l)[:lmax_fit+1]*(1. if F is None else np.asarray(F)[:lmax_fit+1]**2)
    var = np.where(ll >= lmin, R_l/((ll+1)*(2*ll+1)), 0.)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = rng.standard_normal((2, lmax_fit+1, lmax_fit+1))*np.sqrt(var)[None, :, None]
        c[1, :, 0] = 0
        c *= np.arange(lmax_fit+1)[None, None, :] <= ll[None, :, None]
        out.append(np.asarray(pyshtools.SHMagCoeffs.from_array(c, r0=R0).expand(
            a=R0, lat=plat, lon=plon, lmax_calc=lmax_fit))[:, 0])
    return np.array(out)


# ------------------------------------------------------------ coarse basis
def coarse_parents(cell_ids, nside):
    """parent of each cell at NSIDE/2 (HEALPix nesting: a parent holds exactly four children).
    Returns the parent index of each cell and the parent ids (ring numbering at NSIDE/2)."""
    parent_nest = hp.ring2nest(nside, cell_ids) // 4
    parent_ids, parent = np.unique(hp.nest2ring(nside//2, parent_nest), return_inverse=True)
    return parent, parent_ids


def coarsen(G, parent):
    """columns of the coarse cells: sums of their children's columns"""
    A = np.zeros((G.shape[1], parent.max()+1))
    A[np.arange(G.shape[1]), parent] = 1.
    return G @ A


def reuse_cell_columns(old_cells_npz, old_dir, cell, cell_ids, cell_dir):
    """hard-link the column files of an earlier build for every cell whose grid nodes are exactly
    the same (same HEALPix id and the same extent nodes inside it); returns the number reused"""
    os.makedirs(cell_dir, exist_ok=True)
    z = np.load(old_cells_npz)
    old_cell, old_ids = z['cell'], z['cell_ids']

    def node_sets(c):
        flat = np.flatnonzero(c >= 0)
        order = np.argsort(c.ravel()[flat], kind='stable')
        return np.split(flat[order], np.cumsum(np.bincount(c.ravel()[flat]))[:-1])
    old_nodes, new_nodes = node_sets(old_cell), node_sets(cell)
    old_index = {cid: k for k, cid in enumerate(old_ids)}
    n = 0
    for k, cid in enumerate(cell_ids):
        src, dst = f'{old_dir}/{cid}.npy', f'{cell_dir}/{cid}.npy'
        if cid in old_index and not os.path.exists(dst) and os.path.exists(src) \
                and np.array_equal(old_nodes[old_index[cid]], new_nodes[k]):
            os.link(src, dst)
            n += 1
    return n
