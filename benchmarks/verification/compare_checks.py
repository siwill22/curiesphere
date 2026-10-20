"""
Why GCM recovers less of a small checkerboard than CM and DF: the same truths scored like for like.

The GCM and CM-AUS checkerboard tests differ in more than the method: global coefficients against points around
Australia, the noise reference of the error model (MLI_SHA_2E for GCM, MF7 for the local fits), and the tiles
(equal-area HEALPix tiles globally, lon/lat squares for Australia). Here:

  1  GCM's global checkerboard maps (gcm_shrink.npz) scored over Australia's cells only
  2  CM-AUS on the same equal-area tiles (NSIDE 4, 8, 16, 32), with the noise of its own error model (MF7 reference)
     and with the MLI_SHA_2E reference that GCM uses
  3  the two noise spectra, per degree

Run from the repository root (~10 min):
    PYTHONPATH=benchmarks/verification:benchmarks:notebooks:$PYTHONPATH OMP_NUM_THREADS=1 \
        nice -n 15 conda run -n curiesphere python benchmarks/verification/compare_checks.py
Output: benchmarks/verification/results/compare.json, compare.npz
"""
import json
import os

import numpy as np
from scipy.spatial import cKDTree

from cm_checks import CONFIG as CM_CONFIG
from df_checks import RESULTS, log, tiles
from remit.inversion import config, data, errors, gmrf, graph, problem
from remit.inversion.cache import Cache
from remit.inversion.graph import unit

MLI = config.load('inversions/gcm_glc_lcs1_16-133.toml')['errors']['noise']['reference']
NSIDES = (4, 8, 16, 32)


def cm_problem(reference=None):
    cfg = config.load(CM_CONFIG)
    if reference:
        cfg['errors']['noise']['reference'] = reference
    pb = problem.local(cfg, Cache(), log)
    edges, _ = graph.cell_edges(pb.cells)
    Lw = graph.laplacian(edges, 1/graph.edge_lengths_km(edges, pb.cells.clon, pb.cells.clat)**2, len(pb.cells))
    return cfg, pb, Lw


def contrast(v, high):
    return float(v[high].mean() - v[~high].mean())


def main():
    out = dict(gcm_over_australia={}, cm={}, spectra={})
    cfg, pb, Lw = cm_problem()
    lon, lat = pb.cells.clon, pb.cells.clat

    # 1  GCM's maps over Australia: the GCM cell nearest each CM-AUS cell centre (both NSIDE 64 cells)
    z = np.load(os.path.join(RESULTS, 'gcm_shrink.npz'))
    d, idx = cKDTree(unit(z['clon'], z['clat'])).query(unit(lon, lat))
    near = d*6371. < 30.
    log(f'GCM cells matched to {near.sum()} of {len(lon)} CM-AUS cells')
    for ns in NSIDES:
        high = tiles(lon, lat, ns)[1] == 0
        for noise in ('none', 'noise', 'noise and ocean error'):
            v = z[f'{ns}_{noise}'][idx, 0]
            out['gcm_over_australia'][f'{ns}_{noise}'] = contrast(v[near], high[near])
        log(f'  GCM over Australia, tiles NSIDE {ns}: ' + ', '.join(
            f'{n} {out["gcm_over_australia"][f"{ns}_{n}"]:.3f}' for n in ('none', 'noise', 'noise and ocean error')))

    # 2  CM-AUS on the same tiles, noise from MF7 (its own) and from MLI_SHA_2E (GCM's)
    maps = {}
    for ref_name, ref in (('MF7', None), ('MLI_SHA_2E', MLI)):
        if ref is not None:
            cfg, pb, Lw = cm_problem(ref)
        e = np.asarray(pb.W @ pb.draw(np.random.default_rng(20261018), model_error=False))
        out['spectra'][ref_name] = dict(noise_rms_points_nT=pb.info['noise_rms'])
        for ns in NSIDES:
            high = tiles(lon, lat, ns)[1] == 0
            truth = np.where(high, 1.5, 0.5)
            for noise in ('none', 'noise'):
                if noise == 'none' and ref is not None:
                    continue
                y = pb.Gw @ truth + (e if noise == 'noise' else 0.)
                gm = gmrf.GMRF(pb.Gw, y, pb.cells.area_km2, Lw)
                f = gm.fit(np.ones((len(lon), 1)))
                v = gm.posterior(np.ones((len(lon), 1)), f)['v']
                key = f'{ns}_{noise}' + ('' if noise == 'none' else f'_{ref_name}')
                out['cm'][key] = dict(contrast=contrast(v, high), ell_km=float(f['ell'][0]), h=f['h'])
                maps[key] = v
                log(f'  CM-AUS, tiles NSIDE {ns}, {noise} ({ref_name}): contrast {out["cm"][key]["contrast"]:.3f}, '
                    f'ell {f["ell"][0]:.0f} km, h {f["h"]:.2f}')

    # 3  the noise spectra per degree (RMS Br at the reference radius, nT): LCS-1 minus each reference
    lmax = cfg['geometry']['lmax']
    F = data.resolution_filter(cfg['data'].get('filter', 'lcs1'), lmax)
    c_model = data.field_coeffs('lcs1', lmax)
    for ref_name, ref in (('MF7', 'mf7'), ('MLI_SHA_2E', MLI)):
        R = errors.noise_spectrum(c_model, data.field_coeffs(ref, lmax), F)
        out['spectra'][ref_name]['R_l'] = R[:134].tolist()
    with open(os.path.join(RESULTS, 'compare.json'), 'w') as fo:
        json.dump(out, fo, indent=1)
    np.savez(os.path.join(RESULTS, 'compare.npz'), clon=lon, clat=lat, **maps)
    log('compare done')


if __name__ == '__main__':
    main()
