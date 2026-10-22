"""
Maps and tables for the verification document (docs/verification/verification.qmd), drawn from the results of
benchmarks/verification/forward.py and columns.py. Global maps in the Robinson projection, centred on 0 E.

Every case gets one figure: the model on top, then one row per field with 100 km on the left and 300 km on the right.
  reference      the dipole sum
  tested         remit (or the cell columns) to degree L
  difference     tested minus reference
  above L        the field of the same sources above degree L, as remit itself computes it (sign flipped, so it should
                 look the same as the difference row; same colour scale)
  left over      tested to the highest degree minus reference: what neither row above accounts for
Band-limited cases have no field above L, so they show the first three rows only.

Run from the repository root:
    conda run -n curiesphere python docs/verification/make_figures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pygmt
import xarray as xr

RES = Path('benchmarks/verification/results')
OUT = Path('docs/verification/figures')
W = 7.4                                     # map width (cm)
GAP_X = 1.0
ROW = 6.0                                   # vertical step per row (cm): map, title and colour bar
PROJ = f'N0/{W}c'


def for_gmt(da):
    """a lat/lon DataArray as a global gridline grid for GMT: latitudes ascending, the 360 E column and the -90 row
    added (copies of 0 E and of the southernmost row; the Driscoll-Healy grids stop one step short of the pole)"""
    da = da.rename({da.dims[0]: 'lat', da.dims[1]: 'lon'}).sortby('lat')
    da = xr.concat([da, da.isel(lon=[0]).assign_coords(lon=[360.])], 'lon')
    if float(da.lat[0]) > -90:
        da = xr.concat([da.isel(lat=[0]).assign_coords(lat=[-90.]), da], 'lat')
    return da.astype('float64')


def nice(v):
    """a round number at or above v, for a colour scale"""
    if not np.isfinite(v) or v <= 0:
        return 1.
    e = 10**np.floor(np.log10(v))
    return next(m*e for m in (1, 1.5, 2, 3, 5, 7.5, 10) if m*e >= v)


def fmt(x):
    if x == 0:
        return '0'
    if 0.01 <= abs(x) < 1000:
        return f'{x:.3g}'
    m, e = f'{x:.1e}'.split('e')
    return f'{m}@~\\264@~10@+{int(e)}@+'                 # GMT: 1.2 x 10^-13


def panel(fig, da, vmax, title, label, cmap='vik', symbols=None, sub=None):
    pygmt.makecpt(cmap=cmap, series=[-vmax, vmax], background=True)
    fig.grdimage(for_gmt(da), projection=PROJ, region='d', cmap=True,
                 frame=['g', f'+t{title}' + (f'+s{sub}' if sub else '')])
    fig.coast(shorelines='0.15p,gray55', projection=PROJ, region='d')
    if symbols is not None:
        fig.plot(x=symbols[0], y=symbols[1], style='c0.18c', pen='0.6p,black', projection=PROJ, region='d')
    fig.colorbar(position=f'JBC+w{0.7*W}c/0.16c+o0/0.25c+h+e', frame=[f'x+l{label}'])


def stats(a):
    return f'max {fmt(np.abs(a).max())} nT, rms {fmt(np.sqrt(np.mean(a**2)))} nT'


def case_figure(prefix, case, L, title, banded=False, tested='remit', above_from='remit'):
    ds = xr.load_dataset(RES / f'{prefix}_{case}.nc')
    top = int(ds.lmax.max())
    fig = pygmt.Figure()
    pygmt.config(FONT_TITLE='8.5p', FONT_SUBTITLE='7p', FONT_ANNOT_PRIMARY='6.5p', FONT_LABEL='7p',
                 MAP_TITLE_OFFSET='3p', MAP_FRAME_TYPE='plain', MAP_GRID_PEN_PRIMARY='0.15p,gray60', MAP_FRAME_PEN='0.5p')
    symbols = (ds.dipole_lon.values, ds.dipole_lat.values) if 'dipole_lon' in ds else None

    # the model, centred above the two columns; rows below, all at positive offsets from the origin
    n_rows = 3 if banded else 5
    m = ds.model
    with fig.shift_origin(xshift=f'{(W + GAP_X)/2}c', yshift=f'{n_rows*ROW}c'):
        if m.attrs['name'] == 'dipoles':
            fig.basemap(projection=PROJ, region='d', frame=['g', '+tPlanted dipoles (circles)'])
            fig.coast(land='gray90', shorelines='0.2p,gray30')
            fig.plot(x=symbols[0], y=symbols[1], style='c0.22c', fill='black', pen='0.5p,white')
            fig.text(x=symbols[0], y=symbols[1], text=[str(i + 1) for i in range(len(symbols[0]))],
                     font='7p,Helvetica-Bold', offset='0.25c/0.2c', justify='LM')
        else:
            v = np.nanpercentile(np.abs(m.values), 99.5)
            cmap = 'vik' if np.nanmin(m.values) < 0 else 'lajolla'
            vis = for_gmt(m)
            if cmap == 'vik':
                pygmt.makecpt(cmap=cmap, series=[-nice(v), nice(v)], background=True)
            else:
                pygmt.makecpt(cmap=cmap, series=[0, nice(v)], background=True)
            fig.grdimage(vis, projection=PROJ, region='d', cmap=True, nan_transparent=True,
                         frame=['g', f'+tThe model: {m.attrs["name"]}'])
            fig.coast(shorelines='0.2p,gray30', projection=PROJ, region='d')
            if symbols is not None:
                fig.plot(x=symbols[0], y=symbols[1], style='c0.18c', pen='0.6p,black', projection=PROJ, region='d')
            fig.colorbar(position=f'JBC+w{0.7*W}c/0.16c+o0/0.25c+h+e', frame=[f'x+l{m.attrs["units"]}'])

    for ia, alt in enumerate(ds.altitude_km.values):
        dip = ds.dipole_br.isel(altitude_km=ia)
        rem = ds.remit_br.sel(lmax=L).isel(altitude_km=ia)
        diff = rem - dip
        vfield = nice(np.percentile(np.abs(dip.values), 99.9))
        vdiff = nice(np.percentile(np.abs(diff.values), 95))     # saturated, so thin edge lines show; max in subtitle
        rows = [(dip, vfield, f'Dipole sum (reference), {alt} km', None, 'Br (nT)'),
                (rem, vfield, f'{tested}, degrees 1-{L}, {alt} km', None, 'Br (nT)'),
                (diff, vdiff, f'{tested} minus dipole sum', stats(diff.values), 'nT')]
        if not banded:
            above = ds.remit_br.sel(lmax=L).isel(altitude_km=ia) - ds.remit_br.sel(lmax=top).isel(altitude_km=ia)
            left = ds.remit_br.sel(lmax=top).isel(altitude_km=ia) - dip
            rows += [(above, vdiff, f'Minus the field above degree {L}', f'from {above_from}, degrees {L + 1}-{top}',
                      'nT'),
                     (left, nice(np.percentile(np.abs(left.values), 95)), f'{above_from} to degree {top} minus dipole sum',
                      stats(left.values), 'nT')]
        for ir, (da, vmax, t, sub, label) in enumerate(rows):
            x = ia*(W + GAP_X)
            y = (n_rows - 1 - ir)*ROW
            with fig.shift_origin(xshift=f'{x}c', yshift=f'{y}c'):
                panel(fig, da, vmax, t, label, symbols=symbols, sub=sub)
    fig.savefig(OUT / f'{prefix}_{case}.png', dpi=150)
    print(f'{prefix}_{case}.png')


def gallery(prefix='forward', case='dipoles', L=185, tested='remit'):
    """zoomed maps around each planted dipole or cell (azimuthal equidistant, centred on it), one figure per altitude:
    columns as the rows of case_figure, rows one source each"""
    ds = xr.load_dataset(RES / f'{prefix}_{case}.nc')
    top = int(ds.lmax.max())
    w = 3.0
    heads = ['Dipole sum', f'{tested}, degrees 1-{L}', f'{tested} minus dipole sum', f'Minus field above {L}',
             f'To {top}, minus dipole sum']
    for ia, alt in enumerate(ds.altitude_km.values):
        horizon = (6 if alt == 100 else 12) + (2 if case == 'cells' else 0)     # degrees from the source shown
        fig = pygmt.Figure()
        pygmt.config(FONT_TITLE='7p', FONT_ANNOT_PRIMARY='5.5p', FONT_LABEL='6p', MAP_TITLE_OFFSET='2p',
                     MAP_FRAME_TYPE='plain', MAP_GRID_PEN_PRIMARY='0.15p,gray60', MAP_FRAME_PEN='0.5p')
        n = ds.sizes['dipole']
        for k in range(n):
            lon, lat = float(ds.dipole_lon[k]), float(ds.dipole_lat[k])
            proj = f'E{lon}/{lat}/{horizon}/{w}c'
            dist = np.degrees(np.arccos(np.clip(
                np.sin(np.radians(lat))*np.sin(np.radians(ds.lat))
                + np.cos(np.radians(lat))*np.cos(np.radians(ds.lat))*np.cos(np.radians(ds.lon - lon)), -1, 1)))
            near = dist <= horizon                            # colour scales from the shown area only
            dip = ds.dipole_br.isel(altitude_km=ia)
            rem = ds.remit_br.sel(lmax=L).isel(altitude_km=ia)
            hi = ds.remit_br.sel(lmax=top).isel(altitude_km=ia)
            fields = [dip, rem, rem - dip, rem - hi, hi - dip]
            vf = nice(float(np.abs(dip.where(near)).max()))
            vd = nice(float(np.abs((rem - dip).where(near)).max()))
            vl = nice(float(np.abs((hi - dip).where(near)).max()))
            for j, (da, v) in enumerate(zip(fields, [vf, vf, vd, vd, vl])):
                with fig.shift_origin(xshift=f'{j*(w + 0.55)}c', yshift=f'{(n - 1 - k)*(w + 1.6)}c'):
                    pygmt.makecpt(cmap='vik', series=[-v, v], background=True)
                    fig.grdimage(for_gmt(da), projection=proj, region='g', cmap=True, frame=['g5'])
                    fig.coast(shorelines='0.2p,gray30', projection=proj, region='g')
                    fig.plot(x=[lon], y=[lat], style='c0.12c', pen='0.5p,black', projection=proj, region='g')
                    fig.colorbar(position=f'JBC+w{0.8*w}c/0.12c+o0/0.15c+h+e', frame=['xa'])
                    if k == 0:
                        fig.text(position='TC', text=heads[j], font='7p,Helvetica-Bold', justify='BC',
                                 offset='0c/0.75c', no_clip=True)
                    if j == 0:
                        what = '' if case == 'cells' else f'{ds.dipole_direction.values[k]}, '
                        fig.text(position='TL', text=f'{k + 1}. {ds.dipole_label.values[k]}, {what}{alt} km',
                                 font='6.5p,Helvetica-Oblique',
                                 justify='BL', offset='0c/0.2c', no_clip=True)
                    if j in (2, 4):
                        a = da.where(near).values
                        fig.text(position='BC', text=f'max {fmt(np.nanmax(np.abs(a)))} nT', font='6p',
                                 justify='TC', offset='0c/-0.65c', no_clip=True)
        fig.savefig(OUT / f'{prefix}_{case}_zoom_{alt}km.png', dpi=170)
        print(f'{prefix}_{case}_zoom_{alt}km.png')


def tables():
    """markdown tables of the summaries, included by the document"""
    f = pd.read_csv(RES / 'forward_summary.csv')
    f = f[(f.lmax == 185) | (f.lmax == f.groupby('case').lmax.transform('max'))]     # production and highest lmax
    lines = ['| Case | Altitude | lmax | RMS of the field | remit minus dipole sum: max (nT) | rms (nT) | '
             'field above lmax: rms (nT) |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in f.itertuples():
        above = '' if np.isnan(r.rel_rms_above_L) else f'{r.rel_rms_above_L*r.rms_dipole_nT:.2e}'
        lines.append(f'| {r.case} | {r.altitude_km} km | {r.lmax} | {r.rms_dipole_nT:.3g} nT | '
                     f'{r.max_abs_diff_nT:.2e} | {r.rms_diff_nT:.2e} | {above} |')
    (OUT / '_forward_table.md').write_text('\n'.join(lines) + '\n')
    if (RES / 'columns_summary.csv').exists():
        c = pd.read_csv(RES / 'columns_summary.csv')
        c = c[c.altitude_km.isna() | (c.lmax == 185) | (c.lmax == c.groupby('case').lmax.transform('max'))]
        lines = ['| Case | Altitude | lmax | RMS of the field | columns minus dipole sum: max (nT) | rms (nT) | '
                 'field above lmax: rms (nT) |', '|---|---:|---:|---:|---:|---:|---:|']
        for r in c[c.altitude_km.notna()].itertuples():
            above = '' if np.isnan(r.rel_rms_above_L) else f'{r.rel_rms_above_L*r.rms_dipole_nT:.2e}'
            lines.append(f'| {r.case} | {int(r.altitude_km)} km | {r.lmax} | {r.rms_dipole_nT:.3g} nT | '
                         f'{r.max_abs_diff_nT:.2e} | {r.rms_diff_nT:.2e} | {above} |')
        lines += ['', '| Case | Coefficient check | Max relative difference |', '|---|---|---:|']
        for r in c[c.altitude_km.isna()].itertuples():
            lines.append(f'| {r.case} | {r.check} | {r.value:.1e} |')
        (OUT / '_columns_table.md').write_text('\n'.join(lines) + '\n')


FORWARD = [('dipoles', 185, False), ('dipoles_bl', 150, True), ('checker', 185, False), ('checker_bl', 150, True),
           ('checker_radial', 185, False), ('healpix', 185, False), ('healpix_bl', 150, True),
           ('uniform', 185, True), ('geology', 185, False), ('geology_depth', 185, False)]
COLUMNS = [('cells', 185, False), ('cont_checker', 185, False), ('cont_geology', 185, False)]

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    for case, L, banded in FORWARD:
        if (RES / f'forward_{case}.nc').exists():
            if case == 'dipoles':
                gallery()
            else:
                case_figure('forward', case, L, case, banded)
    for case, L, banded in COLUMNS:
        if (RES / f'columns_{case}.nc').exists():
            if case == 'cells':
                gallery('columns', 'cells', L, tested='columns')
            else:
                case_figure('columns', case, L, case, banded, tested='columns', above_from="remit's depth series")
    tables()
