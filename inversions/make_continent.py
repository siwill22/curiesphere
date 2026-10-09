"""
Target-area outlines and domain tables for other continents, as for Australia: the Hasterok et al. (2022) provinces
(gprm) of the continent with continental or transitional crust, less the units in EXCLUDE (outlying arcs, ocean
ridges, units across the dateline). Each province is its own domain (param = its name; margins are their own
provinces). Writes outlines/<tag>.gpkg (the province polygons, lon/lat WGS84) and domains/<tag>_domains.csv (the
columns of australia_domains.csv); edit the CSV by hand to share a value between provinces.

    python make_continent.py nam|sam|afr        (env curiesphere, from inversions/)
"""
import os
import re
import sys
import warnings

warnings.filterwarnings('ignore')

CONTINENTS = {
    'nam': ('North America', [
        # Central America and the Caribbean
        'Chortis Block', 'Mosquito Lowlands', 'Central American Arc', 'Central American Forearc',
        'Chorotega Volcanic Arc', 'Chorotega Backarc', 'Choco Block', 'Nicaragua Plateau', 'Cayman Arc',
        'Greater Antilles Arc', 'Greater Antilles Accretionary Complex', 'South Hispaniola Wedge',
        # Aleutian arc
        'East Aleutian Arc', 'East Aleutian Forearc',
        # across the dateline: Arctic Ocean ridges and basins, Siberian units, the Bering Sea margin
        'Alpha Mendeleev Ridge', 'Lomonosov Ridge', 'Podvodnikov Basin', 'Unnamed Fold Belt',
        'Novosibirsk-Chukhi Fold Belt', 'Bennett-Borrovia Block', 'North Chuchi Basin', 'Bering Sea Margin']),
    'sam': ('South America', []),
    'afr': ('Africa', []),
}


def main(tag):
    from gprm.datasets import Geology
    name, exclude = CONTINENTS[tag]
    prov = Geology.fetch_GlobalTectonicMap()
    sel = prov[(prov.continent == name) & prov.crust_type.isin(['continental', 'transitional'])
               & ~prov.prov_name.isin(exclude)].copy()
    param = sel.prov_name.map(lambda s: re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_'))
    dup = param.duplicated(keep=False)
    sel['param'] = param.where(~dup, param + '_' + sel.index.astype(str))
    os.makedirs('outlines', exist_ok=True)
    out = f'outlines/{tag}.gpkg'
    if os.path.exists(out):
        os.remove(out)
    sel[['prov_name', 'geometry']].set_crs(4326, allow_override=True).to_file(out, driver='GPKG')
    t = sel.reset_index().rename(columns={'index': 'key'})
    t[['key', 'id', 'prov_name', 'prov_type', 'crust_type', 'prov_group', 'param']].to_csv(
        f'domains/{tag}_domains.csv', index=False)
    print(f'{name}: {len(sel)} provinces ({len(exclude)} names excluded) -> {out}, domains/{tag}_domains.csv')


if __name__ == '__main__':
    main(sys.argv[1])
