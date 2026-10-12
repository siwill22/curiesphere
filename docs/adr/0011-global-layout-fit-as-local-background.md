# Local fits take the global layout fit as their background

A local fit subtracts a background for everything outside its target: the ocean model over the oceans, and a model of the continental crust elsewhere. That continental background is now the best global layout fit (GCM with province levels and the ocean-model error, ADRs 0008–0009: `gcm_glc_layouts_oc_lcs1_16-133.nc`) instead of GCM-GLC. It is the better model of the same crust by about 600 log units, it describes the crust outside a target with the same province levels the local fits use inside it, and its ocean-model error term keeps ocean-model errors off coastal continental cells, which are where local fits are most sensitive. With it, every continent's DF fit (Gaussian means) has a higher log evidence (Africa +30, South America +22, North America +9, Australia +5, Antarctica +10, Eurasia +60, Zealandia +10) and a smaller misfit in the ring of data around the target. The comparison is fair because the data are identical and the background is fixed before the local fit.

## Considered Options

- GCM-GLC (one level everywhere), the earlier background: kept as a check. It is the cleaner choice when a local fit tests the polygons against geology-blind layouts, since the layout fit carries polygon information into the crust outside the target.
- DF-GLC (one fractal pattern globally, ADR 0010): not tried as a background yet.

## Consequences

Both backgrounds are fitted to the same data, including the data over the target, so the mild reuse of data is unchanged. New local configs inherit `[fit.background] outside = "results/gcm_glc_layouts_oc_lcs1_16-133.nc"` (the `_laybg` configs).
