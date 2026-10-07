# Cells are the smallest parameter unit

Every parameterisation in the inversion toolbox is a grouping of HEALPix cells (NSIDE 64 by default, finer when needed): its columns are the cached cell columns combined by a fixed matrix, G_param = G_cells S. Domain boundaries therefore follow cell edges, about 100 km at NSIDE 64. We do not compute polygon-exact domains from the 0.1° grid nodes: the data resolve nothing finer than about 150 km, and one column cache then serves every method (cell maps, polygon and free domains, covariates) with no new forward calculation.

## Considered Options

- Domains defined directly by grid nodes (exact polygon edges): rejected, because each domain set needs its own columns and the extra boundary precision is below the data's resolution.
