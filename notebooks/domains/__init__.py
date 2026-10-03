"""
Domain inversion framework (docs/domain-inversion-framework.md): plain functions shared by the
merging, Voronoi optimisation and Voronoi rjMCMC methods.

    region   outline file -> extent on the 0.1 deg grid; Hasterok helper
    basis    cells, cell columns, data points, background, data, noise, coarse basis
    outputs  cell-graph edges, boundary probability per edge, trade-off records
"""
