"""
Inversion toolbox: estimating crustal VIS from satellite field models (docs/inversion-toolbox-spec.md).

Terms are defined in CONTEXT.md at the repository root; decisions in docs/adr/.
  config     inversion configs (TOML, inheritance, explore/final mode defaults), resolved config and its hash
  cache      content-addressed store of expensive intermediate results
  geometry   grid nodes, target areas, cells, layers and direction models
  columns    cell columns: Gauss coefficients of 1 SI.km of VIS in each cell, by node sum or by transform
"""
