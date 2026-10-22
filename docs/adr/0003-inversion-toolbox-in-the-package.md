# The inversion toolbox is part of the installable package

The inversion code moves out of `notebooks/` (where it is imported through PYTHONPATH) into the installable package as `remit.inversion`, and the forward-model parts of `notebooks/depth_models.py` (source depth, slab columns, inducing field, LCS-1 filter) move into `remit`, because the toolbox's columns depend on them. `notebooks/depth_models.py` stays as a thin re-export so the paper notebooks run unchanged; `notebooks/` keeps only run scripts, configs, figures and the paper notebooks. Optional dependencies (pygplates, gprm and ptt for outlines, cvxpy for bounds, torch for the GPU) are imported only where used, so the core installs without them.

## Considered Options

- Keep the toolbox in `notebooks/domains/`: rejected, because it cannot be installed or tested as a unit, and every script depends on PYTHONPATH.
