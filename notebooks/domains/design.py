"""
Design matrices for the linear models (run_linear_models.py, run_null_regression.py): VIS = B theta,
with named columns (see regions/<region>/hypotheses.toml)
  intercept       one mean level (prior LEVEL)
  class           one mean per province class, in place of the intercept (prior LEVEL each)
  crust, lab, heat_flow, heat_flow_<alt>, age   standardised covariates (prior SLOPE per sd)
  undated, craton, group:<prov_group>           indicator offsets (prior SLOPE)
GROUPS are the covariate groups whose subsets make the automatic hypotheses.
"""
import numpy as np

from . import covariates
from .linear import V_LO, V_HI

LEVEL, SLOPE = (V_LO, V_HI), (-2., 2.)
GROUPS = {'class': ['class'], 'crust': ['crust'], 'age': ['age', 'undated'], 'heat_flow': ['heat_flow'],
          'lab': ['lab']}


def columns(groups):
    """design columns for a set of covariate groups (the class means replace the intercept)"""
    cols = [] if 'class' in groups else ['intercept']
    return cols + [c for g in GROUPS if g in groups for c in GROUPS[g]]


def design(cols, cov, area):
    """design matrix, coefficient bounds and names for a list of columns; cov: covariates.npz arrays"""
    n = len(area)
    B, lo, hi, names = [], [], [], []

    def add(x, bounds, name):
        B.append(np.asarray(x, float)); lo.append(bounds[0]); hi.append(bounds[1]); names.append(name)
    for c in cols:
        if c == 'intercept':
            add(np.ones(n), LEVEL, 'intercept')
        elif c == 'class':
            for k, cname in enumerate(covariates.CLASS_NAMES):
                add(cov['cls'] == k, LEVEL, f'mean {cname}')
        elif c == 'undated':
            add(cov['undated'], SLOPE, 'undated')
        elif c == 'craton':
            add(np.isin(cov['group'], covariates.CRATON_GROUPS), SLOPE, 'craton')
        elif c.startswith('group:'):
            add(cov['group'] == c[6:], SLOPE, c)
        else:
            add(covariates.standardise(cov[c], area), SLOPE, c)
    return np.stack(B, 1), np.array(lo), np.array(hi), names


def indicator(label):
    """one column per unit of a partition"""
    lab = np.unique(label, return_inverse=True)[1]
    return np.stack([lab == k for k in range(lab.max() + 1)], 1).astype(float)
