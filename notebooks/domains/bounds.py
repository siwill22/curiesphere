"""
Extremal bounds (Parker's ideal-body idea, on the cells): the smallest and largest values of a linear
summary c'v of the VIS (e.g. the mean of a craton) over all models that fit the data and the bounds,
    min / max c'v  subject to  |Gw v - dw| <= sqrt(chi2_target),  lo <= v <= hi.
This is a second-order cone programme, solved exactly by an interior-point method (cvxpy with
Clarabel). With positivity alone the answer says what the data force, whatever the regularisation.
Optionally a roughness limit |D v| <= rough_max (D: graph.difference_operator, the gradient between
cells) is added, to show how much smoothness it takes to pin a quantity down.
When there are more data than cells the misfit is first reduced exactly with a QR factorisation,
|Gw v - dw|^2 = |R v - Q'dw|^2 + |dw - QQ'dw|^2 (smaller and better conditioned); if Clarabel still
fails, SCS is tried.
"""
import numpy as np
import cvxpy as cp


def extremes(Gw, dw, c, chi2_target, lo=0., hi=4., D=None, rough_max=None, solver='CLARABEL'):
    """(min, max) of c'v, the two extremal models and the misfit of each"""
    v = cp.Variable(Gw.shape[1])
    A, b, target = Gw, dw, chi2_target
    if Gw.shape[0] > Gw.shape[1]:
        Q, A = np.linalg.qr(Gw)
        b = Q.T @ dw
        target = chi2_target - float(np.sum((dw - Q @ b)**2))
        if target <= 0:
            raise ValueError(f'bounds: no model reaches chi2 {chi2_target:.0f} (the part of the data outside '
                             f'the span of Gw alone exceeds it)')
    cons = [cp.norm(A @ v - b) <= np.sqrt(target), v >= lo, v <= hi]
    if rough_max is not None:
        cons.append(cp.norm(D @ v) <= rough_max)
    out = {}
    for name, sense in (('min', cp.Minimize), ('max', cp.Maximize)):
        prob = cp.Problem(sense(c @ v), cons)
        try:
            prob.solve(solver=solver)
        except cp.error.SolverError:
            prob.solve(solver='SCS', eps=1e-7, max_iters=200000)
        if prob.status not in ('optimal', 'optimal_inaccurate'):
            raise ValueError(f'bounds: {prob.status} (no model within the bounds reaches chi2 {chi2_target:.0f}?)')
        x = np.clip(v.value, lo, hi)
        out[name] = dict(value=float(c @ x), v=x, chi2=float(np.sum((Gw @ x - dw)**2)), status=prob.status,
                         rough=float(np.linalg.norm(D @ x)) if D is not None else None)
    return out


def mean_functional(area, members):
    """c with c'v = area-weighted mean of v over the cells in members (boolean)"""
    w = area*members
    return w/np.sum(w)
