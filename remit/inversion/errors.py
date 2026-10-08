"""
The error model (CONTEXT.md): noise and model error, each a per-degree spectrum with an amplitude.

  noise          the Lowes spectrum of (field model - F x reference model), degrees >= 16 (the reference's
                 lowest), F the field model's resolution filter so its damping is not counted as error
  model error    optional: the field model's own spectral shape, with a fixed amplitude (RMS Br, nT)
  amplitudes     'fixed' (the spectrum as it is) or 'evidence' (a scale fitted with the model, by the core)

For coefficient data an isotropic field is diagonal: each coefficient of degree l has variance
R_l/((l+1)(2l+1)) for a Lowes spectrum R_l.
"""
import numpy as np

LMIN_REFERENCE = 16


def lowes(c):
    """Lowes-Mauersberger spectrum R_l of a (2, l, m) coefficient array"""
    l = np.arange(c.shape[1])
    return (l + 1)*(c**2).sum((0, 2))


def noise_spectrum(c, ref, F):
    """R_l of c - F ref, zero below the reference's lowest degree"""
    diff = c - ref*F[None, :, None]
    diff[:, :LMIN_REFERENCE] = 0
    return lowes(diff)


def coefficient_variance(R):
    l = np.arange(len(R))
    return R/((l + 1.)*(2*l + 1.))


def scaled_to_rms_br(R, band, rms):
    """the spectrum R rescaled so that its field in the band has RMS Br = rms (nT)"""
    var = coefficient_variance(R)[band.l]
    return R*rms**2/np.sum(band.mean_square_br()*var)
