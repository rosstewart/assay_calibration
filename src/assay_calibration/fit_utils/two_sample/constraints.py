import scipy.stats as sps
import numpy as np
from . import density_utils
from typing import Tuple


def density_constraint_violated(params_1, params_2, xlims: Tuple[float, float]) -> bool:
    """
    Check if the density ratio of distribution 1 to distribution 2 is monotonic

    Arguments:
    params_1: tuple : (a, loc, scale) : skewness, location, scale parameters of distribution 1
    params_2: tuple : (a, loc, scale) : skewness, location, scale parameters of distribution 2
    xlims: tuple : (xmin, xmax) : range of x values to check the density ratio

    Returns:
    bool : True if the density ratio is not monotonic (constraint violated), False otherwise
    """

    log_pdf_1 = sps.skewnorm.logpdf(np.linspace(*xlims, 1000), *params_1)
    log_pdf_2 = sps.skewnorm.logpdf(np.linspace(*xlims, 1000), *params_2)

    return not np.all(np.diff(log_pdf_1 - log_pdf_2) < 0)


def multicomponent_density_constraint_violated(
    param_sets, xlims: Tuple[float, float], tolerance=0
) -> bool:
    """
    For each pair of distributions i and i+1 in param_sets,
    check if the density ratio of distribution i to distribution i+1 is monotonic

    Arguments:
    param_sets: list of tuples:
        each tuple is (a, loc, scale) : skewness, location, scale parameters
        of a skew normal distribution

    xlims: tuple : (xmin, xmax) : range of x values to check the density ratio
    tolerance: float (default 0) : tolerance to allow for numerical precision errors
                                       when converting parameters from alternate to canonical

    Returns:
    bool : True if any density ratio is not monotonic (constraint violated), False otherwise
    """
    x_values = np.linspace(*xlims, 1000)
    log_pdfs = [sps.skewnorm.logpdf(x_values, *params) for params in param_sets]

    for i in range(len(log_pdfs) - 1):
        diffs = np.diff(log_pdfs[i] - log_pdfs[i + 1])
        if np.any(diffs > tolerance): 
            return True
    
    return False
    
    # for i in range(len(log_pdfs) - 1):
    #     if not np.all(np.diff(log_pdfs[i] - log_pdfs[i + 1]) < 0):
    #         return True

    # return False

import numpy as np
import scipy.special as sc

def skewnorm_logpdf_mu_delta_gamma(x, mu, Delta, Gamma):
    """
    Log-PDF of a skew-normal distribution parameterized by (mu, Delta, Gamma).

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the log-density.
    mu : float
        Location parameter.
    Delta : float
        Skewness scaling parameter.
    Gamma : float
        Variance-like scale parameter (must be > 0).

    Returns
    -------
    logpdf : ndarray
        Log density evaluated at x.
    """
    x = np.asarray(x)
    if Gamma <= 0:
        raise ValueError("Gamma must be positive.")

    z = (Delta / np.sqrt(Gamma)) * (x - mu)

    # log φ part
    log_norm_const = -0.5 * np.log(2 * np.pi * Gamma)
    quad_term = -0.5 * ((x - mu) ** 2) / Gamma

    # log Φ part (use stable log_ndtr)
    log_cdf = sc.log_ndtr(z)

    return log_norm_const + quad_term + np.log(2.0) + log_cdf




def positive_likelihood_ratio_montonicity_constraint_violated(
    component_param_sets,
    weights_pathogenic,
    weights_benign,
    xlims: Tuple[float, float],
    epsilon=1e-12,
):
    x_values = np.linspace(*xlims, 1000)
    f_path = density_utils.mixture_pdf(
        x_values, component_param_sets, weights_pathogenic
    )
    f_benign = density_utils.mixture_pdf(x_values, component_param_sets, weights_benign)
    log_lr_plus = np.log(f_path) - np.log(f_benign)
    return (np.diff(log_lr_plus) > 0).any()
