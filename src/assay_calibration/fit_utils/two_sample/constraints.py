import scipy.stats as sps
import numpy as np
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
    param_sets, xlims: Tuple[float, float]
) -> bool:
    """
    For each pair of distributions i and i+1 in param_sets,
    check if the density ratio of distribution i to distribution i+1 is monotonic

    Arguments:
    param_sets: list of tuples:
        each tuple is (a, loc, scale) : skewness, location, scale parameters
        of a skew normal distribution

    xlims: tuple : (xmin, xmax) : range of x values to check the density ratio

    Returns:
    bool : True if any density ratio is not monotonic (constraint violated), False otherwise
    """
    x_values = np.linspace(*xlims, 1000)
    log_pdfs = [sps.skewnorm.logpdf(x_values, *params) for params in param_sets]

    for i in range(len(log_pdfs) - 1):
        if not np.all(np.diff(log_pdfs[i] - log_pdfs[i + 1]) < 0):
            return True

    return False
