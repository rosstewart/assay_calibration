from typing import List, Tuple
import numpy as np
import scipy.stats as sps


def draw_sample(
    params: List[Tuple[float, float, float]], weights: np.ndarray, sample_size: int = 1
) -> np.ndarray:
    """
    Draw a list of samples from a mixture of skew normal distributions

    Required Arguments:
    --------------------------------
    params -- List[Tuple[float]] len(NComponents)
        The parameters of the skew normal components
    weights -- Ndarray (NComponents,)
        The mixture weights of the components

    Optional Arguments:
    --------------------------------
    sample_size -- int (default 1)
        The number of observations to draw

    Returns:
    --------------------------------
    samples -- Ndarray (sample_size,)
        The drawn sample
    """
    samples = []
    for i in range(sample_size):
        k = np.random.binomial(1, weights[1])
        samples.append(sps.skewnorm.rvs(*params[k]))
    return np.array(samples)
