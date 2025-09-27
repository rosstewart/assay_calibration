from . import density_utils
from .constraints import multicomponent_density_constraint_violated
import numpy as np
import scipy.stats as sps


def em_iteration(
    observations,
    sample_indicators,
    current_canonical_params,
    current_weights,
    constrained,
    xlims,
    **kwargs,
):
    """
    Perform one iteration of the EM algorithm with the density constraint

    Arguments:
    observations: np.array (N,) : observed instances
    sample_indicators: np.array (N, S) : indicator matrix of which sample each observation belongs to
    current_canonical_params: list of tuples : [(a, loc, scale)_1, ..., (a, loc, scale)_K] : skewness, location, scale parameters of each component
    current_weights: np.array (S, K) : weights of each component for each sample
    constrained: bool : whether to enforce the density constraint
    xlims: tuple : (xmin, xmax) : range of x values to check the density ratio

    Returns:
    updated_component_params: list of tuples : [(a, loc, scale)_1, ..., (a, loc, scale)_K] : updated skewness, location, scale parameters of each component
    updated_weights: np.array (S, K) : updated weights of each component for each sample
    """
    
    if constrained and multicomponent_density_constraint_violated(
        current_canonical_params, xlims
    ):
        raise ValueError("density constraint violated at start of em iteration,")
    N, S = sample_indicators.shape
    K = len(current_canonical_params)
    assert current_weights.shape == (S, K)
    sample_indicators = validate_indicators(sample_indicators)
    assert sample_indicators.shape == (N, S)
    responsibilities = sample_specific_responsibilities(
        observations, sample_indicators, current_canonical_params, current_weights
    )
    current_alternate_params = [
        density_utils.canonical_to_alternate(*p) for p in current_canonical_params
    ]
    updated_alternate_params = [
        list(density_utils.canonical_to_alternate(*p)) for p in current_canonical_params
    ]
    for component_num in range(K):
        if constrained and multicomponent_density_constraint_violated(
            [
                density_utils.alternate_to_canonical(*p)
                for p in updated_alternate_params
            ],
            xlims,
        ):
            raise ValueError(
                f"constraint violated at start of component {component_num} update"
            )
        candidate_location = get_location_update(
            observations,
            responsibilities[component_num],
            current_canonical_params[component_num],
        )
        if constrained:
            constrained_updated_loc = get_constrained_location_update(
                candidate_location,
                component_num,
                current_alternate_params,
                updated_alternate_params,
                xlims,
                **kwargs,
            )
        else:
            constrained_updated_loc = candidate_location
        updated_alternate_params[component_num][0] = constrained_updated_loc
        if constrained and multicomponent_density_constraint_violated(
            [
                density_utils.alternate_to_canonical(*p)
                for p in updated_alternate_params
            ],
            xlims,
        ):
            raise ValueError(
                f"density constraint violated after updating loc for component {component_num}"
            )
        candidateDelta = get_Delta_update(
            constrained_updated_loc,
            observations,
            responsibilities[component_num],
            current_canonical_params[component_num],
        )
        if constrained:
            constrained_updated_Delta = get_constrained_Delta_update(
                candidateDelta,
                constrained_updated_loc,
                component_num,
                current_alternate_params,
                updated_alternate_params,
                xlims,
                **kwargs,
            )
        else:
            constrained_updated_Delta = candidateDelta
        updated_alternate_params[component_num][1] = constrained_updated_Delta
        if constrained and multicomponent_density_constraint_violated(
            [
                density_utils.alternate_to_canonical(*p)
                for p in updated_alternate_params
            ],
            xlims,
        ):
            raise ValueError(
                f"density constraint violated after updating Delta for component {component_num}"
            )
        candidateGamma = get_Gamma_update(
            constrained_updated_loc,
            constrained_updated_Delta,
            observations,
            responsibilities[component_num],
            current_canonical_params[component_num],
        )
        if constrained:
            constrained_updated_Gamma = get_constrained_Gamma_update(
                candidateGamma,
                updated_alternate_params,
                component_num,
                xlims,
                **kwargs,
            )
        else:
            constrained_updated_Gamma = candidateGamma

        updated_alternate_params[component_num][2] = constrained_updated_Gamma
        if constrained and multicomponent_density_constraint_violated(
            [
                density_utils.alternate_to_canonical(*p)
                for p in updated_alternate_params
            ],
            xlims,
        ):
            raise ValueError(
                f"constraint violated after updating component {component_num} iter {kwargs.get('iterNum',-1)}\n{updated_alternate_params}\n{current_alternate_params}"
            )
    updated_canonical = [
        density_utils.alternate_to_canonical(*p) for p in updated_alternate_params
    ]
    if constrained and multicomponent_density_constraint_violated(
        updated_canonical, xlims
    ):
        raise ValueError("failed to maintain monotonicity through end of em iteration")
    updated_weights = get_sample_weights(
        observations, sample_indicators, updated_canonical, current_weights  # type: ignore
    )
    return updated_canonical, updated_weights


def get_constrained_location_update(
    candidate_location,
    component_num,
    current_alternate_params,
    updated_alternate_params,
    xlims,
    **kwargs,
):
    """
    Perform binary search to get a location update near the unconstrained update that satisfies the density constraint
    """
    bsearch_params = []
    for ki in range(len(current_alternate_params)):
        if ki < component_num:
            bsearch_params.append(updated_alternate_params[ki])
        else:
            bsearch_params.append(current_alternate_params[ki])
    constrained_updated_loc = binary_search(
        candidate_location,
        bsearch_params,
        component_num,
        0,
        xlims,
        msg=f"loc_{component_num} iter {kwargs.get('iterNum',-1)}",
    )
    return constrained_updated_loc


def get_constrained_Delta_update(
    candidate_Delta,
    constrained_updated_loc,
    component_num,
    current_alternate_params,
    updated_alternate_params,
    xlims,
    **kwargs,
):
    K = len(current_alternate_params)
    bsearch_params = []
    for ki in range(K):
        if ki < component_num:
            bsearch_params.append(updated_alternate_params[ki])
        elif ki > component_num:
            bsearch_params.append(current_alternate_params[ki])
        else:
            currentDelta, currentGamma = current_alternate_params[component_num][1:]
            bsearch_params.append((constrained_updated_loc, currentDelta, currentGamma))
    constrained_updated_Delta = binary_search(
        candidate_Delta,
        bsearch_params,
        component_num,
        1,
        xlims,
        msg=f"Delta_{component_num} iter {kwargs.get('iterNum',-1)}",
    )
    return constrained_updated_Delta


def get_constrained_Gamma_update(
    candidate_Gamma,
    updated_alternate_params,
    component_num,
    xlims,
    **kwargs,
):
    constrained_updated_Gamma = binary_search(
        candidate_Gamma,
        updated_alternate_params,
        component_num,
        2,
        xlims,
        msg=f"Gamma_{component_num} iter {kwargs.get('iterNum',-1)}",
    )
    return constrained_updated_Gamma


def get_sample_weights(
    observations, sample_indicators, updated_canonical_params, current_weights
):
    epsilon = 1e-300  # Small value to prevent zeros
    
    updated_weights = np.zeros_like(current_weights)
    for i in range(current_weights.shape[0]):  # for each sample
        sample_observations = observations[
            sample_indicators[:, i]
        ]  # get all observations for this sample
        posts = density_utils.component_posteriors(
            sample_observations, updated_canonical_params, current_weights[i]
        )  # Get the component posteriors (responsibilities) using the updated component params and current weights
        updatedWeight = posts.mean(1)

        # Prevent zero weights
        updatedWeight = np.maximum(updatedWeight, epsilon)
        updatedWeight = updatedWeight / updatedWeight.sum()  # Renormalize
        
        if np.isnan(updatedWeight).any():
            nanLocs = np.where(np.isnan(posts.T))[0]
            raise ValueError(
                f"about to set updated weight to {updatedWeight}\n{sample_observations[nanLocs]}\n{updated_canonical_params}\n{current_weights[i]}\n{nanLocs}\n{posts.T[nanLocs]}"
            )
        updated_weights[i] = updatedWeight
    return updated_weights


# def get_likelihood(observations, sample_indicators, component_params, weights):
#     Likelihood = 0.0
#     for sample_num, sample_mask in enumerate(sample_indicators.T):
#         X = observations[sample_mask]
#         sample_likelihood = density_utils.joint_densities(
#             X, component_params, weights[sample_num]
#         ).sum(axis=0)
#         Likelihood += np.log(sample_likelihood).sum().item()
#     return Likelihood


def sample_specific_responsibilities(
    observations, sample_indicators, component_params, weights
):
    """
    For each observation calculate the posteriors with respect to each component and that observation's sample's component weights

    Arguments:
    observations: np.array (N,) : observed instances
    sample_indicators: np.array (N, S) : indicator matrix of which sample each observation belongs to
    component_params: list of tuples : [(a, loc, scale)_1, ..., (a, loc, scale)_K] : skewness, location, scale parameters of each component
    weights: np.array (S, K) : weights of each component for each sample

    Returns:
    responsibilities: np.array (K, N) : posterior probabilities of each component given x, conditioned on the observed instance's sample weights
    """
    N_samples = sample_indicators.shape[1]
    N_components = len(component_params)
    N_observations = len(observations)
    assert weights.shape == (N_samples, N_components)

    responsibilities = np.zeros((N_components, N_observations))
    for i, sample_mask in enumerate(sample_indicators.T):
        X = observations[sample_mask]
        responsibilities[:, sample_mask] = density_utils.component_posteriors(
            X, component_params, weights[i]
        )
    return responsibilities


def validate_indicators(Indicators):
    assert Indicators.ndim == 2
    assert (Indicators.sum(1) == 1).all()
    assert np.isin(Indicators, [0, 1]).all()
    return Indicators.astype(bool)


def get_location_update(observations, responsibilities, canonical_component_params):
    """
    Calculate the location update for the given component

    Arguments:
    observations: np.array (N,) : observed instances
    responsibilities: np.array (N,) : posterior probabilities of each component given x, conditioned on the observed instance's sample weights
    canonical_component_params: tuple : (a, loc, scale) : skewness, location, scale parameters of the component from the previous iteration

    Returns:
    updated_loc: float : updated location parameter
    """
    assert observations.shape == responsibilities.shape
    v, w = get_truncated_normal_moments(observations, canonical_component_params)
    (_, Delta, Gamma) = density_utils.canonical_to_alternate(
        *canonical_component_params
    )
    m = observations - v * Delta
    return (m * responsibilities).sum() / responsibilities.sum()


def get_Delta_update(
    updated_loc, observations, responsibilities, canonical_component_params
):
    """
    Calculate the Delta update for the given component

    Arguments:
    updated_loc: float : updated location parameter from this iteration
    observations: np.array (N,) : observed instances
    responsibilities: np.array (N,) : posterior probabilities of each component given x, conditioned on the observed instance's sample weights
    canonical_component_params: tuple : (a, loc, scale) : skewness, location, scale parameters of the component from the previous iteration

    Returns:
    updated_Delta: float : updated Delta parameter
    """

    assert observations.shape == responsibilities.shape
    v, w = get_truncated_normal_moments(observations, canonical_component_params)
    d = v * (observations - updated_loc)
    return (d * responsibilities).sum() / responsibilities.sum()


def get_Gamma_update(
    updated_loc,
    updated_Delta,
    observations,
    responsibilities,
    canonical_component_params,
):
    """
    Calculate the Gamma update for the given component

    Arguments:
    updated_loc: float : updated location parameter from this iteration
    updated_Delta: float : updated Delta parameter from this iteration
    observations: np.array (N,) : observed instances
    responsibilities: np.array (N,) : posterior probabilities of each component given x, conditioned on the observed instance's sample weights
    canonical_component_params: tuple : (a, loc, scale) : skewness, location, scale parameters of the component from the previous iteration

    Returns:
    updated_Gamma: float : updated Gamma parameter
    """
    assert observations.shape == responsibilities.shape
    v, w = get_truncated_normal_moments(observations, canonical_component_params)
    g = (
        (observations - updated_loc) ** 2
        - (2 * updated_Delta * v * (observations - updated_loc))
        + (updated_Delta**2 * w)
    )
    gamma_update = (g * responsibilities).sum() / responsibilities.sum()
    
    return gamma_update


def trunc_norm_moments(mu, sigma):
    """Array trunc norm moments"""
    cdf = sps.norm.cdf(mu / sigma)
    flags = cdf == 0
    pdf = sps.norm.pdf(mu / sigma)
    p = np.zeros_like(pdf)
    p[~flags] = pdf[~flags] / cdf[~flags]
    p[flags] = abs(mu[flags] / sigma)

    m1 = mu + sigma * p
    m2 = mu**2 + sigma**2 + sigma * mu * p
    return m1, m2


def get_truncated_normal_moments(observations, canonical_component_params):
    _delta = density_utils._get_delta(canonical_component_params)
    loc, scale = canonical_component_params[1:]
    truncated_normal_loc = _delta / scale * (observations - loc)
    truncated_normal_scale = np.sqrt(1 - _delta**2)
    v, w = trunc_norm_moments(truncated_normal_loc, truncated_normal_scale)
    return v, w


def binary_search(
    candidate_value,
    current_alternate_params,
    component_index,
    parameter_index,
    xlims,
    msg="",
):
    """
    Perform a binary search to find the value of the parameter that satisfies the pairwise component density ratio constraint

    Arguments:
    candidate_value: float : candidate value of the parameter ** IN ALTERNATE PARAMETERIZATION **
    current_alternate_params: list [tuple : (mu, Delta, Gamma) ]: alternate parameterization of each of the K components
    component_index: int : index of the component to update
    parameter_index: int : index of the parameter to update
    xlims: tuple : (xmin, xmax) : range of x values to check the density ratio

    Returns:
    float : updated parameter value
    """

    if multicomponent_density_constraint_violated(
        [
            list(density_utils.alternate_to_canonical(*param))
            for param in current_alternate_params
        ],
        xlims,
    ):
        raise ValueError(f"constraint already violated before bsearch {msg}")
    lower_bound = current_alternate_params[component_index][parameter_index]
    upper_bound = candidate_value
    search_iter = 0
    while abs(upper_bound - lower_bound) > 1e-12:
        search_iter += 1
        updated_alternate_params = [list(p) for p in current_alternate_params]
        updated_alternate_params[component_index][parameter_index] = lower_bound
        if multicomponent_density_constraint_violated(
            list(
                map(
                    lambda tup: density_utils.alternate_to_canonical(*tup),
                    updated_alternate_params,
                )
            ),
            xlims,
        ):
            raise ValueError(
                f"lower bound does not satisfy constraint at search iter {search_iter} - component {component_index} - parameter {parameter_index}"
            )
        midpoint = (upper_bound + lower_bound) / 2

        updated_alternate_params[component_index][parameter_index] = midpoint
        if multicomponent_density_constraint_violated(
            list(map(lambda tup: density_utils.alternate_to_canonical(*tup), updated_alternate_params)),  # type: ignore
            xlims,
        ):
            upper_bound = midpoint
        else:
            lower_bound = midpoint
    verify_binary_search_result(
        lower_bound, current_alternate_params, component_index, parameter_index, xlims
    )
    return lower_bound


def verify_binary_search_result(
    constrained_Alternate_parameter_value,
    current_alternate_params,
    component_index,
    parameter_index,
    xlims,
):
    current_alternate_component = list(current_alternate_params[component_index])
    current_alternate_component[parameter_index] = constrained_Alternate_parameter_value
    if multicomponent_density_constraint_violated(
        [density_utils.alternate_to_canonical(*p) for p in current_alternate_params],
        xlims,
    ):
        raise ValueError(
            f"The binary search result for the {parameter_index } parameter update for component {component_index} does not satify the constraint"
        )
