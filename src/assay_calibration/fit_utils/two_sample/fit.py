from .update_steps import constrained_em_iteration, get_sample_weights, em_iteration
from .density_utils import get_likelihood
from .initializations import kmeans_init
from . import constraints

import numpy as np
import logging
from tqdm.auto import tqdm


def single_fit(observations, sample_indicators, **kwargs):
    CONSTRAINED = kwargs.get("Constrained", True)
    MAX_N_ITERS = kwargs.get("max_iters", 10000)
    verbose = kwargs.get("verbose", True)
    xlims = (observations.min(), observations.max())
    N_samples = sample_indicators.shape[1]
    N_components = 2
    W = np.ones((N_samples, N_components)) / N_components
    try:
        initial_params, kmeans = kmeans_init(observations, n_clusters=N_components)
    except ValueError:
        logging.warning("Failed to initialize")
        return dict(
            component_params=[[] for _ in range(N_components)],
            weights=W,
            likelihoods=[-1 * np.inf],
        )
    W = get_sample_weights(observations, sample_indicators, initial_params, W)
    history = [dict(component_params=initial_params, weights=W)]
    # initial likelihood
    likelihoods = np.array(
        [
            get_likelihood(observations, sample_indicators, initial_params, W)
            / len(sample_indicators),
        ]
    )
    # Check for bad initialization
    try:
        updated_component_params, updated_weights = constrained_em_iteration(
            observations, sample_indicators, initial_params, W, xlims, iterNum=0
        )
    except ZeroDivisionError:
        logging.warning("ZeroDivisionError")
        return dict(
            component_params=initial_params,
            weights=W,
            likelihoods=[*likelihoods, -1 * np.inf],
            kmeans=kmeans,
        )
    likelihoods = np.array(
        [
            *likelihoods,
            get_likelihood(
                observations,
                sample_indicators,
                updated_component_params,
                updated_weights,
            )
            / len(sample_indicators),
        ]
    )
    # Run the EM algorithm
    if verbose:
        pbar = tqdm(total=MAX_N_ITERS, leave=False, desc="EM Iteration")

    for i in range(MAX_N_ITERS):
        history.append(
            dict(component_params=updated_component_params, weights=updated_weights)
        )
        if np.isnan(likelihoods).any():
            raise ValueError()
        if np.isnan(np.concatenate(updated_component_params)).any():
            raise ValueError()
        if np.isnan(updated_weights).any():
            raise ValueError()
        if np.isnan(np.concatenate(updated_component_params)).any():
            raise ValueError(
                f"NaN in updated component params at iteration {i}\n{updated_component_params}"
            )
        if np.isnan(updated_weights).any():
            raise ValueError(
                f"NaN in updated weights at iteration {i}\n{updated_weights}"
            )
        # observations = np.array([np.random.choice(observation_replicates) for observation_replicates in replicates]).reshape(-1,)
        # try:
        if CONSTRAINED:
            updated_component_params, updated_weights = constrained_em_iteration(
                observations,
                sample_indicators,
                updated_component_params,
                updated_weights,
                xlims,
                iterNum=i + 1,
            )
        else:
            updated_component_params, updated_weights = em_iteration(
                observations,
                sample_indicators,
                updated_component_params,
                updated_weights,
            )
        likelihoods = np.array(
            [
                *likelihoods,
                get_likelihood(
                    observations,
                    sample_indicators,
                    updated_component_params,
                    updated_weights,
                )
                / len(sample_indicators),
            ]
        )
        if not CONSTRAINED and i > 0 and (likelihoods[-1] < likelihoods[-2]):
            raise ValueError(
                f"Likelihood decreased at iteration {i} for unconstrained fit"
            )
        if kwargs.get("verbose", True):
            pbar.set_postfix({"likelihood": f"{likelihoods[-1]:.6f}"})  # type: ignore
            pbar.update(1)  # type: ignore
        if (
            kwargs.get("early_stopping", True)
            and i >= 1
            and (np.abs(likelihoods[-1] - likelihoods[-2]) < 1e-10).all()
        ):
            break
    history.append(
        dict(component_params=updated_component_params, weights=updated_weights)
    )
    if kwargs.get("verbose", True):
        pbar.close()  # type: ignore
    if CONSTRAINED:
        assert not constraints.density_constraint_violated(
            updated_component_params[0], updated_component_params[1], xlims
        )

    return dict(
        component_params=updated_component_params,
        weights=updated_weights,
        likelihoods=likelihoods,
        history=history,
        kmeans=kmeans,
    )
