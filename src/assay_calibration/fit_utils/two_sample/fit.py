from .update_steps import em_iteration, get_sample_weights
from .density_utils import get_likelihood
from .initializations import kmeans_init
from . import constraints

import numpy as np
import logging
from tqdm.auto import tqdm


def single_fit(observations, sample_indicators, N_components, constrained, **kwargs):
    """
    Fit a two-component mixture model to the observations using the EM algorithm.

    Parameters
    ----------
    observations : np.ndarray
        1D array of observations (e.g., scores).
    sample_indicators : np.ndarray
        2D one-hot encoded array indicating sample membership for each observation.
    constrained : bool
        Whether to enforce component-pair density ratio constraints

    Optional Parameters (kwargs)
    -------------------------


    max_em_iters : int, default=10000
        Maximum number of EM iterations.

    verbose : bool, default=True
        Whether to display a progress bar.

    initial_weights : np.ndarray, optional
        Optional initial weights for the samples. If provided along with initial_params, these will be used to initialize the EM algorithm.
            Otherwise, the model will be initialized using k-means initialization.

    initial_params : list of np.ndarray, optional
        Optional initial parameters for the mixture components. If provided along with initial_weights, these will be used to initialize the EM algorithm.
            Otherwise, the model will be initialized using k-means initialization.

    early_stopping : bool, default=True
        Whether to stop the EM algorithm early if the likelihood converges.

    Returns
    -------
    dict
        A dictionary containing:
        - 'component_params': List of parameters for each mixture component.
        - 'weights': Final weights for each sample.
        - 'likelihoods': Array of likelihood values at each iteration.
        - 'history': List of dictionaries containing component parameters and weights at each iteration.
        - 'kmeans': KMeans object used for initialization (if applicable), or None.
    """
    MAX_EM_ITERS = kwargs.get("max_em_iters", 10000)
    verbose = kwargs.get("verbose", True)
    xlims = (observations.min(), observations.max())
    N_samples = sample_indicators.shape[1]
    if (
        kwargs.get("initial_weights", None) is not None
        and kwargs.get("initial_params", None) is not None
    ):
        kmeans = None
        # Start with user provided initialization
        initial_params = kwargs.get("initial_params", [])
        W = np.array(kwargs.get("initial_weights"))
        if W.shape != (N_samples, N_components):
            raise ValueError(
                f"Initial weights shape {W.shape} does not match number of samples {N_samples}"
            )
        if len(initial_params) != N_components:
            raise ValueError(
                f"Initial params length {len(initial_params)} does not match number of components {N_components}"
            )
    else:
        # Run Initialization
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
        updated_component_params, updated_weights = em_iteration(
            observations,
            sample_indicators,
            initial_params,
            W,
            constrained,
            xlims,
            iterNum=0,
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
        pbar = tqdm(total=MAX_EM_ITERS, leave=False, desc="EM Iteration")

    for i in range(MAX_EM_ITERS):
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
        updated_component_params, updated_weights = em_iteration(
            observations,
            sample_indicators,
            updated_component_params,
            updated_weights,
            constrained,
            xlims,
            iterNum=i + 1,
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
        if not constrained and i > 0 and (likelihoods[-1] < likelihoods[-2]):
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
    if constrained and constraints.density_constraint_violated(
        updated_component_params[0], updated_component_params[1], xlims
    ):
        raise ValueError("Final parameters violate density constraint")

    return dict(
        component_params=updated_component_params,
        weights=updated_weights,
        likelihoods=likelihoods,
        history=history,
        kmeans=kmeans,
    )
