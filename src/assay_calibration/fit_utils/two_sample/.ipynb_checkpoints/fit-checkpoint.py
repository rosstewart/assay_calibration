from .update_steps import em_iteration, get_sample_weights
from .density_utils import get_likelihood
from .initializations import kmeans_init, methodOfMomentsInit, fix_to_satisfy_density_constraint
from . import constraints

import numpy as np
import logging
from tqdm.auto import tqdm


def single_fit(
    observations, sample_indicators, N_components, constrained, init_method, **kwargs
):
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
    init_method : kmeans or method_of_moments

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

    submerge_steps : int, optional
        Optional max number of initial steps to explore without constraint. constrained must be set to true.

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
    submerge_steps = kwargs.get("submerge_steps", None)
    if submerge_steps is not None and not constrained:
        raise ValueError("constrained must be True when submerge_steps is not None.")
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
        W = np.ones((N_samples, N_components)) / N_components

        assert init_method == "method_of_moments" or init_method == "kmeans"
        initial_params = None
        
        if init_method == "method_of_moments":
            kmeans = "method_of_moments"
            initial_params = methodOfMomentsInit(observations, N_components)

        # init_method is kmeans or was a failed method of moments (fall back to kmeans)
        if initial_params is None:
            if init_method == "method_of_moments" and verbose:
                print('failed method of moments, falling back to kmeans')
                
            # Run Initialization
            try:
                initial_params, kmeans = kmeans_init(
                    observations, n_clusters=N_components
                )
            except ValueError:
                logging.warning("Failed to initialize")
                return dict(
                    component_params=[[] for _ in range(N_components)],
                    weights=W,
                    likelihoods=[-1 * np.inf],
                )
        # elif verbose:
        #     print("method of moments succeeded")

        # if verbose:
        #     print(f'init_method: {init_method}, params: {initial_params}')
            
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


    # print('submerge_steps',submerge_steps)
    if submerge_steps is None:
        # continue as normal for loop
        for i in range(MAX_EM_ITERS):
            history.append( # i+1 will be index of history dict (because of initial params)
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
            # if i == 1:  # When early stopping triggers
            #     print(f"Likelihood at iter 0: {likelihoods[-2]:.12f}")
            #     print(f"Likelihood at iter 1: {likelihoods[-1]:.12f}")
            #     print(f"Difference: {likelihoods[-1] - likelihoods[-2]:.12e}")
            #     print(f"History: {history}")
            if i > 0 and (likelihoods[-1] < likelihoods[-2]):
                decrease = likelihoods[-2] - likelihoods[-1]
                
                relative_decrease = decrease / abs(likelihoods[-2])
                
                is_numerical_error = relative_decrease < 1e-6
                
                if is_numerical_error:
                    print(f"Iteration {i}: Likelihood decreased by {decrease:.2e} (relative: {relative_decrease:.2e}) - likely numerical rounding error")
                else:
                    print(f"Iteration {i}: Likelihood DECREASED by {decrease:.2e} (relative: {relative_decrease:.2e}) - algorithmic issue")
                
                # Continue if it's just numerical error
                if is_numerical_error:
                    # Small enough to ignore, continue iteration
                    pass
                else:
                    # Return failed fit
                    return dict(
                        component_params=updated_component_params,
                        weights=updated_weights,
                        likelihoods=[*likelihoods, -1 * np.inf],
                        kmeans=kmeans,
                    )
                # raise ValueError(
                #     f"Likelihood decreased at iteration {i} for unconstrained fit"
                # )
            if kwargs.get("verbose", True):
                pbar.set_postfix({"likelihood": f"{likelihoods[-1]:.6f}"})  # type: ignore
                pbar.update(1)  # type: ignore
            if (
                kwargs.get("early_stopping", True)
                and i >= 1
                and (np.abs(likelihoods[-1] - likelihoods[-2]) < 1e-10).all()
            ):
                history.append(
                    dict(component_params=updated_component_params, weights=updated_weights)
                )
                break
    else:
        # submerge mode must rewind; for loop is not suitable
        submerge_mode = False
        i = 0

        history.append(
            dict(component_params=updated_component_params, weights=updated_weights)
        )

        last_surface_idx = -1
        while i < MAX_EM_ITERS:
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

            if not submerge_mode:
                print(f"Before em_iteration {i}:")
                print(f"  Params: {updated_component_params}")
            updated_component_params, updated_weights = em_iteration(
                observations,
                sample_indicators,
                updated_component_params,
                updated_weights,
                constrained=not submerge_mode,
                xlims=xlims,
                iterNum=i + 1
            )
            
            if not submerge_mode:
                # save only last run with constraint for later rewind
                history.append(  # i+1 will be index of history dict (because of initial params)
                    dict(component_params=updated_component_params, weights=updated_weights)
                )
                last_surface_idx = i
            
            if not submerge_mode:
                print(f"After em_iteration {i}:")  
                print(f"  Params: {updated_component_params}\n")


            # if not submerge mode or came back up for air
            if not submerge_mode or (submerge_mode and not constraints.multicomponent_density_constraint_violated(updated_component_params, xlims)):
                # in exploitation mode; continue as normal

                if submerge_mode:
                    # came out of water; append latest history
                    print(f'constraint satisfied: came out of water from i={i} to i={len(history)-2}')
                    submerge_mode = False
                    i = last_surface_idx+1
                    history.append(  # i+1 will be index of history dict (because of initial params)
                        dict(component_params=updated_component_params, weights=updated_weights)
                    )

                print('iteration',i)
                    
            
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
                # if i == 1:  # When early stopping triggers
                #     print(f"Likelihood at iter 0: {likelihoods[-2]:.12f}")
                #     print(f"Likelihood at iter 1: {likelihoods[-1]:.12f}")
                #     print(f"Difference: {likelihoods[-1] - likelihoods[-2]:.12e}")
                #     print(f"History: {history}")
                if i > 0 and (likelihoods[-1] < likelihoods[-2]):
                    decrease = likelihoods[-2] - likelihoods[-1]
                    
                    # Determine if this is numerical rounding error
                    # Machine epsilon for float64 is ~2.2e-16
                    # Relative decrease compared to likelihood magnitude
                    relative_decrease = decrease / abs(likelihoods[-2])
                    
                    # If relative decrease is less than sqrt(machine epsilon) * 100, 
                    # it's likely numerical rounding
                    is_numerical_error = relative_decrease < 1e-6
                    
                    if is_numerical_error:
                        print(f"Iteration {i}: Likelihood decreased by {decrease:.2e} (relative: {relative_decrease:.2e}) - likely numerical rounding error")
                    else:
                        print(f"Iteration {i}: Likelihood DECREASED by {decrease:.2e} (relative: {relative_decrease:.2e}) - algorithmic issue")
                    
                    # Continue if it's just numerical error
                    if is_numerical_error:
                        # Small enough to ignore, continue iteration
                        pass
                    else:
                        # Return failed fit
                        return dict(
                            component_params=updated_component_params,  # Fixed: was [[]*N_components]
                            weights=updated_weights,
                            likelihoods=[*likelihoods, -1 * np.inf],
                            kmeans=kmeans,
                        )
                    # raise ValueError(
                    #     f"Likelihood decreased at iteration {i} for unconstrained fit"
                    # )
                if kwargs.get("verbose", True):
                    pbar.set_postfix({"likelihood": f"{likelihoods[-1]:.6f}"})  # type: ignore
                    pbar.update(1)  # type: ignore
                if (
                    kwargs.get("early_stopping", True)
                    and i >= 1
                    and (np.abs(likelihoods[-1] - likelihoods[-2]) < 1e-10).all()
                ):
                    print('early stopping')
                    break

                submerge_mode = True # go back underwater
            else:
                # in exploration mode; constraint was violated
                if i - last_surface_idx >= submerge_steps:
                    # rewind back to last time constraint was satisfied. perform another constrained iteration, then go back underwater
                    print(f'after {submerge_steps} steps underwater: constraint not satisfied, rewinding from i={i} to i={last_surface_idx} (len(history)={len(history)})')
                    submerge_mode = False
                    i = last_surface_idx # i += 1 at end
                    # updated_component_params, updated_weights = history[-1]['component_params'], history[-1]['weights']
                    # fixed_underwater_params = fix_to_satisfy_density_constraint(updated_component_params, xlims)
                    # if len(fixed_underwater_params) == 0 or any(len(p) == 0 for p in fixed_underwater_params):
                    updated_component_params, updated_weights = history[-1]['component_params'], history[-1]['weights']
                        # print('could not fix underwater params to density constraint')
                    # else:
                    #     updated_component_params = fixed_underwater_params
                    #     print('fixed underwater params to density constraint')

                    if submerge_steps == 1:
                        submerge_steps = 0
                    else:
                        submerge_steps = submerge_steps // 2 # half on each attempt
                    
            
            i += 1
    
    # history.append(
    #     dict(component_params=updated_component_params, weights=updated_weights)
    # )
    if kwargs.get("verbose", True):
        pbar.close()  # type: ignore
    if constrained and constraints.multicomponent_density_constraint_violated(
        updated_component_params, xlims
    ):
        raise ValueError("Final parameters violate density constraint")

    return dict(
        component_params=updated_component_params,
        weights=updated_weights,
        likelihoods=likelihoods,
        history=history,
        kmeans=kmeans,
        xlims=xlims
    )
