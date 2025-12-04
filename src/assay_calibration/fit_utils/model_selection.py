from src.assay_calibration.fit_utils.two_sample.fit import single_fit
from src.assay_calibration.data_utils.dataset import Scoreset
from tqdm.auto import trange
import json
from pathlib import Path
import numpy as np
import scipy.stats as sps
from typing import List, Dict
from joblib import Parallel, delayed
from src.assay_calibration.fit_utils.fit import makeOneHot, sample_specific_bootstrap
from src.assay_calibration.fit_utils.two_sample.density_utils import get_likelihood


def bootstrapped_likelihood_ratio_test(scoreset, N_bootstraps, initial_bootstrap_seed, **kwargs):
    """
    Required Args:
    ------------------
    - scoreset : Scoreset : Scoreset to use in model selection
    - N_bootstraps : int : number of bootstrap iterations to run in the bootstrapped likelihood ratio test
    - initial_bootstrap_seed : int : bootstrap_seed to use for the observed data train/val split

    Optional kwargs:
    ------------------
    - N_restarts : int (default 100) : Number of restarts to in each bootstrap iteration
    - init_method : str in {'kmeans','method_of_moments'} (default 'kmeans'): what method to use in initializing the component parameters
    - constrained : bool (default True) : run the fits enforcing the monotonicity constraint
    - init_constraint_adjustment : str in {'skew','scale'} : which parameter to adjust to initially satisfy density constraints
    - save_filepath : Optional[str|Path] : file where the model selection results are saved

    Returns:
    ---------------
    - selection_results : dict
    """
    N_restarts = kwargs.get("N_restarts", 100)
    constrained = kwargs.get("constrained", True)
    init_method = kwargs.get("init_method", "kmeans")
    init_constraint_adjustment = kwargs.get("init_constraint_adjustment", "scale")
    scores = scoreset.scores
    sample_assignments = scoreset.sample_assignments
    sample_assignments = makeOneHot(sample_assignments)
    mask = sample_assignments.any(1) & (~np.isnan(scores))
    scores = scores[mask]
    sample_assignments = sample_assignments[mask]
    train_indices, val_indices = sample_specific_bootstrap(sample_assignments,bootstrap_seed=initial_bootstrap_seed)
    scores_train, sample_assignments_train = (
        scores[train_indices],
        sample_assignments[train_indices],
    )
    scores_val, sample_assignments_val = (
        scores[val_indices],
        sample_assignments[val_indices],
    )
    model_two_comps = fit_iteration(
        scores_train,
        sample_assignments_train,
        2,
        constrained,
        init_method,
        init_constraint_adjustment,
        N_restarts,
    )
    model_three_comps = fit_iteration(
        scores_train,
        sample_assignments_train,
        2,
        constrained,
        init_method,
        init_constraint_adjustment,
        N_restarts,
    )
    likelihood_two_comp = get_likelihood(
        scores_val,
        sample_assignments_val,
        model_two_comps["component_params"],
        model_two_comps["weights"],
    )
    likelihood_three_comp = get_likelihood(
        scores_val,
        sample_assignments_val,
        model_three_comps["component_params"],
        model_three_comps["weights"],
    )
    observed_test_statistic = -2 * (likelihood_two_comp - likelihood_three_comp)
    sample_sizes = sample_assignments.sum(0)
    test_statistics = np.zeros(N_bootstraps)
    selection_results = {
        "observed_model_2_comp": model_two_comps,
        "observed_model_3_comp": model_three_comps,
        "observed_test_statistic": observed_test_statistic,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    bootstrap_results = Parallel(n_jobs=-1, verbose=N_bootstraps)(
        delayed(run_bootstrap_iter)(
            model_two_comps["component_params"],
            model_two_comps["weights"],
            sample_sizes,
            constrained,
            init_method,
            init_constraint_adjustment,
            N_restarts,
        )
        for bootstrap_num in trange(N_bootstraps)
    )
    selection_results["bootstrap_results"] = bootstrap_results
    for i, result in enumerate(bootstrap_results):
        if result is None:
            raise ValueError(f"Bootstrap iteration {i} failed")
        test_statistics[i] = result["test_statistic"]
    p_value = (1 + (test_statistics >= observed_test_statistic).sum()) / (1 + N_bootstraps)
    selection_results["p_value"] = p_value
    selection_results["kwargs"] = kwargs
    save_filepath = kwargs.get("save_filepath", None)
    if save_filepath is not None:
        save_filepath = Path(save_filepath)
        save_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(save_filepath, "w") as f:
            json.dump(serialize_dict(selection_results), f)
        print(f"Model selection results written to {save_filepath}")
    print(f"Model selection p-value: {p_value}")
    return selection_results


def run_bootstrap_iter(
    component_params,
    weights,
    sample_sizes,
    constrained,
    init_method,
    init_constraint_adjustment,
    N_restarts,
):
    simulated_scores, simulated_sample_assignments = generate_scoreset(
        component_params, weights, sample_sizes
    )
    train_indices, val_indices = sample_specific_bootstrap(simulated_sample_assignments)
    scores_train, sample_assignments_train = (
        simulated_scores[train_indices],
        simulated_sample_assignments[train_indices],
    )
    scores_val, sample_assignments_val = (
        simulated_scores[val_indices],
        simulated_sample_assignments[val_indices],
    )
    bootstrap_model_k2 = fit_iteration(
        scores_train,
        sample_assignments_train,
        2,
        constrained,
        init_method,
        init_constraint_adjustment,
        N_restarts,
    )
    bootstrap_model_k3 = fit_iteration(
        scores_train,
        sample_assignments_train,
        3,
        constrained,
        init_method,
        init_constraint_adjustment,
        N_restarts,
    )
    likelihood_two_comp = get_likelihood(
        scores_val,
        sample_assignments_val,
        bootstrap_model_k2["component_params"],
        bootstrap_model_k2["weights"],
    )
    likelihood_three_comp = get_likelihood(
        scores_val,
        sample_assignments_val,
        bootstrap_model_k3["component_params"],
        bootstrap_model_k3["weights"],
    )

    bootstrap_result = {
        "test_statistic": -2 * (likelihood_two_comp - likelihood_three_comp),
        "bootstrapped_model_2_comp": bootstrap_model_k2,
        "bootstrapped_model_3_comp": bootstrap_model_k3,
    }
    return serialize_dict(bootstrap_result)


def serialize_dict(d: Dict) -> Dict:
    """
    Recursively serializes every value in a dictionary to ensure it can be written using json.dump.
    """
    if isinstance(d, dict):
        return {k: serialize_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [serialize_dict(v) for v in d]
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, float):  # Handles np.float64, np.float32, and Python floats
        return float(d)
    elif isinstance(d, int):  # Handles np.int64, np.int32, and Python ints
        return int(d)
    elif isinstance(d, bool):  # Handles np.bool_ and Python bools
        return bool(d)
    else:
        try:
            json.dumps(d)
        except TypeError:
            return None
        return d


def fit_iteration(
    scores,
    sample_assignments,
    n_components,
    constrained,
    init_method,
    init_constraint_adjustment,
    N_restarts,
):
    fits: List[Dict] = [
        single_fit(
            scores,
            sample_assignments,
            n_components,
            constrained,
            init_method,
            init_constraint_adjustment,
        )
        for _ in trange(N_restarts)
    ]
    # Sort fits by increasing likelihood
    fits.sort(key=lambda d: d["likelihoods"][-1])
    # Find iteration with best likelihood
    best_fit = fits[-1]
    return best_fit


def generate_scoreset(params, weights, sample_sizes):
    samples = []
    sample_assignments = []
    assert weights.shape[1] == len(params)
    assert len(sample_sizes) == weights.shape[0]
    n_samples = len(sample_sizes)
    for sampleNum, (sample_weights, sample_size) in enumerate(
        zip(weights, sample_sizes)
    ):
        comp_sizes = np.round(sample_weights * sample_size).astype(int)
        for compParams, compSize in zip(params, comp_sizes):
            if compSize <= 0:
                continue
            samples.append(
                sps.skewnorm.rvs(
                    compParams[0], loc=compParams[1], scale=compParams[2], size=compSize
                )
            )
            sa = np.zeros((compSize, n_samples), dtype=bool)
            sa[:, sampleNum] = 1
            sample_assignments.append(sa)
    return np.concatenate(samples), np.concatenate(sample_assignments)
