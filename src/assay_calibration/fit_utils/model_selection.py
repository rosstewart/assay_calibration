from src.assay_calibration.fit_utils.two_sample.fit import single_fit
from src.assay_calibration.data_utils.dataset import Scoreset
from tqdm.auto import trange
import json, gzip
from pathlib import Path
import numpy as np
import scipy.stats as sps
from typing import List, Dict, Optional, Tuple
from joblib import Parallel, delayed
from src.assay_calibration.fit_utils.fit import makeOneHot, sample_specific_bootstrap
from src.assay_calibration.fit_utils.two_sample.density_utils import get_likelihood, component_posteriors
import pickle
from scipy import stats

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
    - bootstrap_fits_path : Optional[str|Path] : path to pre-computed bootstrap fits
    - selection_strategy : str in {'best', 'median'} (default 'best') : how to select representative model from bootstrap fits

    Returns:
    ---------------
    - selection_results : dict
    """
    N_restarts = kwargs.get("N_restarts", 100)
    constrained = kwargs.get("constrained", True)
    init_method = kwargs.get("init_method", "kmeans")
    init_constraint_adjustment = kwargs.get("init_constraint_adjustment", "scale")
    selection_strategy = kwargs.get("selection_strategy", "median")
    
    scores = scoreset.scores
    sample_assignments = scoreset.sample_assignments
    sample_assignments = makeOneHot(sample_assignments)
    mask = sample_assignments.any(1) & (~np.isnan(scores))
    scores = scores[mask]
    sample_assignments = sample_assignments[mask]
    scoreset_name = scoreset.scoreset_name
    with open('/data/ross/assay_calibration/val_counts.pkl','rb') as f:
        val_counts = pickle.load(f)

    bootstrap_fits_path = kwargs.get("bootstrap_fits_path", None)
    
    if bootstrap_fits_path is not None:
        # Load pre-computed bootstrap fits
        with gzip.open(bootstrap_fits_path, 'rt') as f:
            boot_results = json.load(f)[scoreset_name]
        
        results_2c = [boot_results[key]["2c"] for key in boot_results.keys()]
        results_3c = [boot_results[key]["3c"] for key in boot_results.keys()]
        
        model_two_fits = [_fit['fit'] for _fit in results_2c]
        model_two_val_lls = np.array([_fit['val_ll'] for _fit in results_2c])
        model_three_fits = [_fit['fit'] for _fit in results_3c]
        model_three_val_lls = np.array([_fit['val_ll'] for _fit in results_3c])
        
        # Select representative models based on strategy
        if selection_strategy == 'best':
            # Select model with best validation likelihood
            best_2c_idx = np.argmax(model_two_val_lls)
            best_3c_idx = np.argmax(model_three_val_lls)
            model_two_comps = model_two_fits[best_2c_idx]
            model_three_comps = model_three_fits[best_3c_idx]
            likelihood_two_comp = model_two_val_lls[best_2c_idx]
            likelihood_three_comp = model_three_val_lls[best_3c_idx]
        elif selection_strategy == 'median':
            # Select model closest to median validation likelihood
            median_2c_idx = np.argsort(model_two_val_lls)[len(model_two_val_lls) // 2]
            median_3c_idx = np.argsort(model_three_val_lls)[len(model_three_val_lls) // 2]
            model_two_comps = model_two_fits[median_2c_idx]
            model_three_comps = model_three_fits[median_3c_idx]
            likelihood_two_comp = model_two_val_lls[median_2c_idx]
            likelihood_three_comp = model_three_val_lls[median_3c_idx]
        else:
            raise ValueError(f"Unknown selection_strategy: {selection_strategy}")

                    
        n_val = np.median([val_counts[(scoreset_name, boot_idx)] 
                          for boot_idx in range(1000) 
                          if (scoreset_name, boot_idx) in val_counts])
        
        likelihood_two_comp *= n_val # correct to whole-dataset log likelihood
        likelihood_three_comp *= n_val
        
        print(f"Using pre-computed bootstrap fits ({selection_strategy} strategy)")
        print(f"  2-comp validation LL: {likelihood_two_comp:.4f}")
        print(f"  3-comp validation LL: {likelihood_three_comp:.4f}")
    
    else:
        # Fit models from scratch
        train_indices, val_indices = sample_specific_bootstrap(
            sample_assignments, bootstrap_seed=initial_bootstrap_seed
        )
        scores_train, sample_assignments_train = (
            scores[train_indices],
            sample_assignments[train_indices],
        )
        scores_val, sample_assignments_val = (
            scores[val_indices],
            sample_assignments[val_indices],
        )
        n_val = len(scores_val)
        
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
            3,  # Fixed: was 2, should be 3
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
    
    # Calculate observed test statistic
    observed_test_statistic = -2 * (likelihood_two_comp - likelihood_three_comp) / n_val
    # observed_test_statistic = -2 * (likelihood_two_comp - likelihood_three_comp)
    print(f"observed_test_statistic: {observed_test_statistic}")
    sample_sizes = sample_assignments.sum(0)
    
    selection_results = {
        "observed_model_2_comp": model_two_comps,
        "observed_model_3_comp": model_three_comps,
        "observed_test_statistic": observed_test_statistic,
        "observed_likelihood_2_comp": float(likelihood_two_comp),
        "observed_likelihood_3_comp": float(likelihood_three_comp),
    }
    
    # Run parametric bootstrap under null hypothesis (2 components)
    print(f"Running {N_bootstraps} parametric bootstrap iterations...")
    bootstrap_results = [
        run_bootstrap_iter(
            model_two_comps["component_params"],
            model_two_comps["weights"],
            sample_sizes,
            constrained,
            init_method,
            init_constraint_adjustment,
            N_restarts,
        )
        for _ in trange(N_bootstraps)
    ]
    
    # Extract test statistics
    test_statistics = np.zeros(N_bootstraps)
    selection_results["bootstrap_results"] = bootstrap_results
    for i, result in enumerate(bootstrap_results):
        if result is None:
            raise ValueError(f"Bootstrap iteration {i} failed")
        test_statistics[i] = result["test_statistic"]
        print(f"bootstrap {i} test statistic: {test_statistics[i]}, ll2 {result['likelihood_two_comp']}, ll3 {result['likelihood_three_comp']}")
    
    # Calculate p-value with continuity correction
    p_value = (1 + (test_statistics >= observed_test_statistic).sum()) / (1 + N_bootstraps)
    selection_results["p_value"] = float(p_value)
    selection_results["test_statistics"] = test_statistics.tolist()
    selection_results["kwargs"] = kwargs
    
    # Save results
    save_filepath = kwargs.get("save_filepath", None)
    if save_filepath is not None:
        save_filepath = Path(save_filepath)
        save_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(save_filepath, "w") as f:
            json.dump(serialize_dict(selection_results), f, indent=2)
        print(f"Model selection results written to {save_filepath}")
    
    print(f"\nLikelihood Ratio Test Results:")
    print(f"  Observed test statistic: {observed_test_statistic:.6f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Decision (α=0.05): {'Reject null (use 3 components)' if p_value < 0.05 else 'Fail to reject (use 2 components)'}")
    
    return selection_results


def bootstrap_paired_test(
    boot_results: Dict,
    k_range: List[int] = [2, 3],
    alpha: float = 0.05,
    verbose: bool = True
) -> Dict:
    """
    Paired statistical test on bootstrap validation likelihoods.
    
    Tests if k=3 is significantly better than k=2 using paired Wilcoxon test.
    More powerful than unpaired tests since bootstrap samples are correlated.
    
    Args:
    -----
    - boot_results : Dict : bootstrap results dictionary
    - k_range : List[int] : should be [2, 3] for this test
    - alpha : float : significance level (default 0.05)
    - verbose : bool : print results
    
    Returns:
    --------
    - dict with test results and selected k

    CONSERVATIVE BEST
    """
    if len(k_range) != 2:
        raise ValueError("Paired test only works for comparing 2 models")
    
    k_low, k_high = sorted(k_range)
    
    # Extract paired validation likelihoods
    val_lls_low = []
    val_lls_high = []
    
    for key in sorted(boot_results.keys(), key=int):
        k_low_str = f"{k_low}c"
        k_high_str = f"{k_high}c"
        
        if k_low_str in boot_results[key] and k_high_str in boot_results[key]:
            val_lls_low.append(boot_results[key][k_low_str]['val_ll'])
            val_lls_high.append(boot_results[key][k_high_str]['val_ll'])
    
    val_lls_low = np.array(val_lls_low)
    val_lls_high = np.array(val_lls_high)
    
    # Paired differences: positive = k_high better
    differences = val_lls_high - val_lls_low
    
    # Wilcoxon signed-rank test (non-parametric paired test)
    # H0: median difference <= 0 (k_high not better)
    # Ha: median difference > 0 (k_high significantly better)
    statistic, p_value = stats.wilcoxon(differences, alternative='greater')
    
    # Also compute effect size and confidence interval
    mean_diff = differences.mean()
    median_diff = np.median(differences)
    std_diff = differences.std()
    
    # 95% confidence interval for mean difference (bootstrap percentile)
    ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
    fifth_percentile = np.percentile(differences, 5)
    
    # Select k based on test
    selected_k = k_high if p_value < alpha else k_low
    conservative_k = k_low if fifth_percentile <= 0 else k_high
    
    if verbose:
        print("\n" + "="*70)
        print("BOOTSTRAP PAIRED TEST")
        print("="*70)
        print(f"\nComparing k={k_high} vs k={k_low} on {len(differences)} paired bootstrap samples")
        print(f"\nValidation LL improvement (k={k_high} - k={k_low}):")
        print(f"  Mean ΔLL:   {mean_diff:+.6f}")
        print(f"  Median ΔLL: {median_diff:+.6f}")
        print(f"  Std ΔLL:    {std_diff:.6f}")
        print(f"  95% CI:     [{ci_low:+.6f}, {ci_high:+.6f}]")
        print(f"\nWilcoxon signed-rank test:")
        print(f"  Test statistic: {statistic:.1f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significance level: {alpha}")
        print(f"\nDecision: ", end="")
        if p_value < alpha:
            print(f"k={k_high} is significantly better (p={p_value:.4f} < {alpha})")
        else:
            print(f"k={k_high} not significantly better (p={p_value:.4f} >= {alpha}), use k={k_low}")
    
    return {
        'selected_k': selected_k,
        'conservative_k': conservative_k,
        'p_value': float(p_value),
        'mean_diff': float(mean_diff),
        'median_diff': float(median_diff),
        'std_diff': float(std_diff),
        'ci_95': (float(ci_low), float(ci_high)),
        'n_samples': len(differences),
        'method': 'wilcoxon_paired',
    }


def information_criterion_model_selection(
    scoreset: Scoreset,
    k_range: List[int],
    initial_bootstrap_seed: int,
    criterion: str = "bic",
    **kwargs
) -> Dict:
    """
    Select optimal number of components using information criteria (BIC/AIC/ICL).
    
    This implements Occam's razor / minimum description length principle:
    trades off model fit against model complexity.
    
    Required Args:
    ------------------
    - scoreset : Scoreset : Scoreset to use in model selection
    - k_range : List[int] : list of component numbers to test (e.g., [2, 3, 4, 5])
    - initial_bootstrap_seed : int : bootstrap_seed for train/val split
    
    Optional kwargs:
    ------------------
    - criterion : str in {'bic', 'aic', 'icl'} (default 'bic') : which information criterion to use
        * BIC (Bayesian Information Criterion): stronger penalty for complexity, preferred for small samples
        * AIC (Akaike Information Criterion): weaker penalty, better for prediction
        * ICL (Integrated Classification Likelihood): BIC + entropy penalty for mixture models
    - N_restarts : int (default 100) : Number of restarts for each model
    - init_method : str in {'kmeans','method_of_moments'} (default 'kmeans')
    - constrained : bool (default True) : enforce monotonicity constraint
    - init_constraint_adjustment : str in {'skew','scale'}
    - save_filepath : Optional[str|Path] : where to save results
    - use_validation : bool (default True) : use validation set for likelihood calculation
    
    Returns:
    ---------------
    - selection_results : dict with keys:
        * 'best_k' : selected number of components
        * 'scores' : dict mapping k -> criterion score
        * 'models' : dict mapping k -> fitted model
        * 'delta_scores' : improvement of each k over the next simpler model
    """
    N_restarts = kwargs.get("N_restarts", 100)
    constrained = kwargs.get("constrained", True)
    init_method = kwargs.get("init_method", "kmeans")
    init_constraint_adjustment = kwargs.get("init_constraint_adjustment", "scale")
    use_validation = kwargs.get("use_validation", True)
    
    # Prepare data
    scores = scoreset.scores
    sample_assignments = scoreset.sample_assignments
    sample_assignments = makeOneHot(sample_assignments)
    mask = sample_assignments.any(1) & (~np.isnan(scores))
    scores = scores[mask]
    sample_assignments = sample_assignments[mask]
    
    n_observations = len(scores)
    n_samples = sample_assignments.shape[1]
    
    # Split into train/val if requested
    if use_validation:
        train_indices, val_indices = sample_specific_bootstrap(
            sample_assignments, bootstrap_seed=initial_bootstrap_seed
        )
        scores_train, sample_assignments_train = (
            scores[train_indices],
            sample_assignments[train_indices],
        )
        scores_val, sample_assignments_val = (
            scores[val_indices],
            sample_assignments[val_indices],
        )
        n_observations_eval = len(scores_val)
    else:
        scores_train = scores
        sample_assignments_train = sample_assignments
        scores_val = scores
        sample_assignments_val = sample_assignments
        n_observations_eval = n_observations
    
    # Fit all candidate models
    print(f"Fitting models for k = {k_range}")
    models = {}
    criterion_scores = {}
    likelihoods = {}
    n_params_dict = {}
    
    for k in trange(len(k_range), desc="Fitting models"):
        k_val = k_range[k]
        
        # Fit model
        model = fit_iteration(
            scores_train,
            sample_assignments_train,
            k_val,
            constrained,
            init_method,
            init_constraint_adjustment,
            N_restarts,
        )
        models[k_val] = model
        
        # Calculate likelihood on evaluation set
        log_likelihood = get_likelihood(
            scores_val,
            sample_assignments_val,
            model["component_params"],
            model["weights"],
        )
        likelihoods[k_val] = log_likelihood
        
        # Calculate number of parameters
        # k components × 3 params each (skew, loc, scale) + (k-1) × n_samples weight params
        n_params = k_val * 3 + (k_val - 1) * n_samples
        n_params_dict[k_val] = n_params
        
        # Calculate information criterion
        if criterion.lower() == "bic":
            score = -2 * log_likelihood + n_params * np.log(n_observations_eval)
        elif criterion.lower() == "aic":
            score = -2 * log_likelihood + 2 * n_params
        elif criterion.lower() == "icl":
            # ICL = BIC - 2 × entropy (requires soft assignments)
            # If soft assignments not available, fall back to BIC
            bic = -2 * log_likelihood + n_params * np.log(n_observations_eval)
            if "soft_assignments" in model:
                entropy = -np.sum(
                    model["soft_assignments"] * np.log(model["soft_assignments"] + 1e-10)
                )
                score = bic - 2 * entropy
            else:
                print(f"Warning: soft assignments not available for k={k_val}, using BIC")
                score = bic
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        criterion_scores[k_val] = score
    
    # Select best model (lowest criterion score)
    best_k = min(criterion_scores, key=criterion_scores.get)
    
    # Calculate delta scores (improvement over simpler model)
    delta_scores = {}
    sorted_k = sorted(k_range)
    for i, k in enumerate(sorted_k[1:], start=1):
        k_prev = sorted_k[i-1]
        # Negative delta = improvement (lower is better for IC)
        delta_scores[k] = criterion_scores[k] - criterion_scores[k_prev]
    
    # Prepare results
    selection_results = {
        "best_k": int(best_k),
        "criterion": criterion,
        "scores": {k: float(v) for k, v in criterion_scores.items()},
        "likelihoods": {k: float(v) for k, v in likelihoods.items()},
        "n_params": n_params_dict,
        "delta_scores": {k: float(v) for k, v in delta_scores.items()},
        "models": models,
        "k_range": k_range,
        "n_observations": int(n_observations),
        "n_observations_eval": int(n_observations_eval),
        "kwargs": kwargs,
    }
    
    # Print summary
    print(f"\n{criterion.upper()} Model Selection Results:")
    print(f"{'k':<5} {'Log-Lik':<12} {'n_params':<10} {criterion.upper():<12} {'Δ{criterion.upper()}':<12}")
    print("-" * 60)
    for k in sorted_k:
        delta_str = f"{delta_scores[k]:+.2f}" if k in delta_scores else "—"
        marker = " ← BEST" if k == best_k else ""
        print(f"{k:<5} {likelihoods[k]:<12.2f} {n_params_dict[k]:<10} "
              f"{criterion_scores[k]:<12.2f} {delta_str:<12}{marker}")
    
    print(f"\nSelected model: k = {best_k}")
    print(f"Interpretation: Model with {best_k} components provides the best "
          f"balance between fit and complexity")
    
    # Save results
    save_filepath = kwargs.get("save_filepath", None)
    if save_filepath is not None:
        save_filepath = Path(save_filepath)
        save_filepath.parent.mkdir(parents=True, exist_ok=True)
        # Don't save the full models (too large), just the metadata
        save_results = {k: v for k, v in selection_results.items() if k != "models"}
        save_results["best_model_params"] = {
            "component_params": models[best_k]["component_params"],
            "weights": models[best_k]["weights"],
        }
        with open(save_filepath, "w") as f:
            json.dump(serialize_dict(save_results), f, indent=2)
        print(f"Model selection results written to {save_filepath}")
    
    return selection_results


def stability_based_selection(
    bootstrap_fits_dict: Dict[int, List[Dict]],
    k_range: List[int],
    min_weight_threshold: float = 0.05,
    verbose: bool = True
) -> Dict:
    """
    Select optimal k based on component stability across bootstrap samples.
    
    Key principle: Real components are stable; spurious components are unstable.
    
    Args:
    -----
    - bootstrap_fits_dict : Dict[int, List[Dict]] : {k: [list of bootstrap fits]}
    - k_range : List[int] : candidate k values
    - min_weight_threshold : float (default 0.05) : minimum weight for "real" component
    
    Returns:
    --------
    - dict with stability scores, selected k, and detailed diagnostics
    """
    results = {}
    
    for k in k_range:
        fits = bootstrap_fits_dict[k]
        n_bootstraps = len(fits)
        
        # Extract parameters and weights from all bootstrap fits
        all_params = []
        all_weights = []
        
        for fit in fits:
            params = np.array(fit['component_params'])  # shape: (k, 3) - skew, loc, scale
            weights = np.array(fit['weights'])  # shape: (n_samples, k)
            
            # CRITICAL: Sort components by location parameter to avoid label switching
            sort_idx = np.argsort(params[:, 1])  # Sort by loc (index 1)
            params = params[sort_idx]
            weights = weights[:, sort_idx]
            
            all_params.append(params)
            all_weights.append(weights.mean(axis=0))  # Average across samples
        
        all_params = np.array(all_params)  # shape: (n_bootstraps, k, 3)
        all_weights = np.array(all_weights)  # shape: (n_bootstraps, k)
        
        # Metric 1: Component persistence (what % of bootstraps has weight > threshold)
        persistence = (all_weights > min_weight_threshold).mean(axis=0)  # shape: (k,)
        min_persistence = persistence.min()
        
        # Metric 2: Weight stability (coefficient of variation of weights)
        weight_means = all_weights.mean(axis=0)
        weight_stds = all_weights.std(axis=0)
        weight_cv = weight_stds / (weight_means + 1e-10)
        max_weight_cv = weight_cv.max()
        
        # Metric 3: Parameter stability (CV of location and scale params)
        # Even with sorting, param CV can be noisy, so use it lightly
        loc_cv = (all_params[:, :, 1].std(axis=0) / 
                  (np.abs(all_params[:, :, 1].mean(axis=0)) + 1e-10))
        scale_cv = (all_params[:, :, 2].std(axis=0) / 
                   (all_params[:, :, 2].mean(axis=0) + 1e-10))
        param_cv = (loc_cv + scale_cv) / 2
        max_param_cv = param_cv.max()
        
        # Combined stability score (lower is better)
        # Primary signal: persistence and weight CV
        # Secondary signal: param CV (downweighted due to label switching issues)
        stability_penalty = (
            (1 - min_persistence) * 3.0 +  # Persistence is critical
            max_weight_cv * 2.0 +           # Weight CV is important
            min(max_param_cv, 5.0) * 0.1    # Param CV is weak signal, cap at 5.0
        )
        
        # Minimal complexity penalty
        complexity_penalty = (k - min(k_range)) * 0.1
        
        total_score = stability_penalty + complexity_penalty
        
        results[k] = {
            'stability_score': float(total_score),
            'min_persistence': float(min_persistence),
            'max_weight_cv': float(max_weight_cv),
            'max_param_cv': float(max_param_cv),
            'persistence_per_component': persistence.tolist(),
            'weight_cv_per_component': weight_cv.tolist(),
            'param_cv_per_component': param_cv.tolist(),
            'mean_weights': weight_means.tolist(),
        }
    
    # Select k with lowest stability score
    best_k = min(results, key=lambda k: results[k]['stability_score'])
    
    # Print detailed results
    if verbose:
        print("\nStability-Based Model Selection Results:")
        print(f"{'k':<5} {'Score':<10} {'MinPersist':<12} {'MaxWtCV':<12} {'MaxParamCV':<12} {'Decision'}")
        print("-" * 70)
        for k in sorted(k_range):
            r = results[k]
            marker = " ← BEST" if k == best_k else ""
            print(f"{k:<5} {r['stability_score']:<10.3f} {r['min_persistence']:<12.3f} "
                  f"{r['max_weight_cv']:<12.3f} {r['max_param_cv']:<12.3f}{marker}")
        
        print(f"\nSelected model: k = {best_k}")
        print("\nInterpretation:")
        print(f"  - Min persistence: {results[best_k]['min_persistence']:.2%} of bootstraps")
        print(f"    (>95% = stable, <80% = unstable)")
        print(f"  - Max weight CV: {results[best_k]['max_weight_cv']:.3f}")
        print(f"    (<0.3 = stable, 0.3-0.5 = moderate, >0.5 = unstable)")
        
        # Add warning if unstable
        for k in k_range:
            r = results[k]
            if r['max_weight_cv'] > 0.5 and r['min_persistence'] < 0.95:
                print(f"\n  ⚠ Warning: k={k} has unstable components (MaxWtCV={r['max_weight_cv']:.3f}, "
                      f"MinPersist={r['min_persistence']:.2%})")
    
    return {
        'best_k': best_k,
        'results': results,
        'method': 'stability',
    }


def model_selection_from_bootstrap_fits(
    scoreset: Scoreset,
    bootstrap_fits_path: str,
    boot_results: Dict,
    scoreset_name: str,
    k_range: List[int] = [2, 3],
    use_stability: bool = True,
    use_ic: bool = True,
    ic_criterion: str = "all",
    verbose: bool =True
) -> Dict:
    """
    Perform model selection using pre-computed bootstrap fits.
    
    Args:
    -----
    - scoreset : Scoreset
    - bootstrap_fits_path : str : path to JSON with bootstrap fits
    - scoreset_name : str : name of scoreset in the JSON
    - k_range : List[int] : candidate k values to compare
    - use_stability : bool : use stability-based selection (SECONDARY)
    - use_ic : bool : compute information criteria from bootstrap fits (PRIMARY)
    - ic_criterion : str : 'icl' (default), 'bic', 'aic', or 'auto'
        * icl: ICL = BIC - 2×entropy, penalizes fuzzy assignments (recommended)
        * bic: Standard BIC (conservative)
        * aic: Standard AIC (liberal)
        * auto: AIC for n<500, BIC otherwise
    
    Returns:
    --------
    - dict with results from all methods and final recommendation
    """
    if boot_results is None:
        # Load bootstrap fits
        with gzip.open(bootstrap_fits_path, 'rt') as f:
            all_results = json.load(f)
            boot_results = all_results[scoreset_name]
    
    # Get actual data
    scores = scoreset.scores
    sample_assignments = scoreset.sample_assignments
    sample_assignments = makeOneHot(sample_assignments)
    mask = sample_assignments.any(1) & (~np.isnan(scores))
    scores = scores[mask]
    sample_assignments = sample_assignments[mask]
    n_actual_observations = mask.sum()
    n_samples = sample_assignments.shape[1]

    # Auto-select criterion if requested
    if ic_criterion == "auto":
        if n_actual_observations < 500:
            ic_criterion = "aic"
            if verbose:
                print(f"Auto-selected AIC (small sample: n={n_actual_observations})")
        else:
            ic_criterion = "bic"
            if verbose:
                print(f"Auto-selected BIC (large sample: n={n_actual_observations})")

    if verbose:
        print(f"Dataset: {n_actual_observations} observations, {n_samples} samples")
    
    # Organize fits by k
    bootstrap_fits_dict = {k: [] for k in k_range}
    
    for key, val in boot_results.items():
        for k in k_range:
            k_str = f"{k}c"
            if k_str in val:
                bootstrap_fits_dict[k].append(val[k_str]['fit'])
    
    results = {}

    # Load validation counts
    with open('/data/ross/assay_calibration/val_counts.pkl','rb') as f:
        val_counts = pickle.load(f)
    
    # Method 1: Information criterion (PRIMARY METHOD)
    if use_ic:
        if verbose:
            print("\n" + "="*70)
            if ic_criterion.lower() == 'icl':
                print("METHOD 1: ICL (Integrated Classification Likelihood)")
                print("(ICL = BIC - 2×entropy, penalizes fuzzy cluster assignments)")
            else:
                print(f"METHOD 1: {ic_criterion.upper()}")
            print("="*70)
        
        ic_results = {}
        for k in k_range:
            fits = bootstrap_fits_dict[k]
            
            # Get validation likelihoods from all bootstraps
            val_lls = [boot_results[key][f"{k}c"]['val_ll'] 
                      for key in boot_results.keys() 
                      if f"{k}c" in boot_results[key]]
            
            if not val_lls:
                continue
            
            n_val = np.median([val_counts[(scoreset_name, boot_idx)] 
                              for boot_idx in range(1000) 
                              if (scoreset_name, boot_idx) in val_counts])
            
            # Use median validation LL (already total LL)
            median_ll = np.median(val_lls) * n_val
            n_params = k * 3 + (k - 1) * n_samples
            
            # Calculate entropy for ICL
            entropy = 0.0
            if ic_criterion.lower() == 'icl' or ic_criterion.lower() == "all":
                # Get median model
                median_idx = np.argsort(val_lls)[len(val_lls) // 2]
                median_model = fits[median_idx]
                
                # Calculate entropy from posterior assignments
                for sample_num in range(n_samples):
                    sample_mask = sample_assignments[:, sample_num].astype(bool)
                    if sample_mask.sum() == 0:
                        continue
                    X_sample = scores[sample_mask]
                    
                    # Get posteriors
                    posteriors = component_posteriors(
                        X_sample,
                        median_model['component_params'],
                        median_model['weights'][sample_num]
                    )
                    
                    # Calculate entropy: -sum(p * log(p))
                    posteriors_safe = np.clip(posteriors, 1e-10, 1.0)
                    sample_entropy = -np.sum(posteriors * np.log(posteriors_safe))
                    entropy += sample_entropy
            
            # Calculate information criterion
            if ic_criterion.lower() == 'icl':
                bic = -2 * median_ll + n_params * np.log(n_val)
                ic_score = bic - 2 * entropy
            elif ic_criterion.lower() == 'bic':
                ic_score = -2 * median_ll + n_params * np.log(n_val)
            elif ic_criterion.lower() == 'aic':
                ic_score = -2 * median_ll + 2 * n_params
            elif ic_criterion.lower() == "all":
                bic = -2 * median_ll + n_params * np.log(n_val)
                icl = bic - 2 * entropy
                aic = -2 * median_ll + 2 * n_params
                ic_score = bic
            else:
                raise ValueError(f"Unknown criterion: {ic_criterion}")
            
            ic_results[k] = {
                'score': float(ic_score),
                'median_ll': float(median_ll),
                'n_params': int(n_params),
                'n_val': int(n_val),
                'entropy': float(entropy) if ic_criterion.lower() == 'icl' else None,
            }

            if ic_criterion.lower() == "all":
                ic_results[k]["bic"] = bic
                ic_results[k]["aic"] = aic
                ic_results[k]["icl"] = icl
                
        
        best_k_ic = min(ic_results, key=lambda k: ic_results[k]['score'])
        if ic_criterion.lower() == "all":
            best_k_bic = min(ic_results, key=lambda k: ic_results[k]['bic'])
            best_k_aic = min(ic_results, key=lambda k: ic_results[k]['aic'])
            best_k_icl = min(ic_results, key=lambda k: ic_results[k]['icl'])
        
        # Print results
        if verbose:
            print(f"\n{ic_criterion.upper()} Results:")
            header = f"{'k':<5} {'Median LL':<12} {'n_params':<10}"
            if ic_criterion.lower() == 'icl':
                header += f" {'Entropy':<12}"
            header += f" {ic_criterion.upper():<12} {'ΔLL':<12}"
            print(header)
            print("-" * (60 + (12 if ic_criterion.lower() == 'icl' else 0)))
        
        sorted_k = sorted(k_range)
        for i, k in enumerate(sorted_k):
            if k in ic_results:
                r = ic_results[k]
                marker = " ← BEST" if k == best_k_ic else ""
                
                # Show likelihood improvement
                if i > 0:
                    k_prev = sorted_k[i-1]
                    delta_ll = ic_results[k]['median_ll'] - ic_results[k_prev]['median_ll']
                    delta_str = f"+{delta_ll:.2f}" if delta_ll > 0 else f"{delta_ll:.2f}"
                else:
                    delta_str = "—"
                
                row = f"{k:<5} {r['median_ll']:<12.2f} {r['n_params']:<10}"
                if ic_criterion.lower() == 'icl':
                    row += f" {r['entropy']:<12.1f}"
                row += f" {r['score']:<12.2f} {delta_str:<12}{marker}"
                if verbose:
                    print(row)
        
        # Additional analysis for ICL
        if ic_criterion.lower() == 'icl' and len(sorted_k) >= 2:
            k_low = sorted_k[0]
            k_high = sorted_k[1]
            delta_entropy = ic_results[k_high]['entropy'] - ic_results[k_low]['entropy']
            if verbose:
                print(f"\nCluster assignment quality:")
                print(f"  Entropy Δ (k={k_high} vs k={k_low}): {delta_entropy:+.1f}")
                print(f"  (negative = more certain/separated assignments)")
        
        results['ic'] = {
            'best_k': best_k_ic,
            'criterion': ic_criterion,
            'results': ic_results,
        }
    
    # Method 2: Stability (SECONDARY)
    if use_stability:
        if verbose:
            print("\n" + "="*70)
            print("METHOD 2: Stability Check (for detecting spurious components)")
            print("="*70)
        stability_results = stability_based_selection(
            bootstrap_fits_dict, k_range, verbose=verbose
        )
        results['stability'] = stability_results
    
    # Make final recommendation
    if verbose:
        print("\n" + "="*70)
        print("FINAL RECOMMENDATION")
        print("="*70)
    
    if use_ic and use_stability:
        k_ic = results['ic']['best_k']
        k_stab = stability_results['best_k']

        if verbose:
            print(f"{ic_criterion.upper()} method selects: k = {k_ic} (PRIMARY)")
            print(f"Stability method selects: k = {k_stab} (SECONDARY)")
        
        stab_data = stability_results['results']
        
        # Check for severe instability
        ic_choice_severely_unstable = (
            stab_data[k_ic]['max_weight_cv'] > 1.0 or
            stab_data[k_ic]['min_persistence'] < 0.70
        )
        
        if ic_choice_severely_unstable:
            final_k = k_stab
            decision = (f"⚠ {ic_criterion.upper()} selected k={k_ic}, but shows SEVERE instability "
                       f"(MaxWtCV={stab_data[k_ic]['max_weight_cv']:.3f}, "
                       f"MinPersist={stab_data[k_ic]['min_persistence']:.2%}). "
                       f"Overriding to k={k_stab}.")
        else:
            final_k = k_ic
            if k_ic == k_stab:
                decision = f"Strong consensus: both methods agree on k={final_k}"
            else:
                decision = (f"Following {ic_criterion.upper()} (k={k_ic}). "
                           f"Stability prefers k={k_stab} but has conservative bias. "
                           f"IC choice shows acceptable stability "
                           f"(MaxWtCV={stab_data[k_ic]['max_weight_cv']:.3f}, "
                           f"MinPersist={stab_data[k_ic]['min_persistence']:.2%}).")
    
    elif use_ic:
        final_k = results['ic']['best_k']
        decision = f"Based on {ic_criterion.upper()}: k={final_k}"
    elif use_stability:
        final_k = stability_results['best_k']
        decision = f"Based on stability: k={final_k}"
    else:
        raise ValueError("Must use at least one selection method")

    if verbose:
        print(f"\n{'='*70}")
        print(f"FINAL SELECTION: k = {final_k}")
        print(f"Reasoning: {decision}")
        print(f"{'='*70}")
    
    final_results = {
        'final_k': final_k,
        'decision': decision,
        'results': results,
    }

    if ic_criterion.lower() == "all":
        final_results["best_k_bic"] = best_k_bic
        final_results["best_k_aic"] = best_k_aic
        final_results["best_k_icl"] = best_k_icl

    return final_results


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
        "test_statistic": -2 * (likelihood_two_comp - likelihood_three_comp) / len(scores_val),
        "bootstrapped_model_2_comp": bootstrap_model_k2,
        "bootstrapped_model_3_comp": bootstrap_model_k3,
        "likelihood_two_comp": likelihood_two_comp,
        "likelihood_three_comp": likelihood_three_comp,
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
    n_jobs=-1,
):
    fits: List[Dict] = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(single_fit)(
            scores,
            sample_assignments,
            n_components,
            constrained,
            init_method,
            init_constraint_adjustment,
        )
        for _ in range(N_restarts)
    )
    
    # Sort fits by increasing likelihood
    fits.sort(key=lambda d: d["likelihoods"][-1])
    # Find iteration with best likelihood
    best_fit = fits[-1]
    return best_fit


def generate_scoreset(params, weights, sample_sizes):
    samples = []
    sample_assignments = []
    weights = np.array(weights)
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
