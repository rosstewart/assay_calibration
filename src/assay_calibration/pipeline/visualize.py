"""
Visualization and calibration result generation
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from joblib import Parallel, delayed
import logging

from ..fit_utils.fit import calculate_score_ranges, thresholds_from_prior
from ..fit_utils.two_sample import density_utils
from ..fit_utils.point_ranges import (
    enforce_monotonicity_point_ranges,
    prior_equation_2c,
    prior_invalid,
    get_fit_prior,
    get_bootstrap_score_ranges,
    remove_insufficient_bootstrap_converage_points,
    check_thresholds_reached,
    extend_points_to_xlims
)
from ..data_utils.dataset import Scoreset, BasicScoreset
from ..fit_utils.utils import serialize_dict
from ..plot_utils.utils import plot_scoreset_example_publication

from .config import PipelineConfig
from .utils import load_dataset_from_df

def generate_visualizations(
    bootstrap_results: Dict,
    config: PipelineConfig,
    selected_components: Dict,
    logger: logging.Logger
) -> Dict:
    """
    Generate visualizations and calibration results
    
    Args:
        bootstrap_results: Dict of bootstrap fits
        config: Pipeline configuration
        selected_components: Dict mapping component keys to component counts
        logger: Logger instance
    
    Returns:
        Dict with calibration results
    """
    
    # Load dataset
    df = pd.read_csv(config.dataset_csv)
    scoreset = load_dataset_from_df(df, config)
    results = {}
    
    # Process each selected component count
    for component_key, n_c in selected_components.items():
        logger.info(f"\nProcessing {component_key}...")
        
        # Extract fits for this component count
        fits = []
        for bootstrap_idx in sorted(bootstrap_results.keys(), key=int):
            if component_key in bootstrap_results[bootstrap_idx]:
                fit_result = bootstrap_results[bootstrap_idx][component_key]
                if fit_result is not None:
                    fits.append(fit_result)
        
        logger.info(f"  Valid fits: {len(fits)}/{len(bootstrap_results)}")
        
        if len(fits) == 0:
            logger.warning(f"  No valid fits for {component_key}, skipping")
            continue
        
        # Process fits and generate calibration
        calibration = process_component_fits(
            fits=fits,
            scoreset=scoreset,
            config=config,
            n_c=n_c,
            logger=logger
        )
        
        results[component_key] = calibration
        
        # Generate visualization
        output_dir = Path(config.output_dir)
        figure_path = output_dir / f"{config.dataset_name}_{component_key}_visualization.png"
        
        try:
            fig = plot_scoreset_example_publication(
                dataset=config.dataset_name,
                scoreset=scoreset,
                indv_summary=calibration,
                fits=fits,
                score_range=calibration['score_range'],
                config=f"({config.benign_method})",
                n_c=component_key,
                n_samples=len([s for s in scoreset.samples]),
                relax=False,  # TODO: Add relax option to config
                flipped=calibration.get('scoreset_flipped', False)
            )
            
            fig.savefig(figure_path, bbox_inches='tight', dpi=300)
            logger.info(f"  Saved visualization: {figure_path}")
            
        except Exception as e:
            logger.error(f"  Failed to generate visualization: {e}")
    
    return results

def process_component_fits(
    fits: List[Dict],
    scoreset: Scoreset,
    config: PipelineConfig,
    n_c: int,
    logger: logging.Logger
) -> Dict:
    """
    Process bootstrap fits to generate calibration thresholds
    
    This is the core calibration logic adapted from the notebooks
    """
    
    # Identify sample indices
    pathogenic_idx, benign_idx, gnomad_idx, synonymous_idx = None, None, None, None
    
    for i, sample_name in enumerate(scoreset.sample_names):
        if scoreset.sample_counts[i] == 0:
            continue
        
        if sample_name == "Pathogenic/Likely Pathogenic":
            pathogenic_idx = i
        elif sample_name == "Benign/Likely Benign":
            benign_idx = i
        elif sample_name == "gnomAD" or sample_name == "population":
            gnomad_idx = i
        elif sample_name == "Synonymous":
            synonymous_idx = i
    
    if pathogenic_idx is None or gnomad_idx is None:
        raise ValueError("Missing required samples (Pathogenic or gnomAD)")
    
    if benign_idx is None and synonymous_idx is None:
        raise ValueError("Missing benign reference (Benign or Synonymous)")

    if synonymous_idx is None and (config.benign_method == 'avg' or config.benign_method == 'synonymous'):
        logger.warning(f"  No synonymous sample, setting benign_method from {config.benign_method} to benign")
        config.benign_method = 'benign'
    elif benign_idx is None and (config.benign_method == 'avg' or config.benign_method == 'benign'):
        logger.warning(f"  No benign sample, setting benign_method from {config.benign_method} to synonymous")
        config.benign_method = 'synonymous'
    
    # Adjust indices if benign is missing
    if benign_idx is None:
        gnomad_idx -= 1
        synonymous_idx -= 1
    
    # Compute priors
    if not config.use_2c_equation or n_c != 2:
        # Use EM estimation
        n_cores = os.cpu_count() or 1
        fit_priors = np.array(Parallel(n_jobs=min(len(fits), n_cores), verbose=0)(
            delayed(get_fit_prior)(fit, scoreset, config.benign_method)
            for fit in fits
        ))
    else:
        # Use 2c equation
        fit_priors = []
        for fit in fits:
            weights = fit['fit']['weights']
            
            if len(weights) == 3:
                w_p = weights[pathogenic_idx]
                w_g = weights[gnomad_idx]
                w_b = weights[benign_idx if benign_idx is not None else synonymous_idx]
                w_s = w_b
            elif len(weights) == 4:
                w_p, w_b, w_g, w_s = weights
            else:
                raise ValueError(f"Unexpected number of samples: {len(weights)}")
            
            if config.benign_method == 'synonymous':
                fit_priors.append(prior_equation_2c(w_p, w_s, w_g))
            elif config.benign_method == 'avg':
                w_bs = (np.array(w_b) + np.array(w_s)) / 2
                fit_priors.append(prior_equation_2c(w_p, w_bs, w_g))
            else:
                fit_priors.append(prior_equation_2c(w_p, w_b, w_g))
        
        fit_priors = np.array(fit_priors)
    
    # Filter invalid priors
    valid_mask = ~(np.isnan(fit_priors) | (fit_priors <= 0) | (fit_priors >= 1))
    fit_priors = fit_priors[valid_mask]
    fits = np.array(fits)[valid_mask].tolist()
    
    logger.info(f"  Valid priors: {len(fit_priors)}")
    
    # Compute prior
    prior = np.nanmedian(fit_priors)
    
    # Setup score range
    observed_scores = scoreset.scores[scoreset._sample_assignments.any(1)]
    score_range = np.linspace(*np.percentile(observed_scores, [0, 100]), 10000)
    
    # Compute log likelihood ratios
    _log_fp = np.stack([
        density_utils.mixture_pdf(
            score_range,
            fit['fit']['component_params'],
            fit['fit']['weights'][pathogenic_idx]
        )
        for fit in fits
    ])
    
    # Determine benign index
    if config.benign_method == "synonymous" and synonymous_idx is not None:
        benign_idx = synonymous_idx
    
    if config.benign_method != 'avg' or benign_idx is None:
        if benign_idx is None:
            benign_idx = synonymous_idx
        
        _log_fb = np.stack([
            density_utils.mixture_pdf(
                score_range,
                fit['fit']['component_params'],
                fit['fit']['weights'][benign_idx]
            )
            for fit in fits
        ])
    else:
        # Average benign and synonymous
        _log_fb = np.stack([
            density_utils.mixture_pdf(
                score_range,
                fit['fit']['component_params'],
                (np.array(fit['fit']['weights'][benign_idx]) + 
                 np.array(fit['fit']['weights'][synonymous_idx])) / 2
            )
            for fit in fits
        ])
    
    # Get bootstrap score ranges
    n_cores = os.cpu_count() or 1
    results = Parallel(n_jobs=min(len(fits), n_cores), verbose=0)(
        delayed(get_bootstrap_score_ranges)(
            fitIdx, fit, fp, fb, score_range, fit_priors, config.point_values
        )
        for fitIdx, (fit, fp, fb) in enumerate(zip(fits, _log_fp, _log_fb))
    )
    
    # Aggregate results
    log_fp = np.full((len(fits), len(score_range)), np.nan)
    log_fb = np.full((len(fits), len(score_range)), np.nan)
    ranges_pathogenic, ranges_benign = [], []
    Cs = []
    
    for fitIdx, log_fp_local, log_fb_local, ranges_p, ranges_b, C in results:
        log_fp[fitIdx] = log_fp_local
        log_fb[fitIdx] = log_fb_local
        ranges_pathogenic.append({key: np.array(value).reshape(-1) for key, value in ranges_p.items()})
        ranges_benign.append({key: np.array(value).reshape(-1) for key, value in ranges_b.items()})
        Cs.append(C)
    
    log_lr_plus = log_fp - log_fb
    
    # Compute C range
    C = np.array([np.nanpercentile(Cs, 5), np.nanpercentile(Cs, 95)])
    
    # Detect if scoreset is flipped
    # if config.benign_method != 'avg':
    #     _isflipped = [
    #         fit["fit"]["weights"][0][0] < fit["fit"]["weights"][1][0] or
    #         fit["fit"]["weights"][0][-1] > fit["fit"]["weights"][1][-1]
    #         for fit in fits
    #     ]
    # else:
    #     _isflipped = [
    #         fit["fit"]["weights"][0][0] < (np.array(fit['fit']['weights'][1][0]) + 
    #                                       np.array(fit['fit']['weights'][3][0])) / 2 or
    #         fit["fit"]["weights"][0][-1] > (np.array(fit['fit']['weights'][1][-1]) +
    #                                        np.array(fit['fit']['weights'][3][-1])) / 2
    #         for fit in fits
    #     ]

    benign_method = config.benign_method
    scoreset_flipped = False
    path_mean_score = np.mean(scoreset.scores[scoreset._sample_assignments[:,pathogenic_idx]])
    if benign_method == 'benign':
        ben_mean_score = np.mean(scoreset.scores[scoreset._sample_assignments[:,benign_idx]])
    elif benign_method == 'synonymous':
        ben_mean_score = np.mean(scoreset.scores[scoreset._sample_assignments[:,synonymous_idx]])
    elif benign_method == 'avg':
        ben_mean_score = (np.mean(scoreset.scores[scoreset._sample_assignments[:,benign_idx]])+np.mean(scoreset.scores[scoreset._sample_assignments[:,synonymous_idx]]))/2
        # _isflipped = [_fit["fit"]["weights"][0][0] < _fit["fit"]["weights"][1][0] or _fit["fit"]["weights"][0][-1] > _fit["fit"]["weights"][1][-1] for _fit in fits] # P 1st weight < B 1st weight or P last weight > B last weight. `or` is for 3c weird cases
    else:
        raise ValueError("invalid benign_method")

    if path_mean_score > ben_mean_score:
        scoreset_flipped = True
        # _isflipped = [_fit["fit"]["weights"][0][0] < (np.array(_fit['fit']['weights'][1][0])+np.array(_fit['fit']['weights'][3][0]))/2 or _fit["fit"]["weights"][0][-1] > (np.array(_fit['fit']['weights'][1][-1])+np.array(_fit['fit']['weights'][3][-1]))/2 for _fit in fits]
    # if np.sum(_isflipped) > len(fits) / 2:
    #     scoreset_flipped = True
    
    # Compute point ranges
    nan_counts = np.isnan(log_lr_plus).sum(0)
    range_subset = nan_counts < log_lr_plus.shape[1]
    
    if config.use_median_prior:
        # Use median prior for unified thresholds
        point_ranges_pathogenic, point_ranges_benign, C = calculate_score_ranges(
            np.nanpercentile(log_lr_plus[:, range_subset], 5, axis=0),
            np.nanpercentile(log_lr_plus[:, range_subset], 95, axis=0),
            prior,
            score_range[range_subset],
            config.point_values,
        )
        point_ranges = {**point_ranges_pathogenic, **point_ranges_benign}
    else:
        # Use 5th percentile conservative thresholds
        p_xaxis_5percentile_conservative = 5 if not scoreset_flipped else 95
        b_xaxis_5percentile_conservative = 95 if not scoreset_flipped else 5
        
        p_max = max if not scoreset_flipped else min
        b_min = min if not scoreset_flipped else max
        p_inf = -np.inf if not scoreset_flipped else np.inf
        b_inf = np.inf if not scoreset_flipped else -np.inf
        
        conservative_thresholds = {}
        
        for point_value in config.point_values:
            conservative_thresholds[point_value] = np.nanpercentile(
                [p_max(ranges_p[point_value]) if len(ranges_p[point_value]) > 0 else p_inf
                 for ranges_p in ranges_pathogenic],
                p_xaxis_5percentile_conservative
            )
            
            conservative_thresholds[-1 * point_value] = np.nanpercentile(
                [b_min(ranges_b[-1 * point_value]) if len(ranges_b[-1 * point_value]) > 0 else b_inf
                 for ranges_b in ranges_benign],
                b_xaxis_5percentile_conservative
            )
        
        # Convert thresholds to ranges
        point_ranges = {}
        valid_scores = score_range[range_subset]
        
        for point_value, threshold in conservative_thresholds.items():
            if np.isnan(threshold) or np.isinf(threshold):
                point_ranges[point_value] = []
                continue
            
            if (point_value > 0 and not scoreset_flipped) or (point_value < 0 and scoreset_flipped):
                # Pathogenic or flipped benign
                if abs(point_value) == max(config.point_values):
                    point_ranges[point_value] = [[valid_scores[0], threshold]]
                else:
                    lower_lim = conservative_thresholds[point_value + 1 if point_value > 0 else point_value - 1]
                    if np.isnan(lower_lim):
                        point_ranges[point_value] = [[valid_scores[0], threshold]]
                    else:
                        point_ranges[point_value] = [[lower_lim, threshold]]
            else:
                # Benign
                if abs(point_value) == max(config.point_values):
                    point_ranges[point_value] = [[threshold, valid_scores[-1]]]
                else:
                    upper_lim = conservative_thresholds[point_value - 1 if point_value < 0 else point_value + 1]
                    if np.isnan(upper_lim):
                        point_ranges[point_value] = [[threshold, valid_scores[-1]]]
                    else:
                        point_ranges[point_value] = [[threshold, upper_lim]]
    
    # Check for insufficient bootstrap coverage
    percent_no_evidence = {point: 0.0 for point in config.point_values + list(-1 * np.array(config.point_values))}
    
    # Enforce monotonicity
    enforce_monotonicity_point_ranges(
        point_ranges,
        config.point_values,
        score_range[range_subset],
        scoreset_flipped=scoreset_flipped,
        liberal=config.liberal_monotonicity
    )
    
    # Extend to limits
    extend_points_to_xlims(
        point_ranges,
        config.point_values,
        score_range[range_subset],
        scoreset_flipped
    )
    
    # Serialize and return
    return serialize_dict({
        'prior': prior,
        'priors': fit_priors,
        'point_ranges': point_ranges,
        'score_range': score_range[range_subset],
        'log_lr_plus': log_lr_plus[:, range_subset],
        'log_fp': log_fp[:, range_subset],
        'log_fb': log_fb[:, range_subset],
        'C': C,
        'scoreset_flipped': scoreset_flipped,
        'n_valid_fits': len(fits),
    })
