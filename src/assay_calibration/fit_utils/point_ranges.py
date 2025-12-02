import sys
sys.path.append('..')
import os
from pathlib import Path
import json
import numpy as np
from typing import Dict, Tuple, List
from joblib import Parallel, delayed
import logging
from .fit import (calculate_score_ranges,thresholds_from_prior)  # noqa: E402
from .two_sample import density_utils  # noqa: E402
from ..data_utils.dataset import Scoreset  # noqa: E402
from .utils import serialize_dict  # noqa: E402
from collections import defaultdict

def enforce_monotonicity_point_ranges(point_ranges, point_values, score_range, scoreset_flipped=False, liberal=False, log_f=None):

    
    if liberal:
        print('enforcing monotonicity in points (liberal)...',file=log_f)
        for i in point_values:
            point = i # pathogenic
            if len(point_ranges[point]) != 0:
                point_ranges[point] = [point_ranges[point][-1]] if not scoreset_flipped else [point_ranges[point][0]]
                    
            point = -i # benign
            if len(point_ranges[point]) != 0:
                point_ranges[point] = [point_ranges[point][0]] if not scoreset_flipped else [point_ranges[point][-1]]

        return
    
    max_path_points = None
    max_ben_points = None
    
    print('enforcing monotonicity in points (std)...',file=log_f)
    for i in point_values:
        point = i # pathogenic

        if abs(point) == 1 and len(point_ranges[point]) != 0 and len(point_ranges[point+1]) == 0: # highest evidence at 1
            l,h = point_ranges[point][0][0], point_ranges[point][-1][-1] # could be more than one range

            if l != score_range[0] and h != score_range[-1]: # evidence not at min or max, evidence goes back to 0. remove
                print(f'supporting evidence ({point}) goes back to no evidence. removing...', file=log_f)
                max_path_points = point

        if max_path_points is not None:
            point_ranges[point] = []
        elif len(point_ranges[point]) > 1: # e.g. --_-

            point_h = point + 1
            if point_h in point_ranges and len(point_ranges[point_h]) != 0 and point_ranges[point_h][0][0] != point_ranges[point][0][-1] and point_ranges[point_h][-1][-1] != point_ranges[point][-1][0]:
                # if dips into no evidence/switches sides and not up into higher point ranges
                idx_to_keep = []
                for range_idx, range_ in enumerate(point_ranges[point]):
                    if range_[0] == point_ranges[point_h][-1][-1] or range_[-1] == point_ranges[point_h][0][0]:
                        # valid range. keep
                        idx_to_keep.append(range_idx)
                print(f'point ranges {point}: before removing dipping {point_ranges[point]}', file=log_f)
                if len(idx_to_keep) == 0:
                    point_ranges[point] = []
                    max_path_points = point
                else:
                    point_ranges[point] = list(np.array(point_ranges[point])[np.array(idx_to_keep)])
                print(f'point ranges {point}: after removing dipping {point_ranges[point]}', file=log_f)

            if len(point_ranges[point]) > 1: # if didn't dip or still needs flattening
                print(f'flattening ({point}): {point_ranges[point]}', file=log_f)
                
                # flatten
                point_ranges[point] = [[point_ranges[point][0][0], point_ranges[point][-1][-1]]]
                if max_path_points is None:
                    max_path_points = point
                
        point = -i # benign

        if abs(point) == 1 and len(point_ranges[point]) != 0 and len(point_ranges[point-1]) == 0: # highest evidence at -1
            l,h = point_ranges[point][0][0], point_ranges[point][-1][-1] # could be more than one range

            if l != score_range[0] and h != score_range[-1]: # evidence not at min or max, evidence goes back to 0. remove
                print(f'supporting evidence ({point}) goes back to no evidence. removing...', file=log_f)
                max_ben_points = point

        if max_ben_points is not None:
            point_ranges[point] = []
        elif len(point_ranges[point]) > 1: # e.g. --_-

            point_h = point - 1
            if point_h in point_ranges and len(point_ranges[point_h]) != 0 and point_ranges[point_h][0][0] != point_ranges[point][0][-1] and point_ranges[point_h][-1][-1] != point_ranges[point][-1][0]:
                # if dips into no evidence/switches sides and not up into higher point ranges
                idx_to_keep = []
                for range_idx, range_ in enumerate(point_ranges[point]):
                    if range_[0] == point_ranges[point_h][-1][-1] or range_[-1] == point_ranges[point_h][0][0]:
                        # valid range. keep
                        idx_to_keep.append(range_idx)
                print(f'point ranges {point}: before removing dipping {point_ranges[point]}', file=log_f)
                if len(idx_to_keep) == 0:
                    point_ranges[point] = []
                    max_ben_points = point
                else:
                    point_ranges[point] = list(np.array(point_ranges[point])[np.array(idx_to_keep)])
                print(f'point ranges {point}: after removing dipping {point_ranges[point]}', file=log_f)

            if len(point_ranges[point]) > 1: # if didn't dip or still needs flattening
                print(f'flattening ({point}): {point_ranges[point]}', file=log_f)
                
                # flatten
                point_ranges[point] = [[point_ranges[point][0][0], point_ranges[point][-1][-1]]]
                if max_ben_points is None:
                    max_ben_points = point



def extend_points_to_xlims(point_ranges, point_values, score_range, scoreset_flipped, log_f=None, inf=False):
    print('extending points to xlims...',file=log_f)
    left = -np.inf if inf else score_range[0]
    right = np.inf if inf else score_range[-1]
    for i in point_values:
        point = i # pathogenic
        if len(point_ranges[point]) != 0:
            j = 1
            all_no_evidence = True
            while point+j in point_ranges:
                if len(point_ranges[point+j]) != 0:
                    all_no_evidence = False
                j += 1
            
            if all_no_evidence:
                # extend to xlims
                point_ranges[point] = [[left, point_ranges[point][-1][-1]]] if not scoreset_flipped else [[point_ranges[point][0][0], right]]
                
        point = -i # benign
        if len(point_ranges[point]) != 0:
            j = 1
            all_no_evidence = True
            while point-j in point_ranges:
                if len(point_ranges[point-j]) != 0:
                    all_no_evidence = False
                j += 1
            
            if all_no_evidence:
                # extend to xlims
                point_ranges[point] = [[left, point_ranges[point][-1][-1]]] if scoreset_flipped else [[point_ranges[point][0][0], right]]

        
def prior_equation_2c(w_p, w_b, w_g):
    return (w_g[1] - w_b[1]) / (w_p[1] - w_b[1])

def prior_invalid(prior):
    return prior <= 0 or prior >= 1

def get_fit_prior(fit, scoreset, benign_method, **kwargs):
    pathogenic_idx = kwargs.get('pathogenic_idx',0)
    benign_idx = kwargs.get('benign_idx',1)
    if benign_method == 'synonymous':
        benign_idx = 3
    params = fit['fit']['component_params']
    weights = fit['fit']['weights']
    population = scoreset.scores[scoreset._sample_assignments[:,2]]
    pathogenic_density = density_utils.joint_densities(population,
                                                       params,
                                                       weights[pathogenic_idx]).sum(axis=0)
    if benign_method != 'avg':
        benign_density = density_utils.joint_densities(population,
                                                       params,
                                                       weights[benign_idx]).sum(axis=0)
    else:
        bs_weights = (np.array(weights[1])+np.array(weights[3]))/2
        benign_density = density_utils.joint_densities(population,
                                                       params,
                                                       bs_weights).sum(axis=0)
        
    assert len(pathogenic_density) == len(population)
    assert len(benign_density) == len(population)
    prior_estimate = 0.5
    converged = False
    em_steps = 0
    max_em_steps = kwargs.get("max_em_steps",10000)
    while not converged:
        em_steps += 1

        with np.errstate(divide='ignore', invalid='ignore', over='ignore',under='ignore'):
            posteriors = 1 / (
                1
                + (1 - prior_estimate)
                / prior_estimate
                * benign_density # type: ignore
                / pathogenic_density
            )
        new_prior = np.nanmean(posteriors)
        prior_estimate = new_prior
        if prior_estimate < 0 or prior_estimate > 1:
            raise ValueError(f"Invalid prior estimate obtained, {prior_estimate}")
        if em_steps >= max_em_steps:
            break
    if prior_estimate <= 0 or prior_estimate >= 1:
        return np.nan
    return prior_estimate

def get_bootstrap_score_ranges(fitIdx, fit, fp, fb, score_range, fit_priors, point_values):
    fit_xmin, fit_xmax = fit['fit']['xlims']
    mask = (score_range >= fit_xmin) & (score_range <= fit_xmax)# & ((fp > -7.0) | (fb > -7.0)) # add min density check

    # log_fp_local = np.zeros_like(fp)
    # log_fb_local = np.zeros_like(fb)

    # CRITICAL: IGNORE BOOTSTRAPS THAT DON'T SPAN DATA POINT. MARKING 0 WILL CAUSE STRANGE LR+ CURVES AT EXTREMES
    log_fp_local = np.full_like(fp, np.nan, dtype=float)
    log_fb_local = np.full_like(fb, np.nan, dtype=float)

    log_fp_local[mask] = fp[mask]
    log_fb_local[mask] = fb[mask]

    lrP = log_fp_local[mask] - log_fb_local[mask]
    s = score_range[mask]

    
    ranges_p, ranges_b, C = calculate_score_ranges(
        lrP, lrP, fit_priors[fitIdx], s, point_values
    )
    C = int(C)
    
    if prior_invalid(fit_priors[fitIdx]):
        log_fp_local = np.full_like(fp, np.nan, dtype=float)
        log_fb_local = np.full_like(fb, np.nan, dtype=float)
        for key in ranges_p:
            ranges_p[key] = []
        for key in ranges_b:
            ranges_b[key] = []
        C = np.nan

    return fitIdx, log_fp_local, log_fb_local, ranges_p, ranges_b, C

def remove_insufficient_bootstrap_converage_points(point_ranges, percent_no_evidence, point_values):

    # P/LP
    for point in point_values:
        if percent_no_evidence[point] > 0.05 and len(point_ranges[point]) > 0:
            if point > 1 : # extend range below
                i = 1
                while point-i != 0:
                    if len(point_ranges[point-i]) > 0:
                        new_range = np.vstack([point_ranges[point-i], point_ranges[point]])[
                                                np.vstack([point_ranges[point-i], point_ranges[point]])[:, 0].argsort()]
                        point_ranges[point-i] = new_range
                        break
                    i += 1
                
            point_ranges[point] = [] # remove strength

    # B/LB
    for point_p in point_values:
        point = -point_p 
        if percent_no_evidence[point] > 0.05 and len(point_ranges[point]) > 0:
            if point < -1 : # extend range below
                i = 1
                while point+i != 0:
                    if len(point_ranges[point+i]) > 0:
                        new_range = np.vstack([point_ranges[point+i], point_ranges[point]])[
                                                np.vstack([point_ranges[point+i], point_ranges[point]])[:, 0].argsort()]
                        point_ranges[point+i] = new_range
                        break
                    i += 1
                
            point_ranges[point] = [] # remove strength



def check_thresholds_reached(lrPlus, tau, point_values, pathogenicOrBenign):
    
    if pathogenicOrBenign == "benign":
        point_values = -1 * np.array(point_values)
    
    reached = {}
    
    for p in point_values:
        if pathogenicOrBenign == "pathogenic":
            # Check if LR+ ever exceeds threshold
            reached[p] = np.any(lrPlus >= tau[abs(p)-1]) # list idx not dict
        else:
            # Check if LR+ ever goes below threshold
            reached[p] = np.any(lrPlus <= tau[abs(p)-1]) # list idx not dict
    
    return reached




