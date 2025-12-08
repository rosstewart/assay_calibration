import sys
import os
from pathlib import Path
import json
import numpy as np
from typing import Dict, Tuple, List
from joblib import Parallel, delayed
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from ..fit_utils.fit import (calculate_score_ranges,thresholds_from_prior)  # noqa: E402
from ..fit_utils.two_sample import density_utils  # noqa: E402
from ..data_utils.dataset import Scoreset  # noqa: E402
from ..fit_utils.utils import serialize_dict  # noqa: E402
import matplotlib.gridspec as gridspec

def plot_scoreset(scoreset:Scoreset, summary: Dict, scoreset_fits: List[Dict], score_range, use_median_prior,use_2c_equation, n_c, benign_method, C):
    fig, ax = plt.subplots(2,scoreset.n_samples, figsize=(5*scoreset.n_samples,10),sharex=True,sharey=False)
    for sample_num in range(scoreset.n_samples):
        sns.histplot(scoreset.scores[scoreset.sample_assignments[:,sample_num]],stat='density',ax=ax[1,sample_num],alpha=.5,color='pink',)
        density = sample_density(score_range, scoreset_fits, sample_num)
        for compNum in range(density.shape[1]):

            compDensity = density[:,compNum,:]
            d = np.nanpercentile(compDensity,[5,50,95],axis=0)
            ax[1,sample_num].plot(score_range,d[1],color=f"C{compNum}",linestyle='--',label=f"Comp {compNum+1}")
            ax[1,sample_num].legend()
        d = np.nansum(density,axis=1)
        d_perc = np.percentile(d,[5,50,95],axis=0)
        ax[1,sample_num].plot(score_range,d_perc[1],color='black',alpha=.5)
        ax[1,sample_num].fill_between(score_range,d_perc[0],d_perc[2],color='gray',alpha=0.3)
        ax[1,sample_num].set_xlabel("Score")
        ax[0,sample_num].set_title(f"{scoreset.sample_names[sample_num]} (n={scoreset.sample_assignments[:,sample_num].sum():,d})")
    point_ranges = sorted([(int(k), v) for k,v in summary['point_ranges'].items()])
    point_values = [pr[0] for pr in point_ranges]
    print(point_ranges)
    for axi in ax[0]:
        for pointIdx,(pointVal, scoreRanges) in enumerate(point_ranges):
            for sr in scoreRanges:
                axi.plot([sr[0], sr[1]], [pointIdx,pointIdx], color='red' if pointVal > 0 else 'blue', linestyle='-', alpha=0.7)
        axi.set_ylim(-1,len(point_values))
        axi.set_ylabel("Points")

        axi.set_yticks(range(len(point_values)),labels=list(map(lambda i: f"{i:+d}" if i!=0 else "0",point_values)))
    ax[0,2].set_title(f"{scoreset.scoreset_name} ({n_c}, median:{use_median_prior},em:{not use_2c_equation}): (gnomAD pop, n={scoreset.sample_assignments[:,2].sum():,d})\nprior {summary['prior']:.3f}, C: {summary['C']}")
    return fig

def plot_scoreset_compare_point_assignments(dataset, scoresets, summary, scoreset_fits, score_ranges, n_samples):
    
    # Determine which scoreset types exist
    scoreset_keys = list(scoresets.keys())
    has_2c = any('2c' in k for k in scoreset_keys)
    has_3c = any('3c' in k for k in scoreset_keys)
    has_4c = any('4c' in k for k in scoreset_keys)
    
    # Build list of scoreset types in order
    scoreset_types = []
    if has_4c:
        scoreset_types.append('4c')
    if has_3c:
        scoreset_types.append('3c')
    if has_2c:
        scoreset_types.append('2c')
    
    n_rows = len(scoreset_types) * 2  # 2 rows per scoreset type (LR+ and fits/points)
    
    # Get scoresets and configs for each type
    scoreset_data = {}
    for st in scoreset_types:
        # Find the scoreset with this type
        st_key = [k for k in scoreset_keys if st in k][0]
        scoreset_data[st] = {
            'scoreset': scoresets[st_key],
            'score_range': score_ranges[st_key],
            'configs': sorted([k for k in summary.keys() if k[1] == st and 'avg' not in k]) + \
                      sorted([k for k in summary.keys() if k[1] == st and 'avg' in k]),
            'fits_key': st_key
        }
    
    # Determine layout dimensions
    max_samples = max(sd['scoreset'].n_samples for sd in scoreset_data.values())
    max_configs = max(len(sd['configs']) for sd in scoreset_data.values())
    n_cols_total = max_samples + max_configs
    
    fig, ax = plt.subplots(n_rows, n_cols_total, figsize=(10*n_cols_total, 10*n_rows), 
                           squeeze=False, gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

    # Process each scoreset type
    for type_idx, st in enumerate(scoreset_types):
        row_lr = type_idx * 2      # LR+ row
        row_fits = type_idx * 2 + 1  # Fits/points row
        
        sd = scoreset_data[st]
        scoreset = sd['scoreset']
        score_range = sd['score_range']
        configs = sd['configs']
        n_samples_st = scoreset.n_samples
        
        # ===== LR+ row: Hide sample columns =====
        for col_idx in range(max_samples):
            ax[row_lr, col_idx].axis('off')
        
        # ===== Fits/points row: Plot fits =====
        for sample_num in range(n_samples_st):
            ax_fit = ax[row_fits, sample_num]
            
            # Plot histogram
            hist_data = sns.histplot(scoreset.scores[scoreset.sample_assignments[:,sample_num]], 
                                     stat='density', ax=ax_fit, alpha=.5, color='pink')
            
            # Get maximum density from histogram patches
            max_hist_density = max([patch.get_height() for patch in ax_fit.patches])
            
            # Plot fitted densities
            density = sample_density(score_range, scoreset_fits[sd['fits_key']], sample_num)
            for compNum in range(density.shape[1]):
                compDensity = density[:,compNum,:]
                d = np.nanpercentile(compDensity,[5,50,95],axis=0)
                ax_fit.plot(score_range, d[1], color=f"C{compNum}", linestyle='--', label=f"Comp {compNum+1}")
            ax_fit.legend(fontsize=8)
            
            d = np.nansum(density, axis=1)
            d_perc = np.percentile(d, [5,50,95], axis=0)
            ax_fit.plot(score_range, d_perc[1], color='black', alpha=.5)
            ax_fit.fill_between(score_range, d_perc[0], d_perc[2], color='gray', alpha=0.3)
            ax_fit.set_title(f"{st}: {scoreset.sample_names[sample_num]}\n(n={scoreset.sample_assignments[:,sample_num].sum():,d})")
            ax_fit.set_xlabel("Score")
            ax_fit.set_ylabel("Density")
            ax_fit.set_ylim([0, max_hist_density * 1.1])
            ax_fit.grid(linewidth=0.5, alpha=0.3)
        
        # Hide unused sample columns
        for col_idx in range(n_samples_st, max_samples):
            ax[row_fits, col_idx].axis('off')
        
        # Get x-limits from fits
        xlim = ax[row_fits, 0].get_xlim()
        
        # ===== LR+ row: Plot LR+ summaries =====
        for config_idx, (config, n_c) in enumerate(configs):
            col_idx = max_samples + config_idx
            ax_lr = ax[row_lr, col_idx]
            
            log_lr_plus = summary[(config, n_c)]['log_lr_plus']
            llr_curves = np.nanpercentile(np.array(log_lr_plus),[5,50,95],axis=0)
            labels = ['5th percentile','Median','95th percentile']
            
            for i, c in enumerate(['red','black','blue']):
                ax_lr.plot(score_range, llr_curves[i], color=c, label=labels[i])
            
            point_values = sorted(list(set([abs(int(k)) for k in summary[(config, n_c)]['point_ranges'].keys()])))
            tauP, tauB, _ = list(map(np.log, thresholds_from_prior(summary[(config, n_c)]['prior'], point_values + [10])))
            priors = np.percentile(np.array(summary[(config, n_c)]['priors']),[5,50,95])
            
            ax_lr.set_title(f"{st} LR+ {config}\nprior: {priors[1]:.3f} ({priors[0]:.3f}-{priors[2]:.3f}), C: {summary[(config, n_c)]['C']}", fontsize=10)
            add_thresholds(tauP[:-1], tauB[:-1], ax_lr)
            ax_lr.set_xlabel("Score")
            ax_lr.set_ylabel("Log LR+")
            ax_lr.legend(fontsize=6, loc='best')
            ax_lr.set_xlim(xlim)
            ax_lr.set_ylim([tauB[-1], tauP[-1]])  # Set y-limits based on ±10 thresholds
            ax_lr.grid(linewidth=0.5, alpha=0.3)
        
        # Hide unused config columns in LR+ row
        for col_idx in range(max_samples + len(configs), n_cols_total):
            ax[row_lr, col_idx].axis('off')
        
        # ===== Fits/points row: Plot point assignments =====
        for config_idx, (config, n_c) in enumerate(configs):
            col_idx = max_samples + config_idx
            ax_points = ax[row_fits, col_idx]
            
            point_ranges = sorted([(int(k), v) for k,v in summary[(config, n_c)]['point_ranges'].items()])
            point_values = [pr[0] for pr in point_ranges]
            
            # Plot all samples on same axis
            for sample_num in range(n_samples_st):
                for pointIdx, (pointVal, scoreRanges) in enumerate(point_ranges):
                    for sr in scoreRanges:
                        ax_points.plot([sr[0], sr[1]], [pointIdx, pointIdx], 
                                     color='red' if pointVal > 0 else 'blue', 
                                     linestyle='-', alpha=0.7, linewidth=2)
            
            ax_points.set_ylim(-1, len(point_values))
            ax_points.set_yticks(range(len(point_values)), 
                               labels=list(map(lambda i: f"{i:+d}" if i!=0 else "0", point_values)))
            ax_points.set_xlabel("Score")
            ax_points.set_ylabel("Points")
            ax_points.set_title(f"{st} Points {config}", fontsize=10)
            ax_points.set_xlim(xlim)
            ax_points.grid(linewidth=0.5, alpha=0.3)
        
        # Hide unused config columns in fits/points row
        for col_idx in range(max_samples + len(configs), n_cols_total):
            ax[row_fits, col_idx].axis('off')
    
    fig.suptitle(f"{scoreset_data[scoreset_types[-1]]['scoreset'].scoreset_name}", fontsize=16, y=0.995)
    
    return fig


def sample_density(x, fits, sampleNum):
    _density = np.stack([density_utils.joint_densities(x, _fit['fit']['component_params'],_fit['fit']['weights'][sampleNum])
                        for _fit in fits])
    density = np.full(_density.shape,np.nan)
    for fitIdx,fit in enumerate(fits):
        fit_xmin,fit_xmax = fit['fit']['xlims']
        mask = (x >= fit_xmin) & (x <= fit_xmax)
        density[fitIdx,:,mask] = _density[fitIdx,:,mask]
    return density


def add_thresholds(tauP, tauB, ax):
    for tp,tb in zip(tauP,tauB):
        ax.axhline(tp,color='red',linestyle='--',alpha=0.5)
        ax.axhline(tb,color='blue',linestyle='--',alpha=0.5)


def plot_summary(scoreset: Scoreset, fits: List[Dict], summary:Dict, score_range, log_fp, log_fb, use_median_prior,use_2c_equation, n_c, benign_method, C, dataset):
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    log_lr_plus = log_fp - log_fb
    llr_curves = np.nanpercentile(np.array(log_lr_plus),[5,50,95],axis=0)
    labels = ['5th percentile','Median','95th percentile']
    for i,c in enumerate(['red','black','blue']):
        ax.plot(score_range,llr_curves[i],color=c,label=labels[i])
    point_values = sorted(list(set([abs(int(k)) for k in summary['point_ranges'].keys()])))
    tauP,tauB,_ = list(map(np.log, thresholds_from_prior(summary['prior'],point_values)) )
    priors = np.percentile(np.array(summary['priors']),[5,50,95])
    ax.set_title(f"{dataset} ({n_c}, median:{use_median_prior},em:{not use_2c_equation}): prior: {priors[1]:.3f} ({priors[0]:.3f}-{priors[2]:.3f}), C: {C}")
    add_thresholds(tauP, tauB, ax)
    ax.set_xlabel("Score")
    ax.set_ylabel("Log LR+")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig


# claude generated. do not want to deal with the matplotlib headache :)
def plot_scoreset_best_config(dataset, scoreset, indv_summary, fits, score_range, config, n_c, n_samples, relax=False, flipped=False, debug=False):
    """
    Plot a single configuration with samples in one row, point assignments below, and LR+ below that.
    
    Parameters:
    -----------
    dataset : str
        Dataset name
    scoreset : Scoreset
        The scoreset object
    indv_summary : dict
        Summary dict for this specific (config, n_c) configuration
    fits : list
        Fitted models
    score_range : np.ndarray
        Score range for plotting
    config : tuple
        Configuration tuple (e.g., ('avg',) or ('benign',))
    n_c : str
        '2c' or '3c'
    n_samples : int
        Number of samples in the scoreset
    relax : bool
        Whether this is a relaxed fit
    flipped : bool
        Whether the scoreset is flipped (higher scores = more pathogenic)
        Pass True if P/LP is on the right side
    """
    
    # Create figure: 3 rows, n_samples columns (all square)
    fig, ax = plt.subplots(3, n_samples, figsize=(7*n_samples, 18), 
                           squeeze=False, gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

    relax_code = "R" if relax else ""
    
    # ===== Row 0: Sample fits =====
    for sample_num in range(n_samples):
        ax_fit = ax[0, sample_num]
        
        # Get sample mask
        sample_mask = scoreset.sample_assignments[:,sample_num]
        
        # Plot based on sample number (which category)
        if sample_num == 0:  # P/LP
            sns.histplot(scoreset.scores[sample_mask], 
                         stat='density', ax=ax_fit, alpha=0.6, color='#CA7682')
        elif sample_num == 1:  # B/LB
            sns.histplot(scoreset.scores[sample_mask], 
                         stat='density', ax=ax_fit, alpha=0.6, color='#1D7AAB')
        elif sample_num == 2:  # gnomAD
            sns.histplot(scoreset.scores[sample_mask], 
                         stat='density', ax=ax_fit, alpha=0.3, color='#A0A0A0')
        elif sample_num == 3:  # Synonymous
            sns.histplot(scoreset.scores[sample_mask], 
                         stat='density', ax=ax_fit, alpha=0.5, color='#6BAA75')
    
        max_hist_density = max([patch.get_height() for patch in ax_fit.patches]) if ax_fit.patches else 1.0
        
        density = sample_density(score_range, fits, sample_num)
        for compNum in range(density.shape[1]):
            compDensity = density[:,compNum,:]
            d = np.nanpercentile(compDensity,[5,50,95],axis=0)
            ax_fit.plot(score_range, d[1], color=f"C{compNum}", linestyle='--', label=f"Comp {compNum+1}")
        ax_fit.legend(fontsize=8)
        
        d = np.nansum(density, axis=1)
        d_perc = np.percentile(d, [5,50,95], axis=0)
        ax_fit.plot(score_range, d_perc[1], color='black', alpha=.5)
        ax_fit.fill_between(score_range, d_perc[0], d_perc[2], color='gray', alpha=0.3)
        ax_fit.set_title(f"{n_c}{relax_code}: {scoreset.sample_names[sample_num].replace('population','gnomAD')}\n(n={scoreset.sample_assignments[:,sample_num].sum():,d})")
        ax_fit.set_xlabel("Score")
        ax_fit.set_ylabel("Density")
        ax_fit.set_ylim([0, max_hist_density * 1.1])
        ax_fit.grid(linewidth=0.5, alpha=0.3)
    
    # Get x-limits from first fit
    xlim = ax[0, 0].get_xlim()
    
    # ===== Row 1: Point assignments (one per sample) =====
    point_ranges = sorted([(int(k), v) for k,v in indv_summary['point_ranges'].items()])
    point_values = [pr[0] for pr in point_ranges]
    
    for sample_num in range(n_samples):
        ax_points = ax[1, sample_num]
        
        # Plot only this sample's point assignments
        for pointIdx, (pointVal, scoreRanges) in enumerate(point_ranges):
            for sr in scoreRanges:
                ax_points.plot([sr[0], sr[1]], [pointIdx, pointIdx], 
                             color='red' if pointVal > 0 else 'blue', 
                             linestyle='-', alpha=0.7, linewidth=2)
        
        ax_points.set_ylim(-1, len(point_values))
        ax_points.set_yticks(range(len(point_values)), 
                           labels=list(map(lambda i: f"{i:+d}" if i!=0 else "0", point_values)))
        ax_points.set_xlabel("Score")
        ax_points.set_ylabel("Points")
        ax_points.set_title(f"Point Assignments", fontsize=11)
        ax_points.set_xlim(xlim)
        ax_points.grid(linewidth=0.5, alpha=0.3)
    
    # ===== Row 2: LR+ summaries (one per sample) =====
    point_values_all = sorted(list(set([abs(int(k)) for k in indv_summary['point_ranges'].keys()])))
    tauP, tauB, _ = list(map(np.log, thresholds_from_prior(indv_summary['prior'], point_values_all + [10])))
    priors = np.percentile(np.array(indv_summary['priors']),[5,50,95])
    
    # Identify which point values have insufficient evidence (empty ranges)
    point_ranges_dict = {int(k): v for k, v in indv_summary['point_ranges'].items()}
    pathogenic_points = [p for p in point_values_all if p > 0]
    benign_points = [-p for p in point_values_all if p > 0]
    
    # Find highest pathogenic point with evidence
    highest_pathogenic_with_evidence = None
    for p in sorted(pathogenic_points, reverse=True):
        if p in point_ranges_dict and len(point_ranges_dict[p]) > 0:
            highest_pathogenic_with_evidence = p
            break
    
    # Find lowest benign point with evidence (most negative)
    lowest_benign_with_evidence = None
    for p in sorted(benign_points):
        if p in point_ranges_dict and len(point_ranges_dict[p]) > 0:
            lowest_benign_with_evidence = p
            break
    
    for sample_num in range(n_samples):
        ax_lr = ax[2, sample_num]
        
        log_lr_plus = indv_summary['log_lr_plus']
        llr_curves = np.nanpercentile(np.array(log_lr_plus),[5,50,95],axis=0)
        labels = ['5th percentile','Median','95th percentile']
        colors = ['red','black','blue']
        
        # Check 5th percentile: find max and its position
        lr_5th = llr_curves[0]
        max_5th_idx = np.nanargmax(lr_5th)
        max_5th = lr_5th[max_5th_idx]
        exceeds_pathogenic = any(max_5th > tau for tau in tauP[:-1])
        
        # Check if insufficient evidence causes cutoff
        insufficient_evidence_pathogenic_idx = None
        if highest_pathogenic_with_evidence is not None and highest_pathogenic_with_evidence < max(pathogenic_points):
            # Find the threshold for the highest point with evidence
            tau_idx = pathogenic_points.index(highest_pathogenic_with_evidence)
            if tau_idx < len(pathogenic_points) - 1:
                tau_cutoff = tauP[tau_idx+1]
                # Find where curve crosses this threshold
                if not flipped:
                    crossing_indices = np.where(lr_5th >= tau_cutoff)[0]
                    if len(crossing_indices) > 0:
                        insufficient_evidence_pathogenic_idx = crossing_indices[-1]  # Last crossing
                else:
                    crossing_indices = np.where(lr_5th >= tau_cutoff)[0]
                    if len(crossing_indices) > 0:
                        insufficient_evidence_pathogenic_idx = crossing_indices[0]  # First crossing
        
        # Check if it comes back down after exceeding and find the crossing point
        should_plot_pathogenic_dotted = False
        pathogenic_crossing_idx = None
        if exceeds_pathogenic:
            # Find the highest threshold exceeded
            highest_tau_exceeded = max([tau for tau in tauP[:-1] if max_5th > tau])
            
            if not flipped:
                # Normal: Use FIRST crossing (when going up towards max from the left)
                before_max = lr_5th[:max_5th_idx+1]
                crossing_indices = np.where(before_max >= highest_tau_exceeded)[0]
                if len(crossing_indices) > 0:
                    # Check if it comes back down after the max
                    if max_5th_idx < len(lr_5th) - 1:
                        comes_back_down = any(lr_5th[max_5th_idx+1:] < highest_tau_exceeded)
                        if comes_back_down:
                            pathogenic_crossing_idx = crossing_indices[0]
                            should_plot_pathogenic_dotted = True
            else:
                # Flipped: Use SECOND crossing (when coming back down from max to the right)
                if max_5th_idx < len(lr_5th) - 1:
                    after_max = lr_5th[max_5th_idx+1:]
                    crossing_indices = np.where(after_max < highest_tau_exceeded)[0]
                    if len(crossing_indices) > 0:
                        # Check if it came from below before the max
                        comes_from_below = any(lr_5th[:max_5th_idx] < highest_tau_exceeded)
                        if comes_from_below:
                            pathogenic_crossing_idx = max_5th_idx + 1 + crossing_indices[0]
                            should_plot_pathogenic_dotted = True
        
        # Check 95th percentile: find min and its position
        lr_95th = llr_curves[2]
        min_95th_idx = np.nanargmin(lr_95th)
        min_95th = lr_95th[min_95th_idx]
        exceeds_benign = any(min_95th < tau for tau in tauB[:-1])
        
        # Check if insufficient evidence causes cutoff
        insufficient_evidence_benign_idx = None
        if lowest_benign_with_evidence is not None and lowest_benign_with_evidence > min(benign_points):
            # Find the threshold for the lowest benign point with evidence
            tau_idx = benign_points.index(lowest_benign_with_evidence)
            if tau_idx < len(benign_points) - 1:
                tau_cutoff = tauB[tau_idx+1]
                # Find where curve crosses this threshold
                if not flipped:
                    crossing_indices = np.where(lr_95th <= tau_cutoff)[0]
                    if len(crossing_indices) > 0:
                        insufficient_evidence_benign_idx = crossing_indices[-1]  # Last crossing
                else:
                    crossing_indices = np.where(lr_95th <= tau_cutoff)[0]
                    if len(crossing_indices) > 0:
                        insufficient_evidence_benign_idx = crossing_indices[0]  # First crossing
        
        # Check if it comes back up after going below and find the crossing point
        should_plot_benign_dotted = False
        benign_crossing_idx = None
        if exceeds_benign:
            # Find the lowest threshold crossed
            lowest_tau_crossed = min([tau for tau in tauB[:-1] if min_95th < tau])
            
            if not flipped:
                # Normal: Use SECOND crossing (when coming back up from min to the right)
                if min_95th_idx < len(lr_95th) - 1:
                    after_min = lr_95th[min_95th_idx+1:]
                    crossing_indices = np.where(after_min > lowest_tau_crossed)[0]
                    if len(crossing_indices) > 0:
                        # Check if it came from above before the min
                        comes_from_above = any(lr_95th[:min_95th_idx] > lowest_tau_crossed)
                        if comes_from_above:
                            benign_crossing_idx = min_95th_idx + 1 + crossing_indices[0]
                            should_plot_benign_dotted = True
            else:
                # Flipped: Use FIRST crossing (when going down towards min from the left)
                before_min = lr_95th[:min_95th_idx+1]
                crossing_indices = np.where(before_min <= lowest_tau_crossed)[0]
                if len(crossing_indices) > 0:
                    # Check if it comes back up after the min
                    if min_95th_idx < len(lr_95th) - 1:
                        comes_back_up = any(lr_95th[min_95th_idx+1:] > lowest_tau_crossed)
                        if comes_back_up:
                            benign_crossing_idx = crossing_indices[0]
                            should_plot_benign_dotted = True
        
        # Debug output
        if debug:
            print(f"\nSample {sample_num}:")
            print(f"  Highest pathogenic with evidence: {highest_pathogenic_with_evidence}, cutoff_idx: {insufficient_evidence_pathogenic_idx}")
            print(f"  Lowest benign with evidence: {lowest_benign_with_evidence}, cutoff_idx: {insufficient_evidence_benign_idx}")
            print(f"  5th percentile: max={max_5th:.3f} at idx={max_5th_idx}, exceeds_pathogenic={exceeds_pathogenic}")
            if exceeds_pathogenic:
                print(f"    highest_tau_exceeded={highest_tau_exceeded:.3f}, crossing_idx={pathogenic_crossing_idx}, should_plot_dotted={should_plot_pathogenic_dotted}")
            print(f"  tauP thresholds: {[f'{t:.3f}' for t in tauP[:-1]]}")
            print(f"  95th percentile: min={min_95th:.3f} at idx={min_95th_idx}, exceeds_benign={exceeds_benign}")
            if exceeds_benign:
                print(f"    lowest_tau_crossed={lowest_tau_crossed:.3f}, crossing_idx={benign_crossing_idx}, should_plot_dotted={should_plot_benign_dotted}")
            print(f"  tauB thresholds: {[f'{t:.3f}' for t in tauB[:-1]]}")
            print(f"  flipped={flipped}")
        
        # Plot curves
        for i, c in enumerate(colors):
            if len(log_lr_plus) == 1 and i != 1:
                continue  # if no bootstraps only plot one curve
            
            curve = llr_curves[i]
            
            # Handle 5th percentile (i=0)
            if i == 0:
                # Handle insufficient evidence
                if insufficient_evidence_pathogenic_idx is not None:
                    if debug:
                        print(f"  Plotting 5th percentile with insufficient evidence cutoff at idx {insufficient_evidence_pathogenic_idx}")
                    if not flipped:
                        # Regular: dotted up to cutoff, solid after
                        ax_lr.plot(score_range[:insufficient_evidence_pathogenic_idx+1], curve[:insufficient_evidence_pathogenic_idx+1], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                        ax_lr.plot(score_range[insufficient_evidence_pathogenic_idx:], curve[insufficient_evidence_pathogenic_idx:], 
                                 color=c, label=labels[i], linewidth=2)
                    else:
                        # Flipped: solid up to cutoff, dotted after
                        ax_lr.plot(score_range[:insufficient_evidence_pathogenic_idx+1], curve[:insufficient_evidence_pathogenic_idx+1], 
                                 color=c, label=labels[i], linewidth=2)
                        ax_lr.plot(score_range[insufficient_evidence_pathogenic_idx:], curve[insufficient_evidence_pathogenic_idx:], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                # Handle non-monotonic
                elif should_plot_pathogenic_dotted and pathogenic_crossing_idx is not None:
                    if debug:
                        print(f"  Plotting 5th percentile with non-monotonic cutoff at idx {pathogenic_crossing_idx}")
                    if not flipped:
                        # Normal: dotted before crossing, solid after
                        ax_lr.plot(score_range[:pathogenic_crossing_idx+1], curve[:pathogenic_crossing_idx+1], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                        ax_lr.plot(score_range[pathogenic_crossing_idx:], curve[pathogenic_crossing_idx:], 
                                 color=c, label=labels[i], linewidth=2)
                    else:
                        # Flipped: solid before crossing, dotted after
                        ax_lr.plot(score_range[:pathogenic_crossing_idx+1], curve[:pathogenic_crossing_idx+1], 
                                 color=c, label=labels[i], linewidth=2)
                        ax_lr.plot(score_range[pathogenic_crossing_idx:], curve[pathogenic_crossing_idx:], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                else:
                    ax_lr.plot(score_range, curve, color=c, label=labels[i], linewidth=2)
            
            # Handle 95th percentile (i=2)
            elif i == 2:
                # Handle insufficient evidence
                if insufficient_evidence_benign_idx is not None:
                    if debug:
                        print(f"  Plotting 95th percentile with insufficient evidence cutoff at idx {insufficient_evidence_benign_idx}")
                    if not flipped:
                        # Regular: solid up to cutoff, dotted after
                        ax_lr.plot(score_range[:insufficient_evidence_benign_idx+1], curve[:insufficient_evidence_benign_idx+1], 
                                 color=c, label=labels[i], linewidth=2)
                        ax_lr.plot(score_range[insufficient_evidence_benign_idx:], curve[insufficient_evidence_benign_idx:], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                    else:
                        # Flipped: dotted up to cutoff, solid after
                        ax_lr.plot(score_range[:insufficient_evidence_benign_idx+1], curve[:insufficient_evidence_benign_idx+1], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                        ax_lr.plot(score_range[insufficient_evidence_benign_idx:], curve[insufficient_evidence_benign_idx:], 
                                 color=c, label=labels[i], linewidth=2)
                # Handle non-monotonic
                elif should_plot_benign_dotted and benign_crossing_idx is not None:
                    if debug:
                        print(f"  Plotting 95th percentile with non-monotonic cutoff at idx {benign_crossing_idx}")
                    if not flipped:
                        # Normal: solid before crossing, dotted after
                        ax_lr.plot(score_range[:benign_crossing_idx+1], curve[:benign_crossing_idx+1], 
                                 color=c, label=labels[i], linewidth=2)
                        ax_lr.plot(score_range[benign_crossing_idx:], curve[benign_crossing_idx:], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                    else:
                        # Flipped: dotted before crossing, solid after
                        ax_lr.plot(score_range[:benign_crossing_idx+1], curve[:benign_crossing_idx+1], 
                                 color=c, linestyle=':', alpha=0.8, linewidth=2)
                        ax_lr.plot(score_range[benign_crossing_idx:], curve[benign_crossing_idx:], 
                                 color=c, label=labels[i], linewidth=2)
                # Plot normally for median (i=1)
                else:
                    ax_lr.plot(score_range, curve, color=c, 
                             label=labels[i] if len(log_lr_plus) != 1 else 'Single fit',
                             linewidth=2)
            
        
        ax_lr.set_title(f"Log LR+\nprior: {priors[1]:.3f}, C: {indv_summary['C']}", fontsize=11)
        add_thresholds(tauP[:-1], tauB[:-1], ax_lr)
        ax_lr.set_xlabel("Score")
        ax_lr.set_ylabel("Log LR+")
        ax_lr.legend(fontsize=8, loc='best')
        ax_lr.set_xlim(xlim)
        ax_lr.set_ylim([tauB[-1], tauP[-1]])
        ax_lr.grid(linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    fig.suptitle(f"{dataset} - {n_c}{relax_code} {config}", fontsize=16, y=0.998)
    
    return fig


def plot_scores_only(dataset, scoreset):

    
    n_samples = len([s for s in scoreset.samples])
    score_range = [min(scoreset.scores), max(scoreset.scores)]
    
    # Create figure: 3 rows, n_samples columns (all square)
    fig, ax = plt.subplots(1, n_samples, figsize=(7*n_samples, 6), 
                           squeeze=False, gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

    
    # ===== Row 0: Sample fits =====
    for sample_num in range(n_samples):
        ax_fit = ax[0, sample_num]
        
        sns.histplot(scoreset.scores[scoreset.sample_assignments[:,sample_num]], 
                     stat='density', ax=ax_fit, alpha=.5, color='pink')

        max_hist_density = max([patch.get_height() for patch in ax_fit.patches])
        
        # density = sample_density(score_range, fits, sample_num)
        # for compNum in range(density.shape[1]):
        #     compDensity = density[:,compNum,:]
        #     d = np.nanpercentile(compDensity,[5,50,95],axis=0)
        #     ax_fit.plot(score_range, d[1], color=f"C{compNum}", linestyle='--', label=f"Comp {compNum+1}")
        # ax_fit.legend(fontsize=8)
        
        # d = np.nansum(density, axis=1)
        # d_perc = np.percentile(d, [5,50,95], axis=0)
        # ax_fit.plot(score_range, d_perc[1], color='black', alpha=.5)
        # ax_fit.fill_between(score_range, d_perc[0], d_perc[2], color='gray', alpha=0.3)
        ax_fit.set_title(f"{scoreset.sample_names[sample_num]}\n(n={scoreset.sample_assignments[:,sample_num].sum():,d})")
        ax_fit.set_xlabel("Score")
        ax_fit.set_ylabel("Density")
        ax_fit.set_ylim([0, max_hist_density * 1.1])
        ax_fit.grid(linewidth=0.5, alpha=0.3)
        
    fig.suptitle(dataset)
    
    return fig

def plot_scoreset_example_publication(dataset, scoreset, indv_summary, fits, score_range, config, n_c, n_samples, relax=False, flipped=False, debug=False):
    """
    Plot each sample in separate vertical subplots with all thresholds overlayed.
    
    Parameters: (same as original)
    """
    
    # Sample colors matching the original plot
    sample_colors = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']  # P/LP, B/LB, gnomAD, Synonymous
    sample_alphas = [0.6, 0.6, 0.3, 0.5]
    
    # Threshold configuration for benign (negative) and pathogenic (positive)
    point_values_to_plot = [1, 2, 3, 4, 8]
    linestyles = ['dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (3, 5, 1, 5))]
    linewidths = [1.5, 1.5, 1.5, 1.5, 1.5]
    labels_thresh = ['Supporting', 'Moderate', 'Moderate+', 'Strong', 'Very Strong']
    
    relax_code = "R" if relax else ""
    
    # Create figure with n_samples rows
    scale = 4
    fig, axes = plt.subplots(n_samples, 1, figsize=(2*scale, scale*n_samples), squeeze=False)
    axes = axes.flatten()
    
    # Get point ranges for threshold plotting
    point_ranges = indv_summary['point_ranges']
    
    # Plot each sample in its own subplot
    for sample_num in range(n_samples):
        ax = axes[sample_num]
        sample_mask = scoreset.sample_assignments[:, sample_num]
        sample_name = scoreset.sample_names[sample_num]
        
        # Plot histogram for this sample
        sns.histplot(scoreset.scores[sample_mask], 
                     stat='density', ax=ax, 
                     alpha=sample_alphas[sample_num], 
                     color=sample_colors[sample_num])
                     # label=sample_name)
        
        max_hist_density = max([patch.get_height() for patch in ax.patches]) if ax.patches else 1.0
        
        # Plot fitted density curves with matching color
        density_sample = sample_density(score_range, fits, sample_num)
        
        # Plot sum of components
        d = np.nansum(density_sample, axis=1)
        d_perc = np.percentile(d, [5, 50, 95], axis=0)
        ax.plot(score_range, d_perc[1], 
               color='black', 
               alpha=0.5,
               linewidth=2)
        ax.fill_between(score_range, d_perc[0], d_perc[2], 
                       color='gray', 
                       alpha=0.3)
        
        import matplotlib.lines as mlines

        # Add threshold lines for all point values (both positive and negative)
        for idx, point_val in enumerate(point_values_to_plot):
            # Find benign threshold (negative point value)
            for pv, score_ranges in point_ranges.items():
                if pv == -point_val:
                    for sr in score_ranges:
                        threshold_score = sr[0] if not flipped else sr[1]
                        ax.axvline(threshold_score, 
                                  color='b',
                                  linestyle=linestyles[idx],
                                  linewidth=linewidths[idx],
                                  alpha=0.7)
                                  # label=f"-{point_val}")
                        break
                    break
            
            # Find pathogenic threshold (positive point value)
            for pv, score_ranges in point_ranges.items():
                if pv == point_val:
                    for sr in score_ranges:
                        threshold_score = sr[1] if not flipped else sr[0]
                        ax.axvline(threshold_score, 
                                  color='r',
                                  linestyle=linestyles[idx],
                                  linewidth=linewidths[idx],
                                  alpha=0.7)
                                  # label=f"+{point_val}")
                        break
                    break

        handles = []
        for idx, point_val in enumerate(point_values_to_plot):
            if len(point_ranges[point_val]) != 0 or len(point_ranges[-point_val]) != 0:
                h = mlines.Line2D(
                    [], [],
                    color='gray',
                    linestyle=linestyles[idx],
                    linewidth=linewidths[idx],
                    label=f"±{point_val}"
                )
                handles.append(h)

        if sample_name != "gnomAD":
            ax.set_title(f"{sample_name} (n={sample_mask.sum():,d})", fontsize=16, fontweight='bold')
        else:
            ax.set_title(f"{sample_name} (n={sample_mask.sum():,d}, prior={indv_summary['prior']:.3f})", fontsize=16, fontweight='bold')
        ax.set_xlabel("Assay score", fontsize=14)
        ax.set_ylabel("Density", fontsize=14)
        ax.set_ylim([0, max_hist_density * 1.1])
        ax.legend(fontsize=11, loc='best', ncol=1, handles=handles)
        ax.grid(linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def plot_scoreset_calibration_comparison(dataset, scoreset, indv_summary, fits, score_range, config, n_c, n_samples, relax=False, flipped=False, debug=False):
    """
    Plot histogram with P/LP, B/LB, all SNVs, threshold lines, and calibration comparisons below.
    """
    
    # Threshold configuration
    point_values_to_plot = [1, 2, 3, 4, 8]
    linestyles = ['dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (3, 5, 1, 5))]
    linewidths = [1.5, 1.5, 1.5, 1.5, 1.5]
    
    # Strength colors for calibration bars
    strenth_color = {
        "BP4_Very Strong": "#4b91a6",
        "BP4_Strong": "#7ab5d1",
        "-3": "#99c8dc",
        "BP4_Moderate": "#d0e8f0",
        "BP4_Supporting": "#e4f1f6",
        "IR": "#e0e0e0",
        "PP3_Supporting": "#e6b1b8",
        "PP3_Moderate": "#d68f99",
        "+3": "#ca7682",
        "PP3_Strong": "#b85c6b",
        "PP3_Very Strong": "#943744"
    }
    
    relax_code = "R" if relax else ""
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[2, 1, 1, 0.3], hspace=0.3)
    
    # Main histogram
    ax1 = plt.subplot(gs[0])
    # Jia 2021 (0-cutoff)
    ax2 = plt.subplot(gs[1])
    # DanZ-based calibration
    ax3 = plt.subplot(gs[2])
    # Legend axis
    leg_ax = plt.subplot(gs[3])
    leg_ax.axis('off')
    
    # Get all scores
    all_scores = scoreset.snv_scores
    x_min = min(all_scores.min(), scoreset.scores.min())
    x_max = max(all_scores.max(), scoreset.scores.max())
    bin_width = (x_max - x_min) / 50
    
    # Get point ranges for threshold plotting
    point_ranges = indv_summary['point_ranges']
    
    # Assume first two samples are P/LP and B/LB
    plp_mask = scoreset.sample_assignments[:, 0]
    blb_mask = scoreset.sample_assignments[:, 1]
    
    # Plot histograms
    sns.histplot(scoreset.scores[blb_mask], 
                 binwidth=bin_width, color='#1D7AAB', alpha=0.6, ax=ax1, label='ClinVar B/LB')
    sns.histplot(scoreset.scores[plp_mask], 
                 binwidth=bin_width, color='#CA7682', alpha=0.6, ax=ax1, label='ClinVar P/LP')
    
    # Overlay all SNVs on secondary axis
    ax1_twin = ax1.twinx()
    sns.histplot(all_scores, binwidth=bin_width, color='#A0A0A0', alpha=0.3, ax=ax1_twin, label='All SNVs')
    
    # Add threshold vertical lines
    threshold_scores_benign = []
    threshold_scores_path = []
    
    for idx, point_val in enumerate(point_values_to_plot):
        # Find benign threshold (negative point value)
        for pv, score_ranges in point_ranges.items():
            if pv == -point_val:
                for sr in score_ranges:
                    threshold_score = sr[0] if not flipped else sr[1]
                    threshold_scores_benign.append(threshold_score)
                    # ax1.axvline(threshold_score, 
                    #           color='b',
                    #           linestyle=linestyles[idx],
                    #           linewidth=linewidths[idx],
                    #           alpha=0.7)
                    break
                break
        
        # Find pathogenic threshold (positive point value)
        for pv, score_ranges in point_ranges.items():
            if pv == point_val:
                for sr in score_ranges:
                    threshold_score = sr[1] if not flipped else sr[0]
                    threshold_scores_path.append(threshold_score)
                    # ax1.axvline(threshold_score, 
                    #           color='r',
                    #           linestyle=linestyles[idx],
                    #           linewidth=linewidths[idx],
                    #           alpha=0.7)
                    break
                break
    
    ax1.set_xlim(x_min, x_max)
    ax1_twin.set_xlim(x_min, x_max)
    ax1.set_xlabel('')
    ax1.set_ylabel('P/B variant count', fontsize=14)
    ax1_twin.set_ylabel('SNV count', fontsize=14)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    
    # Add threshold legend
    import matplotlib.lines as mlines
    threshold_handles = []
    # for idx, point_val in enumerate(point_values_to_plot):
    #     if point_val in point_ranges or -point_val in point_ranges:
    #         h = mlines.Line2D([], [], color='gray', linestyle=linestyles[idx], 
    #                         linewidth=linewidths[idx], label=f"±{point_val}")
    #         threshold_handles.append(h)
    
    ax1.legend(lines1 + lines2 + threshold_handles, labels1 + labels2 + [h.get_label() for h in threshold_handles], 
              loc='best', fontsize=11, ncol=1)
    ax1_twin.get_legend().remove() if ax1_twin.get_legend() else None
    ax1.set_title(f'MSH2 Functional Score', fontsize=22, fontweight='bold')
    
    # Row 2: Scott
    ax2.axvspan(x_min, 0, color=strenth_color['BP4_Strong'], alpha=0.9)
    ax2.axvspan(0, 0.4, color=strenth_color['IR'], alpha=0.9)
    ax2.axvspan(0.4, x_max, color=strenth_color['PP3_Strong'], alpha=0.9)
    
    count_below_0 = (all_scores < 0).sum()
    count_above_0 = (all_scores > 0.4).sum()
    ax2.text(x_min/2, 5, f'{count_below_0}', ha='center', va='center', color='black', fontsize=13)
    ax2.text(x_max/2, 5, f'{count_above_0}', ha='center', va='center', color='black', fontsize=13)
    
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(0, 10)
    ax2.set_ylabel('SNV Count', fontsize=14)
    ax2.set_yticks([])
    ax2.set_xticks([])#[0], ['0'], fontsize=14)
    ax2.set_title('Scott et al. (2022)', loc='left', pad=5, fontsize=18, style='italic')
    ax2.grid(False)
    
    # Row 3: DanZ-based calibration with threshold intervals
    # Build intervals from thresholds
    threshold_scores_benign_sorted = sorted(threshold_scores_benign)  # -8, -4, -3, -2, -1
    threshold_scores_path_sorted = sorted(threshold_scores_path)  # +1, +2, +3, +4, +8
    
    intervals = []
    
    # Benign intervals
    # if len(threshold_scores_benign_sorted) >= 3:
    intervals.append(("BP4_Moderate", x_min, threshold_scores_benign_sorted[0]))  # -3 and below
    intervals.append(("BP4_Supporting", threshold_scores_benign_sorted[0], threshold_scores_benign_sorted[1]))  # -2

    # print(threshold_scores_benign_sorted)
    # print(threshold_scores_path_sorted)
    
    # IR interval
    # if len(threshold_scores_benign_sorted) >= 5 and len(threshold_scores_path_sorted) >= 1:
    intervals.append(("IR", threshold_scores_benign_sorted[1], threshold_scores_path_sorted[0]))
    
    # Pathogenic intervals
    # if len(threshold_scores_path_sorted) >= 3:
    intervals.append(("PP3_Supporting", threshold_scores_path_sorted[0], threshold_scores_path_sorted[1]))  # +1
    intervals.append(("PP3_Moderate", threshold_scores_path_sorted[1], threshold_scores_path_sorted[2]))  # +2
    intervals.append(("+3", threshold_scores_path_sorted[2], x_max))  # +3 and above
    
    for name, start, end in intervals:
        ax3.axvspan(start, end, color=strenth_color[name], alpha=0.9)
        count = ((all_scores >= start) & (all_scores < end)).sum()
        if (end - start) > 0.2:
            ax3.text((start + end) / 2, 5, str(count), ha='center', va='center', 
                    fontsize=13, color='black')
    
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(0, 10)
    ax3.set_ylabel('SNV Count', fontsize=14)
    ax3.set_yticks([])
    
    # Set x-axis ticks at thresholds
    all_thresholds = threshold_scores_benign_sorted + threshold_scores_path_sorted
    ax3.set_xticks([])#all_thresholds, [f'{x:.2f}' for x in all_thresholds], rotation=60, fontsize=12)
    ax3.set_title('Zeiberg et al. (2025)', loc='left', pad=5, fontsize=18, style='italic')
    ax3.grid(False)
    
    # Legend for strength colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=strenth_color[name], label=name) for name in [
        "BP4_Moderate", "BP4_Supporting", "IR", 
        "PP3_Supporting", "PP3_Moderate", "+3"
        # "BP4_Very Strong", "BP4_Strong", "-3", "BP4_Moderate", "BP4_Supporting", "IR", 
        # "PP3_Supporting", "PP3_Moderate", "+3", "PP3_Strong", "PP3_Very Strong"
    ]]
    leg_ax.legend(handles=legend_elements, loc='center', ncol=6, frameon=False, fontsize=12)
    
    # plt.suptitle(f'MSH2 Assay', fontsize=22, fontweight='bold')
    plt.tight_layout()
    
    return fig

import matplotlib
import pandas as pd
def plot_confusion_mat(dataset, scoreset, indv_summary, fits, score_range, config, n_c, n_samples, relax=False, flipped=False, debug=False):
    """
    Plot confusion matrix comparing DanZ calibration vs Jia 2021 (0-cutoff).
    Shows how P/LP and B/LB variants are classified into Benign/IR/Pathogenic ranges.
    """
    
    # Get point ranges
    point_ranges = indv_summary['point_ranges']
    
    # Assume first two samples are P/LP and B/LB
    plp_scores = scoreset.scores[scoreset.sample_assignments[:, 0]]
    blb_scores = scoreset.scores[scoreset.sample_assignments[:, 1]]
    
    # Determine benign, IR, and pathogenic score ranges for DanZ
    benign_ranges = []
    ir_ranges = []
    pathogenic_ranges = []
    
    for pv, score_ranges in point_ranges.items():
        if pv < 0:  # Benign
            benign_ranges.extend(score_ranges)
        elif pv > 0:  # Pathogenic
            pathogenic_ranges.extend(score_ranges)
    
    # Count B/LB variants in each category (DanZ)
    blb_in_benign = sum(any(start <= score <= end for start, end in benign_ranges) for score in blb_scores)
    blb_in_path   = sum(any(start <= score <= end for start, end in pathogenic_ranges) for score in blb_scores)
    
    # IR = NOT benign AND NOT path
    blb_in_ir = sum(
        not any(start <= score <= end for start, end in benign_ranges) and
        not any(start <= score <= end for start, end in pathogenic_ranges)
        for score in blb_scores
    )
    
    # Count P/LP variants in each category (DanZ)
    plp_in_benign = sum(any(start <= score <= end for start, end in benign_ranges) for score in plp_scores)
    plp_in_path   = sum(any(start <= score <= end for start, end in pathogenic_ranges) for score in plp_scores)
    
    plp_in_ir = sum(
        not any(start <= score <= end for start, end in benign_ranges) and
        not any(start <= score <= end for start, end in pathogenic_ranges)
        for score in plp_scores
    )

    
    # Create DanZ DataFrame
    danz = pd.DataFrame({
        'BLB': [blb_in_benign, blb_in_ir, blb_in_path],
        'PLP': [plp_in_benign, plp_in_ir, plp_in_path]
    }).T

    
    # Count for Jia 2021 (0-cutoff)
    blb_below_0 = (blb_scores < 0).sum()
    blb_above_0 = (blb_scores > 0).sum()
    plp_below_0 = (plp_scores < 0).sum()
    plp_above_0 = (plp_scores > 0).sum()
    
    # Create Jia 2021 DataFrame
    auth = pd.DataFrame({
        'BLB': [blb_below_0, 0, blb_above_0],
        'PLP': [plp_below_0, 0, plp_above_0]
    }).T
    
    ind = ['Normal', 'IR', 'Abnormal']
    danz.columns = ind
    auth.columns = ind
    
    print(danz)
    print(auth)
    
    if debug:
        print("DanZ counts:")
        print(danz)
        print("\nJia 2021 counts:")
        print(auth)
    
    # Calculate row-wise percentages
    danz_percent = danz.div(danz.sum(axis=1), axis=0) * 100
    auth_percent = auth.div(auth.sum(axis=1), axis=0) * 100
    difference = (danz_percent - auth_percent)
    
    # Create custom colormap
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    colors = ["gold", "whitesmoke", "purple"]
    cmap_custom = LinearSegmentedColormap.from_list("PurpleYellow", colors)
    max_abs_value = 10  # Maximum absolute value for scale
    step = 1            # Step size between bounds
    
    posbounds = np.arange(0, max_abs_value + step, step)
    negbounds = -np.arange(step, max_abs_value + step, step)[::-1]
    bounds = np.concatenate([negbounds, posbounds])
    norm = BoundaryNorm(bounds, cmap_custom.N)
    
    def format_diff_value(x):
        if abs(x) < 0.1:  # Only show differences ≥ 0.1%
            return f"{x:+.2f}%"
        return f"{x:+.1f}%"
    
    format_array = np.vectorize(format_diff_value)
    annot_data = format_array(difference)
    
    # Create heatmap
    fig = plt.figure(figsize=(10, 3))
    sns.heatmap(difference, 
                annot=annot_data, fmt='',
                cmap=cmap_custom, 
                norm=norm,
                cbar_kws={'label': 'Percentage Point Difference', 'pad': 0.01},
                linewidths=0.5,
                linecolor='gray')
    
    relax_code = "R" if relax else ""
    plt.title(f'{dataset.split("_")[0]}: Zeiberg et al. (2025) vs. Author Difference', 
              fontsize=14, fontweight='bold')
    plt.ylabel('ClinVar Classification', fontsize=12)
    plt.xlabel('Functional Category', fontsize=12)
    plt.tight_layout()
    
    return fig

def plot_scoreset_final_pillar_project(dataset, scoreset, indv_summary, fits, score_range, config, n_c, n_samples, relax=False, flipped=False, debug=False):
    """
    Combined plot with:
    - All samples (P/LP, B/LB, gnomAD, Synonymous) overlayed in one histogram with fitted densities
    - Vertical threshold lines
    - Two calibration interval bars below (Scott et al. and Zeiberg et al.)
    
    Parameters: (same as original)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.lines as mlines
    from matplotlib.patches import Patch
    import seaborn as sns
    import numpy as np
    
    # Sample colors matching the original plot
    sample_colors = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']  # P/LP, B/LB, gnomAD, Synonymous
    sample_alphas = [0.5, 0.5, 0.15, 0.4]  # gnomAD more transparent as background
    
    # Darker versions of sample colors for fitted density lines
    fit_colors = ['#8B3A47', '#0D4A6B', '#505050', '#3A7A45']  # Darker versions
    
    # Threshold configuration
    point_values_to_plot = [1, 2, 3, 4, 8]
    linestyles = ['dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (3, 5, 1, 5))]
    linewidths = [1.5, 1.5, 1.5, 1.5, 1.5]
    
    # Strength colors for calibration bars
    strength_color = {
        "BP4_Very Strong": "#4b91a6",
        "BP4_Strong": "#7ab5d1",
        "-3": "#99c8dc",
        "BP4_Moderate": "#d0e8f0",
        "BP4_Supporting": "#e4f1f6",
        "IR": "#e0e0e0",
        "PP3_Supporting": "#e6b1b8",
        "PP3_Moderate": "#d68f99",
        "+3": "#ca7682",
        "PP3_Strong": "#b85c6b",
        "PP3_Very Strong": "#943744"
    }
    
    relax_code = "R" if relax else ""
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 0.6, 0.6, 0.4], hspace=0.35)
    
    ax_hist = plt.subplot(gs[0])   # Main histogram with all samples
    ax_scott = plt.subplot(gs[1])  # Scott et al. calibration
    ax_zeiberg = plt.subplot(gs[2])  # Zeiberg et al. calibration
    leg_ax = plt.subplot(gs[3])    # Legend axis
    leg_ax.axis('off')
    
    # Get score ranges for x-axis limits
    all_scores = scoreset.snv_scores
    x_min = score_range[0]
    x_max = score_range[-1]
    bin_width = (x_max - x_min) / 50
    
    # Get point ranges for threshold plotting
    point_ranges = indv_summary['point_ranges']
    
    # Build legend handles for samples (histogram + fit)
    sample_handles = []
    
    # Plot each sample's histogram and fitted density
    for sample_num in range(n_samples):
        sample_mask = scoreset.sample_assignments[:, sample_num]
        sample_name = scoreset.sample_names[sample_num]
        color = sample_colors[sample_num]
        fit_color = fit_colors[sample_num]
        alpha = sample_alphas[sample_num]
        
        # Plot histogram for this sample
        sns.histplot(
            scoreset.scores[sample_mask],
            binwidth=bin_width,
            stat='density',
            ax=ax_hist,
            alpha=alpha,
            color=color,
        )
        
        # Plot fitted density curve with darker color for visibility
        density_sample = sample_density(score_range, fits, sample_num)
        d = np.nansum(density_sample, axis=1)
        d_perc = np.percentile(d, [5, 50, 95], axis=0)
        
        fit_alpha = 0.5 if sample_num == 2 else 1.0  # gnomAD fit more subtle
        
        # Darker median line
        ax_hist.plot(
            score_range, d_perc[1],
            color=fit_color,
            alpha=fit_alpha,
            linewidth=2,
            zorder=10
        )
        # Confidence band
        ax_hist.fill_between(
            score_range, d_perc[0], d_perc[2],
            color=fit_color,
            alpha=0.15 if sample_num == 2 else 0.2,
            zorder=5
        )
        
        # Create combined legend handle: patch for histogram + line for fit
        hist_patch = Patch(facecolor=color, alpha=alpha, edgecolor='none')
        fit_line = mlines.Line2D([], [], color=fit_color, linewidth=2, alpha=fit_alpha)
        sample_handles.append((hist_patch, fit_line, f'{sample_name} (n={sample_mask.sum():,d})'))
    
    # Collect thresholds for calibration bars
    threshold_scores_benign = []
    threshold_scores_path = []
    
    # Add threshold vertical lines
    for idx, point_val in enumerate(point_values_to_plot):
        # Find benign threshold (negative point value)
        for pv, score_ranges_pr in point_ranges.items():
            if pv == -point_val:
                for sr in score_ranges_pr:
                    threshold_score = sr[0] if not flipped else sr[1]
                    threshold_scores_benign.append(threshold_score)
                    ax_hist.axvline(
                        threshold_score,
                        color='#2166AC',  # Darker blue
                        linestyle=linestyles[idx],
                        linewidth=linewidths[idx],
                        alpha=0.8
                    )
                    break
                break
        
        # Find pathogenic threshold (positive point value)
        for pv, score_ranges_pr in point_ranges.items():
            if pv == point_val:
                for sr in score_ranges_pr:
                    threshold_score = sr[1] if not flipped else sr[0]
                    threshold_scores_path.append(threshold_score)
                    ax_hist.axvline(
                        threshold_score,
                        color='#B2182B',  # Darker red
                        linestyle=linestyles[idx],
                        linewidth=linewidths[idx],
                        alpha=0.8
                    )
                    break
                break
    
    # Build threshold legend handles
    threshold_handles = []
    for idx, point_val in enumerate(point_values_to_plot):
        if len(point_ranges.get(point_val, [])) != 0 or len(point_ranges.get(-point_val, [])) != 0:
            h = mlines.Line2D(
                [], [],
                color='gray',
                linestyle=linestyles[idx],
                linewidth=linewidths[idx],
                label=f"±{point_val}"
            )
            threshold_handles.append(h)
    
    # Configure main histogram axis
    ax_hist.set_xlim(x_min, x_max)
    ax_hist.set_xlabel('')
    ax_hist.set_ylabel('Density', fontsize=12)
    ax_hist.set_title(f'{dataset.split("_")[0]} Functional Score', fontsize=14, fontweight='bold')
    ax_hist.tick_params(axis='both', labelsize=10)
    
    # Create sample legend (upper left) with histogram + fit distinction
    from matplotlib.legend_handler import HandlerTuple
    sample_legend_handles = [(h[0], h[1]) for h in sample_handles]
    sample_legend_labels = [h[2] for h in sample_handles]
    
    legend1 = ax_hist.legend(
        sample_legend_handles, 
        sample_legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
        loc='upper left',
        fontsize=10,
    )
    ax_hist.add_artist(legend1)
    
    # Create threshold legend (upper right)
    legend2 = ax_hist.legend(
        handles=threshold_handles,
        loc='upper right',
        fontsize=10,
    )
    
    # Row 2: Scott et al. (2022) calibration
    ax_scott.axvspan(x_min, 0, color=strength_color['BP4_Strong'], alpha=0.9)
    ax_scott.axvspan(0, 0.4, color=strength_color['IR'], alpha=0.9)
    ax_scott.axvspan(0.4, x_max, color=strength_color['PP3_Strong'], alpha=0.9)
    
    count_below_0 = (all_scores < 0).sum()
    count_0_to_04 = ((all_scores >= 0) & (all_scores < 0.4)).sum()
    count_above_04 = (all_scores >= 0.4).sum()
    ax_scott.text((x_min + 0) / 2, 0.5, f'{count_below_0}', ha='center', va='center', color='black', fontsize=11)
    ax_scott.text((0 + 0.4) / 2, 0.5, f'{count_0_to_04}', ha='center', va='center', color='black', fontsize=11)
    ax_scott.text((0.4 + x_max) / 2, 0.5, f'{count_above_04}', ha='center', va='center', color='black', fontsize=11)
    
    ax_scott.set_xlim(x_min, x_max)
    ax_scott.set_ylim(0, 1)
    ax_scott.set_yticks([])
    ax_scott.set_xticks([])
    ax_scott.set_title('Scott et al. (2022)', loc='left', pad=3, fontsize=11, style='italic')
    
    # Row 3: Zeiberg et al. (2025) calibration with threshold intervals
    threshold_scores_benign_sorted = sorted(threshold_scores_benign)
    threshold_scores_path_sorted = sorted(threshold_scores_path)
    
    intervals = []
    if len(threshold_scores_benign_sorted) >= 2 and len(threshold_scores_path_sorted) >= 3:
        intervals.append(("BP4_Moderate", x_min, threshold_scores_benign_sorted[0]))
        intervals.append(("BP4_Supporting", threshold_scores_benign_sorted[0], threshold_scores_benign_sorted[1]))
        intervals.append(("IR", threshold_scores_benign_sorted[1], threshold_scores_path_sorted[0]))
        intervals.append(("PP3_Supporting", threshold_scores_path_sorted[0], threshold_scores_path_sorted[1]))
        intervals.append(("PP3_Moderate", threshold_scores_path_sorted[1], threshold_scores_path_sorted[2]))
        intervals.append(("+3", threshold_scores_path_sorted[2], x_max))
    
    for name, start, end in intervals:
        ax_zeiberg.axvspan(start, end, color=strength_color[name], alpha=0.9)
        count = ((all_scores >= start) & (all_scores < end)).sum()
        if (end - start) > 0.2:
            ax_zeiberg.text(
                (start + end) / 2, 0.5, str(count),
                ha='center', va='center',
                fontsize=11, color='black'
            )
    
    ax_zeiberg.set_xlim(x_min, x_max)
    ax_zeiberg.set_ylim(0, 1)
    ax_zeiberg.set_yticks([])
    ax_zeiberg.set_xlabel('Assay Score', fontsize=12)
    ax_zeiberg.tick_params(axis='x', labelsize=10)
    ax_zeiberg.set_title('Zeiberg et al. (2025)', loc='left', pad=3, fontsize=11, style='italic')
    
    # Legend for strength colors - ordered from benign to pathogenic
    # Define the desired order
    legend_order = ["BP4_Strong", "BP4_Moderate", "BP4_Supporting", "IR", "PP3_Supporting", "PP3_Moderate", "+3", "PP3_Strong"]
    
    # Collect all used strength names
    used_strengths = {"BP4_Strong", "IR", "PP3_Strong"}  # Scott always uses these
    for name, _, _ in intervals:
        used_strengths.add(name)
    
    # Build legend in order
    all_legend = [
        Patch(facecolor=strength_color[name], label=name, edgecolor='none')
        for name in legend_order
        if name in used_strengths
    ]
    
    leg_ax.legend(handles=all_legend, loc='upper center', ncol=len(all_legend), frameon=False, fontsize=9)
    
    plt.tight_layout()
    
    return fig


def plot_scoreset_final_pillar_project_v2(dataset, scoreset, indv_summary, fits, score_range, config, n_c, n_samples, relax=False, flipped=False, debug=False):
    """
    Combined plot with:
    - Top row: Individual sample fits (one per column) showing mixture components
    - Second row: All samples overlayed in one histogram (no fits)
    - Third row: Scott et al. calibration
    - Fourth row: Zeiberg et al. calibration
    - Bottom: Legend
    
    Parameters: (same as original)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.lines as mlines
    from matplotlib.patches import Patch
    import seaborn as sns
    import numpy as np
    
    # Sample colors matching the original plot
    sample_colors = ['#CA7682', '#1D7AAB', '#A0A0A0', '#6BAA75']  # P/LP, B/LB, gnomAD, Synonymous
    sample_alphas = [0.5, 0.5, 0.15, 0.4]  # gnomAD more transparent as background
    
    # Component colors for fits (colorblind-friendly palette)
    component_colors = plt.cm.Set2(np.linspace(0, 1, n_c))
    
    # Threshold configuration
    point_values_to_plot = [1, 2, 3, 4, 8]
    linestyles = ['dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (3, 5, 1, 5))]
    linewidths = [1.25, 1.25, 1.25, 1.25, 1.25]
    
    # Strength colors for calibration bars
    strength_color = {
        "BP4_Very Strong": "#4b91a6",
        "BP4_Strong": "#7ab5d1",
        "-3": "#99c8dc",
        "BP4_Moderate": "#d0e8f0",
        "BP4_Supporting": "#e4f1f6",
        "IR": "#e0e0e0",
        "PP3_Supporting": "#e6b1b8",
        "PP3_Moderate": "#d68f99",
        "+3": "#ca7682",
        "PP3_Strong": "#b85c6b",
        "PP3_Very Strong": "#943744"
    }
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(18, 13.17460317))
    gs = gridspec.GridSpec(5, n_samples, height_ratios=[2, 2, 1, 1, 0.3], hspace=0.3, wspace=0.15)
    
    # Top row: individual fit panels (one per sample)
    ax_fits = [plt.subplot(gs[0, i]) for i in range(n_samples)]
    
    # Second row: combined histogram spanning all columns
    ax_hist = plt.subplot(gs[1, :])
    
    # Third row: Scott et al. calibration
    ax_scott = plt.subplot(gs[2, :])
    
    # Fourth row: Zeiberg et al. calibration
    ax_zeiberg = plt.subplot(gs[3, :])
    
    # Bottom: Legend axis
    leg_ax = plt.subplot(gs[4, :])
    leg_ax.axis('off')
    
    # Add row title for top row
    fig.text(0.5, 0.902, 'Model Fits', ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Get score ranges for x-axis limits
    all_scores = scoreset.snv_scores
    x_min = score_range[0]
    x_max = score_range[-1]
    bin_width = (x_max - x_min) / 50
    
    # Get point ranges for threshold plotting
    point_ranges = indv_summary['point_ranges']
    
    # Pre-compute thresholds for use in fit panels
    threshold_scores_benign = []
    threshold_scores_path = []
    threshold_info = []  # Store (threshold_score, color, linestyle, linewidth) tuples
    
    for idx, point_val in enumerate(point_values_to_plot):
        for pv, score_ranges_pr in point_ranges.items():
            if pv == -point_val:
                for sr in score_ranges_pr:
                    threshold_score = sr[0] if not flipped else sr[1]
                    threshold_scores_benign.append(threshold_score)
                    threshold_info.append((threshold_score, '#2166AC', linestyles[idx], linewidths[idx]))
                    break
                break
        
        for pv, score_ranges_pr in point_ranges.items():
            if pv == point_val:
                for sr in score_ranges_pr:
                    threshold_score = sr[1] if not flipped else sr[0]
                    threshold_scores_path.append(threshold_score)
                    threshold_info.append((threshold_score, '#B2182B', linestyles[idx], linewidths[idx]))
                    break
                break
    
    # ===== TOP ROW: Individual fits with components =====
    for sample_num in range(n_samples):
        ax = ax_fits[sample_num]
        sample_mask = scoreset.sample_assignments[:, sample_num]
        sample_name = scoreset.sample_names[sample_num]
        color = sample_colors[sample_num]
        alpha = sample_alphas[sample_num]
        
        hist_data = scoreset.scores[sample_mask]
        n_count = sample_mask.sum()
        
        # Plot histogram for this sample - full visibility for all
        sns.histplot(
            hist_data,
            binwidth=bin_width,
            stat='density',
            ax=ax,
            alpha=0.4,
            color=color,
        )
        
        density_sample = sample_density(score_range, fits, sample_num)  # shape: (n_bootstraps, n_c, n_scores)
        
        # Plot sum of components (total fit) in black - solid line
        d_total = np.nansum(density_sample, axis=1)  # shape: (n_bootstraps, n_scores)
        d_total_perc = np.percentile(d_total, [5, 50, 95], axis=0)  # shape: (3, n_scores)

        ax.fill_between(score_range, d_total_perc[0], d_total_perc[2], 
                       color='gray', 
                       alpha=0.3)
        ax.plot(score_range, d_total_perc[1], 
               color='black', 
               alpha=0.65,
               linewidth=2)
        
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel('')
        ax.set_ylabel('Density' if sample_num == 0 else '', fontsize=12)
        ax.tick_params(axis='both', labelsize=9)
        
        # Add threshold lines to fit panels
        for thresh_score, thresh_color, thresh_ls, thresh_lw in threshold_info:
            ax.axvline(
                thresh_score,
                color=thresh_color,
                linestyle=thresh_ls,
                linewidth=thresh_lw,
                alpha=0.8
            )
        
        # Create legend inside each panel with sample name and count
        hist_patch = Patch(facecolor=color, alpha=0.4, edgecolor='none')
        if sample_num == 2:  # gnomAD - add prior
            legend_label = f'{sample_name}\n(n={n_count:,d}, prior={indv_summary["prior"]:.3f})'
        else:
            legend_label = f'{sample_name}\n(n={n_count:,d})'
        
        ax.legend([hist_patch], [legend_label], loc='upper right' if sample_num != 0 else 'upper left', fontsize=9, framealpha=0.9)
    
    # ===== SECOND ROW: Combined histogram =====
    sample_handles = []
    
    for sample_num in range(n_samples):
        sample_mask = scoreset.sample_assignments[:, sample_num]
        sample_name = scoreset.sample_names[sample_num]
        color = sample_colors[sample_num]
        alpha = sample_alphas[sample_num]
        
        # For gnomAD (sample 2), use all SNV scores and rename to "All SNVs"
        if sample_num == 2:
            hist_data = all_scores
            display_name = 'All SNVs'
            n_count = len(all_scores)
        else:
            hist_data = scoreset.scores[sample_mask]
            display_name = sample_name
            n_count = sample_mask.sum()
        
        sns.histplot(
            hist_data,
            binwidth=bin_width,
            stat='density',
            ax=ax_hist,
            alpha=alpha,
            color=color,
        )
        
        hist_patch = Patch(facecolor=color, alpha=alpha, edgecolor='none')
        sample_handles.append((hist_patch, f'{display_name} (n={n_count:,d})'))
    
    ax_hist.set_xlim(x_min, x_max)
    ax_hist.set_xlabel('')
    ax_hist.set_ylabel('Density', fontsize=12)
    ax_hist.set_title(f'{dataset.split("_")[0]} Functional Score', fontsize=14, fontweight='bold')
    ax_hist.tick_params(axis='both', labelsize=10)
    
    # Sample legend (upper left)
    sample_legend_handles = [h[0] for h in sample_handles]
    sample_legend_labels = [h[1] for h in sample_handles]
    
    ax_hist.legend(
        sample_legend_handles,
        sample_legend_labels,
        loc='upper left',
        fontsize=10,
    )
    
    # ===== THIRD ROW: Scott et al. calibration =====
    ax_scott.axvspan(x_min, 0, color=strength_color['BP4_Strong'], alpha=0.9)
    ax_scott.axvspan(0, 0.4, color=strength_color['IR'], alpha=0.9)
    ax_scott.axvspan(0.4, x_max, color=strength_color['PP3_Strong'], alpha=0.9)
    
    count_below_0 = (all_scores < 0).sum()
    count_0_to_04 = ((all_scores >= 0) & (all_scores < 0.4)).sum()
    count_above_04 = (all_scores >= 0.4).sum()
    ax_scott.text((x_min + 0) / 2, 0.5, f'{count_below_0}', ha='center', va='center', color='black', fontsize=11)
    ax_scott.text((0 + 0.4) / 2, 0.5, f'{count_0_to_04}', ha='center', va='center', color='black', fontsize=11)
    ax_scott.text((0.4 + x_max) / 2, 0.5, f'{count_above_04}', ha='center', va='center', color='black', fontsize=11)
    
    ax_scott.set_xlim(x_min, x_max)
    ax_scott.set_ylim(0, 1)
    ax_scott.set_yticks([])
    ax_scott.set_ylabel('SNV Count', fontsize=12)
    ax_scott.set_xticks([])
    ax_scott.set_title('Scott et al. (2022)', loc='left', pad=3, fontsize=14, style='italic')
    
    # ===== FOURTH ROW: Zeiberg et al. calibration =====
    threshold_scores_benign_sorted = sorted(threshold_scores_benign)
    threshold_scores_path_sorted = sorted(threshold_scores_path)
    
    intervals = []
    if len(threshold_scores_benign_sorted) >= 2 and len(threshold_scores_path_sorted) >= 3:
        intervals.append(("BP4_Moderate", x_min, threshold_scores_benign_sorted[0]))
        intervals.append(("BP4_Supporting", threshold_scores_benign_sorted[0], threshold_scores_benign_sorted[1]))
        intervals.append(("IR", threshold_scores_benign_sorted[1], threshold_scores_path_sorted[0]))
        intervals.append(("PP3_Supporting", threshold_scores_path_sorted[0], threshold_scores_path_sorted[1]))
        intervals.append(("PP3_Moderate", threshold_scores_path_sorted[1], threshold_scores_path_sorted[2]))
        intervals.append(("+3", threshold_scores_path_sorted[2], x_max))
    
    for name, start, end in intervals:
        ax_zeiberg.axvspan(start, end, color=strength_color[name], alpha=0.9)
        count = ((all_scores >= start) & (all_scores < end)).sum()
        if (end - start) > 0.2:
            ax_zeiberg.text(
                (start + end) / 2, 0.5, str(count),
                ha='center', va='center',
                fontsize=11, color='black'
            )
    
    ax_zeiberg.set_xlim(x_min, x_max)
    ax_zeiberg.set_ylim(0, 1)
    ax_zeiberg.set_ylabel('SNV Count', fontsize=12)
    ax_zeiberg.set_yticks([])
    ax_zeiberg.set_xlabel('Assay Score', fontsize=12)
    ax_zeiberg.tick_params(axis='x', labelsize=10)
    ax_zeiberg.set_title('Zeiberg et al. (2025)', loc='left', pad=3, fontsize=14, style='italic')
    
    # ===== BOTTOM: Strength legend =====
    legend_order = ["BP4_Strong", "BP4_Moderate", "BP4_Supporting", "IR", "PP3_Supporting", "PP3_Moderate", "+3", "PP3_Strong"]
    
    used_strengths = {"BP4_Strong", "IR", "PP3_Strong"}
    for name, _, _ in intervals:
        used_strengths.add(name)
    
    all_legend = [
        Patch(facecolor=strength_color[name], label=name, edgecolor='none')
        for name in legend_order
        if name in used_strengths
    ]
    
    leg_ax.legend(handles=all_legend, loc='upper center', ncol=len(all_legend), frameon=False, fontsize=12)
    
    plt.tight_layout()
    
    return fig