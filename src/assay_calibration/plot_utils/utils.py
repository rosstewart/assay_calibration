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
# def plot_scoreset_best_config(dataset, scoreset, indv_summary, fits, score_range, config, n_c, n_samples, relax=False):
    # """
    # Plot a single configuration with samples in one row, point assignments below, and LR+ below that.
    
    # Parameters:
    # -----------
    # dataset : str
    #     Dataset name
    # scoreset : Scoreset
    #     The scoreset object
    # indv_summary : dict
    #     Summary dict for this specific (config, n_c) configuration
    # fits : list
    #     Fitted models
    # score_range : np.ndarray
    #     Score range for plotting
    # config : tuple
    #     Configuration tuple (e.g., ('avg',) or ('benign',))
    # n_c : str
    #     '2c' or '3c'
    # n_samples : int
    #     Number of samples in the scoreset
    # """
    
    # # Create figure: 3 rows, n_samples columns (all square)
    # fig, ax = plt.subplots(3, n_samples, figsize=(7*n_samples, 18), 
    #                        squeeze=False, gridspec_kw={'hspace': 0.3, 'wspace': 0.3})

    # relax_code = "R" if relax else ""
    
    # # ===== Row 0: Sample fits =====
    # for sample_num in range(n_samples):
    #     ax_fit = ax[0, sample_num]
        
    #     sns.histplot(scoreset.scores[scoreset.sample_assignments[:,sample_num]], 
    #                  stat='density', ax=ax_fit, alpha=.5, color='pink')

    #     max_hist_density = max([patch.get_height() for patch in ax_fit.patches])
        
    #     density = sample_density(score_range, fits, sample_num)
    #     for compNum in range(density.shape[1]):
    #         compDensity = density[:,compNum,:]
    #         d = np.nanpercentile(compDensity,[5,50,95],axis=0)
    #         ax_fit.plot(score_range, d[1], color=f"C{compNum}", linestyle='--', label=f"Comp {compNum+1}")
    #     ax_fit.legend(fontsize=8)
        
    #     d = np.nansum(density, axis=1)
    #     d_perc = np.percentile(d, [5,50,95], axis=0)
    #     ax_fit.plot(score_range, d_perc[1], color='black', alpha=.5)
    #     ax_fit.fill_between(score_range, d_perc[0], d_perc[2], color='gray', alpha=0.3)
    #     ax_fit.set_title(f"{n_c}{relax_code}: {scoreset.sample_names[sample_num]}\n(n={scoreset.sample_assignments[:,sample_num].sum():,d})")
    #     ax_fit.set_xlabel("Score")
    #     ax_fit.set_ylabel("Density")
    #     ax_fit.set_ylim([0, max_hist_density * 1.1])
    #     ax_fit.grid(linewidth=0.5, alpha=0.3)
    
    # # Get x-limits from first fit
    # xlim = ax[0, 0].get_xlim()
    
    # # ===== Row 1: Point assignments (one per sample) =====
    # point_ranges = sorted([(int(k), v) for k,v in indv_summary['point_ranges'].items()])
    # point_values = [pr[0] for pr in point_ranges]
    
    # for sample_num in range(n_samples):
    #     ax_points = ax[1, sample_num]
        
    #     # Plot only this sample's point assignments
    #     for pointIdx, (pointVal, scoreRanges) in enumerate(point_ranges):
    #         for sr in scoreRanges:
    #             ax_points.plot([sr[0], sr[1]], [pointIdx, pointIdx], 
    #                          color='red' if pointVal > 0 else 'blue', 
    #                          linestyle='-', alpha=0.7, linewidth=2)
        
    #     ax_points.set_ylim(-1, len(point_values))
    #     ax_points.set_yticks(range(len(point_values)), 
    #                        labels=list(map(lambda i: f"{i:+d}" if i!=0 else "0", point_values)))
    #     ax_points.set_xlabel("Score")
    #     ax_points.set_ylabel("Points")
    #     ax_points.set_title(f"Point Assignments", fontsize=11)
    #     ax_points.set_xlim(xlim)
    #     ax_points.grid(linewidth=0.5, alpha=0.3)
    
    # # ===== Row 2: LR+ summaries (one per sample) =====
    # point_values_all = sorted(list(set([abs(int(k)) for k in indv_summary['point_ranges'].keys()])))
    # tauP, tauB, _ = list(map(np.log, thresholds_from_prior(indv_summary['prior'], point_values_all + [10])))
    # priors = np.percentile(np.array(indv_summary['priors']),[5,50,95])
    
    # for sample_num in range(n_samples):
    #     ax_lr = ax[2, sample_num]
        
    #     log_lr_plus = indv_summary['log_lr_plus']
    #     llr_curves = np.nanpercentile(np.array(log_lr_plus),[5,50,95],axis=0)
    #     labels = ['5th percentile','Median','95th percentile']
        
    #     for i, c in enumerate(['red','black','blue']):
    #         if len(log_lr_plus) == 1 and i != 1:
    #             continue # if no bootstraps only plot one curve
    #         ax_lr.plot(score_range, llr_curves[i], color=c, label=labels[i] if len(log_lr_plus) != 1 else 'Single fit')
        
    #     ax_lr.set_title(f"Log LR+\nprior: {priors[1]:.3f}, C: {indv_summary['C']}", fontsize=11)
    #     add_thresholds(tauP[:-1], tauB[:-1], ax_lr)
    #     ax_lr.set_xlabel("Score")
    #     ax_lr.set_ylabel("Log LR+")
    #     ax_lr.legend(fontsize=8, loc='best')
    #     ax_lr.set_xlim(xlim)
    #     ax_lr.set_ylim([tauB[-1], tauP[-1]])  # Set y-limits based on ±10 thresholds
    #     ax_lr.grid(linewidth=0.5, alpha=0.3)
    
    # plt.tight_layout()
    # fig.suptitle(f"{dataset} - {n_c}{relax_code} {config}", fontsize=16, y=0.998)
    
    # return fig

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
        
        sns.histplot(scoreset.scores[scoreset.sample_assignments[:,sample_num]], 
                     stat='density', ax=ax_fit, alpha=.5, color='pink')

        max_hist_density = max([patch.get_height() for patch in ax_fit.patches])
        
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
        ax_fit.set_title(f"{n_c}{relax_code}: {scoreset.sample_names[sample_num]}\n(n={scoreset.sample_assignments[:,sample_num].sum():,d})")
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
            
            # Handle 5th percentile (i=0) - exceeding pathogenic thresholds
            if i == 0 and should_plot_pathogenic_dotted and pathogenic_crossing_idx is not None:
                if debug:
                    print(f"  Plotting 5th percentile with dotted line from crossing at idx {pathogenic_crossing_idx}")
                if not flipped:
                    # Normal: dotted before first crossing, solid after
                    ax_lr.plot(score_range[:pathogenic_crossing_idx+1], curve[:pathogenic_crossing_idx+1], 
                             color=c, linestyle=':', alpha=0.8, linewidth=2)
                    ax_lr.plot(score_range[pathogenic_crossing_idx:], curve[pathogenic_crossing_idx:], 
                             color=c, label=labels[i], linewidth=2)
                else:
                    # Flipped: solid before second crossing, dotted after
                    ax_lr.plot(score_range[:pathogenic_crossing_idx+1], curve[:pathogenic_crossing_idx+1], 
                             color=c, label=labels[i], linewidth=2)
                    ax_lr.plot(score_range[pathogenic_crossing_idx:], curve[pathogenic_crossing_idx:], 
                             color=c, linestyle=':', alpha=0.8, linewidth=2)
            
            # Handle 95th percentile (i=2) - going below benign thresholds
            elif i == 2 and should_plot_benign_dotted and benign_crossing_idx is not None:
                if debug:
                    print(f"  Plotting 95th percentile with dotted line from crossing at idx {benign_crossing_idx}")
                if not flipped:
                    # Normal: solid before second crossing, dotted after
                    ax_lr.plot(score_range[:benign_crossing_idx+1], curve[:benign_crossing_idx+1], 
                             color=c, label=labels[i], linewidth=2)
                    ax_lr.plot(score_range[benign_crossing_idx:], curve[benign_crossing_idx:], 
                             color=c, linestyle=':', alpha=0.8, linewidth=2)
                else:
                    # Flipped: dotted before first crossing, solid after
                    ax_lr.plot(score_range[:benign_crossing_idx+1], curve[:benign_crossing_idx+1], 
                             color=c, linestyle=':', alpha=0.8, linewidth=2)
                    ax_lr.plot(score_range[benign_crossing_idx:], curve[benign_crossing_idx:], 
                             color=c, label=labels[i], linewidth=2)
            
            # Plot normally for median (i=1) or when no threshold exceeded
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