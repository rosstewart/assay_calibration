"""
Statistical model selection for component counts
"""
import numpy as np
from scipy import stats
from typing import Dict, List

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
    
    Selection methods:
    - selected_k: Based on p-value test (standard)
    - conservative_k: Based on 5th percentile of differences (conservative)
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
    
    # Compute effect size and confidence interval
    mean_diff = differences.mean()
    median_diff = np.median(differences)
    std_diff = differences.std()
    
    # 95% confidence interval for mean difference (bootstrap percentile)
    ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
    fifth_percentile = np.percentile(differences, 5)
    
    # Select k based on test
    selected_k = k_high if p_value < alpha else k_low
    
    # Conservative selection: use k_high only if 5th percentile is positive
    # This means 95% of bootstrap samples show improvement
    conservative_k = k_high if fifth_percentile > 0 else k_low
    
    if verbose:
        print("\n" + "="*70)
        print("BOOTSTRAP PAIRED TEST")
        print("="*70)
        print(f"\nComparing k={k_high} vs k={k_low} on {len(differences)} paired bootstrap samples")
        print(f"\nValidation LL improvement (k={k_high} - k={k_low}):")
        print(f"  Mean ΔLL:   {mean_diff:+.6f}")
        print(f"  Median ΔLL: {median_diff:+.6f}")
        print(f"  Std ΔLL:    {std_diff:.6f}")
        print(f"  5th %ile:   {fifth_percentile:+.6f}")
        print(f"  95% CI:     [{ci_low:+.6f}, {ci_high:+.6f}]")
        print(f"\nWilcoxon signed-rank test:")
        print(f"  Test statistic: {statistic:.1f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significance level: {alpha}")
        print(f"\nDecision (p-value): ", end="")
        if p_value < alpha:
            print(f"k={k_high} is significantly better (p={p_value:.4f} < {alpha})")
        else:
            print(f"k={k_high} not significantly better (p={p_value:.4f} >= {alpha}), use k={k_low}")
        
        print(f"\nDecision (conservative): ", end="")
        if fifth_percentile > 0:
            print(f"k={k_high} (5th percentile > 0, improvement in 95% of bootstraps)")
        else:
            print(f"k={k_low} (5th percentile <= 0, insufficient evidence for k={k_high})")
    
    return {
        'selected_k': int(selected_k),
        'conservative_k': int(conservative_k),
        'p_value': float(p_value),
        'mean_diff': float(mean_diff),
        'median_diff': float(median_diff),
        'std_diff': float(std_diff),
        'fifth_percentile': float(fifth_percentile),
        'ci_95': (float(ci_low), float(ci_high)),
        'n_samples': len(differences),
        'method': 'wilcoxon_paired',
        'k_low': int(k_low),
        'k_high': int(k_high),
    }
