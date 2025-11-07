#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys, os, json, pickle, gzip, logging, warnings
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from joblib import Parallel, delayed

import matplotlib
matplotlib.use('Agg')
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', module='matplotlib')

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['path.simplify'] = True

sys.path.append("..")
from src.assay_calibration.fit_utils.fit import (calculate_score_ranges,thresholds_from_prior)  # noqa: E402
from src.assay_calibration.fit_utils.two_sample import density_utils  # noqa: E402
from src.assay_calibration.fit_utils.point_ranges import (enforce_monotonicity_point_ranges, prior_equation_2c, prior_invalid, get_fit_prior, get_bootstrap_score_ranges, remove_insufficient_bootstrap_converage_points, check_thresholds_reached)  # noqa: E402
from src.assay_calibration.data_utils.dataset import Scoreset  # noqa: E402
from src.assay_calibration.fit_utils.utils import serialize_dict  # noqa: E402
from src.assay_calibration.plot_utils.utils import plot_scoreset, plot_scoreset_compare_point_assignments, plot_summary, plot_scoreset_best_config

import contextlib

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# In[ ]:


constrained = True

if constrained:
    results_name = 'initial_datasets_results_1000bootstraps_100fits'
    with gzip.open(f'/data/ross/assay_calibration/{results_name}.json.gz', 'rt', encoding='utf-8') as f:
        results = json.load(f)
    
    results_name = 'clinvar_circ_datasets_results_1000bootstraps_100fits'
    with gzip.open(f'/data/ross/assay_calibration/{results_name}.json.gz', 'rt', encoding='utf-8') as f:
        results = {**results, **json.load(f)}

    results_name = 'clinvar_2018_datasets_results_1000bootstraps_100fits'
    with gzip.open(f'/data/ross/assay_calibration/{results_name}.json.gz', 'rt', encoding='utf-8') as f:
        results = {**results, **json.load(f)}


    results_name = 'semifinal_run_datasets_results_1000bootstraps_100fits'
    with gzip.open(f'/data/ross/assay_calibration/{results_name}.json.gz', 'rt', encoding='utf-8') as f:
        results_relax = json.load(f)

else:
    results_name = 'unconstrained_rerun_initial_datasets_results_1000bootstraps_100fits'
    with gzip.open(f'/data/ross/assay_calibration/{results_name}.json.gz', 'rt', encoding='utf-8') as f:
        results = json.load(f)

del results['KCNH2_Kozek_Glazer_2020'] # ignore this dataset 


# In[ ]:


dataset_configs = {
    "ASPA_Grønbæk-Thygesen_2024_abundance": ("3c", "avg"),
    "ASPA_Grønbæk-Thygesen_2024_toxicity": ("3c", "avg"),
    "BARD1_unpublished": ("2c", "avg"),
    "CALM1_CALM2_CALM3_Weile_2017": ("3c", "avg"),
    "CARD11_Meitlis_2020_DMSO_no_introns": ("3c", "avg"),
    "CARD11_Meitlis_2020_Ibrutinib_no_introns": ("2c", "avg"),
    "CBS_Sun_2020_high_B6": ("3c", "avg"),
    "CBS_Sun_2020_low_B6": ("2c", "avg"),
    "CHK2_Gebbia_2024": ("2c", "benign"),
    "CRX_Shepherdson_2024": ("2c", "avg"),
    "CTCF_unpublished": ("2c", "avg"),
    "F9_Popp_2025_carboxy_F9_specific": ("3c", "avg"),
    "F9_Popp_2025_carboxy_gla_motif": ("3c", "avg"),
    "F9_Popp_2025_heavy_chain": ("3c", "avg"),
    "F9_Popp_2025_light_chain": ("3c", "benign"),
    "F9_Popp_2025_strep_2": ("3c", "avg"),
    "FKRP_Ma_2024": ("3c", "avg"),
    "G6PD_unpublished": ("2c", "avg"),
    "GCK_Gersing_2023_complementation": ("2c", "avg"),
    "GCK_Gersing_2024_abundance": ("2c", "avg"),
    "HMBS_van_Loggerenberg_2023_combined": ("3c", "avg"),
    "HMBS_van_Loggerenberg_2023_erythroid": ("3c", "avg"),
    "HMBS_van_Loggerenberg_2023_ubquitous": ("2c", "avg"),
    "JAG1_Gilbert_2024": ("3c", "avg"),
    "KCNE1_Muhammad_2024_absence_of_WT": ("3c", "avg"),
    "KCNE1_Muhammad_2024_potassium_flux": ("2c", "avg"),
    "KCNE1_Muhammad_2024_presence_of_WT": ("2c", "avg"),
    "KCNH2_Jiang_2022": ("2c", "avg"),
    "KCNH2_O_Neill_2024_surface_expression": ("3c", "avg"),
    "KCNQ4_Zheng_2022_current_homozygous": ("3c", "avg"),
    "KCNQ4_Zheng_2022_v12_homozygous": ("2c", "avg"),
    "LARGE1_Ma_2024": ("3c", "avg"),
    "MSH2_Jia_2021": ("2c", "avg"),
    "NDUFAF6_Sung_2024": ("2c", "avg"),
    "OTC_Lo_2023": ("3c", "avg"),
    "PALB2_unpublished": ("2c", "avg"),
    "PAX6_McDonnell_2024_BLX_geneticin": ("3c", "avg"),
    "PAX6_McDonnell_2024_BLX_no_geneticin": ("3c", "avg"),
    "PAX6_McDonnell_2024_LE9_geneticin": ("2c", "avg"),
    "PAX6_McDonnell_2024_LE9_no_geneticin": ("2c", "avg"),
    "RAD51D_unpublished": ("3c", "avg"),
    "RHO_Wan_2019": ("2c", "avg"),
    "SCN5A_Glazer_2020": ("2c", "avg"),
    "SCN5A_Ma_2024_current_density": ("2c", "avg"),
    "TP53_Boettcher_2019": ("2c", "avg"),
    "TP53_Fortuno_2021_Kato_meta": ("2c", "avg"),
    "TP53_Giacomelli_2018_combined_score": ("2c", "avg"),
    "TP53_Giacomelli_2018_p53WT_Nutlin3": ("2c", "avg"),
    "TP53_Giacomelli_2018_p53null_Nutlin3": ("2c", "avg"),
    "TP53_Giacomelli_2018_p53null_etoposide": ("2c", "avg"),
    "TP53_Kato_2003_AIP1nWT": ("3c", "avg"),
    "TP53_Kato_2003_BAXnWT": ("2c", "avg"),
    "TP53_Kato_2003_GADD45nWT": ("2c", "avg"),
    "TP53_Kato_2003_MDM2nWT": ("3c", "avg"),
    "TP53_Kato_2003_NOXAnWT": ("3c", "avg"),
    "TP53_Kato_2003_P53R2nWT": ("2c", "avg"),
    "TP53_Kato_2003_WAF1nWT": ("3c", "avg"),
    "TP53_Kato_2003_h1433snWT": ("3c", "avg"),
    "TPK1_Weile_2017": ("2c", "avg"),
    "TSC2_rapgap_unpublished": ("3c", "avg"),
    "TSC2_tuberin_unpublished": ("3c", "avg"),
    "XRCC2_unpublished": ("2c", "avg"),
    "BAP1_Waters_2024": ("3c", "avg"),
    "BRCA1_Adamovich_2022_Cisplatin": ("2c", "avg"),
    "BRCA1_Adamovich_2022_HDR": ("2c", "avg"),
    "BRCA1_Findlay_2018": ("2c", "avg"),
    "BRCA2_Hu_2024": ("2c", "avg"),
    "BRCA2_Sahu_2023_exon13_Cisplatin": ("3c", "benign"),
    "BRCA2_Sahu_2023_exon13_Olaparib": ("2c", "avg"),
    "BRCA2_Sahu_2023_exon13_SGE": ("3c", "benign"),
    "BRCA2_Sahu_2023_exon13_global_score": ("2c", "avg"),
    "BRCA2_Sahu_2025_HDR": ("2c", "avg"),
    "BRCA2_unpublished": ("3c", "avg"),
    "DDX3X_Radford_2023_cLFC_day15": ("2c", "benign"),
    # "PTEN_Matreyek_2018": ("2c", "avg"), # replace with filtered nonsense
    "PTEN_Mighell_2018": ("2c", "avg"),
    "RAD51C_Olvera-León_2024_z_score_D4_D14": ("2c", "avg"),
    "VHL_Buckley_2024": ("2c", "benign"),
    "BAP1_Waters_2024_clinvar_2018": ("3c", "avg"),
    "BRCA1_Adamovich_2022_Cisplatin_clinvar_2018": ("2c", "avg"),
    "BRCA1_Adamovich_2022_HDR_clinvar_2018": ("2c", "avg"),
    "BRCA1_Findlay_2018_clinvar_2018": ("2c", "avg"),
    "BRCA2_Hu_2024_clinvar_2018": ("2c", "avg"),
    "BRCA2_Sahu_2023_exon13_Cisplatin_clinvar_2018": ("3c", "avg"),
    "BRCA2_Sahu_2023_exon13_Olaparib_clinvar_2018": ("2c", "avg"),
    "BRCA2_Sahu_2023_exon13_SGE_clinvar_2018": ("3c", "avg"),
    "BRCA2_Sahu_2023_exon13_global_score_clinvar_2018": ("3c", "avg"),
    "BRCA2_Sahu_2025_HDR_clinvar_2018": ("2c", "avg"),
    "BRCA2_unpublished_clinvar_2018": ("2c", "avg"),
    "DDX3X_Radford_2023_cLFC_day15_clinvar_2018": ("2c", "avg"),
    # "PTEN_Matreyek_2018_clinvar_2018": ("2c", "avg"), # replace with filtered nonsense
    "PTEN_Mighell_2018_clinvar_2018": ("2c", "avg"),
    "RAD51C_Olvera-León_2024_z_score_D4_D14_clinvar_2018": ("2c", "avg"),
    "VHL_Buckley_2024_clinvar_2018": ("2c", "benign")
}

dataset_relax_configs = {
    "BARD1_unpublished": ("2c", "avg"),
    "DDX3X_Radford_2023_cLFC_day15": ("2c", "avg"),
    "DDX3X_Radford_2023_cLFC_day15_clinvar_2018": ("2c", "avg"),
    "FKRP_Ma_2024": ("3c", "avg"),
    "G6PD_unpublished": ("3c", "avg"),
    "HMBS_van_Loggerenberg_2023_combined": ("2c", "avg"),
    "HMBS_van_Loggerenberg_2023_erythroid": ("3c", "avg"),
    "HMBS_van_Loggerenberg_2023_ubquitous": ("3c", "avg"),
    "KCNE1_Muhammad_2024_presence_of_WT": ("2c", "avg"),
    # "KCNH2_O_Neill_2024_surface_expression": ("3c", "avg"), # KEEP CONSTRAINED 3c avg # only compute lr+ when densities are high enough, extend
    # "LARGE1_Ma_2024": ("3c", "avg"), # KEEP CONSTRAINED
    "PTEN_Matreyek_2018_filtered_nonsense": ("2c", "avg"),
    "PTEN_Matreyek_2018_filtered_nonsense_clinvar_2018": ("2c", "avg"),
    "RAD51C_Olvera-León_2024_z_score_D4_D14": ("3c", "avg"), # ENFORCE 10e-3 on normalized densities 0-1 
    "RAD51C_Olvera-León_2024_z_score_D4_D14_clinvar_2018": ("3c", "avg"),
    "RAD51D_unpublished": ("2c", "avg"),
    "TSC2_Calhoun_cliPE_unpublished": ("2c", "avg"),
    "TSC2_Calhoun_immuneSGE_unpublished": ("2c", "avg"),
    "TSC2_tuberin_unpublished": ("3c", "avg"),
    "VHL_Buckley_2024": ("3c", "avg"),
    "VHL_Buckley_2024_clinvar_2018": ("3c", "avg"),
    "XRCC2_unpublished": ("2c", "avg")
}


# In[ ]:


plot_save_dir = '/data/ross/assay_calibration/calibrations_11_06_25'

def process_dataset(dataset, config, plot_save_dir, relax=False):

    if not relax:
        if dataset in dataset_relax_configs:
            return # will be overwritten by better fit
        
        mode = "point_assignment_comparison"
        if dataset.endswith('_clinvar_2018'):
            mode = "point_assignment_clinvar_2018"
    else:
        mode = "point_assignment_semifinal_rerun"
    

    use_median_prior, use_2c_equation = True, False
    n_c, benign_method = config

    points_save_dir = f'/data/ross/assay_calibration/{mode}/{dataset}'
    
    os.makedirs(plot_save_dir, exist_ok=True)
    plot_save_f = f'{plot_save_dir}/{dataset}.png'
    json_save_f = f'{plot_save_dir}/{dataset}.json'

    experiment_code = f'{dataset}_{n_c}_{"median" if use_median_prior else "5-percentile"}_{"equation" if use_2c_equation else "em"}{"_"+benign_method if benign_method != "benign" else ""}'
    
    pkl_filepath = f'{points_save_dir}/{experiment_code}.pkl'

    if not os.path.exists(pkl_filepath):
        pkl_filepath = pkl_filepath.replace("_avg","")
        benign_method = "benign"
        assert os.path.exists(pkl_filepath)

    
    log_f = pkl_filepath.replace('.pkl','.log')
    scoreset_flipped = False
    with open(log_f,'r') as f:
        for line in f:
            if line.strip() == "scoreset_flipped: True": 
                scoreset_flipped = True
    # if scoreset_flipped:
    #     print(dataset, 'scoreset_flipped:', scoreset_flipped)
    
    with open(pkl_filepath,'rb') as f:
        scoreset, indv_summary, fits, score_range, _, n_c = pickle.load(f)
    n_samples = len([s for s in scoreset.samples])

    with open(json_save_f,'w') as f:
        obj = {k: indv_summary[k] for k in ['prior','point_ranges']}
        obj['dataset'] = dataset
        obj['relax'] = relax
        obj['n_c'] = n_c
        obj['benign_method'] = benign_method
        obj['clinvar_2018'] = dataset.endswith('_clinvar_2018')
        obj['scoreset_flipped'] = scoreset_flipped
        json.dump(obj, f, indent=2)


    with suppress_output():
        scoreset_figure = plot_scoreset_best_config(dataset, scoreset, indv_summary, fits, score_range, f'({benign_method})', n_c, n_samples, relax=relax, flipped=scoreset_flipped)
        scoreset_figure.savefig(plot_save_f,bbox_inches='tight',dpi=300)
        # plt.show(scoreset_figure)
        plt.close(scoreset_figure)
        # print(dataset,'done')

n_cores = os.cpu_count() or 1

with suppress_output():
    Parallel(n_jobs=min(len(dataset_configs), n_cores), verbose=10)(delayed(process_dataset)(dataset, config, plot_save_dir, relax=False)
                                               for dataset, config in dataset_configs.items())
    
    Parallel(n_jobs=min(len(dataset_relax_configs), n_cores), verbose=10)(delayed(process_dataset)(dataset, config, plot_save_dir, relax=True)
                                               for dataset, config in dataset_relax_configs.items()])

# for dataset, config in dataset_configs.items():
#     process_dataset(dataset, config, plot_save_dir, relax=False)

# for dataset, config in dataset_relax_configs.items():
#     process_dataset(dataset, config, plot_save_dir, relax=True)


# ### prepare dataset config labels

# In[ ]:


# print('{\n    "',end='')
# print(*results.keys(),sep='": ("n_c", "benign_or_avg"),\n    "', end='')
# print(': ("n_c", "benign_or_avg")\n}')


# In[ ]:


# print("""{
#     "BAP1_Waters_2024_clinvar_2018": (“3c”, “avg”),
#     "BRCA1_Adamovich_2022_Cisplatin_clinvar_2018": (“2c”, "avg"),
#     "BRCA1_Adamovich_2022_HDR_clinvar_2018": (“2c”, "avg"),
#     "BRCA1_Findlay_2018_clinvar_2018": (“2c”, "avg"),
#     "BRCA2_Hu_2024_clinvar_2018": (“2c”, "avg"),
#     "BRCA2_Sahu_2023_exon13_Cisplatin_clinvar_2018": (“3c”, "avg"),
#     "BRCA2_Sahu_2023_exon13_Olaparib_clinvar_2018": (“2c”, “avg”),
#     "BRCA2_Sahu_2023_exon13_SGE_clinvar_2018": (“3c”, “avg”),
#     "BRCA2_Sahu_2023_exon13_global_score_clinvar_2018": (“3c”, “avg”),
#     "BRCA2_Sahu_2025_HDR_clinvar_2018": (“2c”, “avg”),
#     "BRCA2_unpublished_clinvar_2018": (“2c”, “avg”),
#     "DDX3X_Radford_2023_cLFC_day15_clinvar_2018": (“2c”, “avg”),
#     "PTEN_Matreyek_2018_clinvar_2018": (“2c”, “avg”),
#     "PTEN_Mighell_2018_clinvar_2018": (“2c”, “avg”),
#     "RAD51C_Olvera-León_2024_z_score_D4_D14_clinvar_2018": (“2c”, “avg”),
#     "VHL_Buckley_2024_clinvar_2018: (“2c”, “benign”)
# }""".replace('“','"').replace('”','"'))

