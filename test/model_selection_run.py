import json
from typing import Dict
import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from src.assay_calibration.fit_utils.fit import Fit
from src.assay_calibration.data_utils.dataset import Scoreset
from src.assay_calibration.fit_utils.model_selection import bootstrapped_likelihood_ratio_test
import argparse

def load_fits_file(fits_filepath: str|Path)->Dict:
    fits_filepath = Path(fits_filepath)
    if not fits_filepath.exists():
        raise ValueError(f"Fit filepath {fits_filepath} does not exists")
    with open(fits_filepath) as f:
        fits = json.load(f)
    return fits

def organize_fits(fits : Dict) -> Dict[str, Dict[int,dict]]:
    organized_fits = {}
    for scoreset_name, scoreset_fits in fits.items():
        if scoreset_name not in organize_fits:
            organized_fits[scoreset_name] = {}
        for fit_key, fit in scoreset_fits.items():
            if fit['bootstrap_seed'] not in organized_fits[scoreset_name]:
                organized_fits[scoreset_name][fit['bootstrap_seed']] = [fit,]
            else:
                organized_fits[scoreset_name][fit['bootstrap_seed']].append(fit)
        for bootstrap_seed,bootstrap_seed_fits in organized_fits[scoreset_name].items():
            organized_fits[scoreset_name][bootstrap_seed] = sorted(bootstrap_seed_fits,
                                                                   key=lambda fit: fit['fit_idx'])
    # { scoreset_name : 
    #       {bootstrap_seed : [bootstrap_seed_fit_0,
    #                          bootstrap_seed_fit_1,...,],
    #       }
    # }
    return organized_fits

def load_scoreset(dataframe_filepath: str|Path, scoreset_name: str,**kwargs) -> Scoreset:
    dataframe_filepath = Path(dataframe_filepath)
    if not dataframe_filepath.exists():
        raise ValueError(f"{dataframe_filepath} does not exist")
    df = pd.read_csv(dataframe_filepath)
    dataset_df = df[df.Dataset == scoreset_name]
    min_clinvar_stars = kwargs.get('min_clinvar_stars',1)
    clinvar_release = kwargs.get('clinvar_release','2025')
    scoreset = Scoreset(dataset_df,
                        min_clinvar_stars=min_clinvar_stars,
                        clinvar_release=clinvar_release)
    return scoreset

def model_selection_run(dataframe_filepath,scoreset_name, save_filepath,N_bootstraps=100,initial_bootstrap_seed=0):
    scoreset = load_scoreset(dataframe_filepath,scoreset_name)
    _ = bootstrapped_likelihood_ratio_test(scoreset, N_bootstraps, initial_bootstrap_seed,save_filepath=save_filepath)

if __name__ == "__main__":
    DEBUG = True
    if DEBUG:
        initial_bootstrap_seed = 0
        N_bootstraps = 10
        model_selection_run("/Users/dz/Documents/research/assay_calibration/final_pillar_data_with_clinvar_18_25_gnomad_wREVEL_wAM_wspliceAI_wMutpred2_wtrainvar_expanded_111225.csv.gz",
                            "BARD1_unpublished",
                            f"/Users/dz/Documents/research/assay_calibration/model_selection_results/BARD1_unpublished.model_selection.initial_bootstrap_seed_{initial_bootstrap_seed}.N_bootstraps_{N_bootstraps}.json")
    else:
        parser = argparse.ArgumentParser(description="Run model selection with bootstrapped likelihood ratio test.")
        parser.add_argument("dataframe_filepath", type=str, help="Path to the input dataframe file.")
        parser.add_argument("scoreset_name", type=str, help="Name of the scoreset to process.")
        parser.add_argument("save_filepath", type=str, help="Path to save the results.")
        parser.add_argument("--N_bootstraps", type=int, default=100, help="Number of bootstraps to perform (default: 100).")
        parser.add_argument("--initial_bootstrap_seed", type=int, default=0, help="Initial seed for bootstrapping (default: 0).")

        args = parser.parse_args()

        model_selection_run(
            dataframe_filepath=args.dataframe_filepath,
            scoreset_name=args.scoreset_name,
            save_filepath=args.save_filepath,
            N_bootstraps=args.N_bootstraps,
            initial_bootstrap_seed=args.initial_bootstrap_seed
        )
