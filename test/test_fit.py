import sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.assay_calibration.data_utils.dataset import (
    PillarProjectDataframe,
    Scoreset,
    BasicScoreset,
)
from src.assay_calibration.fit_utils.fit import Fit
import json


def test_fit():
    df = PillarProjectDataframe(Path(__file__).parent / "example_data.csv")
    ds = Scoreset(df.dataframe[df.dataframe.Dataset == "BRCA1_Adamovich_2022_HDR"])
    print(ds)
    fit = Fit(ds)
    fit.run(core_limit=1, num_fits=1, component_range=[2, 3], bootstrap=False)
    result = fit.to_dict()
    print(json.dumps(result, indent=4))


def test_basic_scoreset():
    abnormal = stats.norm(loc=-5, scale=3)
    normal = stats.norm(loc=0, scale=3)
    scores = np.concatenate([abnormal.rvs(100), normal.rvs(150), abnormal.rvs(50)])
    sample_assignments = np.zeros_like(scores)
    sample_assignments[:100] = 1
    sample_assignments[100:200] = 2
    sample_assignments[200:] = 1
    scoreset = BasicScoreset(scores, sample_assignments)
    fit = Fit(scoreset)
    fit.run(core_limit=1, num_fits=1, component_range=[2, 3])
    result = fit.to_dict()
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    test_fit()
    print("Test completed successfully.")
