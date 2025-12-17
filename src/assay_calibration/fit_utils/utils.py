from typing import Dict
import numpy as np
from sklearn.cluster import KMeans
import pickle


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
    elif isinstance(d, (np.floating, float)):  # Handles np.float64, np.float32, and Python floats
        return float(d)
    elif isinstance(d, (np.integer, int)):  # Handles np.int64, np.int32, and Python ints
        return int(d)
    elif isinstance(d, (np.bool_, bool)):  # Handles np.bool_ and Python bools
        return bool(d)
    elif isinstance(d,KMeans):
        return {"cluster_centers": d.cluster_centers_.tolist()}
    else:
        return d
