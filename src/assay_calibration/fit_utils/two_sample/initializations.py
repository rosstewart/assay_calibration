from .constraints import density_constraint_violated
import numpy as np
from sklearn.cluster import KMeans
import scipy.stats as sps


def kmeans_init(X, **kwargs):
    """
    Initialize the parameters of the skew normal mixture model using kmeans and the method of moments

    Arguments:
    X: np.array (N,): observed instances

    Optional Keyword Arguments:
    - n_clusters: int: number of clusters to use in kmeans. Default: 2
    - kmeans_init: str: initialization method for kmeans. Options: ["random", "k-means++"]. Default: "random"
    - skewnorm_init_method: str: method to use for fitting the skew normal distribution. Options: ["mle", "mm"]. Default: "mle"
    """
    repeat = 0
    while repeat < 1000:
        n_clusters = kwargs.get("n_clusters", 2)
        init = kwargs.get("kmeans_init", "random")
        kmeans = KMeans(n_clusters=n_clusters, init=init)

        X = np.array(X).reshape((-1, 1))
        kmeans.fit(X)
        cluster_assignments = kmeans.predict(X)

        component_parameters = []
        for i in range(n_clusters):
            X_cluster = X[cluster_assignments == i]
            loc, scale = sps.norm.fit(X_cluster)
            a = np.random.uniform(-0.25, 0.25)
            component_parameters.append((a, float(loc), float(scale)))
        component_parameters = fix_to_satisfy_density_constraint(
            component_parameters, (X.min(), X.max())
        )
        if not len(component_parameters[0]):
            repeat += 1
        else:
            return component_parameters, kmeans
    raise ValueError("Failed to initialize")


def methodOfMomentsInit(
    X,
    n_components,
):
    # return [(skew_1, loc_1, sigma_1), ..., (skew_k, loc_k, sigma_k)]
    raise NotImplementedError("Implement Method of Moments")
    cutPoints = np.random.uniform(X.min(), X.max(), n_components - 1)


def fix_to_satisfy_density_constraint(component_parameters, xlims):
    n_components = len(component_parameters)
    rep_failed = False
    for compI, compJ in zip(range(0, n_components - 1), range(1, n_components)):
        if rep_failed:
            break
        for _ in range(300):
            if not density_constraint_violated(
                component_parameters[compI], component_parameters[compJ], xlims
            ):
                break
            component_parameters[compI] = [
                component_parameters[compI][0]
                - 0.05 * abs(component_parameters[compI][0]),
                component_parameters[compI][1],
                component_parameters[compI][2],
            ]
            component_parameters[compJ] = [
                component_parameters[compJ][0]
                + 0.05 * abs(component_parameters[compJ][0]),
                component_parameters[compJ][1],
                component_parameters[compJ][2],
            ]

        if density_constraint_violated(
            component_parameters[compI], component_parameters[compJ], xlims
        ):
            rep_failed = True
            break
    if rep_failed:
        return [[] for _ in range(n_components)]
    assert not density_constraint_violated(
        component_parameters[0], component_parameters[1], xlims
    )
    return component_parameters
