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
        kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=10) 

        X = np.array(X).reshape((-1, 1))
        kmeans.fit(X)
        cluster_assignments = kmeans.predict(X)

        component_parameters = []
        for i in range(n_clusters):
            X_cluster = X[cluster_assignments == i]
            loc, scale = sps.norm.fit(X_cluster)
            a = np.random.uniform(-0.25, 0.25)
            component_parameters.append((a, float(loc), float(scale)))
        # Sort by location
        component_parameters = sorted(component_parameters, key=lambda x: x[1])
        component_parameters = fix_to_satisfy_density_constraint(
            component_parameters, (X.min(), X.max())
        )
        if len(component_parameters) == 0 or any(len(p) == 0 for p in component_parameters):
            repeat += 1
        else:
            return component_parameters, kmeans
    raise ValueError("Failed to initialize")


def sn_method_of_moments_init(X):
    '''
    Skew normal method of moments estimator for a given component.

    Arguments:
    X: np.array (N,): observed instances

    Returns:
    (skew, loc, scale) or empty list upon failure
    '''
    
    # calculate moments
    m1 = np.mean(X)
    m2 = np.var(X)
    m3 = sps.skew(X)
    
    # Ensure minimum variance
    if m2 < 1e-10:
        return [], [], []
    
    # constants
    a1 = np.sqrt(2/np.pi)
    b1 = (4/np.pi - 1) / a1
    
    try:
        delta = np.sign(m3) / np.sqrt(a1**2 + m2 * (b1 / np.abs(m3))**(2/3))
        
        if np.isnan(delta) or np.abs(delta) >= 0.99:
            lambda_init = np.random.uniform(-0.25, 0.25)
            # Ensure minimum scale
            scale = max(np.sqrt(m2), 1e-6)
            return lambda_init, m1, scale
        
        # initial scale - check denominator first
        denom = 1 - a1**2 * delta**2
        if denom <= 1e-10:  # Too close to 0
            # Fall back to simple scale
            return m3, m1, max(np.sqrt(m2), 1e-6)
        
        sigma = np.sqrt(m2 / denom)
        
        # Ensure minimum scale
        sigma = max(sigma, 1e-6)
        
        # initial location
        mu = m1 - a1 * delta * sigma
        
        # initial lambda
        lambda_init = m3
        
        if np.any(np.isnan([mu, sigma, lambda_init])) or np.any(np.isinf([mu, sigma, lambda_init])):
            print('MoM nan param error:')
            return []
            
        return lambda_init, mu, sigma
        
    except (ZeroDivisionError, RuntimeWarning) as e:
        print('MoM zero divide error:',e)
        return []
    
def methodOfMomentsInit(X, n_components, max_attempts=1000, **kwargs):
    '''
    Initialize method of moments component samples and run `sn_method_of_moments_init` for each sample.
    
    Arguments:
    X: np.array (N,): observed instances
    n_components: int: number of components to give initial params for
    max_attempts: max attempts to determine stable cut points

    Returns:
    [(skew_1, loc_1, sigma_1), ..., (skew_k, loc_k, sigma_k)] or None upon failure
    '''
    
    # k-means++ style intialization
    for attempt in range(max_attempts):
        if np.random.rand() < 0.7:  # 70% of the time use smart init
            # Percentiles with random jitter
            base_percentiles = np.linspace(0, 100, n_components + 1)[1:-1]
            percentile_range = np.percentile(X, 75) - np.percentile(X, 25)  # IQR
            jitter = np.random.normal(0, percentile_range * 0.1, len(base_percentiles))
            cutPoints = np.percentile(X, np.sort(np.clip(base_percentiles + jitter, 1, 99)))
        else:  # 30% random exploration
            # Pure random for diversity
            cutPoints = np.sort(np.random.uniform(
                np.percentile(X, 5),  # Avoid extreme tails
                np.percentile(X, 95), 
                n_components - 1
            ))
        
        component_parameters = []
        success = True
        
        for i in range(n_components):
            if i == 0:
                X_component = X[X <= cutPoints[0]]
            elif i == n_components - 1:
                X_component = X[X > cutPoints[-1]]
            else:
                X_component = X[(X > cutPoints[i-1]) & (X <= cutPoints[i])]

            
            if len(X_component) < 3:  # Need at least 3 points for skewness
                success = False
                break
            
            params = sn_method_of_moments_init(X_component)
            
            if len(params) == 0:
                success = False
                break
                
            component_parameters.append(params)
        
        if success and all(len(params) > 0 for params in component_parameters):

            # enforce constraint and return
            # if kwargs.
            component_parameters = fix_to_satisfy_density_constraint(
                component_parameters, (X.min(), X.max())
            )

            if len(component_parameters) == 0 or any(len(p) == 0 for p in component_parameters):
                # print("constraint failed for MoM")
                continue
                
            return component_parameters

    print("MoM constraint failed")
    return None


# def fix_to_satisfy_density_constraint(component_parameters, xlims):
#     n_components = len(component_parameters)
#     rep_failed = False
#     for compI, compJ in zip(range(0, n_components - 1), range(1, n_components)):
#         if rep_failed:
#             break
#         for _ in range(300):
#             if not density_constraint_violated(
#                 component_parameters[compI], component_parameters[compJ], xlims
#             ):
#                 break
#             component_parameters[compI] = [
#                 component_parameters[compI][0]
#                 - 0.05 * abs(component_parameters[compI][0]),
#                 component_parameters[compI][1],
#                 component_parameters[compI][2],
#             ]
#             component_parameters[compJ] = [
#                 component_parameters[compJ][0]
#                 + 0.05 * abs(component_parameters[compJ][0]),
#                 component_parameters[compJ][1],
#                 component_parameters[compJ][2],
#             ]

#         if density_constraint_violated(
#             component_parameters[compI], component_parameters[compJ], xlims
#         ):
#             rep_failed = True
#             break
#     if rep_failed:
#         return [[] for _ in range(n_components)]
#     assert not density_constraint_violated(
#         component_parameters[0], component_parameters[1], xlims
#     )
#     return component_parameters

def fix_to_satisfy_density_constraint(component_parameters, xlims):
    n_components = len(component_parameters)

    if any(len(p) == 0 for p in component_parameters):
        return [[] for _ in range(n_components)]
    
    # Ensure components are ordered by location first  
    component_parameters = sorted(component_parameters, key=lambda x: x[1])

    # Validate all scales before starting
    for i in range(n_components):
        if len(component_parameters[i]) >= 3 and component_parameters[i][2] < 1e-6:
            # Set minimum scale
            component_parameters[i] = list(component_parameters[i])
            component_parameters[i][2] = 1e-6
    
    if n_components == 3:
        # Adjust middle component to satisfy both neighbors
        for _ in range(300):
            v01 = density_constraint_violated(component_parameters[0], component_parameters[1], xlims)
            v12 = density_constraint_violated(component_parameters[1], component_parameters[2], xlims)
            
            if not v01 and not v12:
                break
            
            if v01:
                # Fix 0-1
                component_parameters[0] = [
                    component_parameters[0][0] - 0.05 * abs(component_parameters[0][0]),
                    component_parameters[0][1],
                    component_parameters[0][2],
                ]
                component_parameters[1] = [
                    component_parameters[1][0] + 0.025 * abs(component_parameters[1][0]),  # Smaller adjustment
                    component_parameters[1][1],
                    component_parameters[1][2],
                ]
            
            if v12:
                # Fix 1-2
                component_parameters[1] = [
                    component_parameters[1][0] - 0.025 * abs(component_parameters[1][0]),  # Smaller adjustment
                    component_parameters[1][1],
                    component_parameters[1][2],
                ]
                component_parameters[2] = [
                    component_parameters[2][0] + 0.05 * abs(component_parameters[2][0]),
                    component_parameters[2][1],
                    component_parameters[2][2],
                ]
        
        # Check final
        if density_constraint_violated(component_parameters[0], component_parameters[1], xlims) or \
           density_constraint_violated(component_parameters[1], component_parameters[2], xlims):
            return [[] for _ in range(n_components)]
    else:
        rep_failed = False
        for compI, compJ in zip(range(0, n_components - 1), range(1, n_components)):
            if rep_failed:
                break
                
            # Try to fix this pair
            attempts = 0
            for attempts in range(300):
                if not density_constraint_violated(
                    component_parameters[compI], component_parameters[compJ], xlims
                ):
                    break
                component_parameters[compI] = [
                    component_parameters[compI][0] - 0.05 * abs(component_parameters[compI][0]),
                    component_parameters[compI][1],
                    component_parameters[compI][2],
                ]
                component_parameters[compJ] = [
                    component_parameters[compJ][0] + 0.05 * abs(component_parameters[compJ][0]),
                    component_parameters[compJ][1],
                    component_parameters[compJ][2],
                ]
            
            # Check if we failed after all attempts
            still_violated = density_constraint_violated(
                component_parameters[compI], component_parameters[compJ], xlims
            )
            
            # print(f"Pair ({compI}, {compJ}): attempts={attempts+1}, still_violated={still_violated}")
            
            if still_violated:
                rep_failed = True
                # print(f"Setting rep_failed=True for pair ({compI}, {compJ})")
                break
        
        # print(f"Final rep_failed={rep_failed}")
        
        if rep_failed:
            return [[] for _ in range(n_components)]
    
    # Check all adjacent pairs are satisfied
    for i in range(n_components - 1):
        violated = density_constraint_violated(
            component_parameters[i], component_parameters[i + 1], xlims
        )
        assert not violated, f"Components {i} and {i+1} violate constraint"
    
    return component_parameters

# def fix_to_satisfy_density_constraint(component_parameters, xlims):
#     n_components = len(component_parameters)
#     rep_failed = False
    
#     # Check all consecutive pairs AND all non-consecutive pairs
#     for compI in range(n_components - 1):
#         for compJ in range(compI + 1, n_components):
#             if rep_failed:
#                 break
#             for _ in range(300):
#                 if not density_constraint_violated(
#                     component_parameters[compI], component_parameters[compJ], xlims
#                 ):
#                     break
#                 component_parameters[compI] = [
#                     component_parameters[compI][0]
#                     - 0.05 * abs(component_parameters[compI][0]),
#                     component_parameters[compI][1],
#                     component_parameters[compI][2],
#                 ]
#                 component_parameters[compJ] = [
#                     component_parameters[compJ][0]
#                     + 0.05 * abs(component_parameters[compJ][0]),
#                     component_parameters[compJ][1],
#                     component_parameters[compJ][2],
#                 ]
#             if density_constraint_violated(
#                 component_parameters[compI], component_parameters[compJ], xlims
#             ):
#                 rep_failed = True
#                 break
    
#     if rep_failed:
#         return [[] for _ in range(n_components)]
    
#     # Check all pairs one more time to assert
#     for i in range(n_components - 1):
#         for j in range(i + 1, n_components):
#             assert not density_constraint_violated(
#                 component_parameters[i], component_parameters[j], xlims
#             )
    
#     return component_parameters
