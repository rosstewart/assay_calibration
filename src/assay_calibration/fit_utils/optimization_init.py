import numpy as np
import scipy.stats as stats
import scipy.optimize as opt
from typing import Tuple, List

# Compute log PDF of skew normal distribution
def log_skewnorm_pdf(x, skewness, loc, scale):
    return stats.skewnorm.logpdf(x, skewness, loc=loc, scale=scale)


# Compute KL-Divergence D(p || q) = E_p[ log(p/q) ]
def kl_divergence(params_p, params_q, x_samples):
    skew_p, loc_p, log_scale_p = params_p
    skew_q, loc_q, log_scale_q = params_q

    scale_p, scale_q = np.exp(log_scale_p), np.exp(log_scale_q)  # Convert log scale

    log_p = log_skewnorm_pdf(x_samples, skew_p, loc_p, scale_p)
    log_q = log_skewnorm_pdf(x_samples, skew_q, loc_q, scale_q)

    return np.sum(np.exp(log_p) * (log_p - log_q))


# Compute derivative of log-density ratio
def log_density_ratio_derivative(params_f, params_g, x_samples):
    skew_f, loc_f, log_scale_f = params_f
    skew_g, loc_g, log_scale_g = params_g

    scale_f, scale_g = np.exp(log_scale_f), np.exp(log_scale_g)

    log_f = log_skewnorm_pdf(x_samples, skew_f, loc_f, scale_f)
    log_g = log_skewnorm_pdf(x_samples, skew_g, loc_g, scale_g)
    log_ratio = log_f - log_g

    # Compute numerical derivative
    d_log_ratio_dx = np.gradient(log_ratio, x_samples)
    return d_log_ratio_dx


# Define the monotonicity constraint (log density ratio must be decreasing)
def monotonicity_constraint(params, x_samples):
    n_components = len(params) // 3
    param_sets = [params[i : i + 3] for i in range(0, len(params), 3)]

    constraint_value = []
    for i in range(n_components - 1):
        params_f = param_sets[i]
        params_g = param_sets[i + 1]
        d_log_ratio_dx = log_density_ratio_derivative(params_f, params_g, x_samples)
        constraint_value.append(-np.diff(d_log_ratio_dx) + 1e-4)
    return np.concatenate(constraint_value)  # Ensure non-increasing trend


# Objective function (Sum of KL divergences: D_kl(f0 || f1) + D_kl(g0 || g1)) + ... + D_kl(h0 || h1))
def objective(params, initialParams, x_samples):
    param_sets = [params[i : i + 3] for i in range(0, len(params), 3)]
    divergence = 0.0
    for i in range(len(initialParams)):
        divergence += kl_divergence(initialParams[i], param_sets[i], x_samples)
    return divergence


# Optimize distributions under monotonicity constraint (density ratio decreasing)
def optimize_distributions(
    initialParams: List[Tuple[float, float, float]], x_range=(-5, 5), num_samples=100
) -> List[Tuple[float, float, float]]:
    x_samples = np.linspace(x_range[0], x_range[1], num_samples)

    # Convert scale to log-scale for better numerical stability
    initialParams_log = [
        [params[0], params[1], np.log(params[2])] for params in initialParams
    ]

    initial_params = np.array(sum(initialParams_log, []))  # Concatenate initial params

    # Define constraints
    constraints = {
        "type": "ineq",
        "fun": lambda params: monotonicity_constraint(params, x_samples),
    }

    # Perform constrained optimization
    result = opt.minimize(
        objective,
        initial_params,
        args=(initialParams_log, x_samples),
        method="SLSQP",
        constraints=constraints,
    )

    if result.success:
        optimized_params = result.x
        optimized_params = list(
            tuple(
                [
                    optimized_params[i],
                    optimized_params[i + 1],
                    np.exp(optimized_params[i + 2]),
                ]
            )
            for i in range(0, len(optimized_params), 3)
        )
        return optimized_params
    else:
        raise RuntimeError("Optimization failed: " + result.message)


# Verify monotonicity
def check_monotonicity(f1, g1, x_samples):
    d_log_ratio_dx = log_density_ratio_derivative(f1, g1, x_samples)
    return np.all(np.diff(d_log_ratio_dx) <= 0)  # Check decreasing trend


if __name__ == "__main__":
    # Example Initial Distributions
    f0 = (2.0, 0.0, 1.0)  # (skewness, loc, scale)
    g0 = (-1.0, 1.0, 1.2)  # (skewness, loc, scale)
    h0 = (0.0, 1.2, 1.1)  # (skewness, loc, scale)

    # Optimize
    f1, g1, h1 = optimize_distributions([f0, g0, h0])

    # Check monotonicity
    x_samples = np.linspace(-5, 5, 100)
    result_is_monotonic = check_monotonicity(f1, g1, x_samples), check_monotonicity(
        g1, h1, x_samples
    )

    # Display results
    print(f"Initial f0: {f0}")
    print(f"Initial g0: {g0}")
    print(f"Initial h0: {h0}")
    initial_is_monotonic = check_monotonicity(f0, g0, x_samples), check_monotonicity(
        g0, h0, x_samples
    )
    print(f"Initial density ratio monotonically decreasing: {initial_is_monotonic}")
    print(f"Optimized f1: {f1}")
    print(f"Optimized g1: {g1}")
    print(f"Optimized h1: {h1}")
    print(f"Density ratio monotonically decreasing: {result_is_monotonic}")
