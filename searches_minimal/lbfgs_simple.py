"""
Minimal L-BFGS Example
----------------------

This script shows how to use scipy's L-BFGS-B optimizer directly with
autofit's Model and Analysis objects, bypassing the full NonLinearSearch
wrapper class.

This is useful for quickly testing a search on a problem before investing
in a full autofit integration.

Requirements:
    scipy (included with autofit)
"""

import time

import numpy as np
import autofit as af

# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class Gaussian:
    def __init__(self, centre=30.0, normalization=1.0, sigma=5.0):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    def model_data_from(self, xvalues, xp=np):
        return xp.multiply(
            xp.divide(self.normalization, self.sigma * xp.sqrt(2.0 * xp.pi)),
            xp.exp(-0.5 * xp.square(xp.divide(xvalues - self.centre, self.sigma))),
        )


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

xvalues = np.arange(100)
true_gaussian = Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
data = true_gaussian.model_data_from(xvalues=xvalues)
noise_map = np.full(data.shape, 0.01)
data += np.random.normal(0.0, 0.01, data.shape)

# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


class Analysis(af.Analysis):
    def __init__(self, data, noise_map):
        super().__init__()
        self.data = data
        self.noise_map = noise_map

    def log_likelihood_function(self, instance):
        model_data = instance.model_data_from(xvalues=xvalues)
        residual_map = self.data - model_data
        chi_squared_map = (residual_map / self.noise_map) ** 2.0
        log_likelihood = -0.5 * sum(chi_squared_map)
        return log_likelihood


analysis = Analysis(data=data, noise_map=noise_map)

# --------------------------------------------------------------------------
# autofit Model (provides priors and parameter mapping)
# --------------------------------------------------------------------------

model = af.Model(Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

# --------------------------------------------------------------------------
# Direct scipy L-BFGS-B interface
# --------------------------------------------------------------------------

from scipy import optimize


def chi_squared(params):
    """Return chi-squared (negative 2x log likelihood) for minimisation."""
    instance = model.instance_from_vector(vector=params)
    log_like = analysis.log_likelihood_function(instance)
    return -2.0 * log_like


# Extract bounds from the uniform priors.
bounds = [(0.0, 100.0), (0.0, 50.0), (0.0, 50.0)]

# Start near the true values.
x0 = np.array([45.0, 20.0, 8.0])

t_start = time.time()
result = optimize.minimize(
    fun=chi_squared,
    x0=x0,
    method="L-BFGS-B",
    bounds=bounds,
)
t_elapsed = time.time() - t_start

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

best_instance = model.instance_from_vector(vector=result.x)

print("\n--- L-BFGS-B Results ---")
print(
    f"Best fit:  centre={best_instance.centre:.4f}  normalization={best_instance.normalization:.4f}  sigma={best_instance.sigma:.4f}"
)
print(f"True:      centre=50.0000  normalization=25.0000  sigma=10.0000")
print(f"Chi-squared:   {result.fun:.2f}")
print(f"Converged:     {result.success}")
print(f"\n--- Performance ---")
print(f"Wall time:          {t_elapsed:.4f} s")
print(f"Function evals:     {result.nfev}")
print(f"Gradient evals:     {result.njev}")
print(f"Iterations:         {result.nit}")
print(f"Time per eval:      {t_elapsed / result.nfev * 1e3:.3f} ms")
