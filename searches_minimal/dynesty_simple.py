"""
Minimal Dynesty Example
-----------------------

This script shows how to use the Dynesty nested sampler directly with
autofit's Model and Analysis objects, bypassing the full NonLinearSearch
wrapper class.

This is useful for quickly testing a search on a problem before investing
in a full autofit integration.

Requirements:
    pip install dynesty
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
# Direct Dynesty interface
# --------------------------------------------------------------------------

from dynesty import NestedSampler


def prior_transform(cube):
    """Map unit cube to physical parameters via the model's priors."""
    return np.array(model.vector_from_unit_vector(cube))


n_likelihood_calls = 0


def log_likelihood(params):
    """Evaluate log likelihood for a physical parameter vector."""
    global n_likelihood_calls
    n_likelihood_calls += 1
    instance = model.instance_from_vector(vector=params)
    return analysis.log_likelihood_function(instance)


sampler = NestedSampler(
    loglikelihood=log_likelihood,
    prior_transform=prior_transform,
    ndim=model.prior_count,
    nlive=200,
)

t_start = time.time()
sampler.run_nested(print_progress=True)
t_elapsed = time.time() - t_start

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

results = sampler.results

best_idx = np.argmax(results.logl)
best_params = results.samples[best_idx]
best_instance = model.instance_from_vector(vector=best_params)

print("\n--- Dynesty Results ---")
print(
    f"Best fit:  centre={best_instance.centre:.4f}  normalization={best_instance.normalization:.4f}  sigma={best_instance.sigma:.4f}"
)
print(f"True:      centre=50.0000  normalization=25.0000  sigma=10.0000")
print(f"Log evidence:  {results.logz[-1]:.2f}")
print(f"\n--- Performance ---")
print(f"Wall time:          {t_elapsed:.2f} s")
print(f"Likelihood calls:   {n_likelihood_calls}")
print(f"Time per call:      {t_elapsed / n_likelihood_calls * 1e3:.3f} ms")
print(f"Total samples:      {results.niter}")
print(f"Effective samples:  {len(results.samples)}")
