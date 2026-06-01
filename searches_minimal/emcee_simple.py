"""
Minimal Emcee Example
---------------------

This script shows how to use the Emcee MCMC sampler directly with
autofit's Model and Analysis objects, bypassing the full NonLinearSearch
wrapper class.

This is useful for quickly testing a search on a problem before investing
in a full autofit integration.

Requirements:
    pip install emcee
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
# Direct Emcee interface
# --------------------------------------------------------------------------

import emcee


n_likelihood_calls = 0


def log_posterior(params):
    """Evaluate log posterior = log likelihood + log prior."""
    global n_likelihood_calls
    log_prior = sum(model.log_prior_list_from_vector(vector=params))

    if not np.isfinite(log_prior):
        return -np.inf

    n_likelihood_calls += 1
    instance = model.instance_from_vector(vector=params)
    log_like = analysis.log_likelihood_function(instance)

    return log_like + log_prior


nwalkers = 30
nsteps = 2000
ndim = model.prior_count

# Initialize walkers near the true values with small scatter.
initial_positions = np.array([50.0, 25.0, 10.0])
walkers = initial_positions + 1e-2 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(
    nwalkers=nwalkers,
    ndim=ndim,
    log_prob_fn=log_posterior,
)

t_start = time.time()
sampler.run_mcmc(walkers, nsteps=nsteps, progress=True)
t_elapsed = time.time() - t_start

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

# Discard burn-in and flatten the chain.
flat_samples = sampler.get_chain(discard=500, thin=10, flat=True)
flat_log_prob = sampler.get_log_prob(discard=500, thin=10, flat=True)

best_idx = np.argmax(flat_log_prob)
best_params = flat_samples[best_idx]
best_instance = model.instance_from_vector(vector=best_params)

medians = np.median(flat_samples, axis=0)
stds = np.std(flat_samples, axis=0)
labels = ["centre", "normalization", "sigma"]

print("\n--- Emcee Results ---")
print(
    f"Best fit:  centre={best_instance.centre:.4f}  normalization={best_instance.normalization:.4f}  sigma={best_instance.sigma:.4f}"
)
print(f"True:      centre=50.0000  normalization=25.0000  sigma=10.0000")

print("\nPosterior summary (median +/- 1 sigma):")
for label, med, std in zip(labels, medians, stds):
    print(f"  {label:15s} = {med:.4f} +/- {std:.4f}")

total_samples = nwalkers * nsteps

print(f"\n--- Performance ---")
print(f"Wall time:          {t_elapsed:.2f} s")
print(f"Likelihood calls:   {n_likelihood_calls}")
print(f"Total samples:      {total_samples}  ({nwalkers} walkers x {nsteps} steps)")
print(f"Time per call:      {t_elapsed / n_likelihood_calls * 1e3:.3f} ms")
print(f"Effective samples:  {len(flat_samples)}  (after burn-in & thinning)")
