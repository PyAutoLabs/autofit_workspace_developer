"""
Minimal NSS Example (JAX JIT)
------------------------------

This script shows how to use the NSS (Nested Slice Sampling) JAX-based
nested sampler with a pure JAX likelihood, enabling full jax.jit
compilation for GPU acceleration.

The Gaussian model and likelihood are written entirely in jax.numpy so
the JIT-compiled inner loop runs without Python callbacks.

Requirements:
    pip install git+https://github.com/yallup/nss.git
    (pulls handley-lab/blackjax fork with nested sampling support)
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
from nss.ns import run_nested_sampling
from blackjax.ns.utils import log_weights

# --------------------------------------------------------------------------
# Data (generated with NumPy, converted to JAX arrays)
# --------------------------------------------------------------------------

import numpy as np

np.random.seed(1)

xvalues = jnp.arange(100, dtype=jnp.float64)
true_centre = 50.0
true_normalization = 25.0
true_sigma = 10.0

true_data = jnp.multiply(
    jnp.divide(true_normalization, true_sigma * jnp.sqrt(2.0 * jnp.pi)),
    jnp.exp(-0.5 * jnp.square(jnp.divide(xvalues - true_centre, true_sigma))),
)
noise_map = jnp.full(true_data.shape, 0.01)
data = true_data + jnp.array(np.random.normal(0.0, 0.01, true_data.shape))

# --------------------------------------------------------------------------
# Pure JAX model and likelihood
# --------------------------------------------------------------------------

# Parameter order: [centre, normalization, sigma]
# Priors: centre ~ U(0, 100), normalization ~ U(0, 50), sigma ~ U(0, 50)

prior_lower = jnp.array([0.0, 0.0, 0.0])
prior_upper = jnp.array([100.0, 50.0, 50.0])


def log_prior(params):
    """Uniform box prior in log-density form."""
    in_bounds = jnp.all((params >= prior_lower) & (params <= prior_upper))
    log_vol = jnp.sum(jnp.log(prior_upper - prior_lower))
    return jnp.where(in_bounds, -log_vol, -jnp.inf)


def log_likelihood(params):
    """Chi-squared log-likelihood for a 1D Gaussian model."""
    centre, normalization, sigma = params[0], params[1], params[2]
    model_data = jnp.multiply(
        jnp.divide(normalization, sigma * jnp.sqrt(2.0 * jnp.pi)),
        jnp.exp(-0.5 * jnp.square(jnp.divide(xvalues - centre, sigma))),
    )
    residuals = data - model_data
    chi_squared = jnp.sum(jnp.square(residuals / noise_map))
    return -0.5 * chi_squared


# --------------------------------------------------------------------------
# Run NSS
# --------------------------------------------------------------------------

n_live = 500
n_dim = 3

# Draw initial samples from the uniform prior.
rng_key = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng_key)
initial_samples = jax.random.uniform(
    init_key, shape=(n_live, n_dim), minval=prior_lower, maxval=prior_upper
)

print("Running NSS (JAX JIT) nested sampling...")
print(f"  n_live={n_live}, n_dim={n_dim}")
print(f"  JIT compilation will happen on first step (may take a moment)\n")

t_start = time.time()
final_state, results = run_nested_sampling(
    rng_key,
    loglikelihood_fn=log_likelihood,
    prior_logprob=log_prior,
    num_mcmc_steps=5,
    initial_samples=initial_samples,
    num_delete=1,
    termination=-3,
)
t_elapsed = time.time() - t_start

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

# Extract weighted posterior samples.
logw = log_weights(rng_key, final_state)
positions = final_state.particles.position
log_likelihoods = final_state.particles.loglikelihood
n_evals = int(results.evals)
max_logl = float(jnp.max(log_likelihoods))

# Best fit = highest likelihood sample.
best_idx = jnp.argmax(log_likelihoods)
best_params = positions[best_idx]

# Weighted mean from posterior.
w = jnp.exp(logw.mean(axis=-1))
w = w / w.sum()
weighted_mean = jnp.sum(positions * w[:, None], axis=0)

summary = f"""\
--- NSS (JAX JIT) Results ---
Best fit:        centre={best_params[0]:.4f}  normalization={best_params[1]:.4f}  sigma={best_params[2]:.4f}
True:            centre=50.0000  normalization=25.0000  sigma=10.0000
Max log L:       {max_logl:.4f}
Log evidence:    {float(results.logZs.mean()):.4f}

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (includes JIT compilation / warmup)
Sampling time:       {float(results.time):.2f} s     (excludes JIT warmup)
Likelihood evals:    {n_evals}
Time per eval:       {float(results.time) / max(n_evals, 1) * 1e3:.3f} ms
ESS:                 {float(results.ess):.1f}
Posterior samples:   {len(positions)}
n_live / n_dim:      {n_live} / {n_dim}

--- Convergence ---
Converged:           yes (NSS termination=-3)
Evals to ML:         n/a (pure JAX path; intermediate evals not exposed without host callback)
Time to ML:          n/a
"""

print()
print(summary)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
summary_path = output_dir / f"{Path(__file__).stem}_summary.txt"
summary_path.write_text(summary)
print(f"Summary written to: {summary_path}")
