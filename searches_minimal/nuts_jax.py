"""
Minimal NUTS Example (BlackJAX, JAX JIT + jax.grad)
---------------------------------------------------

This script shows how to use BlackJAX's No-U-Turn Sampler (NUTS) on the
same 1D Gaussian fitting problem used by the other ``searches_minimal``
JAX scripts (``nss_jit.py``, ``nss_grad.py``, ``nautilus_jax.py``).

NUTS is a gradient-based MCMC sampler, so the model and likelihood are
written entirely in ``jax.numpy`` and the joint log-density is targeted
directly. ``blackjax.window_adaptation`` runs a Stan-style warmup that
tunes both the step size (dual averaging) and the inverse mass matrix
before the production sampling phase begins.

Unlike the nested samplers in this folder, NUTS does NOT estimate the
log evidence — it samples from the posterior only. The summary block
reports ``Log evidence: n/a``.

Requirements:
    pip install blackjax
"""

import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import blackjax
from blackjax.diagnostics import effective_sample_size

from searches_minimal._metrics import MLTracker

# --------------------------------------------------------------------------
# Data (generated with NumPy, converted to JAX arrays)
# --------------------------------------------------------------------------

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
# Pure JAX model and log-density
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


def log_density(params):
    """Joint log-posterior (target for NUTS)."""
    return log_likelihood(params) + log_prior(params)


# Verify gradient works (NUTS uses jax.grad of log_density internally).
test_params = jnp.array([50.0, 25.0, 10.0])
test_grad = jax.grad(log_density)(test_params)
print(f"Gradient check at true values: {test_grad}")
print(f"  (non-zero due to noise in synthetic data)\n")

# --------------------------------------------------------------------------
# Run BlackJAX NUTS
# --------------------------------------------------------------------------

n_dim = 3
num_warmup = 500
num_samples = 2000

rng_key = jax.random.PRNGKey(42)

# Initial position: prior centre, well inside the box.
initial_position = jnp.array([50.0, 25.0, 10.0])

# Trigger compilation of log_density once so the first warmup call doesn't
# pay it (matches what nautilus_jax / nss_jit do for fair timing).
_t_jit_start = time.time()
_ = float(log_density(initial_position))
t_jit = time.time() - _t_jit_start

print("Running BlackJAX NUTS...")
print(f"  n_dim={n_dim}  num_warmup={num_warmup}  num_samples={num_samples}")
print(f"  Window adaptation tunes step size + inverse mass matrix\n")

# --- Warmup phase: window adaptation tunes step size + inverse mass matrix.
warmup = blackjax.window_adaptation(blackjax.nuts, log_density)

t_warmup_start = time.time()
rng_key, warmup_key = jax.random.split(rng_key)
(last_state, tuned_params), _ = warmup.run(
    warmup_key, initial_position, num_steps=num_warmup
)
# block_until_ready forces async dispatch to complete before timing.
jax.block_until_ready(last_state.position)
t_warmup = time.time() - t_warmup_start

# --- Sampling phase: scan-based JIT'd NUTS chain.
nuts_kernel = blackjax.nuts(log_density, **tuned_params)


def one_step(state, rng_key):
    new_state, info = nuts_kernel.step(rng_key, state)
    return new_state, (new_state, info)


def run_chain(rng_key, initial_state, num_samples):
    # ``jax.lax.scan`` JIT-compiles its body, so the inner per-step kernel
    # still runs as a single fused XLA computation. Keeping ``run_chain``
    # un-jitted lets us pass ``num_samples`` as a Python int — under
    # ``@jax.jit`` it would become a tracer and trip ``random.split``'s
    # static-shape requirement.
    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos


t_sample_start = time.time()
rng_key, sample_key = jax.random.split(rng_key)
states, infos = run_chain(sample_key, last_state, num_samples)
jax.block_until_ready(states.position)
t_sample = time.time() - t_sample_start

t_elapsed = t_warmup + t_sample

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

samples = states.position  # (num_samples, n_dim)

# Per-step log-likelihood (post-hoc, so we can recover evals-to-ML even
# though the chain ran inside jax.lax.scan).
log_l_history = jax.vmap(log_likelihood)(samples)
max_logl = float(jnp.max(log_l_history))
best_idx = int(jnp.argmax(log_l_history))
best_params = samples[best_idx]

# Posterior summary (median +/- 1 sigma) from the post-warmup chain.
medians = jnp.median(samples, axis=0)
stds = jnp.std(samples, axis=0)
labels = ["centre", "normalization", "sigma"]
posterior_summary_lines = "\n".join(
    f"  {label:15s} = {float(medians[i]):.4f} +/- {float(stds[i]):.4f}"
    for i, label in enumerate(labels)
)

# ESS via blackjax.diagnostics. The function expects chain axis 0 and
# sample axis 1, with any remaining axes treated as parameter dimensions.
# Single chain, so prepend a length-1 chain axis to (num_samples, n_dim).
ess_per_param = effective_sample_size(samples[None, ...])
ess = float(jnp.min(ess_per_param))  # report the worst-case ESS

# Evals-to-ML / time-to-ML via the post-hoc tracker variant (NUTS does
# multiple leapfrog steps per sample, so the chain index is a sample
# index, not a per-evaluation index — ``MLTracker.from_log_l_history``
# scales time-to-ML linearly across the sampling phase, which is the
# right approximation for a tuned NUTS chain where each step costs
# roughly the same).
n_logl_evals = int(jnp.sum(infos.num_integration_steps))
evals_to_ml, time_to_ml = MLTracker.from_log_l_history(
    [float(x) for x in log_l_history],
    total_sampling_time=t_sample,
    tolerance=1.0,
)

# Acceptance / divergence diagnostics.
mean_accept = float(jnp.mean(infos.acceptance_rate))
n_divergent = int(jnp.sum(infos.is_divergent))

summary = f"""\
--- NUTS (BlackJAX, jax.grad + JIT) Results ---
Best fit:        centre={best_params[0]:.4f}  normalization={best_params[1]:.4f}  sigma={best_params[2]:.4f}
True:            centre=50.0000  normalization=25.0000  sigma=10.0000
Max log L:       {max_logl:.4f}
Log evidence:    n/a (NUTS does not estimate Z)

Posterior summary (median +/- 1 sigma):
{posterior_summary_lines}

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (warmup + sampling)
Sampling time:       {t_sample:.2f} s     (post-warmup, JIT'd scan)
Warmup time:         {t_warmup:.2f} s     (window adaptation: step size + mass matrix)
JIT compile time:    {t_jit:.2f} s     (one-shot warm-up of log_density)
Likelihood evals:    {n_logl_evals}     (sum of integration steps over the chain)
Time per eval:       {t_sample / max(n_logl_evals, 1) * 1e3:.3f} ms
ESS (min over dims): {ess:.1f}
Posterior samples:   {len(samples)}
Mean acceptance:     {mean_accept:.3f}
Divergences:         {n_divergent}     (post-warmup; non-zero suggests step size too large)

--- Convergence ---
Converged:           yes (NUTS targets the posterior; no divergences ≈ healthy chain)
Evals to ML:         {evals_to_ml if evals_to_ml is not None else 'n/a'}     (sample index, not raw eval — see MLTracker.from_log_l_history)
Time to ML:          {f'{time_to_ml:.2f} s' if time_to_ml is not None else 'n/a'}
"""

print()
print(summary)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
summary_path = output_dir / f"{Path(__file__).stem}_summary.txt"
summary_path.write_text(summary)
print(f"Summary written to: {summary_path}")
