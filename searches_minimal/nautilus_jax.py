"""
Minimal Nautilus Example (JAX JIT likelihood)
---------------------------------------------

This script shows how to use the Nautilus nested sampler with a pure-JAX,
``jax.jit``-compiled log likelihood. Nautilus itself is a NumPy-based
sampler, so the JAX likelihood is wrapped in a thin Python adapter that
converts the parameter vector to a JAX array, calls the JIT-compiled
likelihood, and returns a Python float.

This is the Nautilus analogue of ``nss_jit.py``: same model, same priors,
same noise realisation, same ``n_live`` as ``nautilus_simple.py``, but
with the inner likelihood evaluation moved onto JAX so the comparison
versus NSS is on a level footing.

Requirements:
    pip install nautilus-sampler
"""

import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from nautilus import Sampler

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
# Pure JAX log likelihood (JIT-compiled)
# --------------------------------------------------------------------------

prior_lower = np.array([0.0, 0.0, 0.0])
prior_upper = np.array([100.0, 50.0, 50.0])


@jax.jit
def jit_log_likelihood(params):
    """Chi-squared log-likelihood for a 1D Gaussian model (pure JAX)."""
    centre, normalization, sigma = params[0], params[1], params[2]
    model_data = jnp.multiply(
        jnp.divide(normalization, sigma * jnp.sqrt(2.0 * jnp.pi)),
        jnp.exp(-0.5 * jnp.square(jnp.divide(xvalues - centre, sigma))),
    )
    residuals = data - model_data
    chi_squared = jnp.sum(jnp.square(residuals / noise_map))
    return -0.5 * chi_squared


# Trigger compilation up front so the first sampler call doesn't pay it.
_t_jit_start = time.time()
_ = float(jit_log_likelihood(jnp.asarray([50.0, 25.0, 10.0])))
t_jit = time.time() - _t_jit_start

# --------------------------------------------------------------------------
# Nautilus interface
# --------------------------------------------------------------------------


def prior_transform(cube):
    """Map unit cube to physical parameters via uniform priors."""
    return prior_lower + cube * (prior_upper - prior_lower)


n_likelihood_calls = 0
tracker = MLTracker()


def log_likelihood(params):
    """Adapter: NumPy in, JIT'd JAX likelihood, Python float out."""
    global n_likelihood_calls
    n_likelihood_calls += 1
    log_l = float(jit_log_likelihood(jnp.asarray(params)))
    tracker.record(log_l)
    return log_l


n_live = 200
n_dim = 3

sampler = Sampler(
    prior=prior_transform,
    likelihood=log_likelihood,
    n_dim=n_dim,
    n_live=n_live,
)

t_start = time.time()
sampler.run(verbose=True)
t_elapsed = time.time() - t_start

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

points, log_w, log_l = sampler.posterior()

best_idx = np.argmax(log_l)
best_params = points[best_idx]
max_logl = float(np.max(log_l))
evals_to_ml, time_to_ml = tracker.finalise(max_log_l=max_logl, tolerance=1.0)

summary = f"""\
--- Nautilus (JAX JIT) Results ---
Best fit:        centre={best_params[0]:.4f}  normalization={best_params[1]:.4f}  sigma={best_params[2]:.4f}
True:            centre=50.0000  normalization=25.0000  sigma=10.0000
Max log L:       {max_logl:.4f}
Log evidence:    {float(sampler.log_z):.4f}

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (excludes JIT compilation, run ahead of time)
Sampling time:       {t_elapsed:.2f} s     (Nautilus does not split warmup)
JIT compile time:    {t_jit:.2f} s     (one-shot warm-up before sampling)
Likelihood evals:    {n_likelihood_calls}
Time per eval:       {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms
ESS:                 {float(sampler.n_eff):.1f}
Posterior samples:   {len(points)}
n_live / n_dim:      {n_live} / {n_dim}

--- Convergence ---
Converged:           yes (Nautilus default n_eff / f_live)
Evals to ML:         {evals_to_ml if evals_to_ml is not None else 'n/a'}     (first eval within 1 nat of max log L)
Time to ML:          {f'{time_to_ml:.2f} s' if time_to_ml is not None else 'n/a'}
"""

print()
print(summary)

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
summary_path = output_dir / f"{Path(__file__).stem}_summary.txt"
summary_path.write_text(summary)
print(f"Summary written to: {summary_path}")
