"""
Minimal NSS Example (autofit interface)
----------------------------------------

This script shows how to use the NSS (Nested Slice Sampling) JAX-based
nested sampler directly with autofit's Model and Analysis objects,
bypassing the full NonLinearSearch wrapper class.

The NumPy-based likelihood from autofit is wrapped with jax.pure_callback
so that it can be called inside NSS's JIT-compiled sampling loop.

Requirements:
    pip install git+https://github.com/yallup/nss.git
    (pulls handley-lab/blackjax fork with nested sampling support)
"""

import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import autofit as af
from nss.ns import run_nested_sampling
from blackjax.ns.utils import log_weights

from searches_minimal._metrics import MLTracker

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

np.random.seed(1)

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
# Direct NSS interface
# --------------------------------------------------------------------------

n_likelihood_calls = 0
tracker = MLTracker()


def numpy_log_likelihood(params_np):
    """Evaluate log likelihood via autofit (NumPy)."""
    global n_likelihood_calls
    n_likelihood_calls += 1
    instance = model.instance_from_vector(vector=params_np.tolist())
    log_l = float(analysis.log_likelihood_function(instance))
    tracker.record(log_l)
    return np.float64(log_l)


def numpy_log_prior(params_np):
    """Evaluate log prior via autofit (NumPy)."""
    log_priors = model.log_prior_list_from_vector(vector=params_np.tolist())
    return np.float64(sum(log_priors))


def log_likelihood(params):
    """JAX wrapper around the NumPy likelihood via pure_callback."""
    return jax.pure_callback(
        lambda p: jnp.float64(numpy_log_likelihood(np.asarray(p))),
        jax.ShapeDtypeStruct((), jnp.float64),
        params,
        vmap_method="sequential",
    )


def log_prior(params):
    """JAX wrapper around the NumPy prior via pure_callback."""
    return jax.pure_callback(
        lambda p: jnp.float64(numpy_log_prior(np.asarray(p))),
        jax.ShapeDtypeStruct((), jnp.float64),
        params,
        vmap_method="sequential",
    )


# Production-realistic settings, scaled down so the pure_callback path
# can still converge in tens of minutes. The callback path has heavy JAX
# <-> Python overhead per evaluation; n_live=200 + termination=-2 lands
# a good evidence estimate without the runaway eval count of n_live=500
# / termination=-3 (which would take 1+ hours via this path).
n_live = 200
rng_key = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng_key)

prior_lower = jnp.array([0.0, 0.0, 0.0])
prior_upper = jnp.array([100.0, 50.0, 50.0])
initial_samples = jax.random.uniform(
    init_key, shape=(n_live, model.prior_count), minval=prior_lower, maxval=prior_upper
)

print("Running NSS (autofit interface) nested sampling...")
print(f"  n_live={n_live}, n_dim={model.prior_count}")
print(f"  Using jax.pure_callback for NumPy likelihood\n")

t_start = time.time()
# num_delete=10 / num_mcmc=2 keeps per-step callback overhead bounded;
# termination=-2 (delta-logZ < 0.01) gives a good evidence estimate while
# fitting inside a reasonable wall-time budget for the callback path.
final_state, results = run_nested_sampling(
    rng_key,
    loglikelihood_fn=log_likelihood,
    prior_logprob=log_prior,
    num_mcmc_steps=2,
    initial_samples=initial_samples,
    num_delete=10,
    termination=-2,
)
t_elapsed = time.time() - t_start

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

positions = final_state.particles.position
log_likelihoods = final_state.particles.loglikelihood

best_idx = jnp.argmax(log_likelihoods)
best_params = positions[best_idx]
best_instance = model.instance_from_vector(vector=np.asarray(best_params).tolist())
max_logl = float(jnp.max(log_likelihoods))
evals_to_ml, time_to_ml = tracker.finalise(max_log_l=max_logl, tolerance=1.0)

summary = f"""\
--- NSS (autofit) Results ---
Best fit:        centre={best_instance.centre:.4f}  normalization={best_instance.normalization:.4f}  sigma={best_instance.sigma:.4f}
True:            centre=50.0000  normalization=25.0000  sigma=10.0000
Max log L:       {max_logl:.4f}
Log evidence:    {float(results.logZs.mean()):.4f}

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (includes JIT compilation / warmup)
Sampling time:       {float(results.time):.2f} s     (excludes JIT warmup)
Likelihood evals:    {n_likelihood_calls}
Time per eval:       {t_elapsed / max(n_likelihood_calls, 1) * 1e3:.3f} ms
ESS:                 {float(results.ess):.1f}
Posterior samples:   {len(positions)}
n_live / n_dim:      {n_live} / {model.prior_count}

--- Convergence ---
Converged:           yes (NSS termination=-2)
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
