"""
Minimal Nautilus Example
------------------------

This script shows how to use the Nautilus nested sampler directly with
autofit's Model and Analysis objects, bypassing the full NonLinearSearch
wrapper class.

This is useful for quickly testing a search on a problem before investing
in a full autofit integration.

Requirements:
    pip install nautilus-sampler
"""

import time
from pathlib import Path

import numpy as np
import autofit as af

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
# Direct Nautilus interface
# --------------------------------------------------------------------------

from nautilus import Sampler


def prior_transform(cube):
    """Map unit cube to physical parameters via the model's priors."""
    return np.array(model.vector_from_unit_vector(cube))


n_likelihood_calls = 0
tracker = MLTracker()


def log_likelihood(params):
    """Evaluate log likelihood for a physical parameter vector."""
    global n_likelihood_calls
    n_likelihood_calls += 1
    instance = model.instance_from_vector(vector=params)
    log_l = float(analysis.log_likelihood_function(instance))
    tracker.record(log_l)
    return log_l


n_live = 200
n_dim = model.prior_count

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
best_instance = model.instance_from_vector(vector=best_params)
max_logl = float(np.max(log_l))
evals_to_ml, time_to_ml = tracker.finalise(max_log_l=max_logl, tolerance=1.0)

summary = f"""\
--- Nautilus (NumPy) Results ---
Best fit:        centre={best_instance.centre:.4f}  normalization={best_instance.normalization:.4f}  sigma={best_instance.sigma:.4f}
True:            centre=50.0000  normalization=25.0000  sigma=10.0000
Max log L:       {max_logl:.4f}
Log evidence:    {float(sampler.log_z):.4f}

--- Performance ---
Wall time:           {t_elapsed:.2f} s     (no separate JIT compilation)
Sampling time:       {t_elapsed:.2f} s     (Nautilus does not split warmup)
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
