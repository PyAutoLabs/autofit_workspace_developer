"""
Nautilus Overhead Profile
-------------------------

Phase 1 of the nautilus-nn-bottleneck research task
(https://github.com/Jammy2211/autofit_workspace_developer/issues/18).

Instruments nautilus 1.0.5 to break sampler wall time into its overhead
components, on problems where the likelihood itself is fast — the regime
(e.g. MGE-only source on GPU) where sampler overhead dominates:

    - NN training        time inside NeuralNetworkEmulator.train
                         (n_networks sklearn MLPs per ellipsoid, per bound)
    - bound geometry     NautilusBound.compute minus NN training
                         (ellipsoid union fitting / splitting / volume MC)
    - proposal sampling  Sampler.sample_shell (draw + shell association)
    - likelihood         Sampler.evaluate_likelihood (transform + callback)
    - residue            everything else (bookkeeping, shell updates, I/O)

Two scenarios:
    gaussian_1d_data  the standard minimal 3-parameter Gaussian-fit problem
                      (identical data/seed to nautilus_simple.py, so rows
                      are comparable to output/comparison.txt)
    fast_gauss_10d    a 10-dim correlated Gaussian log-likelihood costing
                      ~1 us/call — emulates the fast-GPU-likelihood regime
                      at a dimensionality typical of lens models

Requirements:
    pip install nautilus-sampler
"""

import time
from pathlib import Path

import numpy as np
import autofit as af

from searches_minimal._metrics import MLTracker

# --------------------------------------------------------------------------
# Timing instrumentation (monkey-patched wrappers around nautilus internals)
# --------------------------------------------------------------------------

import nautilus
from nautilus import Sampler
from nautilus.neural import NeuralNetworkEmulator
from nautilus.bounds import NautilusBound


class Timings:
    """Accumulate wall time and call counts per instrumented component."""

    def __init__(self):
        self.totals = {}
        self.counts = {}
        self.per_call = {}

    def add(self, key, dt):
        self.totals[key] = self.totals.get(key, 0.0) + dt
        self.counts[key] = self.counts.get(key, 0) + 1
        self.per_call.setdefault(key, []).append(dt)


TIMINGS = Timings()

_train_original = NeuralNetworkEmulator.train.__func__
_compute_original = NautilusBound.compute.__func__
_sample_shell_original = Sampler.sample_shell
_evaluate_likelihood_original = Sampler.evaluate_likelihood


def _train_timed(cls, *args, **kwargs):
    t0 = time.time()
    result = _train_original(cls, *args, **kwargs)
    TIMINGS.add("nn_train", time.time() - t0)
    return result


def _compute_timed(cls, *args, **kwargs):
    t0 = time.time()
    result = _compute_original(cls, *args, **kwargs)
    TIMINGS.add("bound_compute", time.time() - t0)
    n_ellipsoids = len(result.neural_bounds)
    TIMINGS.per_call.setdefault("ellipsoids_per_bound", []).append(n_ellipsoids)
    return result


def _sample_shell_timed(self, *args, **kwargs):
    t0 = time.time()
    result = _sample_shell_original(self, *args, **kwargs)
    TIMINGS.add("sample_shell", time.time() - t0)
    return result


def _evaluate_likelihood_timed(self, *args, **kwargs):
    t0 = time.time()
    result = _evaluate_likelihood_original(self, *args, **kwargs)
    TIMINGS.add("evaluate_likelihood", time.time() - t0)
    return result


NeuralNetworkEmulator.train = classmethod(_train_timed)
NautilusBound.compute = classmethod(_compute_timed)
Sampler.sample_shell = _sample_shell_timed
Sampler.evaluate_likelihood = _evaluate_likelihood_timed

# --------------------------------------------------------------------------
# Scenario 1 — the standard minimal problem (3 parameters, comparable rows)
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


np.random.seed(1)

xvalues = np.arange(100)
true_gaussian = Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
data = true_gaussian.model_data_from(xvalues=xvalues)
noise_map = np.full(data.shape, 0.01)
data += np.random.normal(0.0, 0.01, data.shape)

model = af.Model(Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)


def gaussian_prior_transform(cube):
    return np.array(model.vector_from_unit_vector(cube))


def gaussian_log_likelihood(params):
    instance = model.instance_from_vector(vector=params)
    model_data = instance.model_data_from(xvalues=xvalues)
    residual_map = data - model_data
    chi_squared_map = (residual_map / noise_map) ** 2.0
    return float(-0.5 * np.sum(chi_squared_map))


# --------------------------------------------------------------------------
# Scenario 2 — fast 10-dim correlated Gaussian (fast-GPU-likelihood regime)
# --------------------------------------------------------------------------

N_DIM_FAST = 10

rng_setup = np.random.default_rng(1)
_a = rng_setup.normal(size=(N_DIM_FAST, N_DIM_FAST))
_cov = _a @ _a.T + N_DIM_FAST * np.eye(N_DIM_FAST)
_cov_inv = np.linalg.inv(_cov)
_mean = np.full(N_DIM_FAST, 0.5)


def fast_prior_transform(cube):
    return cube


def fast_log_likelihood(params):
    d = params - _mean
    return float(-0.5 * d @ _cov_inv @ d)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def run_scenario(name, prior_transform, log_likelihood, n_dim, n_live=200):
    TIMINGS.totals.clear()
    TIMINGS.counts.clear()
    TIMINGS.per_call.clear()

    tracker = MLTracker()
    n_calls = {"n": 0}

    def wrapped(params):
        n_calls["n"] += 1
        log_l = log_likelihood(params)
        tracker.record(log_l)
        return log_l

    # Time the raw likelihood so the callback overhead is separable.
    test_point = prior_transform(np.full(n_dim, 0.5))
    t0 = time.time()
    for _ in range(1000):
        log_likelihood(test_point)
    raw_likelihood_ms = (time.time() - t0)

    sampler = Sampler(
        prior=prior_transform,
        likelihood=wrapped,
        n_dim=n_dim,
        n_live=n_live,
        seed=1,
    )

    t_start = time.time()
    sampler.run(verbose=True)
    wall = time.time() - t_start

    points, log_w, log_l = sampler.posterior()
    max_logl = float(np.max(log_l))
    evals_to_ml, time_to_ml = tracker.finalise(max_log_l=max_logl, tolerance=1.0)

    nn = TIMINGS.totals.get("nn_train", 0.0)
    bound = TIMINGS.totals.get("bound_compute", 0.0)
    shell = TIMINGS.totals.get("sample_shell", 0.0)
    like = TIMINGS.totals.get("evaluate_likelihood", 0.0)
    geometry = bound - nn
    residue = wall - bound - shell - like
    ellipsoids = TIMINGS.per_call.get("ellipsoids_per_bound", [])
    n_mlp_fits = TIMINGS.counts.get("nn_train", 0)

    def row(label, seconds, extra=""):
        return f"{label:<22s} {seconds:>8.2f} s   {100 * seconds / wall:>5.1f} %   {extra}"

    summary = f"""\
--- Nautilus Overhead Profile: {name} ---
n_dim / n_live:      {n_dim} / {n_live}
Wall time:           {wall:.2f} s
Likelihood evals:    {n_calls['n']}
Raw likelihood:      {raw_likelihood_ms:.3f} ms/call (timed outside sampler, 1000 calls)

--- Wall-time breakdown ---
{row("nn_train", nn, f"({n_mlp_fits} emulator trainings, {sum(ellipsoids)} ellipsoid-ensembles x 4 MLPs)")}
{row("bound_geometry", geometry, f"({TIMINGS.counts.get('bound_compute', 0)} bounds; ellipsoids/bound: {ellipsoids})")}
{row("sample_shell", shell, f"({TIMINGS.counts.get('sample_shell', 0)} calls)")}
{row("evaluate_likelihood", like, f"({TIMINGS.counts.get('evaluate_likelihood', 0)} batched calls)")}
{row("residue", residue, "(bookkeeping, shell updates)")}

--- Quality (MLTracker contract) ---
Max log L:           {max_logl:.4f}
Log evidence:        {float(sampler.log_z):.4f}
ESS:                 {float(sampler.n_eff):.1f}
Evals to ML:         {evals_to_ml if evals_to_ml is not None else 'n/a'}
Time to ML:          {f'{time_to_ml:.2f} s' if time_to_ml is not None else 'n/a'}
"""
    print()
    print(summary)
    return summary


if __name__ == "__main__":
    summaries = []
    summaries.append(
        run_scenario(
            "gaussian_1d_data (3 params)",
            gaussian_prior_transform,
            gaussian_log_likelihood,
            n_dim=model.prior_count,
        )
    )
    summaries.append(
        run_scenario(
            f"fast_gauss_{N_DIM_FAST}d",
            fast_prior_transform,
            fast_log_likelihood,
            n_dim=N_DIM_FAST,
        )
    )

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{Path(__file__).stem}_summary.txt"
    summary_path.write_text("\n".join(summaries))
    print(f"Summary written to: {summary_path}")
