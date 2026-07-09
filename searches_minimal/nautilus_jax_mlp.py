"""
Nautilus JAX-MLP Emulator Prototype
-----------------------------------

Phase 3 of the nautilus-nn-bottleneck research task
(https://github.com/Jammy2211/autofit_workspace_developer/issues/18).

Replaces nautilus's sklearn `MLPRegressor` ensemble (`nautilus/neural.py`)
with a JAX implementation and measures the end-to-end effect on the same
scenarios as nautilus_profile.py. Design choices, and why:

    - The whole ensemble trains as ONE jitted program: `jax.vmap` over the
      n_networks parameter sets, full-batch Adam for a fixed number of steps.
      sklearn trains the 4 MLPs serially, each pinned to a single thread.
    - Training inputs are padded to power-of-two bucket sizes with masked
      sample weights, so the JIT cache is reused across bounds instead of
      recompiling for every new training-set size.
    - `predict` is also jitted + padded — it sits on the sampling hot path
      (`NeuralBound.contains` calls it for every proposal batch).
    - Same architecture and normalisation as nautilus's default
      (hidden layers (100, 50, 20), relu, inputs standardised), so the
      comparison isolates the training engine, not the model.
    - Fixed-step training replaces sklearn's early stopping
      (`tol=0, n_iter_no_change=10`); N_STEPS below was chosen so the
      emulator matches sklearn's end-to-end sampling quality (log Z / ESS),
      which the benchmark verifies.

This is a *prototype*: HDF5 checkpoint write/read of the emulator is not
implemented, so it must not be used with `filepath=`. Promotion paths
(PyAutoFit wrapper hook vs upstream nautilus PR) are the Phase 4 verdict.

On this laptop JAX is CPU-only; on a GPU node the ensemble training is
expected to shrink further. The benchmark prints the measured numbers only.
"""

import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import optax

from searches_minimal.nautilus_profile import (
    TIMINGS,
    Sampler,
    MLTracker,
    NeuralNetworkEmulator,
    _train_original,
    gaussian_prior_transform,
    gaussian_log_likelihood,
    fast_prior_transform,
    fast_log_likelihood,
    model,
    N_DIM_FAST,
)

HIDDEN_LAYERS = (100, 50, 20)
N_STEPS = 600
LEARNING_RATE = 3e-3


def _pad_size(n, minimum=64):
    size = minimum
    while size < n:
        size *= 2
    return size


def _init_params(key, n_dim):
    sizes = (n_dim,) + HIDDEN_LAYERS + (1,)
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        key, subkey = jax.random.split(key)
        w = jax.random.normal(subkey, (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        params.append((w, jnp.zeros(n_out)))
    return params


def _forward(params, x):
    for w, b in params[:-1]:
        x = jax.nn.relu(x @ w + b)
    w, b = params[-1]
    return (x @ w + b)[..., 0]


@jax.jit
def _train_ensemble(params_ensemble, x, y, weights):
    optimizer = optax.adam(LEARNING_RATE)

    def loss_fn(params):
        pred = _forward(params, x)
        return jnp.sum(weights * (pred - y) ** 2) / jnp.sum(weights)

    def train_one(params):
        opt_state = optimizer.init(params)

        def step(carry, _):
            params, opt_state = carry
            grads = jax.grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), None

        (params, _), _ = jax.lax.scan(
            step, (params, opt_state), None, length=N_STEPS)
        return params

    return jax.vmap(train_one)(params_ensemble)


@jax.jit
def _predict_ensemble(params_ensemble, x):
    return jnp.mean(jax.vmap(_forward, in_axes=(0, None))(params_ensemble, x),
                    axis=0)


class JaxNeuralNetworkEmulator:
    """Drop-in for nautilus.neural.NeuralNetworkEmulator (train/predict only)."""

    _seed_counter = 0

    @classmethod
    def train(cls, x, y, n_networks=4, neural_network_kwargs={}, pool=None):
        emulator = cls()
        emulator.mean = np.mean(x, axis=0)
        emulator.scale = np.std(x, axis=0)

        x_t = (x - emulator.mean) / emulator.scale
        n, n_dim = x_t.shape
        n_pad = _pad_size(n)
        x_pad = np.zeros((n_pad, n_dim))
        x_pad[:n] = x_t
        y_pad = np.zeros(n_pad)
        y_pad[:n] = y
        weights = np.zeros(n_pad)
        weights[:n] = 1.0

        cls._seed_counter += 1
        keys = jax.random.split(
            jax.random.PRNGKey(cls._seed_counter), n_networks)
        params = jax.vmap(lambda k: _init_params(k, n_dim))(keys)

        emulator.params = _train_ensemble(
            params, jnp.array(x_pad), jnp.array(y_pad), jnp.array(weights))
        jax.block_until_ready(emulator.params)
        return emulator

    def predict(self, x):
        x_t = (x - self.mean) / self.scale
        n = x_t.shape[0]
        n_pad = _pad_size(n)
        x_pad = np.zeros((n_pad, x_t.shape[1]))
        x_pad[:n] = x_t
        result = _predict_ensemble(self.params, jnp.array(x_pad))
        return np.asarray(result)[:n]


def _use_jax_emulator():
    def train_timed(cls, *args, **kwargs):
        t0 = time.time()
        result = JaxNeuralNetworkEmulator.train(*args, **kwargs)
        TIMINGS.add("nn_train", time.time() - t0)
        return result

    NeuralNetworkEmulator.train = classmethod(train_timed)


def _use_sklearn_emulator():
    def train_timed(cls, *args, **kwargs):
        t0 = time.time()
        result = _train_original(cls, *args, **kwargs)
        TIMINGS.add("nn_train", time.time() - t0)
        return result

    NeuralNetworkEmulator.train = classmethod(train_timed)


def run_config(name, prior_transform, log_likelihood, n_dim, n_live=200):
    TIMINGS.totals.clear()
    TIMINGS.counts.clear()
    TIMINGS.per_call.clear()

    tracker = MLTracker()

    def wrapped(params):
        log_l = log_likelihood(params)
        tracker.record(log_l)
        return log_l

    sampler = Sampler(
        prior=prior_transform,
        likelihood=wrapped,
        n_dim=n_dim,
        n_live=n_live,
        seed=1,
    )
    t_start = time.time()
    sampler.run(verbose=False)
    wall = time.time() - t_start

    points, log_w, log_l = sampler.posterior()
    evals_to_ml, _ = tracker.finalise(max_log_l=float(np.max(log_l)),
                                      tolerance=1.0)

    nn = TIMINGS.totals.get("nn_train", 0.0)
    bound = TIMINGS.totals.get("bound_compute", 0.0)
    result = dict(
        name=name,
        wall=wall,
        nn=nn,
        geometry=bound - nn,
        shell=TIMINGS.totals.get("sample_shell", 0.0),
        n_evals=int(sampler.n_like),
        log_z=float(sampler.log_z),
        ess=float(sampler.n_eff),
        evals_to_ml=evals_to_ml,
    )
    print(f"  {name:<14s} wall={wall:6.2f} s  nn={nn:6.2f} s  "
          f"shell={result['shell']:5.2f} s  logZ={result['log_z']:8.3f}  "
          f"ESS={result['ess']:7.1f}  evals={result['n_evals']}")
    return result


def format_rows(title, results):
    lines = [
        f"--- {title} ---",
        f"{'config':<14s} {'wall s':>7s} {'nn s':>6s} {'geom s':>6s} "
        f"{'shell s':>7s} {'evals':>7s} {'log Z':>9s} {'ESS':>7s} {'ev->ML':>7s}",
    ]
    for r in results:
        lines.append(
            f"{r['name']:<14s} {r['wall']:>7.2f} {r['nn']:>6.2f} "
            f"{r['geometry']:>6.2f} {r['shell']:>7.2f} {r['n_evals']:>7d} "
            f"{r['log_z']:>9.3f} {r['ess']:>7.1f} "
            f"{r['evals_to_ml'] if r['evals_to_ml'] is not None else 'n/a':>7}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    scenarios = [
        ("fast_gauss_10d", fast_prior_transform, fast_log_likelihood,
         N_DIM_FAST),
        ("gaussian_1d_data", gaussian_prior_transform, gaussian_log_likelihood,
         model.prior_count),
    ]

    tables = []
    for title, prior_transform, log_likelihood, n_dim in scenarios:
        print(f"== {title} ==")
        results = []
        _use_sklearn_emulator()
        results.append(
            run_config("sklearn", prior_transform, log_likelihood, n_dim))
        _use_jax_emulator()
        results.append(
            run_config("jax_mlp", prior_transform, log_likelihood, n_dim))
        # Second JAX run reuses the padded-shape JIT cache — the steady-state
        # cost a long fit would see.
        results.append(
            run_config("jax_mlp_warm", prior_transform, log_likelihood, n_dim))
        tables.append(format_rows(title, results))

    output = "\n\n".join(tables) + "\n"
    print()
    print(output)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{Path(__file__).stem}_summary.txt"
    summary_path.write_text(output)
    print(f"Summary written to: {summary_path}")
