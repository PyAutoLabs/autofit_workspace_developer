"""
Nautilus Zero-Code Tuning Sweep
-------------------------------

Phase 2 of the nautilus-nn-bottleneck research task
(https://github.com/Jammy2211/autofit_workspace_developer/issues/18).

Sweeps the tuning surface nautilus 1.0.5 already exposes — no sampler code
changes — and measures wall time against sampling quality, using the same
instrumentation and scenarios as nautilus_profile.py:

    baseline        n_networks=4, serial training (the PyAutoFit JAX-path
                    default: fit_x1_cpu passes pool=None)
    nets_2          halve the MLP ensemble
    nets_0          no NN at all — pure ellipsoid bounds; sampling efficiency
                    drops but each extra eval is ~free in the fast regime
    pool_s_4        pool=(None, 4): parallel MLP training via a sampler pool,
                    likelihood stays serial/vectorized (what fit_x1_cpu could
                    pass but doesn't)
    small_mlp       one hidden layer (32,) instead of (100, 50, 20)
    n_update_2x     retrain bounds half as often (n_update = 2 * n_live)
    vectorized      batched likelihood callback (numpy-vectorized here; the
                    stand-in for a vmapped GPU likelihood)

Quality is judged on log Z / ESS / evals-to-ML agreement with the baseline
(reference log Z for the 3-parameter Gaussian problem: ~-57.5).
"""

import time
from pathlib import Path

import numpy as np

from searches_minimal.nautilus_profile import (
    TIMINGS,
    Sampler,
    MLTracker,
    gaussian_prior_transform,
    gaussian_log_likelihood,
    fast_prior_transform,
    fast_log_likelihood,
    model,
    _mean,
    _cov_inv,
    N_DIM_FAST,
)


def fast_log_likelihood_vectorized(params):
    """Batched variant of the fast 10-d Gaussian: (n_batch, n_dim) -> (n_batch,)."""
    d = params - _mean
    return -0.5 * np.einsum("bi,ij,bj->b", d, _cov_inv, d)


def run_config(name, prior_transform, log_likelihood, n_dim, n_live=200,
               **sampler_kwargs):
    TIMINGS.totals.clear()
    TIMINGS.counts.clear()
    TIMINGS.per_call.clear()

    tracker = MLTracker()
    vectorized = sampler_kwargs.get("vectorized", False)

    if vectorized:
        def wrapped(params):
            log_l = log_likelihood(params)
            for value in np.atleast_1d(log_l):
                tracker.record(float(value))
            return log_l
    else:
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
        **sampler_kwargs,
    )

    t_start = time.time()
    sampler.run(verbose=False)
    wall = time.time() - t_start

    points, log_w, log_l = sampler.posterior()
    max_logl = float(np.max(log_l))
    evals_to_ml, _ = tracker.finalise(max_log_l=max_logl, tolerance=1.0)

    nn = TIMINGS.totals.get("nn_train", 0.0)
    bound = TIMINGS.totals.get("bound_compute", 0.0)
    result = dict(
        name=name,
        wall=wall,
        nn=nn,
        geometry=bound - nn,
        shell=TIMINGS.totals.get("sample_shell", 0.0),
        like=TIMINGS.totals.get("evaluate_likelihood", 0.0),
        n_bounds=TIMINGS.counts.get("bound_compute", 0),
        n_evals=int(sampler.n_like),
        log_z=float(sampler.log_z),
        ess=float(sampler.n_eff),
        evals_to_ml=evals_to_ml,
    )

    for pool_attr in ("pool_l", "pool_s"):
        pool = getattr(sampler, pool_attr, None)
        if pool is not None:
            try:
                pool.close()
                pool.join()
            except Exception:
                pass

    print(f"  {name:<12s} wall={wall:6.2f} s  nn={nn:6.2f} s  "
          f"logZ={result['log_z']:8.3f}  ESS={result['ess']:7.1f}  "
          f"evals={result['n_evals']}")
    return result


def format_table(title, results, baseline):
    lines = [
        f"--- {title} ---",
        f"{'config':<12s} {'wall s':>7s} {'x':>5s} {'nn s':>6s} {'geom s':>6s} "
        f"{'shell s':>7s} {'like s':>6s} {'bounds':>6s} {'evals':>7s} "
        f"{'log Z':>9s} {'ESS':>7s} {'ev->ML':>7s}",
    ]
    for r in results:
        speedup = baseline["wall"] / r["wall"]
        lines.append(
            f"{r['name']:<12s} {r['wall']:>7.2f} {speedup:>4.1f}x {r['nn']:>6.2f} "
            f"{r['geometry']:>6.2f} {r['shell']:>7.2f} {r['like']:>6.2f} "
            f"{r['n_bounds']:>6d} {r['n_evals']:>7d} {r['log_z']:>9.3f} "
            f"{r['ess']:>7.1f} "
            f"{r['evals_to_ml'] if r['evals_to_ml'] is not None else 'n/a':>7}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    small_mlp = dict(hidden_layer_sizes=(32,), n_iter_no_change=5)

    print("== fast_gauss_10d (fast-GPU-likelihood regime) ==")
    fast_args = (fast_prior_transform, fast_log_likelihood, N_DIM_FAST)
    fast_results = [
        run_config("baseline", *fast_args),
        run_config("nets_2", *fast_args, n_networks=2),
        run_config("nets_0", *fast_args, n_networks=0),
        run_config("pool_s_4", *fast_args, pool=(None, 4)),
        run_config("small_mlp", *fast_args, neural_network_kwargs=small_mlp),
        run_config("n_update_2x", *fast_args, n_update=400),
        run_config(
            "vectorized",
            fast_prior_transform,
            fast_log_likelihood_vectorized,
            N_DIM_FAST,
            vectorized=True,
        ),
    ]

    print("== gaussian_1d_data (3 params; reference log Z ~ -57.5) ==")
    gauss_args = (
        gaussian_prior_transform,
        gaussian_log_likelihood,
        model.prior_count,
    )
    gauss_results = [
        run_config("baseline", *gauss_args),
        run_config("nets_0", *gauss_args, n_networks=0),
        run_config("pool_s_4", *gauss_args, pool=(None, 4)),
        run_config("small_mlp", *gauss_args, neural_network_kwargs=small_mlp),
    ]

    tables = [
        format_table("fast_gauss_10d", fast_results, fast_results[0]),
        format_table("gaussian_1d_data", gauss_results, gauss_results[0]),
    ]
    output = "\n\n".join(tables) + "\n"
    print()
    print(output)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{Path(__file__).stem}_summary.txt"
    summary_path.write_text(output)
    print(f"Summary written to: {summary_path}")
