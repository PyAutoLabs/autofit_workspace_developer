"""
NSS Example
-----------

This script demonstrates the NSS (Nested Slice Sampling) sampler that was
previously bundled with PyAutoFit as ``af.NSS``. It was removed from the
library on 2026-07-11 (PyAutoFit#1356); the full implementation is preserved in
this directory (``search.py`` and friends) as a drop-in ``af.NonLinearSearch``.

NSS must be installed separately — see ``README.md`` in this directory for the
exact pinned install recipe (a ``handley-lab/blackjax`` fork + ``yallup/nss``).
The ``NSS`` class here imports from autofit's base classes, so once its
dependencies are installed it works as a drop-in replacement for any boundary
sampler (``af.Nautilus``, ``af.DynestyStatic``, …).

__Analysis Must Be JAX-Traceable__

NSS runs the log-likelihood inside ``jax.jit``. The boundary-based samplers are
happy with a plain NumPy ``af.ex.Analysis(data, noise_map)`` — but when we hand
the same analysis to NSS the JIT trace hits the NumPy paths and raises
``TracerArrayConversionError``. The fix is to build the analysis with
``use_jax=True``, which makes its arithmetic dispatch through ``jax.numpy``.
"""

import matplotlib.pyplot as plt
import numpy as np
from os import path

import autofit as af

# The parked NSS implementation living alongside this example.
from searches.nss import NSS

"""
__Data__

This example fits a single 1D Gaussian, we therefore load and plot data
containing one Gaussian.
"""
dataset_path = path.join("dataset", "example_1d", "gaussian_x1")

data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
noise_map = af.util.numpy_array_from_json(
    file_path=path.join(dataset_path, "noise_map.json")
)

"""
__Model + Analysis__

A single ``Gaussian`` (dimensionality N=3). The analysis is built with
``use_jax=True`` so its likelihood is JAX-traceable — the precondition for NSS.
"""
model = af.Model(af.ex.Gaussian)

model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.LogUniformPrior(lower_limit=1e-2, upper_limit=1e2)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=30.0)

analysis = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=True)

"""
__Search: NSS__

Swapping ``af.Nautilus(...)`` for ``NSS(...)`` in an existing script is a
one-line change — NSS exposes the same ``result.samples`` interface. See the
kwarg table in ``README.md`` for what each setting does.
"""
search = NSS(
    path_prefix=path.join("searches"),
    name="NSS",
    n_live=200,  # live particles maintained throughout the run
    num_mcmc_steps=5,  # slice-MCMC inner steps per dead-point batch
    num_delete=50,  # particles removed per outer iteration
    termination=-3.0,  # delta-logZ stopping criterion
    seed=42,  # JAX PRNG seed for reproducible runs
    checkpoint_interval=100,  # SLURM-friendly resume — see README
)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

Confirm NSS converged to roughly the same posterior as the boundary samplers.
On this trivial 1D likelihood there is no speed advantage (the per-call cost is
dominated by Python overhead, and the first run pays a one-off ~25-30s JIT
compile) — the per-eval win only shows on a real JAX-traceable autolens /
autogalaxy MGE likelihood.
"""
model_data = result.max_log_likelihood_instance.model_data_from(
    xvalues=np.arange(data.shape[0])
)

plt.errorbar(
    x=range(data.shape[0]),
    y=data,
    yerr=noise_map,
    linestyle="",
    color="k",
    ecolor="k",
    elinewidth=1,
    capsize=2,
)
plt.plot(range(data.shape[0]), model_data, color="r")
plt.title("NSS model fit to 1D Gaussian dataset.")
plt.xlabel("x values of profile")
plt.ylabel("Profile normalization")
plt.show()
plt.close()

print(f"NSS log evidence:    {result.samples.samples_info['log_evidence']:.4f}")
print(f"NSS max log L:       {max(result.samples.log_likelihood_list):.4f}")
