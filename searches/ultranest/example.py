"""
UltraNest Example
-----------------

This script demonstrates how to use the UltraNest nested sampler that was
previously bundled with PyAutoFit. UltraNest must be installed separately:

    pip install ultranest

NOTE: UltraNest's Cython extensions must be compiled against the same NumPy
version you have installed. If you get a ``ValueError: numpy.dtype size
changed`` error, reinstall UltraNest so it recompiles:

    pip install --force-reinstall ultranest

The search class in this directory (search.py) imports from autofit's base
classes and can be used as a drop-in replacement.
"""

import numpy as np

from autoconf import conf

# Register the config directory shipped with this repo so that
# UltraNest can find its YAML defaults.
workspace_path = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
conf.instance.push(new_path=Path(workspace_path) / "config")

from searches.ultranest.search import UltraNest

import autofit as af
from pathlib import Path

# --- Define a simple 1D Gaussian model ---


class Gaussian:
    def __init__(
        self,
        centre=30.0,
        normalization=1.0,
        sigma=5.0,
    ):
        self.centre = centre
        self.normalization = normalization
        self.sigma = sigma

    def model_data_from(self, xvalues, xp=np):
        return xp.multiply(
            xp.divide(self.normalization, self.sigma * xp.sqrt(2.0 * xp.pi)),
            xp.exp(-0.5 * xp.square(xp.divide(xvalues - self.centre, self.sigma))),
        )


# --- Generate some example data ---

xvalues = np.arange(100)
gaussian = Gaussian(centre=50.0, normalization=25.0, sigma=10.0)
data = gaussian.model_data_from(xvalues=xvalues)
noise_map = np.random.normal(0.0, 0.1, data.shape)
data += noise_map
noise_map = np.full(data.shape, 0.1)


# --- Define the Analysis class ---


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


# --- Set up the model ---

model = af.Model(Gaussian)
model.centre = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
model.normalization = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)
model.sigma = af.UniformPrior(lower_limit=0.0, upper_limit=50.0)

# --- Run UltraNest ---

search = UltraNest(
    path_prefix="output",
    name="ultranest_example",
)

analysis = Analysis(data=data, noise_map=noise_map)

result = search.fit(model=model, analysis=analysis)

print(f"Maximum likelihood instance: {result.max_log_likelihood_instance}")
print(f"Log evidence: {result.samples.log_evidence}")
