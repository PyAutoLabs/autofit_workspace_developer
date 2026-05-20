"""
Simulator: 1D Gaussian datasets for graphical-model scale-up testing.

Generates N independent 1D Gaussian profiles sharing a common centre. For
each dataset writes:

    data.json, noise_map.json    - the simulated profile + per-pixel noise sigma
    info.json                    - human-readable dataset descriptor
    model.json                   - serialized truth Gaussian
    ground_truth.json            - truth params + log L at truth (consumed by fit.py)
    image.png                    - quick-look plot

Plus a sample-level `_sample/ground_truth.json` aggregating the shared
centre and the sum of per-dataset truth log likelihoods.

Usage:
    python3 graphical/simulator.py [--total_datasets 30]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import autofit as af
from autoconf.dictable import to_dict


PACKAGE_ROOT = Path(__file__).parent
SAMPLE_NAME = "toy__gaussian_x1__low_snr"

# Toy 1D Gaussian truth — shared across all datasets in the sample. The
# graphical fit recovers `centre` as a shared parameter; `normalization`
# and `sigma` are dataset-specific (but identical in truth here).
TRUE_CENTRE = 50.0
TRUE_NORMALIZATION = 0.5
TRUE_SIGMA = 5.0
N_PIXELS = 100
SIGNAL_TO_NOISE_RATIO = 25.0
NOISE_SIGMA = 1.0 / SIGNAL_TO_NOISE_RATIO


def simulate_dataset(dataset_index, dataset_path):
    xvalues = np.arange(N_PIXELS, dtype=float)

    gaussian = af.ex.Gaussian(
        centre=TRUE_CENTRE,
        normalization=TRUE_NORMALIZATION,
        sigma=TRUE_SIGMA,
    )
    model_data = gaussian.model_data_from(xvalues=xvalues)

    rng = np.random.default_rng(seed=dataset_index)
    noise = rng.normal(0.0, NOISE_SIGMA, N_PIXELS)
    data = model_data + noise
    noise_map = NOISE_SIGMA * np.ones(N_PIXELS)

    dataset_path.mkdir(parents=True, exist_ok=True)

    af.util.numpy_array_to_json(array=data,
                                file_path=dataset_path / "data.json",
                                overwrite=True)
    af.util.numpy_array_to_json(array=noise_map,
                                file_path=dataset_path / "noise_map.json",
                                overwrite=True)

    # Truth-evaluated log likelihood: what a perfect-knowledge oracle
    # would compute for THIS noisy realisation. The fit's max log
    # likelihood should sit at or slightly above this value.
    analysis = af.ex.Analysis(data=data, noise_map=noise_map, use_jax=False)
    truth_log_likelihood = float(analysis.log_likelihood_function(instance=gaussian))

    with open(dataset_path / "ground_truth.json", "w") as f:
        json.dump({
            "centre": TRUE_CENTRE,
            "normalization": TRUE_NORMALIZATION,
            "sigma": TRUE_SIGMA,
            "n_pixels": N_PIXELS,
            "signal_to_noise_ratio": SIGNAL_TO_NOISE_RATIO,
            "noise_sigma": NOISE_SIGMA,
            "truth_log_likelihood": truth_log_likelihood,
        }, f, indent=2)

    info = {
        "domain": "toy",
        "n_pixels": N_PIXELS,
        "signal_to_noise_ratio": SIGNAL_TO_NOISE_RATIO,
        "true_centre": TRUE_CENTRE,
        "true_normalization": TRUE_NORMALIZATION,
        "true_sigma": TRUE_SIGMA,
    }
    with open(dataset_path / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(dataset_path / "model.json", "w") as f:
        try:
            json.dump(to_dict(gaussian), f, indent=2)
        except (TypeError, ValueError):
            pass

    plt.errorbar(x=xvalues, y=data, yerr=noise_map, linestyle="",
                 color="k", ecolor="k", elinewidth=1, capsize=2)
    plt.title(f"dataset_{dataset_index} (truth centre={TRUE_CENTRE})")
    plt.xlabel("xvalues")
    plt.ylabel("profile normalization")
    plt.savefig(dataset_path / "image.png")
    plt.close()

    return truth_log_likelihood


def main():
    parser = argparse.ArgumentParser(
        description="Simulate 1D Gaussian datasets for graphical model fits."
    )
    parser.add_argument("--total_datasets", type=int, default=30)
    args = parser.parse_args()

    sample_path = PACKAGE_ROOT / "dataset" / SAMPLE_NAME
    sample_path.mkdir(parents=True, exist_ok=True)

    truth_lls = []
    for i in range(args.total_datasets):
        truth_lls.append(simulate_dataset(i, sample_path / f"dataset_{i}"))

    sample_meta_path = sample_path / "_sample"
    sample_meta_path.mkdir(parents=True, exist_ok=True)
    with open(sample_meta_path / "ground_truth.json", "w") as f:
        json.dump({
            "shared_centre": TRUE_CENTRE,
            "true_normalization": TRUE_NORMALIZATION,
            "true_sigma": TRUE_SIGMA,
            "n_datasets": args.total_datasets,
            "total_truth_log_likelihood": float(sum(truth_lls)),
            "per_dataset_truth_log_likelihoods": [float(v) for v in truth_lls],
        }, f, indent=2)

    print(f"Simulated {args.total_datasets} datasets -> {sample_path}/")
    print(f"  total truth log likelihood: {sum(truth_lls):.3f}")


if __name__ == "__main__":
    main()
