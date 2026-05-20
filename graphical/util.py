"""
Shared utilities for the `graphical` example package.

Provides argparse, autofit-config wiring, dataset discovery, and a small
profile-summary writer used by both `simulator.py` and `fit.py`.
"""

import argparse
import json
import os
from pathlib import Path

from autoconf import conf

import autofit as af


PACKAGE_ROOT = Path(__file__).parent
DEFAULT_SAMPLE = "toy__gaussian_x1__low_snr"


def parse_sample_args(default_total=30):
    parser = argparse.ArgumentParser(description="Run a graphical-model toy fit.")
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--total_datasets", type=int, default=default_total)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--number_of_cores", type=int, default=1)
    parser.add_argument("--name", default=None,
                        help="Autofit run name (default: derived from N).")
    return parser.parse_args()


def setup_config():
    """Point autofit at this package's config/ and output/ dirs."""
    conf.instance.push(
        new_path=PACKAGE_ROOT / "config",
        output_path=PACKAGE_ROOT / "output",
    )
    return PACKAGE_ROOT


def discover_datasets(sample_path):
    """List dataset_N subdirectories under a sample, sorted numerically.

    Skips `_sample/` and any other underscore-prefixed entries.
    """
    dirs = [
        d.name for d in sorted(sample_path.iterdir())
        if d.is_dir() and not d.name.startswith("_")
    ]
    dirs.sort(key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else x)
    return dirs


def load_dataset(dataset_path):
    """Load (data, noise_map, ground_truth) for a single dataset directory."""
    data = af.util.numpy_array_from_json(file_path=dataset_path / "data.json")
    noise_map = af.util.numpy_array_from_json(file_path=dataset_path / "noise_map.json")
    with open(dataset_path / "ground_truth.json") as f:
        ground_truth = json.load(f)
    return data, noise_map, ground_truth


def output_dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                pass
    return total


def write_profile_summary(profiles_dir, name, **fields):
    profiles_dir.mkdir(parents=True, exist_ok=True)
    out_path = profiles_dir / f"{name}_summary.json"
    with open(out_path, "w") as f:
        json.dump({"name": name, **fields}, f, indent=2)
    return out_path
