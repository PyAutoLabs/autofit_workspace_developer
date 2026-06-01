"""
Expectation propagation toy fit: shared Gaussian centre across N datasets
via per-factor Dynesty fits + LaplaceOptimiser cavity updates.

Adapted from `z_projects/concr/scripts/toy/ep.py`. New behaviour:

- Loads `ground_truth.json` per dataset.
- Measures wall time, peak RSS, and output-dir disk size around the
  full EP loop.
- Runs end-of-run sanity checks (parameter recovery) by indexing the
  posterior mean-field with the original prior objects.
- Writes `profiles/<name>_summary.json` for cross-N aggregation.

Usage:
    python3 ep/fit.py [--sample=toy__gaussian_x1__low_snr]
                      [--total_datasets=3] [--name=N3]
"""

import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import autofit as af

from util import (
    parse_sample_args,
    setup_config,
    discover_datasets,
    load_dataset,
    output_dir_bytes,
    write_profile_summary,
    PACKAGE_ROOT,
)
from sanity import check_parameter_recovery, print_sanity_summary


def fit(args):
    setup_config()
    run_name = args.name or f"N{args.total_datasets}"

    sample_path = PACKAGE_ROOT / "dataset" / args.sample
    dataset_names = discover_datasets(sample_path)[: args.total_datasets]
    if not dataset_names:
        raise SystemExit(
            f"No datasets found at {sample_path}. "
            "Run `python3 ep/simulator.py` first."
        )

    data_list, noise_map_list, truths = [], [], []
    for name in dataset_names:
        data, noise_map, ground_truth = load_dataset(sample_path / name)
        data_list.append(data)
        noise_map_list.append(noise_map)
        truths.append(ground_truth)

    print(f"Loaded {len(data_list)} datasets from {args.sample}")

    analysis_list = [
        af.ex.Analysis(data=data, noise_map=noise_map, use_jax=True)
        for data, noise_map in zip(data_list, noise_map_list)
    ]

    centre_shared_prior = af.GaussianPrior(mean=50.0, sigma=30.0)

    model_list = []
    for _ in data_list:
        gaussian = af.Model(af.ex.Gaussian)
        gaussian.centre = centre_shared_prior
        gaussian.normalization = af.TruncatedGaussianPrior(
            mean=0.5, sigma=2.0, lower_limit=0.0
        )
        gaussian.sigma = af.TruncatedGaussianPrior(mean=5.0, sigma=5.0, lower_limit=0.0)
        model_list.append(af.Collection(gaussian=gaussian))

    paths = af.DirectoryPaths(
        path_prefix=Path(args.sample),
        name=run_name,
    )
    search = af.DynestyStatic(paths=paths, nlive=100, sample="rwalk")

    analysis_factor_list = [
        af.AnalysisFactor(
            prior_model=model,
            analysis=analysis,
            optimiser=search,
            name=f"dataset_{i}",
        )
        for i, (model, analysis) in enumerate(zip(model_list, analysis_list))
    ]
    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

    laplace = af.LaplaceOptimiser()

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    result = factor_graph.optimise(
        optimiser=laplace,
        paths=paths,
        ep_history=af.EPHistory(kl_tol=0.05),
        max_steps=5,
    )
    wall_time = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_kb = max(rss_before, rss_after)

    mean_field = result.updated_ep_mean_field.mean_field

    sanity = []
    centre_msg = mean_field[centre_shared_prior]
    sanity.append(
        check_parameter_recovery(
            "shared centre",
            float(centre_msg.mean),
            float(centre_msg.sigma),
            truths[0]["centre"],
        )
    )

    for i, (model, gt) in enumerate(zip(model_list, truths)):
        norm_msg = mean_field[model.gaussian.normalization]
        sanity.append(
            check_parameter_recovery(
                f"dset_{i} normalization",
                float(norm_msg.mean),
                float(norm_msg.sigma),
                gt["normalization"],
            )
        )
        sig_msg = mean_field[model.gaussian.sigma]
        sanity.append(
            check_parameter_recovery(
                f"dset_{i} sigma",
                float(sig_msg.mean),
                float(sig_msg.sigma),
                gt["sigma"],
            )
        )

    sanity_pass = print_sanity_summary(sanity, header=f"EP N={len(truths)}")

    output_path = PACKAGE_ROOT / "output" / args.sample / run_name
    disk_bytes = output_dir_bytes(output_path)

    write_profile_summary(
        PACKAGE_ROOT / "profiles",
        name=run_name,
        sampler="DynestyStatic+Laplace(EP)",
        n_datasets=len(truths),
        wall_time_s=wall_time,
        peak_rss_kb=peak_rss_kb,
        output_dir_bytes=disk_bytes,
        truth_log_likelihood_sum=float(sum(g["truth_log_likelihood"] for g in truths)),
        sanity_pass=sanity_pass,
    )

    print(f"\nwall_time: {wall_time:.2f} s")
    print(f"peak_rss:  {peak_rss_kb / 1024:.1f} MB")
    print(f"disk_size: {disk_bytes / (1024 * 1024):.2f} MB")
    return result


if __name__ == "__main__":
    fit(parse_sample_args())
