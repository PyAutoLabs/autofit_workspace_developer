"""
Graphical model toy fit: shared Gaussian centre across N datasets via a
single joint Dynesty fit on `factor_graph.global_prior_model`.

Adapted from `z_projects/concr/scripts/toy/graphical.py`. New behaviour:

- Loads `ground_truth.json` per dataset.
- Measures wall time, peak RSS, and output-dir disk size around the fit.
- Runs end-of-run sanity checks (parameter recovery + max log likelihood)
  via the `sanity` module. Reports PASS/FAIL but does not raise.
- Writes `profiles/<name>_summary.json` for cross-N aggregation.

Usage:
    python3 graphical/fit.py [--sample=toy__gaussian_x1__low_snr]
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
from sanity import (
    check_parameter_recovery,
    check_max_log_likelihood,
    print_sanity_summary,
)


def fit(args):
    setup_config()
    run_name = args.name or f"N{args.total_datasets}"

    sample_path = PACKAGE_ROOT / "dataset" / args.sample
    dataset_names = discover_datasets(sample_path)[: args.total_datasets]
    if not dataset_names:
        raise SystemExit(
            f"No datasets found at {sample_path}. "
            "Run `python3 graphical/simulator.py` first."
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
            mean=0.5, sigma=2.0, lower_limit=0.0)
        gaussian.sigma = af.TruncatedGaussianPrior(
            mean=5.0, sigma=5.0, lower_limit=0.0)
        model_list.append(af.Collection(gaussian=gaussian))

    analysis_factor_list = [
        af.AnalysisFactor(prior_model=model, analysis=analysis)
        for model, analysis in zip(model_list, analysis_list)
    ]
    factor_graph = af.FactorGraphModel(*analysis_factor_list, use_jax=True)

    search = af.DynestyStatic(
        path_prefix=Path(args.sample),
        name=run_name,
        sample="rwalk",
        number_of_cores=args.number_of_cores,
    )

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    result = search.fit(model=factor_graph.global_prior_model,
                        analysis=factor_graph)
    wall_time = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_kb = max(rss_before, rss_after)

    max_log_likelihood = float(result.samples.max_log_likelihood_sample.log_likelihood)
    truth_ll_sum = float(sum(g["truth_log_likelihood"] for g in truths))

    median_instance = result.samples.median_pdf()
    errors_instance = result.samples.errors_at_sigma(sigma=1.0)

    sanity = []
    # Shared centre — pulled from the first factor; identical across all i.
    centre_mean = float(median_instance[0].gaussian.centre)
    centre_lower, centre_upper = errors_instance[0].gaussian.centre
    centre_sigma = float((centre_lower + centre_upper) / 2.0)
    sanity.append(check_parameter_recovery(
        "shared centre", centre_mean, centre_sigma, truths[0]["centre"]))

    # Per-dataset normalization and sigma.
    for i, gt in enumerate(truths):
        norm_mean = float(median_instance[i].gaussian.normalization)
        nl, nu = errors_instance[i].gaussian.normalization
        norm_sigma = float((nl + nu) / 2.0)
        sanity.append(check_parameter_recovery(
            f"dset_{i} normalization", norm_mean, norm_sigma,
            gt["normalization"]))

        sig_mean = float(median_instance[i].gaussian.sigma)
        sl, su = errors_instance[i].gaussian.sigma
        sig_sigma = float((sl + su) / 2.0)
        sanity.append(check_parameter_recovery(
            f"dset_{i} sigma", sig_mean, sig_sigma, gt["sigma"]))

    sanity.append(check_max_log_likelihood(
        max_log_likelihood, truth_ll_sum, len(truths)))

    sanity_pass = print_sanity_summary(
        sanity, header=f"graphical N={len(truths)}")

    output_path = PACKAGE_ROOT / "output" / args.sample / run_name
    disk_bytes = output_dir_bytes(output_path)

    write_profile_summary(
        PACKAGE_ROOT / "profiles",
        name=run_name,
        sampler="DynestyStatic",
        n_datasets=len(truths),
        wall_time_s=wall_time,
        peak_rss_kb=peak_rss_kb,
        output_dir_bytes=disk_bytes,
        max_log_likelihood=max_log_likelihood,
        truth_log_likelihood_sum=truth_ll_sum,
        sanity_pass=sanity_pass,
    )

    print(f"\nwall_time: {wall_time:.2f} s")
    print(f"peak_rss:  {peak_rss_kb / 1024:.1f} MB")
    print(f"disk_size: {disk_bytes / (1024 * 1024):.2f} MB")
    print(f"max_log_likelihood: {max_log_likelihood:.3f}")
    print(f"truth_log_likelihood_sum: {truth_ll_sum:.3f}")
    return result


if __name__ == "__main__":
    fit(parse_sample_args())
