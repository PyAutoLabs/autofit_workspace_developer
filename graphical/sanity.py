"""
Sanity checks that compare recovered fit results against per-dataset truths.

Failures are reported as PASS/FAIL prints — never raised — so they remain
visible to future scale-up changes without gating runs.
"""


def check_parameter_recovery(name, recovered_mean, recovered_sigma, truth, k=3.0):
    """Pass iff |recovered_mean - truth| <= k * recovered_sigma."""
    if recovered_sigma is None or recovered_sigma == 0.0:
        passed = abs(recovered_mean - truth) < 1e-6
        msg = (
            f"{name}: mean={recovered_mean:.4g} truth={truth:.4g} "
            f"sigma=0 (deterministic) -> {'PASS' if passed else 'FAIL'}"
        )
        return passed, msg
    z = abs(recovered_mean - truth) / recovered_sigma
    passed = z <= k
    msg = (
        f"{name}: mean={recovered_mean:.4g} ± {recovered_sigma:.4g} "
        f"truth={truth:.4g} |z|={z:.2f} (k={k}) -> "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed, msg


def check_max_log_likelihood(
    max_ll, truth_ll_sum, n_datasets, tolerance_per_dataset=5.0
):
    """Pass iff max_ll >= truth_ll_sum - n_datasets * tolerance.

    A fit's max-LL should sit at or slightly above the truth-evaluated LL
    (the optimiser can fit a bit of the noise). We allow a few nats of
    slack per dataset for numerical wiggle and sampler imperfection.
    """
    threshold = truth_ll_sum - n_datasets * tolerance_per_dataset
    passed = max_ll >= threshold
    msg = (
        f"max_log_likelihood: {max_ll:.3f} vs truth_sum={truth_ll_sum:.3f} "
        f"(N={n_datasets}, tol={tolerance_per_dataset}/dset, "
        f"threshold={threshold:.3f}) -> {'PASS' if passed else 'FAIL'}"
    )
    return passed, msg


def print_sanity_summary(results, header="Sanity checks"):
    """`results` is a list of (passed: bool, msg: str). Returns aggregate pass."""
    n_pass = sum(1 for p, _ in results if p)
    n_total = len(results)
    print(f"\n=== {header} ({n_pass}/{n_total} pass) ===")
    for passed, msg in results:
        prefix = "  ok  " if passed else "  !!  "
        print(prefix + msg)
    aggregate = n_pass == n_total
    print(
        f"SANITY: {'PASS' if aggregate else 'FAIL'} " f"({n_pass}/{n_total} checks)\n"
    )
    return aggregate
