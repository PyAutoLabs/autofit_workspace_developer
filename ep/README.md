# ep — toy EP fit (scale-up baseline)

A self-contained 1D-Gaussian expectation-propagation example used as the
baseline for the scale-up effort tracked in
[issue #16](https://github.com/Jammy2211/autofit_workspace_developer/issues/16).

The companion scoping document
[`PyAutoPrompt/graphical_ep/ep_scoping.md`](../../PyAutoPrompt/graphical_ep/ep_scoping.md)
uses the profile data this package emits to confirm/refute the IC50 tier
breakdown and rank EP-specific follow-up tasks.

## Quick start

```bash
source ~/Code/PyAutoLabs-wt/<task>/activate.sh
cd autofit_workspace_developer

# 1. Generate 30 datasets.
python3 ep/simulator.py --total_datasets=30

# 2. EP fits at the scaling N's. Each runs the full EP loop
#    (per-factor Dynesty + LaplaceOptimiser cavity updates, max_steps=5).
python3 ep/fit.py --total_datasets=3  --name=N3
python3 ep/fit.py --total_datasets=10 --name=N10
python3 ep/fit.py --total_datasets=30 --name=N30

# 3. cProfile attribution at N=10 (the EP scoping doc's bottleneck source).
python3 -m cProfile -o ep/profiles/N10.pstats \
    ep/fit.py --total_datasets=10 --name=N10_cprof
```

Sanity at the end of each run compares the EP mean-field posterior for the
shared centre and each factor's normalization and sigma against the truth.
EP does not produce a joint max-log-likelihood, so the max-LL check is
skipped for this package.

## What's in this package

```
ep/
  __init__.py
  util.py        - argparse, autofit-config wiring, dataset loaders
  sanity.py      - parameter recovery checks (PASS/FAIL prints)
  simulator.py   - 1D Gaussian simulator, writes ground_truth.json per dataset
  fit.py         - factor_graph.optimise() EP loop + sanity + profile writer
  config/        - minimal autofit config (DynestyStatic + general/output)
  dataset/       - simulator output (gitignored)
  output/        - autofit run output (gitignored)
  profiles/      - profile_summary.json + cProfile dumps
```

The simulator is byte-identical to `graphical/simulator.py` and the
seeded RNG guarantees identical datasets on disk across the two packages,
so the EP and graphical baseline runs are an apples-to-apples comparison.

## Posterior extraction

EP returns marginal messages keyed by the original prior objects:

```python
mean_field = result.updated_ep_mean_field.mean_field
mean_field[centre_shared_prior].mean   # shared centre posterior mean
mean_field[centre_shared_prior].sigma  # shared centre posterior sigma
mean_field[model_list[i].gaussian.normalization].mean   # per-factor
```

Each message is a `NormalMessage` (or truncated variant) with `.mean` and
`.sigma` attributes.
