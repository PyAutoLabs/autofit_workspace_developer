# graphical — toy graphical-model fit (scale-up baseline)

A self-contained 1D-Gaussian graphical-model example used as the baseline
for the scale-up effort tracked in
[issue #16](https://github.com/Jammy2211/autofit_workspace_developer/issues/16).

The companion scoping document
[`PyAutoPrompt/graphical_ep/graphical_scoping.md`](../../PyAutoPrompt/graphical_ep/graphical_scoping.md)
uses the profile data this package emits to rank scale-up follow-up tasks.

## Quick start

```bash
source ~/Code/PyAutoLabs-wt/<task>/activate.sh
# or, from a clean checkout:
#   export PYTHONPATH=<PyAutoConf>:<PyAutoFit>:$PYTHONPATH

cd autofit_workspace_developer

# 1. Generate 30 datasets (writes data + ground_truth.json per dataset).
python3 graphical/simulator.py --total_datasets=30

# 2. Run joint Dynesty fits at the scaling N's used by the scoping doc.
python3 graphical/fit.py --total_datasets=3  --name=N3
python3 graphical/fit.py --total_datasets=10 --name=N10
python3 graphical/fit.py --total_datasets=30 --name=N30

# 3. cProfile attribution at N=10 for the scoping doc's bottleneck inventory.
python3 -m cProfile -o graphical/profiles/N10.pstats \
    graphical/fit.py --total_datasets=10 --name=N10_cprof
```

The fit script prints a sanity-check block at the end (parameter recovery
within k·σ of the truth + max log likelihood vs the truth-evaluated LL) and
writes `profiles/<name>_summary.json` with wall time, peak RSS, output-dir
disk size, and the sanity verdict. Failures are reported but do not raise —
the goal is regression visibility, not gating.

## What's in this package

```
graphical/
  __init__.py
  util.py        - argparse, autofit-config wiring, dataset loaders
  sanity.py      - parameter recovery + max-LL checks (PASS/FAIL prints)
  simulator.py   - 1D Gaussian simulator, writes ground_truth.json per dataset
  fit.py         - joint Dynesty fit + sanity + profile_summary writer
  config/        - minimal autofit config (DynestyStatic + general/output)
  dataset/       - simulator output (gitignored)
  output/        - autofit run output (gitignored)
  profiles/      - profile_summary.json + cProfile dumps
```

The `ep/` sibling package is the EP counterpart with an identical simulator
(same seed → identical data) and the same sanity infrastructure.

## Ground truth schema

`dataset/<sample>/dataset_N/ground_truth.json`:

```json
{
  "centre": 50.0, "normalization": 0.5, "sigma": 5.0,
  "n_pixels": 100, "signal_to_noise_ratio": 25.0, "noise_sigma": 0.04,
  "truth_log_likelihood": <af.ex.Analysis.log_likelihood_function at truth>
}
```

And `dataset/<sample>/_sample/ground_truth.json` aggregates `shared_centre`,
`total_truth_log_likelihood`, and the per-dataset truth LL list. `fit.py`
uses both for its end-of-run sanity checks.

## Sandboxed runs

When running from a restricted environment, source `activate.sh` (which
sets `NUMBA_CACHE_DIR` and `MPLCONFIGDIR` to writable temp paths) or set
them manually:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib \
    python3 graphical/fit.py --total_datasets=3
```
