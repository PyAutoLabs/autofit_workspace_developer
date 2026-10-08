# autofit_workspace_developer

Developer workspace for [PyAutoFit](https://github.com/PyAutoLabs/PyAutoFit):
search prototypes, minimal sampler examples, expectation-propagation and graphical
model experiments.

## The `searches/` archive

`searches/` holds non-linear search implementations that are **not** in PyAutoFit,
either because they were removed from the library or because they have not been
promoted into it. They are preserved here so they are not lost and can still be
used by advanced users.

A search lives in the archive only while PyAutoFit does not ship it. When a search
is (re-)mainlined into PyAutoFit, its archive copy is deleted rather than kept in
parallel, so the library is the one maintained implementation. NSS is an example:
it was parked here and is now `af.NSS` in PyAutoFit.

`searches_minimal/` is not part of the archive: it holds minimal scripts that call
samplers directly, bypassing the `NonLinearSearch` wrapper, for prototyping and
benchmarking.

## Archived searches

### UltraNest

Reactive nested sampling via [UltraNest](https://github.com/JohannesBuchner/UltraNest).

- Source: `searches/ultranest/search.py`
- Example: `searches/ultranest/example.py`

### PySwarms

Particle swarm optimisation via [PySwarms](https://github.com/ljvmiranda921/pyswarms).
Includes global-best and local-best variants.

- Source: `searches/pyswarms/abstract.py`, `globe.py`, `local.py`
- Example: `searches/pyswarms/example.py`

## Requirements

These searches require `autofit` to be installed. They import from
`autofit.non_linear.search` base classes.

```bash
pip install autofit
pip install ultranest    # for UltraNest
pip install pyswarms     # for PySwarms
```

## Extracted From

PyAutoFit at commit on `main` branch, April 2026.
