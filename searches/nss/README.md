# NSS — Nested Slice Sampling (parked implementation)

This directory preserves the **`af.NSS`** nested slice sampler that was
previously bundled with PyAutoFit at
`autofit/non_linear/search/nest/nss/`. It was removed from the library on
2026-07-11 (PyAutoFit#1356) because the bespoke install / CI / build machinery
it required was not justified by its measured performance — but the sampler
works, and this is the full, working implementation kept here so it can be
re-mainlined cheaply once `nss` ships as a genuine PyPI package.

Like `../ultranest/` and `../pyswarms/`, `search.py` imports only from
`autofit`'s base classes, so it is a drop-in `af.NonLinearSearch`. The
intra-package imports were converted to **relative** imports (`from .samples
import NSSamples`, …), so the module works both from here (`from searches.nss
import NSS`) and verbatim if the folder is dropped back into the library.

## What NSS is

A JAX-native nested sampler whose inner sampling loop runs end-to-end inside
`jax.jit`. When the log-likelihood is itself JAX-traceable it avoids the
Python ↔ JAX boundary that Nautilus and Dynesty cross on every call. On the
production lensing likelihoods that motivated it the per-evaluation cost was
roughly **30× lower than Nautilus** (MGE). The catch — and the reason it was
parked rather than kept — is that it is **OOM-prone on pixelization / Delaunay
inversions** via the vmap fan-out, so it was never a clear across-the-board
win.

## Why it couldn't be a normal `pip install autofit[nss]`

Its real dependencies are two **pinned git forks**, and PyPI/TestPyPI reject
direct `git+https://` URLs inside an uploaded wheel
(`400 Can't have direct dependency: blackjax @ git+...`). So the `[nss]` extra
could only carry `fastprogress<1.1`, and the two forks had to be installed
manually afterwards.

### The exact install recipe (needed to run this again)

```bash
pip install "fastprogress<1.1"   # keeps fastprogress from pulling python-fasthtml (1.1.5+)
pip install \
  "blackjax @ git+https://github.com/handley-lab/blackjax.git@ef45acd2f2fa0cca15adbdcd3ff7cb3a98987cb5" \
  "nss @ git+https://github.com/yallup/nss.git@69159b0f4a3a53123b9eec7df91e4ed3885e4dc4"
```

- The **`handley-lab/blackjax`** fork carries the `blackjax.ns.adaptive.init`
  entrypoint that mainline blackjax lacks. SHA `ef45acd2` is the May 2026
  "Merge PR #60 — double_compile" revision, locally validated against the
  Phase 1–3 work.
- **`yallup/nss`** (the sampler itself) sits on top of that fork.
- This fork **conflicts with the mainline `blackjax`** pinned in autofit's
  `[optional]` extra — install NSS in its own environment, do not naively
  combine or bump the two.

## Requirement: a JAX-traceable analysis

NSS runs the likelihood inside `jax.jit`, so the analysis must be built with
`use_jax=True` (its arithmetic then dispatches through `jax.numpy`). Handing it
a plain NumPy analysis raises `TracerArrayConversionError`. See `example.py`.

## Key kwargs

| kwarg | production default | meaning |
|-------|--------------------|---------|
| `n_live` | 200 | live particles maintained throughout the run |
| `num_mcmc_steps` | 5 | slice-MCMC inner steps per dead-point batch |
| `num_delete` | 50 | particles removed per outer iteration (larger ⇒ less JIT overhead, coarser coverage) |
| `termination` | -3.0 | stop on `logZ_live - logZ` (≈ remaining evidence < 1e-3) |
| `checkpoint_interval` | — | outer iterations between resumable disk checkpoints (SLURM-friendly) |
| `iterations_per_quick_update` | None | if set, calls `analysis.visualize(...)` every N outer iterations |

## Re-mainlining checklist (when `nss` becomes a real pip package)

1. Move this folder back to `PyAutoFit/autofit/non_linear/search/nest/nss/`
   (relative imports mean the module files need no edits).
2. Restore `tests/` to `PyAutoFit/test_autofit/non_linear/search/nest/nss/`
   — the tests were kept **verbatim** against the library import paths
   (`autofit.non_linear.search.nest.nss.*`, including the `mock.patch` target
   strings), so they run unchanged from that location.
3. Re-add to `autofit/__init__.py`:
   `from .non_linear.search.nest.nss.search import NSS`.
4. Add the dependency to `pyproject.toml`. If `nss` is a real PyPI package by
   then, this is finally a plain `nss = ["nss>=X"]` extra — the whole reason
   this was removed. Otherwise restore the git+ recipe above.
5. Restore the `Search: NSS` tutorial section in
   `autofit_workspace/scripts/searches/nest.py` (see PyAutoFit#1356 diff).

Provenance: removed in PyAutoFit#1356 (2026-07-11); the removal diff is the
authoritative record of every touch-point across the six repos.
