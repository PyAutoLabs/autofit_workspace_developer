# AGENTS.md

This file is for AI coding agents (Claude Code, Codex, Cursor, etc.) and humans
working in this repository. It is the agent-agnostic source of truth; Claude Code
loads it via the `@AGENTS.md` import in `CLAUDE.md`.

## What This Repo Is

**autofit_workspace_developer** is the developer workspace for prototyping new PyAutoFit features, search-interface experiments, and minimal sampler examples. It is not a user-facing workspace — see `../autofit_workspace` for example scripts and tutorials, and `../autofit_workspace_test` for the integration test suite.

Dependencies: `autofit`, plus optional sampler backends (`nautilus-sampler`, `blackjax`, `dynesty`, `emcee`, `pyswarms`, `ultranest`). Python version: 3.11.

## Workspace Structure

```
searches_minimal/            Minimal direct-sampler examples (Nautilus, Dynesty,
                             Emcee, LBFGS) that bypass the NonLinearSearch wrapper.
                             Outputs land in searches_minimal/output/.
searches/                    Search-interface prototypes (pyswarms, ultranest).
projects/                    Example PyAutoFit projects (cosmology, ...).
scripts/                     Tutorial-style developer scripts (howtofit).
config/                      YAML configuration files (non_linear/...).
```

## Running Scripts

Scripts run from the repository root:

```bash
python searches_minimal/nautilus_jax.py
```

Each `searches_minimal/*.py` writes a standardised summary block (best fit, max log L, log evidence, wall time, evaluation count, ESS, posterior sample count) both to stdout and to `searches_minimal/output/<script_name>_summary.txt` so runs can be diffed across samplers without re-running.

**Codex / sandboxed runs**: when running from Codex or any restricted environment, set writable cache directories so `numba` and `matplotlib` do not fail on unwritable home or source-tree paths:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib python searches_minimal/nautilus_jax.py
```

This workspace is often imported from `/mnt/c/...` and Codex may not be able to write to module `__pycache__` directories or `/home/jammy/.cache`, which can cause import-time `numba` caching failures without this override.

## Line Endings — Always Unix (LF)

All files **must use Unix line endings (LF, `\n`)**. Never write `\r\n` line endings.

<!-- repos_sync:history:begin -->
## Never rewrite history

NEVER perform these operations on any repo with a remote:

- `git init` in a directory already tracked by git
- `rm -rf .git && git init`
- Commit with subject "Initial commit", "Fresh start", "Start fresh", "Reset
  for AI workflow", or any equivalent message on a branch with a remote
- `git push --force` to `main` (or any branch tracked as `origin/HEAD`)
- `git filter-repo` / `git filter-branch` on shared branches
- `git rebase -i` rewriting commits already pushed to a shared branch

If the working tree needs a clean state, the **only** correct sequence is:

    git fetch origin
    git reset --hard origin/main
    git clean -fd

This applies equally to humans, local Claude Code, cloud Claude agents, Codex,
and any other agent. The "Initial commit — fresh start for AI workflow" pattern
that appeared independently on origin and local for three workspace repos is
exactly what this rule prevents — it costs ~40 commits of redundant local work
every time it happens.
<!-- repos_sync:history:end -->
