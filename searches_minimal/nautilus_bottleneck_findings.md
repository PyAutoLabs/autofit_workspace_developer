# Nautilus NN-training bottleneck — findings and verdict

Research task: https://github.com/Jammy2211/autofit_workspace_developer/issues/18
Scripts: `nautilus_profile.py` (anatomy), `nautilus_sweep.py` (zero-code knobs),
`nautilus_jax_mlp.py` (JAX emulator prototype). nautilus 1.0.5, `n_live=200`,
`seed=1`, laptop CPU (WSL2, 8 cores). Wall times vary ±40 % between sessions
(thermal/contention); every comparison below is within-run.

## 1. Where the time goes (Phase 1)

With a fast likelihood, Nautilus wall time is almost entirely sampler overhead,
and it is **not only the neural networks**:

| component | 3-param Gaussian fit | fast 10-d likelihood (5 µs/call) |
|---|---:|---:|
| NN ensemble training | 51.7 % | 40.2 % |
| bound geometry (ellipsoid unions + 1000-pt volume MC) | 25.2 % | 38.2 % |
| proposal sampling | 5.2 % | 18.0 % |
| likelihood (incl. transform) | 14.3 % | **0.6 %** |

Mechanics (all in nautilus source): each bound update trains one
`NeuralNetworkEmulator` **per ellipsoid** — `n_networks=4` sklearn MLPs
(hidden layers 100/50/20, `tol=0`, early stop after 10 flat epochs), each
pinned to a single thread (`neural.py`, `threadpool_limits(1)`). Training
parallelises only across the ensemble via the sampler pool (`pool_s`).

**PyAutoFit wrapper finding** (`autofit/non_linear/search/nest/nautilus/search.py`,
read-only this task): `fit_x1_cpu` — the JAX/GPU path — passes `pool=None`, so
in exactly the fast-GPU-likelihood regime every MLP trains serially.

## 2. Zero-code knobs (Phase 2)

All configs preserved quality (log Z within 0.02 nats of baseline both
scenarios, ESS ~10 000, reference log Z −57.5 for the Gaussian fit):

| config | fast 10-d | 3-param fit | note |
|---|---:|---:|---|
| `n_networks=0` (pure ellipsoid bounds) | **3.2×** | **2.0×** | +1–10 % evals; log Z −57.451 vs −57.455 |
| `n_update=2*n_live` (retrain half as often) | 3.0× | — | quality intact |
| `pool=(None, 4)` (parallel MLP training) | 2.3× | 1.1× | `os.fork()` + JAX threads ⇒ deadlock risk |
| `n_networks=2` | 2.2× | — | |
| small MLP `(32,)` | 2.0× | 1.0× | |
| `vectorized=True` (batched callback) | 2.0× | — | the hook a vmapped GPU likelihood needs |
| `n_networks=0` + `n_update=2x` combined | 2.7× | 1.9× | ≈ `n_networks=0` alone; they don't stack |

`n_networks=0` is an explicitly supported nautilus mode
(`bounds/neural.py:74`): bounds fall back to the ellipsoid union, i.e.
MultiNest-style region sampling. Its cost is sampling efficiency — here only
+1 % (10-d) to +10–27 % (3-param) more likelihood calls — which is precisely
the cheap currency in the fast-likelihood regime.

**Caveat that decided against it (see §5):** evals-to-ML on the 3-param fit
degraded 4278 → 6481 (+50 %) — the sampler takes materially longer to *find
the peak* without the NN bounds, even on an easy unimodal problem. Bounds
never bias the result (acceptance is always the exact likelihood), but on
real curved/degenerate lens posteriors at 20–30 parameters the
ellipsoid-only efficiency loss is expected to be much larger than these toy
numbers, plausibly erasing the wall-time win entirely.

## 3. JAX/GPU MLP swap prototype (Phase 3) — negative on CPU

`nautilus_jax_mlp.py` replaces the sklearn ensemble with a single jitted JAX
program (vmap over the 4 networks, full-batch Adam, padded shapes so the JIT
cache survives across bounds, jitted predict on the proposal hot path). Same
architecture/normalisation; end-to-end quality matched sklearn.

Result: **4–6× slower** than sklearn on CPU (fast 10-d: 5.3 s → 24–28 s NN
time; 3-param fit: 37 s → 118 s+). sklearn's mini-batch Adam with early
stopping does far less arithmetic than fixed-step full-batch training, and
CPU JAX gains nothing from the ensemble vmap. A GPU would flip the arithmetic
cost, but dispatch/transfer latency on thousands of tiny fits is exactly
where GPUs disappoint, and §1 caps the achievable win: NN training is ≤ 52 %
of wall, so even free training beats it by at most ~2× — less than
`n_networks=0` already delivers for zero code. **Parked**; revisit only if a
GPU-node measurement of a real lens fit contradicts this.

## 4. Can nautilus be JAX-ified wholesale? No.

The sampler's control flow is adaptive and host-side by design:
data-dependent while loops (ellipsoid splitting until a volume target,
rejection sampling until `n_batch` accepted), a dynamically growing point
store, per-bound sklearn objects, HDF5 checkpointing. Porting that to XLA's
bounded-shape world is not a port — it is writing a different sampler, which
is what NSS already is (measured 7.5× faster per eval on MGE in the A100
campaign, with its own quality/OOM caveats). The long-term answer remains a
JAX-native sampler swap; nothing found here changes that, and nothing here
requires waiting for it.

## 5. Verdict (human decision 2026-07-09)

The `n_networks=0` route was **rejected by the maintainer** on review: for
real lens-model sampling the NN bounds are what keep Nautilus's sampling
efficiency high, and the measured +50 % evals-to-ML on even a toy unimodal
fit (§2 caveat) supports the expectation that ellipsoid-only bounds would
hurt peak-finding on real curved/degenerate posteriors. No PyAutoFit config
change and no GPU validation follow-up were filed.

What stands from this investigation:

1. **The overhead anatomy (§1) is the reference** for any future Nautilus
   speed-up work: NN training 40–52 %, bound geometry 25–38 %, and the two
   must be attacked together for a win beyond ~2×.
2. **Do not wire a process pool into the JAX path** (`pool=(None, N)`):
   measured win is modest and `os.fork()` under a JAX-threaded process risks
   deadlock.
3. **The JAX/GPU MLP swap is parked with a measured negative** (§3); the
   burden of proof for reopening it is a GPU-node measurement on a real fit.
4. **The long-term answer remains a JAX-native sampler swap** (§4) — none of
   the interim knobs measured here beats staying the course on that.
5. If Nautilus overhead work is ever revisited, the next targets in order:
   bound geometry (~0.9 s/bound; profile `Union.compute`/split vs the
   1000-pt volume MC first), then early-stopping-preserving training
   parallelism that does not use process forks (threads or an async bound
   builder).
