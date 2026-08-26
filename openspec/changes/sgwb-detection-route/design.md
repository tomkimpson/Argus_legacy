## Context

See `proposal.md` for motivation. What already exists and constrains the approach:

- **Working array analysis.** Two-stage noise (single-pulsar → array with empirical priors) plus the
  `gw_parameterization = ridge` pivot-log-PSD basis converges cleanly at 33 pulsars: Stage C reached
  r̂ ≤ 1.004 with 0% divergences in ~9.5 h on A100, and covered the injected amplitude at −0.35σ.
  This part of the machinery is done and is not revisited here.
- **No usable detection statistic.** `workflows/ng15_sgwb_demo/scripts/logz_lhm.py` (learned harmonic
  mean) works at 2-D and 18-D but is degenerate at 68-D. It stays as a low-D cross-check only.
- **Filter paths.** `_run_kalman_filter_scan` (sequential, augmented state) supports masks via
  `H_eff = M H`, `R_eff = M R Mᵀ + diag(1−M)`; `_run_kalman_filter_marginal` (Rao-Blackwellized over
  the timing model, the default and 1.4–2× faster) does not. `jax_kalman_filter.py:719–731` currently
  auto-falls back to sequential on masked data and raises if the marginal path is demanded.
- **The ORF is an input matrix.** `data["hd_correlation"]` is built once in `data_loader` from pulsar
  sky positions and handed to the filter. Swapping it is a one-line intervention — which is what made
  the earlier CURN run possible, and is what makes both sky scrambles and a correlation-path
  extension cheap to implement.
- **Compute reality.** OzSTAR A100 on `milan-gpu`; NUTS chains run *sequentially* per GPU, so 4 chains
  want `gpu:4`; jobs are walltime-capped and Argus currently writes results only after sampling
  completes, so a timeout loses the whole run. The 68-pulsar union grid is ~41.6% occupied.

## Goals / Non-Goals

**Goals**

- One model extension that yields *both* candidate evidence estimators, so the bake-off is a choice
  between estimators rather than between implementations.
- A null calibration that is a genuine time-domain analogue of the field's standard, and whose cost
  is bounded well enough to actually run.
- Enough engineering headroom (masks on the fast path, checkpointing) that the 68-pulsar run is a
  scheduling problem, not a research problem.

**Non-Goals**

- No change to the likelihood, the OU generative model, the ridge parameterization, or the two-stage
  noise design. All goldens stay intact.
- No joint noise+GW reformulation (#115) and no parallel filter (#108).
- No attempt to make the OU kernel reach γ = 13/3; the mismatch is measured, not removed.

## Decisions

### D1 — One correlation path, two estimators, instead of a product-space hypermodel

Task M1.6 named a *product-space hypermodel*. Rejected: the classic Carlin–Chib construction samples
a **discrete** model index, which NUTS cannot do, and its pseudo-priors need per-parameter tuning
that is unmanageable at 68-D — exactly the regime where LHM already failed.

Instead, embed both models in one continuous family by interpolating the overlap reduction function:

```
C(ε) = (1 − ε)·I + ε·C_HD ,    ε ∈ [0, 1]
```

`ε = 0` is CURN exactly, `ε = 1` is HD exactly, and every other parameter, prior and data input is
shared. From this single extension come two estimators:

- **(A) Generalised Savage–Dickey — one run, cheap.** Sample `ε` alongside everything else under a
  uniform prior and read off
  `lnB(HD/CURN) = [ln p(ε=1|d) − ln p(ε=1)] − [ln p(ε=0|d) − ln p(ε=0)]`.
  Valid because the nuisance priors are identical at both endpoints. Weakness: when the data strongly
  favour HD, `p(ε=0|d)` is in the far tail and its density estimate is the whole answer — the same
  tail-estimation failure mode that killed LHM, so it must be diagnosed, not assumed.
- **(B) Path sampling / thermodynamic integration — many runs, robust.** Fix `ε` on a ladder and
  integrate `lnB = ∫₀¹ E_ε[∂ ln L/∂ε] dε`. The integrand comes free from JAX autodiff. Each `ε` is an
  independent NUTS run, so the ladder is embarrassingly parallel across GPUs, and the estimator has
  no tail-density step. Cost is the ladder length × array-run cost, and the discretisation error is
  measurable by refining the ladder.

The bake-off is therefore A vs B on the MDC2 Stage C case, where a decisively positive answer is
expected. If they agree, A is frozen as the production procedure (one run) with B retained as the
periodic audit; if they disagree, B is frozen and A is discarded. This is strictly better than the
original plan: shared implementation, mutual cross-check, and no discrete sampling.

Bonus: `p(ε | d)` from run A is itself an interpretable "how HD-like is the observed correlation"
measurement, independent of any evidence machinery.

**Alternatives considered.** Nested sampling — already parked (10–30× slower than NUTS, decision of
2026-07-10); do not revisit. Bridge sampling — needs two well-overlapping posteriors and its own
tail diagnostics; strictly harder than B for no gain here.

### D2 — Sky scrambles, not phase shifts, as the null

Phase shifts randomise the phases of Fourier coefficients — there is no clean analogue in a
time-domain state-space likelihood, and inventing one would need its own validation. Sky scrambles,
by contrast, are a natural fit: draw random sky positions, rebuild the ORF, and pass the new matrix
in at the same seam CURN already uses. Per-pulsar noise, epochs and residuals are untouched, which is
exactly what the spec requires.

Scrambled ORFs are accepted only below a match threshold against the true HD matrix
(the standard `|C_scr·C_HD| / (|C_scr||C_HD|)` criterion), so the ensemble is quasi-orthogonal to the
true correlation and does not accidentally re-test HD.

### D3 — The null ensemble is made affordable by warm-starting, not by cutting corners

A naive ensemble is unaffordable: 100 scrambles × 9.5 h × 4 GPUs is ~4000 GPU-hours at 33 pulsars and
worse at 68. Three levers, in order of preference:

1. **Skip warmup.** Every scramble shares geometry with the real run: reuse the tuned step size and
   mass matrix from the accepted run and sample without re-adaptation. Warmup is roughly half the
   wall-clock, and the adaptation is the part that does not need repeating.
2. **Size chains by the statistic, not by the posterior.** The ensemble needs the *statistic* to a
   useful precision, not per-parameter ESS ≥ 400. Chains are sized from a measured
   statistic-variance-vs-length curve on a handful of pilot scrambles.
3. **Calibrate at 33 pulsars, extrapolate deliberately at 68.** The full ensemble is run on MDC2,
   where it also validates the machinery. At 68 pulsars the ensemble is sized to whatever the compute
   allows, and the p-value is reported at that resolution with the limitation stated — which the
   `sgwb/null-calibration` spec already requires rather than papering over.

Estimator A (one run per realisation) is what makes this feasible at all; if the bake-off forces
estimator B, the null calibration is re-costed before the full-array stage, and may be reported at
33 pulsars only.

### D4 — Masks on the marginal filter mirror the sequential trick

Reuse the sequential path's approach rather than inventing a second one: zero the masked rows of `H`
and inflate the corresponding measurement variances, keeping array shapes static so `scan` and JIT are
unaffected and no gather/pad is needed. In the marginalized filter the same masking must be applied
consistently to the timing-model information accumulators, not only to the innovation — that is the
one place where the port is not mechanical and where the all-ones-mask golden test earns its keep.

Correctness is pinned three ways (see `kalman/masked-marginal-filter`): all-ones mask reproduces the
existing goldens bit-for-bit, masked marginal agrees with masked sequential at matched prior settings,
and an all-zeros pulsar contributes only a parameter-independent constant.

**Alternative considered.** Leaving the sequential fallback in place for M3. Rejected: it silently
gives up the marginal filter's 1.4–2× speedup on the single most expensive run of the project, and
the fallback is only reached on real data — precisely where the cost matters.

### D5 — Checkpointing by chunked sampling, not by intercepting the sampler

NumPyro exposes `post_warmup_state`, so a run can be executed as a sequence of segments, saving
sampler state plus accumulated samples after each. This is the supported idiom and touches only the
run harness. Rejected: host callbacks inside the sampling loop (fragile under JIT and across the
pinned JAX version).

Resume refuses on any model/sampler configuration mismatch, and partial output is written with an
explicit incomplete marker so it can never be mistaken for a finished run.

### D6 — Scale up in three gated stages, freeze after the second

`MDC2 33 (known truth) → NG15 wideband long-baseline subset (~30–40 pulsars, ≥8 yr) → NG15 wideband 68`.

The subset stage is the addition to #111's route. It is the first time the frozen procedure meets real
data — real noise, real cadence, real masks — at a scale that still runs in about a day, and its
amplitude is checkable against the published result. Discovering a real-data problem there costs a
day; discovering it in the 68-pulsar run costs the run. The subset also isolates the funnel/stuck-chain
behaviour previously attributed to short-baseline pulsars.

The procedure freezes when the subset stage passes its gate. The full-array run then differs only in
dataset, pulsar list and resources, and records a config diff proving it.

### D7 — Kernel systematic measured from matched injection pairs

`scripts/inject_powerlaw_gwb.py` already injects either a γ = 13/3 power law or Argus's own OU process
on a given geometry. Run the matched pair through the *frozen* end-to-end procedure at the MDC2
geometry and report the difference in band-referenced amplitude (dex and σ, at both pivots) and in
`lnB`. This turns the known f⁻⁴-vs-f⁻¹³ᐟ³ ceiling into two numbers that ship with the result, and it
runs in parallel with everything else because it depends only on the frozen procedure.

## Risks / Trade-offs

- **Both estimators fail at 68-D** → The ladder-based estimator B has no tail-density step, so its
  failure would be discretisation error, which is measurable and fixable by refining the ladder.
  Detected on MDC2, where the answer is known, before any real-data cost is incurred.
- **The null ensemble is unaffordable at 68 pulsars** → Reported at the resolution the compute
  supports, with the bound stated (`p < 1/N`), and the full-resolution calibration banked at 33
  pulsars. The claim degrades gracefully instead of silently.
- **Warm-starting biases the null ensemble** → Validated against a small number of
  full-warmup realisations before the ensemble is trusted; if they disagree, warm-starting is dropped
  and the ensemble shrinks.
- **Masks on the marginal path change the likelihood subtly** → Three independent correctness pins,
  one of which is the existing golden. If they cannot be satisfied, the sequential fallback stays and
  M3 pays the speed penalty — a cost, not a blocker.
- **The intermediate subset finds a real-data problem** → That is the point of the stage; it is a
  day, not a week, and it is discovered before the expensive run.
- **68-pulsar geometry is harder than 33 even with the ridge basis** → The ridge basis was designed
  against exactly this ridge at 2-D and 68-D; if the array run still fails to converge, the deferred
  joint formulation (#115) is promoted, and the subset result is banked as the reported result.
- **Compute contention** → `milan-gpu` has had backlogs and outages. Checkpointing (D5) exists partly
  for this; long runs must survive requeue.

## Migration Plan

Branch per milestone off `main`, PR each, in this order — the first three are independent and can
proceed in parallel:

1. `feat/orf-path-evidence` — the `ε` correlation path plus both estimators, validated on MDC2.
2. `feat/masked-marginal-filter` — mask port, guard removal, goldens intact.
3. `feat/nuts-checkpointing` — chunked run harness with resume.
4. `feat/sgwb-null-calibration` — sky scrambles and the p-value machinery, calibrated on MDC2.
5. `m2-ng15-subset` — long-baseline real subset; the procedure freezes at the end of this branch.
6. `m3-ng15-full-array` — the frozen procedure on 68 pulsars, plus the write-up.

No rollback concerns: everything new is behind configuration, the existing filter paths and goldens
are unchanged, and each milestone's outputs are additive.

## Open Questions

- Exact `ε`-ladder length and spacing for estimator B — decided from the measured
  `E_ε[∂ ln L/∂ε]` curvature on the MDC2 pilot; does not change the specs or the task breakdown.
- Which specific NG15 pulsars form the long-baseline subset — falls out of the ≥8 yr baseline cut
  applied to the wideband release during preparation.
- Whether the null calibration also needs a CURN-injected mock (as opposed to a no-signal mock) as a
  second validation point; deferrable until the MDC2 calibration is in hand.
