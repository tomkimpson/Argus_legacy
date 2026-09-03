# Research log

## 2026-09-04 — Positive control on MDC2 1b FAILS: the two-stage noise procedure absorbs the GWB

**Goal.** Decide whether the low MDC2 Bayes factor (lnB = 0.175 on dataset 2b) was the data or
our pipeline, and act on the answer.

**What was tried.** Three diagnostics were queued, cheapest first, on the reasoning that a
published benchmark on a public dataset beats standing up a reference implementation.

The **literature check** came first and answered the original question outright. Hazboun et al.
(arXiv:1912.12939) report MDC2 open dataset 2b — identified exactly against our local truth
JSON on amplitude, spectral index, pulsar count, baseline, cadence and the b variant — as a
**non-detection under every search method they tried**: Bayes factors 1.1-2.6 against their own
declared threshold of 3, and an amplitude *upper limit* of <1.4e-15 against the 1.3e-15
injection. Their B is HD-vs-noise-only where ours is HD-vs-CURN; ratioing their HD and CRN rows
gives B(HD/CURN) ~ 0.88, lnB ~ -0.13, against our +0.175. Both indistinguishable from zero.

That exonerated the pipeline on 2b and, more usefully, pointed at a better test. Their dataset
1b has a *smaller* injected amplitude (0.66e-15 vs 1.3e-15) but **no per-pulsar red noise**, and
was strongly detected: B = 40 (free WN) to infinity (fixed WN), with a real amplitude
measurement 0.7 +0.4/-0.3 e-15 against 0.66 injected. Red noise is covariant with a common red
background, and that — not amplitude — is what makes 2b undetectable for everyone.

So 1b became the **positive control**: a dataset where a working pipeline must return a decisive
answer, carrying an independent published number to check against. Ingested `dataset_1b` to
feathers, staged 33 per-pulsar directories, ran the full frozen procedure — Stage A (33 jobs,
all PASS, no railing), empirical-prior extraction, then the 5-rung path-sampling ladder.

**What was learned.** **The control failed, and the failure is ours.**

    ln B(HD/CURN) = 0.0527 +/- 0.0039   [reliable, every gate passes]
    integrand flat at ~0.05 across the whole eps path (0.044, 0.051, 0.045, 0.057, 0.065)

The decisive evidence is not the Bayes factor — that comparison needs care, since their headline
B is a different and easier question. It is the **amplitude**. The prior on the pivot log-PSD is
N(-9.0, 1.333); the injected truth is -6.908, i.e. +1.57 prior-sigma from the centre. At all five
rungs the posterior sd is **95-99% of the prior sd** and the centre is shifted less than half a
prior-sigma, *away* from truth. The posterior is the prior. The published analysis measures this
amplitude; we recover no information about it at all.

**The mechanism.** Stage A fits each pulsar alone with the GW fixed at log10_ha = -20, so by
design its red-noise posterior absorbs the *total* per-pulsar red power — the config header says
exactly this. Stage C then uses those posteriors as PRIORS on per-pulsar red noise. The joint fit
therefore begins from a state where all the common power is already explained as 33 independent
noise processes; adding GW amplitude would over-explain the data, so the GW drifts back to its
prior. We subtract the signal, then look for it in the residual.
`empirical_prior_inflation = 2.0` was the intended safeguard, but it widens the priors without
moving their *centres*, and the centres are what is wrong.

The mechanism predicts the effect is worst where the GWB is a larger fraction of the red power
Stage A absorbs, and that prediction holds: 1b (GWB is all of it) gives 0.053, 2b (diluted by
real injected red noise) gives 0.175 — three times larger on the *dirtier* dataset. Stage A's own
numbers corroborate the dilution: median log10_sigma_p = -16.70 on 1b against -16.01 on 2b, with
2b higher in 24 of 33 pulsars.

**Decisions / dead ends.** **The 2b result no longer validates anything.** Its agreement with the
literature is plausibly coincidental — both near zero, for different reasons — so it cannot be
cited as evidence the pipeline works. **M1's "Stage C truth gate PASS at -0.35 sigma" is
reinterpreted as passing by being uninformative**: a +/-1.8 dex posterior covers the truth
because it covers everything. Breadth was read as success when it was the symptom. And M1 never
had a positive control; that omission is what hid this for two months.

Deliberately bounded what is *not* implicated: the likelihood (goldens bit-identical, masked and
marginal paths agree to 6e-12), the estimators (validated analytically, mutually consistent at
0.73 sigma), the sampler and ridge geometry (44 of 44 SLURM runs converged, 0% divergences,
r_hat <= 1.02), and the OU kernel mismatch (~0.2 dex, not a factor of 100). The defect is
confined to the noise-modelling procedure between the data and the likelihood.

The **prior-inflation sweep and louder-injection test, planned as the next experiments, were both
superseded** — the first because the literature said the 2b confusion was intrinsic, the second
because 1b is strictly better than a synthetic injection (it carries a published reference value).
Running the cheap literature check first saved both.

**Open threads.** The mechanism is strongly indicated but **not yet proven by removal**. The
decisive test is one configuration change: run 1b at eps=0 and eps=1 with
`red_noise_prior = flat`, dropping the empirical priors (~9 h GPU). Same 68 dimensions as the
runs that just sampled cleanly, so the sampling risk is low. If the amplitude moves from -9.5
toward -6.9, diagnosis and fix are confirmed together.

If confirmed, the noise treatment must be replaced: flat/weak per-pulsar priors sampled jointly
(the field standard — NANOGrav fixes white noise only and samples red noise jointly via PTMCMC),
or a hierarchical population prior learned from the array itself rather than pre-committed from
single-pulsar fits. Both return issue #115 (joint noise+GW formulation) to the critical path.

Worth re-testing rather than assuming: the T3.5 verdict that joint sampling is *unsamplable*
predates the ridge parameterization that fixed one of its two named pathologies, was at ~142-D
rather than 68-D, and — the part that matters most — **wide priors cost nothing in dimension**.
Flat and empirical per-pulsar priors sample the same 68 parameters at 33 pulsars; the empirical
scheme narrows priors, it does not reduce dimensionality.

The acceptance gate in `sgwb/model-selection` also needs re-pointing: from 2b (a published
non-detection, unreachable by anyone) to 1b, with a positive control made a standing requirement,
since the gate as written can be passed by an uninformative posterior.

## 2026-09-01 — Correlation-path evidence machinery; MDC2 HD-vs-CURN lnB = 0.175 (gate FAILS)

**Goal.** Replan the route to an SGWB detection (#111) into an executable plan, then execute
its critical path: replace the learned harmonic mean with an evidence estimator that
survives at array scale, and get a calibrated HD-vs-CURN Bayes factor on MDC2 dataset 2b.

**What was tried.** Planning first: the roadmap was re-cut as the OpenSpec change
`openspec/changes/sgwb-detection-route`, on the judgement that #111's flow was overtaken by
its own results — LHM was known broken at 68-D, M2's core question was already half-answered
by T2.4, and M1->M3 was a single leap from 33 mock to 68 real pulsars. The change adds an
estimator bake-off, an empirical null calibration, an intermediate NG15 subset stage, and
demotes M2 from blocking gate to parallel systematic.

Execution then took the whole critical path. Both hypotheses were embedded in one model by
interpolating the ORF, `C(eps) = (1-eps)*I + eps*C_HD`
(`gravitational_waves.correlation_path`, config `orf_path` in `[PriorModel]`), so CURN and HD
became two points of a single posterior. Two estimators were built over it: generalised
Savage-Dickey on the endpoint density ratio (one run) and path sampling over a fixed-eps
ladder (five runs). Both were validated against a shared analytic problem
(`ln Z = a*eps^3`, so lnB = a exactly) before touching real data, and both were given
reliability gates that refuse rather than report.

Supporting work that the campaign needed: masks ported into the marginalized Kalman filter
(the union-grid fallback to the sequential path was silently costing 5.4x on exactly the runs
that matter); NUTS checkpoint/resume; the sky-scramble null generator and warm starts; and
the matched power-law/OU injection pair at the MDC2 geometry for the kernel systematic.

Six A100 runs on MDC2 Stage C: five ladder rungs at eps = 0, 0.25, 0.5, 0.75, 1.0 (~5h40m
each) and one eps-sampled run (9h47m).

**What was learned.** **lnB(HD/CURN) = 0.1754 +/- 0.0171**, reliable on every diagnostic
(Romberg residual 0.0000 against a 0.1 ceiling, min integrand ESS 279). Odds 1.19:1. That
**fails the change's own lnB >= 3 MDC2 gate**, which per `sgwb/array-analysis-procedure`
blocks M3 until diagnosed.

The diagnosis is that the amplitude itself is only marginally detected: at the pure-HD rung
the pivot log-PSD is -6.714 with 16-84% [-9.837, -6.083], and **27% of the posterior lies
below -9, i.e. GW-negligible**. Amplitude lives in common auto-power that CURN reproduces
exactly; HD-vs-CURN rests only on the weaker cross-correlations. So 0.175 is arithmetically
right given the posterior, and the estimators are not at fault. This also reframes M1's
"truth gate PASS at -0.35 sigma" as a weak statement — coverage with a +/-1.76 dex error bar.

Sampling was excellent throughout: **0% divergences in all six runs**, max r_hat 1.02, all
four chains tracking together. The ridge basis plus the correlation path sample cleanly at
69-D, in sharp contrast to the direct-basis Stage B/C runs (r_hat 1.5-2.3, one chain parked
at high amplitude in every case).

Estimator A failed in a way that was not predicted. The design expected eps to pile against 1
on a signal-bearing dataset, emptying the CURN endpoint and tripping `endpoint_occupancy`.
Instead the **eps posterior came out essentially uniform** (mean 0.512, both endpoints
populated at ~5%), and A failed on fold agreement — it cannot resolve a quantity of size 0.1
to better than ~0.3. The near-uniform eps posterior is the finding in its most direct form:
the array carries almost no information about how HD-like the correlation is.

Four defects surfaced, three of them mine and one pre-existing and serious:

1. **The masked likelihood was wrong**, in the sequential mask path from PR #113. Absent
   observations get a unit-variance placeholder, but the PD jitter was scaled as
   `1e-9 * trace(S)/n` — so the placeholders, not the data, set the jitter, inflating it
   ~1e12x against ~1e-12 innovation variances. MDC2 at 41.6% occupancy: marginal-vs-sequential
   agreement went 7e-4 -> 2.3e-6; the offset from masking one pulsar out went -65.50 ->
   -11.0273, which is exactly the predicted -0.5*n*ln(2pi). Unmasked results and both goldens
   bit-for-bit unchanged.
2. Savage-Dickey at Silverman's bandwidth is biased low by ~0.16 nats (over-smoothing flattens
   the endpoint ratio); h->0 extrapolation cuts it ~4x.
3. My first discretisation diagnostic for path sampling was **vacuous**: on a uniform grid
   Simpson *is* the Richardson extrapolation of the trapezoid rule, so their difference is
   identically zero. Replaced with a genuine Romberg estimate.
4. `evaluate_integrand` vmapped every draw at once, so peak memory scaled with run length —
   200 draws worked, 400 was OOM-killed, and the 4000-draw posteriors would never have fitted.

**Decisions / dead ends.** **Path sampling frozen as the production estimator**; Savage-Dickey
kept as a cheap one-run first look. Worth recording that the pre-committed bake-off rule ("if
A and B agree within combined uncertainties, freeze A with B as audit") was **underspecified
and did not decide this case**: it assumed both estimators would be reliable and never
anticipated A agreeing on the value (0.105 +/- 0.095, 0.73 sigma from B) while refusing to
certify it. Freezing A would have adopted an estimator that declines to report in exactly the
regime the project is in, so B was frozen on the rule's intent rather than its letter.

**Product-space / hypermodel sampling was rejected**, despite being the shortlisted
PTA-favourite and the name in M1's task M1.6. The classic Carlin-Chib construction samples a
discrete model index, which NUTS cannot do, and its pseudo-priors need per-parameter tuning at
68-D — the exact regime where LHM already failed. Do not revisit it.

**LHM is not merely broken at high-D, it is biased high.** Its uncalibrated matched-shrinkage
range on this same Stage C pair was 1.8 to 8.6 against a calibrated 0.175 — an order of
magnitude, in the direction of a spurious detection. It now refuses with named diagnostics
instead of returning `nan`. The 6-psr NG15 lnB = +2.1 from T3.4 (2026-07-12) was 18-D and
passed its diagnostics, but has never been cross-checked against path sampling; it should be
re-derived on the correlation path before being quoted.

A guard I had invented — rejecting all-zero-mask pulsars outright — was removed as
contradicting the spec, and survives scoped to `diffuse` mode only, where Lambda = A really is
singular. The OU corner for the injection pair was moved from 10^-8.5 to 10^-9.0: over a 15 yr
baseline the former sits only ~4x below the lowest sampled frequency, leaving a 0.024 dex
imprint of the corner placement on a measurement meant to isolate spectral shape.

**Open threads.** The question blocking M3: **is the marginal amplitude intrinsic to MDC2 2b
at 33 pulsars, or is the two-stage empirical-prior noise treatment absorbing the common signal
into per-pulsar red noise?** Single-pulsar posteriors are known to be overconfident for exactly
this reason, which is why `empirical_prior_inflation = 2.0` exists — and whether 2.0 is enough
has never been tested. Three diagnostics, cheapest first: (1) a literature check for published
MDC2 detection statistics, since MDC2 is a public challenge and a quoted number would settle
"is it us or the data" for the cost of a reading session; (2) an inflation sweep at 1/2/4 on
the eps = 0 and 1 rungs; (3) a louder-injection scaling test using the pair built for task 5.1.

If all three say "machinery fine, data quiet", the remaining decision is a scope one: whether
M3's claim rests on amplitude plus a scramble-calibrated significance rather than a decisive
Bayes factor, which would mean rewriting the acceptance gate in `sgwb/model-selection`.

Also untested at scale: the masked marginal filter (the 68-pulsar benchmark needs M3's
feathers), the scramble generator, and the whole null-calibration campaign.

## 2026-07-18 — T3.5: full-68 NUTS ruled unsamplable; strategic pivot (full-array SGWB is a stepping stone)

**Goal.** Fire the final full-68 convergence lever — keep the dense GW block, raise
`target_accept_prob` 0.95→0.99 to suppress the 39% divergences from the previous run — and, per a
pre-committed decision rule, either launch production if it converged or stop the full-array grind
if it didn't. Framed by a pulse-check at the top of the session: the user reaffirmed that the
full-array SGWB detection is a *stepping stone*, not the destination.

**What was tried.** Edited `configs/ng15_config_full_qlblock.ini` (`target_accept_prob = 0.99`,
dense block retained), submitted job 14360470 (4×A100, 500+500, `max_tree_depth=7`). It ran clean
to COMPLETED in 22.5 h (94% of the 24 h walltime — brushed the cap as predicted but finished).
Iteration speed stayed in the depth-7 band (55–78 s/it), so the depth-10 trajectory-stall risk did
not recur. On completion read `numpyro_diagnostics/mcmc_diagnostics.txt` and computed per-chain
stats directly from the `.nc`.

**What was learned.** The 0.99 lever traded one pathology for a worse one. Divergences fell
**39% → 1.8%** — but the chains **froze**: every parameter's within-chain sd = 0 (e.g. `log10_ha`
per-chain means [−13.09, −12.84, −13.13, −12.65] with sd = [0,0,0,0]), r̂ ≈ **5.9e15**, ESS ≈ **4**
uniformly across all 352 params. Diagnosis: pushing `target_accept` to 0.99 on this ill-conditioned
geometry drove NUTS's dual-averaging step size toward zero, so nearly every proposal was a no-op —
the sampler stopped moving, which trivially suppresses divergences (a frozen chain can't diverge).
This is step-size collapse, not "needs more samples."

**Decisions / dead ends.** **Full-joint NUTS is now ruled UNSAMPLABLE on the 68-pulsar geometry.**
Three levers triangulate it: diagonal mass → ridge (r̂≈18); dense block @0.95 → 39% divergences
(curvature overshoot); dense block @0.99 → frozen chains (step-size collapse). Low accept →
divergences; high accept → no mixing; **no `target_accept` threads the needle** at ~142-D with the
h_a↔γ_a ridge + short-baseline hierarchical funnel. Fixing it would need a *different formulation*
(marginalize per-pulsar red noise à la enterprise, or a different sampler) — real research work.
Per the stepping-stone framing, that is not worth it: the 6-psr amplitude (T3.3) + 6-psr HD-vs-CURN
lnB=+2.1 (T3.4) already validate that Argus recovers the signal, and this also drops the
evidence-at-scale problem (Savage-Dickey/product-space) off the critical path.

**Open threads.** One decision deferred to the user: (a) run a converged long-baseline subset
(~30–40 psr ≥8 yr, ~1 day) as a stronger validation before pivoting, vs. (b) declare validation
done now on the 6-psr results. Either way the next real direction is the *differentiator* — joint
CW+SGWB, non-stationarity, or online/real-time inference — where the state-space formulation is the
actual nugget. Session closed by writing up the branch for a PR (honest framing: validated pipeline
+ two real-data results + a documented full-array sampling limitation).

## 2026-07-16 — T3.5: dense-mass GW block vs the h_a↔γ_a ridge (PARTIAL — linear fix works, curvature/funnel remains)

**Goal.** Fix the full-array (68-pulsar) NG15 SGWB convergence failure — the quick-look run's
`log10_ha↔log10_gamma_a` ridge left r̂≈18.6, min ESS≈4 — *before* committing ~3 days of 4×A100,
by decorrelating the GW corner with a per-block dense mass matrix (the notes' "try first" lever).

**What was tried.**
1. Wrote `scripts/diagnose_gw_ridge.py` (read-only) on the existing quick-look `.nc`: confirmed one
   outlier chain (chain 1 at `log10_ha` −12.65 vs −13.41), a **clean linear ridge** (corr +0.75),
   and that every high-r̂ `log10_σp` followed that same chain (fraction 1.00) → one problem.
2. Added a `dense_mass_blocks` config key parsed in `bayesian_inference.setup_nuts_kernel`
   (list-of-tuples → NumPyro per-block dense mass; boolean `dense_mass` still works). 7 tests pass
   incl. golden likelihood 63618.93 intact. A tiny CPU probe confirmed NumPyro accepts
   `dense_mass=[('log10_ha_prime','log10_gamma_a_prime')]` end-to-end on the real model.
3. Ran the convergence check (job 14302678, `ng15_full_qlblock`, 4×A100, 500+500, 18.7 h).

**What was learned.** The dense block did exactly its job on the *linear* problem: corr **+0.75 →
+0.14**, `log10_ha` r̂ **9.8 → 1.9**, ESS **4 → 44**, max r̂ **18.6 → 2.3**. **But divergences
EXPLODED 3% → 39%** (777/2000), concentrated in **one stuck chain (chain 1: 482/500 = 96%
divergent**; others 67/43/185). Divergence-location analysis (`scratchpad/divloc.py`) showed the
residual tracks **curvature + a mild hierarchical funnel** (weak signatures in `log10_ratio_std`
low-end / `log10_gamma_p_std` high-end), NOT linear correlation. Whitening the ridge let NUTS take
bigger steps that overshoot the *curved* OU band-amplitude/shape degeneracy → the divergence blow-up.

**Decisions / dead ends.** **A hand-rolled linear rotation reparam is now RULED OUT** — it is
mathematically equivalent to the dense block just run, so it would reproduce the same 39%
divergences. The linear fix is exhausted; the remainder is genuinely nonlinear/curvature + funnel.

**Open threads.** Next lever (not started): cheapest = keep the block + raise `target_accept_prob`
0.95 → 0.99 (config-only, attacks curvature/funnel divergences directly). Principled = a *nonlinear*
GW reparam to (band-amplitude, shape) coords in `sample_gw_parameters`. Pragmatic hedge =
long-baseline subset (~30–40 psr ≥8 yr) — the funnel/stuck chain likely lives in the short-baseline
pulsars; smaller nx, ~1 day. All code changes uncommitted on `t3.5-full-array` (worktree).

## 2026-07-12 — T3.4: HD-vs-CURN Bayes factor (PASS — lnB=+2.1, method-proving, not a detection)

**Goal.** Execute Stage-3 T3.4: get the Bayes factor between HD (Hellings-Downs
correlated SGWB) and CURN (common uncorrelated red noise, identity ORF) on the real
6-pulsar NG15 subset, by reusing the NUTS posteriors (nested sampling stays parked).

**What was tried.** Built a self-contained learned-harmonic-mean (LHM) logZ estimator
(`scripts/logz_lhm.py`, CPU-only, arviz/numpy/scipy — deliberately no `harmonic`
dependency to avoid the pinned-jax 0.4.38 fragility). A pre-planning Explore + Plan pass
surfaced the enabling discovery: the `.nc` files already store the per-draw
`log_likelihood` group *and* the unit-Gaussian latent sites (`*_prime`/`*_raw`), and NUTS
shares the same `numpyro_model` as the blackjax NS engine — so logZ is pure post-processing
in the latent space and directly comparable to the NS anchor, with no Kalman re-eval.
Target φ = shrunk full-covariance Gaussian on a train fold, estimate on a disjoint test
fold, shrinkage sweep + 2-fold swap as reliability probes. For the CURN run, `run_curn.py`
monkeypatches the data loader to replace `data["hd_correlation"]` with the identity at
runtime (no library edit; verified HD diagonal = 1, so identity is the correct CURN).
Launched CURN as a 4×A100 SLURM job cloned from the T3.3 production script.

**What was learned.**
- **Both validation gates passed.** GATE: LHM reproduced the trusted 2-D MDC2 NS anchor
  63780.97±0.16 → got 63781.09±0.02 (Δ+0.11, within 3σ), flat shrinkage plateau. Route-B
  correctness check (needed because the 2-D anchor can't exercise the 18-D extraction):
  reconstructed `log_density` matched `logprior+loglik` to <5e-7 across all 4 chains.
- **Result: HD favoured, lnB = +2.1 ± 0.1** (odds ≈8:1, Kass–Raftery "positive", NOT
  decisive). At the posterior median HD also fits +4.3 nat better than CURN. CURN converged
  cleanly (r_hat 1.002, 0.03% div) and recovers log10_ha=-13.99±0.74 — broader/lower than
  HD's -13.40±0.20, sensible since without cross-correlations more power goes to per-pulsar
  noise.
- **The matched-shrinkage trick is what makes lnB trustworthy.** Per-model 18-D LHM has low
  importance-sampling ESS (~4%, vs ~90% at 2-D — the dimensionality curse showing up early).
  But 16 of 18 dims (red-noise + hierarchical) are shared between HD and CURN, so at *matched*
  shrinkage the difficulty cancels: lnB is stable across s=0.6–0.9 (2.16, 2.24, 2.10, 1.92)
  and ~2.5× tighter than either absolute logZ. Upgraded `logz_lhm.py --compare-to` to report
  lnB from the matched-shrinkage plateau and auto-exclude degenerate low-shrink values.

**Decisions / dead ends.** Scope fixed to HD-vs-CURN only (CURN-vs-noise deferred). The TI
fallback was not needed (gate passed). Confirmed the "identity ORF" is physically the correct
CURN (not just a convenient stand-in) because the HD self-correlation is exactly 1. An early
dry-likelihood check misleadingly showed HD==CURN — that was because the test noise params sat
in a GW-negligible regime (LL≈-1340); at the real posterior median (LL≈+6490) the ORF matters
(+4.3 nat), confirming the override is live.

**Open threads.** The evidence estimator will not survive the jump to the full array (~150-D):
LHM's shrunk-Gaussian target can't contain that posterior (ESS already cratering 2D→18D).
Tom flagged Savage-Dickey as the PTA-recommended route; discussed that SD sidesteps the
evidence integral (high-D-friendly) but needs *nested* models — HD-vs-CURN would need a
reparameterized ORF (Γ=(1-ε)I+εΓ_HD, SD at ε=0), while CURN-vs-noise is a free SD from the
existing posterior. Product-space / hypermodel sampling is the non-nested alternative. To be
worked out in detail as its own task before/alongside T3.5.

## 2026-07-11 — T3.3: real-data NG15 SGWB subset run (PASS — amplitude recovered, not a detection)

**Goal.** Execute Stage-3 T3.3: run the committed production config on the REAL epoch-aligned
NG15 6-pulsar subset and verify convergence + amplitude recovery against the published SGWB.

**What was tried.** T3.3 was pure execution — T3.1 (config) and T3.2 (SLURM script) were already
committed. Pre-flight (6 aligned feathers present, milan-gpu up, `sbatch --test-only` schedules),
then submitted `slurm_scripts/ng15_production_run.sh` (job 14108726, 4×A100). Queued ~behind a
higher-priority 6-node job; ran on gina14 once a slot backfilled. Monitored via a persistent
squeue/log monitor; env pre-flight confirmed the load-bearing bits — `argus.model` resolves to
the worktree checkout and the q11 fix is live (`uses γ**2: True`), 4 CUDA devices. While it ran,
built the verification tooling: extended `scripts/compare_ou_recovery.py` with a `--published`
mode (the real run has no injection-truth sidecar, so it band-references the recovered OU
amplitude against the published NG15 power law instead — computes T_span from the feathers,
reports bias/σ + HDI overlap at f=1/(5yr) and 1/yr vs BOTH refs: fixed γ=13/3 log10_A=-14.6 and
free-γ 3.2 log10_A=-14.19).

**What was learned.** Run completed in 70m49s (GPU 85.6% avg). Convergence was *better* than the
T2.4 hi-res confirm: max r_hat 1.003, log10_ha ESS 2434, only **4 divergences/8000 (0.05%)** vs
~50 at confirm — target_accept=0.95 did its job. Recovered `log10_ha = -13.40 ± 0.20`, tight,
off both prior edges (2.4 dex clear of the -11 top); log10_gamma_a = -7.73, not railed. Band
amplitude is consistent with the fixed-γ=13/3 published SGWB to <1σ at both pivots (-0.09σ at
1/yr — essentially exact; -0.86σ at 1/(5yr)); log10_ha brackets the T2.4-mapped published -13.5
to ~0.5σ. Bonus consistency check: the fixed-γ=13/3 published PSD at 1/(5yr) (-5.747) equals the
Stage-2 injection's `log10_psd_injected` — the injection was at the same SMBHB amplitude, so the
comparison sits on validated footing. Corner plot rendered fine (the old low-dim
`utils.corner_plot` "range not valid" bug did not bite this run).

**Decisions / dead ends.** Framed the result honestly (via scientific-critical-thinking): this is
**amplitude recovery / end-to-end method validation on real data, NOT a detection.** The run
*fixed* HD in the model rather than testing it, computed no HD-vs-CURN or signal-vs-noise Bayes
factor, and 6 pulsars (15 pairs) is far short of NANOGrav's 67 (2211 pairs). A well-constrained
off-rails amplitude posterior under an assumed HD model ≠ evidence for the quadrupolar
correlation. The strict 94%-HDI-overlap flag reads False at 1/(5yr) only because the log-PSD core
is tight (published value 0.055 dex outside a narrow HDI) — a sub-σ miss, reported as such.

**Open threads.** T3.4 (HD-vs-CURN contrast) is the actual detection test — rerun with
`data["hd_correlation"]` overridden to identity (diagnostic-script override, no library edit),
Bayes factor via NUTS + posterior-reuse (learned harmonic mean first, validated against the MDC2
anchor logZ=63780; NS parked). Honesty flag stands: a decisive HD factor likely needs the full
array (T3.5); on 6 pulsars expect a weak factor / upper limit. Picking up T3.4 next session.

## 2026-07-10 — T2.6: blackjax nested-sampling GWB evidence engine (PASS)

**Goal.** Build the kill-gated T2.6 spike — a JAX-native nested sampler behind the
`run_nested_sampling` GWB stub — to get Bayesian evidence (logZ) and decide whether it can
unlock the HD-vs-CURN Bayes factor (T3.4). Was NUTS-only ⇒ no evidence ⇒ "RISK B" ceiling.

**What was tried.** Resolved deps first: `blackjax.nss` (nested slice sampling, Yallup et al.
2026) is absent from the PyPI wheels (1.4/1.5) but present in blackjax-devs `main`. Installed
`blackjax 1.6.dev --no-deps` into the `Argus` env, keeping jax pinned at 0.4.38 — the `jax>=0.9`
requirement is only for other blackjax modules; the NS code uses long-stable APIs. One shim needed
(`jax.shard_map = jax.experimental.shard_map.shard_map`) because the blackjax package `__init__`
(via `eca.py`) imports the top-level `jax.shard_map` promoted only in jax≥0.5. Argus imports
blackjax nowhere else ⇒ zero blast radius. Implemented `run_blackjax_nested_sampling` +
`_blackjax_ns_evidence` + `_import_blackjax_ns` in `bayesian_inference.py`, dispatched via
`sampler=blackjax` in `workflow.py`. Key simplification: every free `numpyro.sample` site in the
GWB model is `Normal(0,1)` (physical params are `deterministic` transforms), so the NS prior is an
isotropic unit Gaussian — Jacobian-free — and the likelihood is `numpyro.log_density(model) −
unit_normal_logprior`, reusing the model with no transform re-implementation. Validated in three
gates: analytic Gaussian logZ (d=2,5,15); then the GWB likelihood on GPU (A100) — the venue, since
the joint 32-pulsar Kalman likelihood is ~5 s/eval on CPU but 0.075 s on GPU.

**What was learned.** All three gates pass. On MDC2 dataset_2b (noise fixed → 2 free GW params,
which has a NUTS baseline) the NS posterior reproduces NUTS *exactly* — log10_ha −12.880±0.050 vs
−12.881±0.046, log10_gamma_a −8.106±0.131 vs −8.081±0.125 — and returns logZ=63781±0.2 (NUTS gives
none). Three real issues surfaced only by running it: (1) a numerical pathology — free slice
exploration reaches a latent-tail region where the Kalman innovation covariance is near-singular,
giving a spurious ~2.2e6 log-likelihood that a 33×33 grid probe localised to |z|=6.5 (outside a 6σ
box); fixed with a bounded 6σ latent prior (evidence unchanged, ~2e-9/dim mass loss) + non-finite
guard. (2) The termination guard `i > n_live` forced ≥500 steps under batch deletion (bug); fixed
to `i > 1`. (3) Posterior extraction OOM'd the 80 GB A100 (~106 GB) because `to_physical` traces
the full model (runs the likelihood) and `jax.vmap` over thousands of draws materialised all
intermediates; fixed with sequential `jax.lax.map`. Confirmed from source that `blackjax.nss` is
vectorised — the inner slice kernel is `jax.vmap`-ed over `num_delete` particles
(`blackjax/ns/from_mcmc.py:108`), so the GPU-batch width *is* `num_delete`: `num_delete=1` ran
serial (>2 h, unfinished), `num_delete=25` hit 95% GPU util.

**Decisions / dead ends.** Env: installed into the shared `Argus` env in-place (user's call) but
`--no-deps` so nothing cascaded; the feared jax 0.10 upgrade was avoided entirely. Validation
target: had to substitute MDC2 for the task's preferred OU-injected synthetic — the OU feathers are
gitignored and unrecoverable (no raw feathers anywhere; the ingest step needs `enterprise` and
isn't a committed workflow script). MDC2 gives an exact NUTS cross-check, so the substitution is
sound. The 2-D MDC2 problem is *not* representative of cost on the real ~15–20-D hierarchical model.

**Open threads.** (1) NS cost scaling with dimension D, N_pulsars, and (num_live, num_delete,
num_inner_steps) — flagged by Tom as required before committing NS to T3.4; slice-NS scales worse
with D than gradient NUTS. (2) `utils.corner_plot` errors on 2-param GWB posteriors ("range not
valid") — the workflow's built-in per-run corner plot failed, worked around by plotting from the
saved `.nc`. (3) Timing: NS ~23 min sampling / ~27 min total vs NUTS ~5/~13 on the 2-D MDC2 problem
at num_delete=25 (not cost-tuned) — widening num_delete is the lever.

## 2026-07-08 — SGWB injector (T2.1), a get_Q_block bug, and MDC2 GPU re-validation (T0.1)

**Goal.** Build the Stage-2 GWB injector (`workflows/ng15_sgwb_demo`, T2.1): inject a
synthetic HD-correlated SGWB into the epoch-aligned NG15 feathers to de-risk the central
question — can Argus's single-corner OU recovery absorb a true power-law GWB without biasing
the amplitude?

**What was tried.** Designed `scripts/inject_powerlaw_gwb.py` (CPU, numpy-only) with two
modes: `powerlaw` (true `f^-13/3` via a frequency-domain Fourier-sum GP, enterprise PSD
convention) and `ou` (forward-sim of Argus's own OU generative model as the self-consistent
control). Both replace residuals with pure-synthetic signal on the real geometry, keep all
other feather fields, add fixed-value white noise, and record injected truth (PSD at the
Fourier freqs + pivot amplitudes) for a shape-agnostic comparison. Framing agreed with the
user first: neither power-law nor OU is "the truth"; a PTA only constrains ~1 decade of
frequency, so the robust observable is the band-referenced amplitude, not the spectral index.

Building the OU control exposed a bug: its residuals came out ~500× larger than the
power-law injection (75–294 µs vs 200–520 ns). Traced to `python/argus/model.py`
`get_Q_block`: the integrated-OU **position** process-noise `q11` was divided by `γ**3`
instead of `γ**2`, inflating it by exactly `1/γ` (~1e9 at PTA `γ~1e-9`). Confirmed against the
exact integral `∫₀^dt[(1-e^{-γτ})/γ]²dτ` three ways (series, quadrature, `dt³/3` limit);
`q12`/`q22` were correct. Fixed on an isolated branch `fix-qblock-q11-normalization` (PR #101
to `main`), merged into `ng15-sgwb-demo`; added regression tests and bumped the MDC2 golden
log-likelihood (55963.86→63618.93). With the corrected `q_block`, the OU control naturally
matched the power-law RMS at the default `log10_ha=-14.35`.

Then re-validated on GPU (T0.1): a lite MDC2 GWB+HD+NUTS run (only `log10_ha`/`log10_gamma_a`
sampled; red+white noise fixed). Took four SLURM submissions to get a valid run.

**What was learned.** The fix is confirmed on GPU: likelihood 63618.81, 0 divergences,
`r_hat` 1.00–1.01, robust interior posterior `log10_ha≈-12.88`, `log10_gamma_a≈-8.08` (narrow
and widened priors agree → genuine mode, not a runaway). Crucially, the recovered amplitude
shifted from the buggy run's −15.5 to −12.88 because the fix changes the `ha`→residual-amplitude
scaling: r-noise is now `∝ ha²·γa` (was `∝ ha²`, `γa` cancelled). This is exactly the
`(ha,γa)`↔physical-amplitude mapping the PLAN said Stage 2 must establish — the bug had been
corrupting it. Consequence: all Stage-2/3 `log10_ha` priors must re-centre to ~−12…−13.

**Decisions / dead ends.** (1) First SLURM job failed instantly (0% resource, 63 s) — `set -e`
aborts on the benign non-zero return from `~/.bashrc`/conda-init before any output; removed it
(existing example scripts omit it too). (2) `milan-gpu` briefly flapped `PartitionDown`; a
`--partition=milan-c` override was silently ignored (site GPU-routing forces `milan-gpu`), but
it came back up. (3) The first "COMPLETED" run was a FALSE PASS: it printed the *old* likelihood
55963.87 because `argus` is pip-installed **editable** pointing at the main checkout
`/fred/oz022/tkimpson/Argus` (no fix), and `run_analysis.py` only `sys.path.append`s the repo
`python/` dir — the append loses to the editable install. So GPU runs silently ignore
treehouse-worktree edits. Worked around with `export PYTHONPATH=<worktree>/python` (prepends) +
a log line proving `argus.model.__file__` and the q11 divisor. This contradicts PLAN §3's
"Argus is not pip-installed" claim (to be corrected). The clean long-term fix is merging PR #101
so the editable target serves the fix.

**Open threads.** PR #101 awaiting merge. Next task T2.2 (lite injection-recovery config →
`data/inject_powerlaw`/`data/inject_ou`) needs the re-centred `log10_ha` prior and, until #101
merges, the PYTHONPATH hack in its SLURM script. Then T2.3/T2.4 are the actual OU-vs-power-law
decision gate. The injector's red-noise mode is built but off by default (needs per-pulsar
γp/σp, not in `ng15_psr_noise.json`).

## 2026-06-09 — /simplify pass on the CW inference PR (quality only, behavior-preserving)

**Goal.** Reduce duplication/complexity in the continuous-waves branch's CW additions
(~2.5k lines across `python/argus/`) without changing numerical results, inference, or speed.

**What was tried.** Ran a 4-angle review (reuse / simplification / efficiency / altitude) over
`git diff main...HEAD`, verified every finding against the live source, then applied only the
safe, behavior-preserving ones. Built a strict equivalence harness that captured 21 reference
arrays before any edit (both `compute_cw_signal_single_pulsar` variants × 3 branches, all
numpyro sample-site values from a seeded `Predictive` draw, and the jaxns CW prior-name order
obtained by manually driving the generator and intercepting `yield`ed `Prior.name`s).

Changes applied: (1) collapsed the CW scalar reparameterize-or-fix block — copy-pasted 3× in
`parameter_sampling.py` (numpyro `sample_cw_parameters`, jaxns `build_jaxns_cw_prior_model`,
and the key list in `count_free_parameters`) — into one module-level `CW_SCALAR_PARAMS` spec +
a `_sample_cw_scalar_numpyro` helper; removed a dead `import math`. (2) Extracted shared
`_cw_earth_term` from the two CW waveform functions in `gravitational_waves.py`. (3) Deleted the
dead SMC / replica-exchange diagnostics block in `workflow.py` (modules removed in c553794;
`smc_results`/`re_results` never assigned). (4) Minor: `KPC_TO_SECONDS` → module constant,
hoisted two in-`@jax.jit`-body imports, deduped `is_cw` in `utils.py`.

**What was learned.** Equivalence held exactly (21/21 arrays at rtol=1e-12, atol=0); full test
suite 227 passed (CW subset 62). Warm-jit A/B microbenchmark of the waveform functions:
before 259.8/255.3 µs, after 259.3/264.0 µs per call — within ±2-3% run-to-run noise, i.e. no
hot-path cost (jit inlines the helper to the same XLA graph).

**Decisions / dead ends.** Deliberately shipped **none** of the efficiency-agent's hot-path
rewrites. Precomputing polarization tensors is a no-op (XLA LICM already hoists the
loop-invariant out of the antenna vmap). Precomputing `n_hat`/`geometric_factors` at init is
impossible — they depend on `alpha_gw`/`delta_gw`, which are *sampled*, so they change every
likelihood call (the efficiency agent's premise was wrong). `R_scalars` masking and
`pulsar_direction` precompute are immeasurable against the per-pulsar Kalman core that
dominates `_cw_likelihood`. Doing nothing there is what guarantees "as fast or faster." Larger
altitude ideas (a `KalmanFilter`/`Likelihood` base class, a `PriorSpecification` dataclass)
were left out as out-of-scope architecture.

**Open threads.** Pre-existing working-tree edits to `test/conftest.py` and
`test/test_data_loader.py` were present at session start and are unrelated to this pass. PR on
`continuous-waves` not yet merged to main.

## 2026-06-02 — Validated the Argus CW likelihood; f_gw grid recovers the injection

**Goal.** Settle whether the Argus Kalman-filter CW likelihood agrees with the standard
PTA tool (enterprise) on IPTA MDC2 dataset_3b, then attack the pulsar-term f_gw
multimodality with a fixed-f_gw grid search — as a step toward running on real data.

**What was tried.**
1. Built a "Level-1" agreement gate (`workflows/cw_shared/level1_likelihood_agreement.py`):
   sweep each of the 7 CW params through the injection in both codes (Argus earth-term vs a
   local fixed-noise earth-term enterprise PTA) and compare normalized 1-D profile shapes.
2. The gate appeared to show Argus failing (shallow profiles, peaks off injection) while
   enterprise peaked at injection. Treated as a bug and ran systematic debugging.
3. Ruled out hypotheses one at a time: (a) my flat default noise — refuted, the existing
   Level-3 NUTS posterior misses the injection identically and fit sigma_p->~1e-18;
   (b) red-noise modelling — refuted by `diag_noise_hypothesis.py`: enterprise with red
   noise OFF still recovers the injection; (c) waveform amplitude — verified correct
   (Argus CW signal RMS ~0.45us, matches analytic h0/(2*Omega) x antenna patterns).
4. Looked at scales: data residual RMS ~10us, white noise ~4.6us, CW signal ~0.45us ->
   matched-filter SNR ~7. Argus profile depth (~30) matched that; my enterprise profiling
   depth (~7000) was ~100x too confident. Checked the official Level-2 PTMCMC posteriors:
   BOTH Argus and enterprise detect, nail log10_f_gw, and land on the SAME biased sky mode
   (~alpha=1.46), missing the injection together.
5. Wrote a controlled injection test (`workflows/cw_shared/controlled_injection_test.py`):
   inject a loud known CW (SNR~50) into clean white-noise residuals on the real geometry,
   same residuals to both codes, check peak recovery + detection statistic vs SNR^2/2.
6. Built and ran the f_gw grid (`workflows/cw_level4b_fgw_grid/`): 40 fixed-f_gw,
   pulsar-term, phase-reparam NUTS runs on A100s (SLURM array 12746776), then aggregated
   into a max-logL profile + detection statistics.

**What was learned.**
- **Argus's CW likelihood is correct and correctly normalized.** In the controlled test it
  recovers all 7 params essentially exactly and gives a detection statistic = 0.95 x the
  theoretical SNR^2/2 (the 5% is timing-model marginalization). The Kalman-filter likelihood
  is sound.
- The MDC2 "failure" is **real-data hardness, not a bug**: marginal SNR (~7) + injected
  power-law red noise + genuine multimodality of single-source CW sky/incl/psi localization.
  Both official codes agree with each other and miss the injection together.
- **The f_gw grid works**: the max-logL profile shows a sharp clean peak at the injection
  (peak at log10_f=-8.28 vs injection -8.215), ~80-120 logL above a flat baseline, with
  divergences lowest at the peak and high off-peak (sampler thrashes at wrong f).

**Decisions / dead ends.**
- The Level-1 1-D-profile gate is the WRONG validation for a multimodal likelihood: sweeping
  through the injection only tests whether the injection is a *local* mode, missing deeper
  modes. Superseded by the controlled injection test as the validation of record.
- My enterprise comparison harness (fixed-noise white-only PTA + overriding `psr._residuals`)
  is UNRELIABLE — in the controlled test it gave a *negative* detection statistic
  (impossible), because overriding residuals leaves enterprise's internal caching/design
  matrix inconsistent. Do not trust it; the trustworthy enterprise reference is the official
  Level-2 PTMCMC run on real data.
- The per-point Savage-Dickey BF in the grid aggregator is confounded by pulsar-term chi-phase
  overfitting (reads inf at low f). Use the max-logL profile as the detection signal.
- Grid index 15 (-8.231, the exact-injection point) timed out twice (slowest geometry) but is
  bracketed by 14 and 16, so the peak is fully resolved; not worth backfilling.
- Path gotcha: grid configs live in runs/ (one level deeper than level4a's config), so the
  template's `../data` relative path failed; `generate_configs.py` now absolutises data paths.

**Open threads.**
- The sky/inclination/polarization multimodality is still unaddressed (inherent to single
  source at this SNR; the grid fixes frequency only). Would need restarts or a tempered
  angular kernel for full recovery.
- Real-data pivot (the actual goal): target NANOGrav 15yr. Prerequisites surfaced —
  ECORR/jitter (Argus white noise = (efac*err)^2 + equad^2, no jitter term), noise
  marginalization (OU process can't fit steep power-law red noise -> fix noise from
  single-pulsar runs), and the detected GWB foreground (must be modelled or it biases the CW
  search). Recommended first step: single-pulsar NANOGrav 15yr ingestion test to check the
  loader handles real ECORR/DMX before a full search.
