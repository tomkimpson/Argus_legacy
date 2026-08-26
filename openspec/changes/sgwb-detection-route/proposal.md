## Why

M1 (PR #116) delivered the half of the detection that was never in doubt: with the ridge GW
reparameterization and two-stage noise, Argus recovers the injected MDC2 amplitude at −0.35σ.
It did **not** deliver the half that constitutes a detection. The learned-harmonic-mean (LHM)
evidence estimator went degenerate at 68-D (max-weight-fraction ≈ 0.9, no contained shrinkage
plateau, `nan`), so there is still no calibrated `lnB(HD/CURN)` — the number the claim rests on.
The critical path is therefore the **detection statistic**, not the amplitude.

Issue #111 remains the right destination but its route is no longer the shortest one. Three of
its premises have been overtaken by results:

1. **LHM is the planned estimator.** It is now known to fail at array scale. A replacement has to
   be chosen and validated against a case with a known answer before it is trusted on real data.
2. **M2 (γ = 13/3 expressivity) is a blocking, month-long gate.** T2.4 already answered its core
   question: OU's steepest attainable slope is f⁻⁴ against the true f⁻⁴·³³; the band-referenced
   amplitude is absorbed within ~0.5σ, the spectral index is not recovered. This is a deliberate
   modelling difference — Argus fits a slightly different model of the SGWB than the standard
   power-law — not a defect to be gated on. It becomes a *quantified systematic* reported
   alongside the result, and it runs in parallel.
3. **M1 → M3 is a single leap from 33 mock to 68 real pulsars.** The 68-pulsar union grid is only
   41.6% observed, and masks do not yet work on the (now default) marginalized filter, so M3 would
   silently fall back to the slower sequential path; long runs also still lose everything on a
   walltime timeout. These are known, addressable engineering gaps that should be closed *before*
   the expensive run, not discovered during it.

## What Changes

- **Replace LHM with an evidence estimator that survives 68-D**, chosen by a short head-to-head
  bake-off on the MDC2 Stage C case (product-space hypermodel vs continuous-ORF Savage-Dickey),
  then frozen as the single M3 procedure. LHM is retained only as a low-dimensional cross-check.
- **Add an empirical false-alarm calibration** for the chosen statistic — sky scrambles and/or
  phase shifts — so the reported `lnB` carries a null distribution rather than only a
  Kass–Raftery adjective. Calibrated on MDC2 first, where the answer is known.
- **Quantify the OU-vs-power-law kernel systematic at array scale** (bias on the band-referenced
  amplitude *and* on `lnB`), and fix the claim scope to amplitude + HD correlation. Spectral index
  is explicitly out of scope for the result.
- **Close the M3 engineering gaps**: missing-observation masks in the marginalized filter (lifting
  the #113 guard / #115 deferral), and checkpoint-and-resume for long NUTS runs so a walltime
  timeout costs hours, not the whole run.
- **Stage the scale-up** rather than leaping: MDC2 33 (done, plus evidence) → a long-baseline real
  NG15 subset (~30–40 pulsars, ≥8 yr) → the full 68-pulsar wideband array, with the procedure
  frozen after the subset stage and a named gate at each boundary.
- **Supersede #111's M2 and the M1→M3 ordering**; the milestone structure below replaces them.

**Non-goals:** the joint noise+GW sampling formulation (#115), the sqrt-parallel filter (#108),
recovering the SGWB spectral index, and per-backend/ECORR/DM noise modelling.

## Capabilities

### New Capabilities
- `sgwb/model-selection`: computing a calibrated HD-vs-CURN Bayes factor from Argus posteriors at
  array scale (68-D and above), including the estimator bake-off, the validation gate against a
  known-answer case, and the frozen production procedure.
- `sgwb/null-calibration`: producing an empirical false-alarm distribution for the detection
  statistic via sky scrambles and/or phase shifts, and the significance threshold derived from it.
- `sgwb/kernel-fidelity`: measuring and reporting the systematic induced by fitting an OU kernel
  to a γ = 13/3 power-law background, on both amplitude and evidence, and the resulting bound on
  what may be claimed.
- `sgwb/array-analysis-procedure`: the end-to-end two-stage (single-pulsar noise → array) SGWB
  analysis procedure, its staged scale-up 33 → subset → 68, and the acceptance gate at each stage.
- `kalman/masked-marginal-filter`: missing-observation mask support on the marginalized
  (Rao-Blackwellized) Kalman filter path, removing the sequential-fallback guard.
- `inference/run-checkpointing`: checkpointing and resumption of long NUTS runs so partial results
  survive a walltime timeout or preemption.

### Modified Capabilities
<!-- None: openspec/specs/ is currently empty, so every capability above is new. -->

## Impact

- **Library** (`python/argus/`): `jax_kalman_filter.py` (mask support in the marginal path, removal
  of the `use_marginal` guard at lines ~719–731), `bayesian_inference.py` and
  `parameter_sampling.py` (model-index / ORF-mixing parameter for the evidence estimator;
  checkpoint hooks), `gravitational_waves.py` and `data_loader.py` (scrambled/shifted ORF
  injection for null calibration), `io_manager.py` (checkpoint I/O).
- **Workflow** (`workflows/ng15_sgwb_demo/`): new evidence, scramble and kernel-systematic scripts
  alongside `logz_lhm.py`; new configs and SLURM scripts for the subset and full-array stages;
  `TASKS.md` re-pointed at this plan.
- **Compute**: A100 (`milan-gpu`) SLURM. The dominant costs are the null calibration (many
  scrambles) and the 68-pulsar array run; checkpointing exists specifically to make the latter
  survivable.
- **Data**: MDC2 dataset 2b (staged); NG15 **wideband** par/tim → aligned feathers on the union
  grid (mask-emitting) for the subset and full-array stages.
- **Issues**: supersedes the M2 and M1→M3 sections of #111; consumes the M1.6 follow-up; leaves
  #115 (joint formulation) and #108 (parallel filter) deferred as before.
