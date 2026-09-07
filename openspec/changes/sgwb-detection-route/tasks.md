> Environment: `conda activate Argus`. CPU work prefixed `JAX_PLATFORMS=cpu`; all NUTS work via
> SLURM on A100 (`--partition=milan-gpu --gres=gpu:N`, N = num_chains). Branch per group (see
> design.md — Migration Plan); commit only via `/commit`; no AI attribution anywhere.
> Groups 1–3 are independent and can run in parallel.

## 1. Correlation path and the evidence bake-off (branch `feat/orf-path-evidence`)

- [x] 1.1 Add a correlation-path option that replaces the ORF with `C(ε) = (1−ε)·I + ε·C_HD`, with `ε`
      either fixed by config or sampled under a uniform prior; verify by unit tests showing `ε=1`
      reproduces the HD matrix exactly, `ε=0` the identity, and that the MDC2 golden log-likelihoods
      (63618.93 informative, 59420.06 diffuse) are unchanged when the path is off.
- [x] 1.2 Verify the `ε=0` path is numerically identical to the existing CURN construction
      (`run_curn.py`'s identity-ORF override) by comparing log-likelihoods at a fixed parameter vector.
- [x] 1.3 Implement estimator A (generalised Savage–Dickey on sampled `ε`) as a post-processing
      script over a saved posterior, emitting `lnB`, its uncertainty, the endpoint density
      diagnostics and a `reliable` verdict; verify on a synthetic posterior with a known analytic
      answer.
- [x] 1.4 Implement estimator B (path sampling over a fixed-`ε` ladder) as a post-processing script
      over a set of runs, emitting `lnB`, ladder-discretisation error and a `reliable` verdict;
      verify on the same synthetic case as 1.3 and confirm the two agree.
- [x] 1.5 Add the reliability gate to `logz_lhm.py` so it returns `reliable: false` with the named
      failing diagnostic instead of `nan`; verify it reports `reliable: false` on the stored 68-D
      MDC2 Stage C posterior and `reliable: true` on the 2-D MDC2 anchor.
- [x] 1.6 Run estimator A on MDC2 Stage C (single array run with `ε` sampled, ridge basis, empirical
      priors) and record `lnB`, `p(ε|d)` and diagnostics; verify the run converges (r̂ ≤ 1.01 on
      sampled sites, divergences ≤ 1%).
- [x] 1.7 Run the estimator B ladder on MDC2 Stage C (pilot ladder first, then refined from the
      measured `∂lnL/∂ε` curvature); verify the discretisation error is smaller than the reported
      `lnB` uncertainty.
- [x] 1.8 Decide the bake-off in writing: if A and B agree within combined uncertainties, freeze A as
      the production procedure with B as audit; otherwise freeze B and discard A. Verify by a
      committed decision note recording both numbers and the rule that was applied.
- [ ] 1.9 Run the frozen estimator on the MDC2 no-injection control and verify it returns `lnB`
      consistent with zero and `reliable: true` — the negative control the injected case cannot give.
      NOT closed by the 4.2b sky scramble: a scramble mis-describes a correlation that is present, so
      it lands below zero by construction. This needs a dataset with no correlated signal, and MDC2
      group1 has none (dataset1/2 are GWB, dataset3 is a CW source) — so it needs a synthesised
      signal-free set at the 1b geometry, or an explicit decision to drop it.
- [ ] 1.10 Record the frozen evidence configuration as a versioned config file plus a one-line
      provenance stamp written into every run's outputs; verify a run records which frozen version it
      used.

## 2. Masks on the marginalized filter (branch `feat/masked-marginal-filter`)

- [x] 2.1 Port masked updates into `_run_kalman_filter_marginal` using the sequential path's
      `H_eff = M H`, `R_eff = M R Mᵀ + diag(1−M)` construction, applied consistently to the
      timing-model information accumulators; verify shapes stay static under `scan`/JIT.
- [x] 2.2 Verify an all-ones mask on the marginal path reproduces the existing goldens (63618.93 and
      59420.06) bit-for-bit.
- [x] 2.3 Verify masked marginal and masked sequential log-likelihoods agree on the same masked
      dataset at matched timing-prior settings, to the tolerance documented for the unmasked case.
- [x] 2.4 Verify an all-zeros-mask pulsar changes the log-likelihood only by a constant independent of
      the sampled parameters (test against the same dataset with that pulsar removed).
- [x] 2.5 Verify an epoch with no observed pulsars propagates the state without an update and leaves
      the log-likelihood unchanged.
- [x] 2.6 Verify gradients of the masked marginal log-likelihood are finite for every sampled
      parameter, including at partial and empty epochs.
- [x] 2.7 Remove the `use_marginal` fallback guard (`jax_kalman_filter.py:719–731`) so masked data
      select the marginal path with no warning and an explicit request succeeds; verify by updating
      the existing guard tests to the new behaviour and running the full suite.
- [x] 2.8 Benchmark masked marginal vs masked sequential on the 68-pulsar union grid and record the
      speedup; verify the marginal path is at least as fast before the guard removal is merged.

## 3. Checkpointing for long runs (branch `feat/nuts-checkpointing`)

- [x] 3.1 Restructure the run harness to sample in segments via NumPyro's `post_warmup_state`, saving
      sampler state plus accumulated samples after each segment; verify a run with checkpointing
      disabled produces output identical to the current harness.
- [x] 3.2 Add resume-from-checkpoint, refusing on any model or sampler configuration mismatch with the
      mismatch named; verify by a test that attempts a resume against a deliberately altered config.
- [x] 3.3 Verify statistical equivalence: a short run executed uninterrupted and the same run
      interrupted-and-resumed under the same seed give posteriors agreeing within Monte Carlo error.
- [x] 3.4 Mark output from an incomplete run with an explicit partial flag and the sample count drawn;
      verify loaders surface the flag and that partial output fails the usability check.
- [x] 3.5 Measure checkpointing overhead at the configured interval on a representative array run and
      record it; verify the overhead is within the documented tolerance.

## 4. Null calibration (branch `feat/sgwb-null-calibration`)

- [x] 4.1 Add a sky-scramble ORF generator that draws random sky positions, rebuilds the correlation
      matrix and accepts a scramble only below the ORF match threshold; verify the accepted ensemble's
      match distribution and that per-pulsar noise, epochs and residuals are untouched.
- [x] 4.2 Add a warm-start mode that reuses a completed run's tuned step size and mass matrix and
      samples without re-adaptation; verify it reproduces the parent run's posterior on the unscrambled
      ORF.
- [x] 4.2b Run ONE full-warmup scramble at 33 pulsars end-to-end through the frozen estimator as a
      falsification check on the injected-signal result; verify the estimator responds to the
      correlation pattern and that flat priors are not manufacturing HD evidence. DONE on MDC2 1b:
      `lnB = -0.766 +/- 0.010` (`reliable: true`) against `+3.043 +/- 0.015` with the true ORF, with
      the integrand negative and monotonic at every rung. Negative, not zero, is the correct answer
      here — a wrong correlation pattern fits worse than none on data that contains a real one. This
      is a falsification check, NOT a false-alarm probability: N=1 bounds only `p < 1`. Cold ladder,
      so it is also the full-warmup baseline 4.3 compares against. See
      `workflows/ng15_sgwb_demo/notes/RESULTS_null_calibration_scramble.md`.
- [ ] 4.3 Run a small pilot set of scrambles at 33 pulsars both warm-started and with full warmup;
      verify the two statistic distributions agree before warm-starting is used for the ensemble.
- [ ] 4.4 Measure statistic variance vs chain length on the pilot scrambles and choose the ensemble
      chain length from it; verify by a recorded variance-vs-length curve.
- [ ] 4.5 Record the target ensemble size, the resulting p-value resolution and the estimated compute
      cost before launching; verify the plan is committed ahead of the runs.
- [ ] 4.6 Run the MDC2 scramble ensemble and compute the false-alarm probability for the observed
      statistic, reporting `p < 1/N` when the observed value exceeds every realisation; verify the
      observed statistic lands in the extreme upper tail.
- [ ] 4.7 Run the ensemble on the matched no-injection control; verify the observed statistic sits in
      the bulk and the recovered p-values are consistent with uniform.

## 5. Kernel systematic (parallel; depends on the frozen procedure from 1.10)

- [x] 5.1 Generate the matched injection pair at the MDC2 geometry (γ = 13/3 power law and Argus's own
      OU process, same noise and epochs) with `scripts/inject_powerlaw_gwb.py`; verify both load
      cleanly through `get_processed_residuals` and their truth JSONs record the pivot amplitudes.
- [ ] 5.2 Run both injections end-to-end through the frozen procedure; verify both converge to the
      usability thresholds.
- [ ] 5.3 Report the amplitude systematic in dex and in units of posterior width at both pivot
      frequencies, and the `lnB` systematic; verify by a committed comparison artefact.
- [ ] 5.4 Apply the decision rule: if the systematic is subdominant to the statistical uncertainty and
      does not flip the decisiveness of `lnB`, retain the single-corner OU kernel and record the
      numbers; otherwise open a separate change for a richer kernel and hold claims. Verify by a
      committed decision note.
- [ ] 5.5 Add the kernel statement (OU's f⁻⁴ ceiling vs the standard f⁻¹³ᐟ³ expectation) and the
      measured systematic to the result-artefact generator; verify a generated summary carries both.

## 6. Intermediate real-data stage — NG15 wideband long-baseline subset (branch `m2-ng15-subset`)

- [ ] 6.1 Select the subset by a ≥ 8 yr baseline cut on the NG15 wideband release (~30–40 pulsars) and
      record the pulsar list; verify each selected pulsar's baseline against its `.tim`.
- [ ] 6.2 Ingest and epoch-align the subset onto a common union grid with masks; verify the reported
      joint-epoch count, per-pulsar retention and grid occupancy clear the documented floors.
- [ ] 6.3 Run the single-pulsar noise stage on every subset pulsar; verify 100% converge to the
      documented thresholds and that none rail against a prior edge.
- [ ] 6.4 Run the array stage with empirical priors, ridge basis, marginal filter and checkpointing
      enabled; verify convergence on sampled sites and that no chain is stuck by the per-chain-median
      check.
- [ ] 6.5 Apply the frozen evidence procedure; verify it reports `reliable: true` and record `lnB`.
- [ ] 6.6 Verify the recovered band-referenced amplitude is consistent with NANOGrav's published
      wideband result for comparable data, reporting the offset in dex and σ.
- [ ] 6.7 Run the null calibration on the subset at whatever ensemble size the compute supports;
      verify the p-value is reported at its true resolution with the limitation stated.
- [ ] 6.8 Freeze the full analysis configuration and commit it as the versioned production procedure;
      verify the freeze is recorded and that group 7 can diff against it.

## 7. Full array — NG15 wideband 68 pulsars (branch `m3-ng15-full-array`)

- [ ] 7.1 Ingest and epoch-align the full 68-pulsar wideband array with masks; verify occupancy and
      per-pulsar retention are reported and clear the floors.
- [ ] 7.2 Run the single-pulsar noise stage on all 68; verify 100% converge and none rail.
- [ ] 7.3 Launch the array run under the frozen configuration; verify by a committed config diff that
      it differs from the frozen version only in dataset paths, pulsar list and resource requests.
- [ ] 7.4 Verify the array run converges on sampled sites with divergences within tolerance, resuming
      from checkpoints as needed rather than restarting.
- [ ] 7.5 Apply the frozen evidence procedure and verify it reports `reliable: true`; record `lnB` and
      `p(ε|d)`.
- [ ] 7.6 Run the null calibration at the affordable ensemble size and report the false-alarm
      probability at its true resolution.
- [ ] 7.7 Verify the recovered amplitude is consistent with NANOGrav's published wideband result,
      reported in dex and σ, with the measured kernel systematic attached.
- [ ] 7.8 Produce the result artefact: amplitude with systematic, `lnB` with reliability diagnostics,
      false-alarm probability with its resolution, the kernel statement, and the scope bound
      (amplitude + correlation only, no spectral-index claim). Verify against the
      `sgwb/kernel-fidelity` and `sgwb/null-calibration` reporting requirements.

## 8. Close-out

- [ ] 8.1 Update issue #111 to point at this change, marking its M2 section and its M1→M3 ordering
      superseded; verify the issue reflects the executed route.
- [ ] 8.2 Update `workflows/ng15_sgwb_demo/TASKS.md` and the repo's handoff/log to the completed
      route; verify a fresh session can pick up from them.
