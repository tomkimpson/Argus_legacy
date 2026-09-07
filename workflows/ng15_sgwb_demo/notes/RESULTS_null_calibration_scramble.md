# Null calibration: a single sky scramble on MDC2 1b

**Branch:** `feat/sgwb-null-calibration` · **Issue:** #111 · **Date:** 2026-09-07

## Question

The flat-prior ladder on MDC2 dataset 1b returned `lnB(HD/CURN) = 3.043 +/- 0.015` with the
pivot log-PSD recovered onto the injected truth. That number does not by itself distinguish

  (a) *we detect the Hellings-Downs correlation*, from
  (b) *flat red-noise priors let the GW claim any common power going, and the correlation
      structure is incidental.*

The way to separate them is to destroy the correlation pattern and change nothing else.

## What was run

One accepted sky scramble of the 33-pulsar array: random sky positions, ORF rebuilt from
them, the same residuals, epochs, per-pulsar noise, priors, ridge basis and NUTS settings.
Full 5-rung path-sampling ladder, cold (full warmup), through the **frozen** estimator.

    scramble       data/scrambles/mdc2_d1_scrambles.npz (seed 0, index 0)
    ORF match      0.0025 against the true ORF (threshold 0.2) -- effectively orthogonal
    conditioning   cond 7.34 vs the true ORF's 7.66, min eigenvalue 0.514
    rungs          eps = 0, 0.25, 0.5, 0.75, 1.0; SLURM 16275833_[0-4]
    sampling       0 divergences on every rung; 1h38m-2h21m each on 4x A100

The scrambled ORF is injected at runtime by `run_scrambled.py`, monkeypatching
`get_processed_residuals` exactly as `run_curn.py` does for the identity. No library edit,
and **no change to `lnb_path_sampling.py`** -- the production estimator is frozen, so
editing it would have marked the 3.043 stale. `scripts/lnb_scrambled.py` installs the same
override and delegates, leaving the estimator byte-identical.

## Result

    ln B(HD/CURN)      NULL -0.7656 +/- 0.0097      FLAT +3.0431 +/- 0.0148
    reliable           true                         true
    failed diagnostics []                           []

    integrand d lnZ/d eps
      eps      null       flat
      0.00   -0.386     +4.695
      0.25   -0.574     +3.859
      0.50   -0.789     +2.966
      0.75   -0.952     +2.220
      1.00   -1.130     +1.558

**The integrand flips sign completely, smoothly and monotonically.** The null integral is
not a small number assembled from large values that cancel; it is negative at every rung.
Estimator gates clear with room: Romberg residual 0.0008 against a 0.1 ceiling, min
integrand ESS 348 against a floor of 50, endpoints covered.

## Reading it

**Hypothesis (b) is dead.** Flat priors do not manufacture Hellings-Downs evidence. The
amplitude recovery and the +3.043 both stand.

**The null is NOT zero, and should not be.** -0.766 sits 79 sigma from zero in the
estimator's own uncertainty. That is the correct answer, not a failure: imposing a *wrong*
correlation pattern on data that contains a real one fits actively worse than assuming no
correlation at all, so a scramble on signal-bearing data must land below zero. A near-zero
result would have been the surprise. The sign of the penalty is itself evidence that the
estimator responds to geometry rather than to prior width.

**A supporting observation.** At eps = 0.75 the amplitude posterior is 3.3x wider under the
scramble (sd 0.972 vs 0.292) at essentially the same centre. The centre is expected to be
stable: scrambling preserves the ORF diagonal, so each pulsar's auto-power and hence the
common signal is untouched. Only the cross-correlation information is destroyed, and the
widening is what losing it costs.

## Correctness check that came for free

At eps = 0 the ORF is the identity for *any* geometry, so `mdc2_d1_null_eps000` and
`mdc2_d1_flat_eps000` sample the same target. They agree **bitwise** -- all 13 posterior
variables, 4000 draws, max absolute difference exactly 0 -- across two genuinely distinct
files written three days apart on different nodes, with the null run's log confirming the
scramble *was* installed. The override cannot perturb what it must not, and the pipeline
reproduces deterministically.

Pre-flight also confirmed, on the real data, that the patched and unpatched loads differ in
`hd_correlation` and nothing else: residuals, metadata, design matrices and parameter
covariances are all value-identical. That is the `sgwb/model-selection` "only substituted
quantity" requirement, checked rather than asserted.

## What this does NOT establish

**There is no false-alarm probability here.** One scramble is a falsification check, not a
null distribution. The honest bound from N = 1 is `p < 1`, which is no significance at all.
The estimator's +/- 0.0097 is its own numerical precision and says nothing about
scramble-to-scramble scatter, which is unmeasured.

A real p-value needs the ensemble (tasks 4.3-4.7). At ~40 A100-hours per realisation
(5 rungs x ~2 h x 4 GPUs) a 100-scramble ensemble is ~4000 A100-hours and is almost
certainly unaffordable; the null-calibration spec anticipates exactly this and requires the
claim be scaled to whatever ensemble the compute supports. Warm starts
(`checkpointing.py:239`) exist to halve that cost but must first be validated against
full-warmup runs -- this ladder is that baseline.

**This is the sky-scramble test (group 4), not the no-injection control (task 1.9).**
"lnB consistent with zero" is the criterion for a dataset with no correlated signal to
mis-describe. MDC2 group1 has no such dataset -- `group1_gw_parameters.json` holds only
dataset1 (GWB), dataset2 (GWB) and dataset3 (a CW source) -- so task 1.9 still needs either
a synthesised signal-free dataset at the 1b geometry (via `inject_powerlaw_gwb.py`, which is
6-pulsar NG15 machinery needing a port) or an explicit decision to drop it.

## Operational notes

- The readout must go through `scripts/lnb_scrambled.py`, never `lnb_path_sampling.py`
  directly: `--evaluate` rebuilds the data from the config and would integrate against the
  TRUE ORF, returning a well-formed and meaningless number that nothing downstream catches.
  The ORF fingerprint in `bayesian_inference` guards *resume*, not the estimator.
- **Run the estimator as a SLURM CPU job, not on the login node.** This one took 6h33m
  (~85 min per rung) holding 36 GB resident, against the ~30 min the flat ladder recorded,
  purely because the login node was cgroup-limited to 1 core at load average 17.5. It
  survived, but a multi-hour 36 GB login-node process can be reaped at any moment.
- Artefacts: `outputs/lnb_path_sampling_d1_null.json`, `outputs/mdc2_d1_null_eps*/`,
  `data/scrambles/mdc2_d1_scrambles.npz` (+ `_summary.json`, both tracked -- the accepted
  ORF *is* the definition of this null and the result is not reproducible without it).
