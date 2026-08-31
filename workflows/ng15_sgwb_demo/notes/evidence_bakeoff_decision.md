# Task 1.8 — the evidence-estimator bake-off, decided

## Result

Both estimators were run on the MDC2 dataset 2b Stage C configuration (33 pulsars,
two-stage noise with empirical priors, ridge GW basis).

| estimator | ln B(HD/CURN) | verdict |
|---|---|---|
| A — generalised Savage-Dickey (1 run, 9h47m) | 0.105 ± 0.095 | **NOT USABLE** — failed `fold_agreement` |
| B — path sampling, 5-rung ladder (5 runs, ~5h40m each) | **0.1754 ± 0.0171** | **reliable** |

They agree: the gap is 0.070 against a combined uncertainty of 0.096, i.e. 0.73σ. B is
5.5× more precise and passes every gate.

## Decision: freeze B as the production procedure; keep A as a cheap first look

The pre-committed rule was "if A and B agree within combined uncertainties, freeze A
with B as audit; otherwise freeze B and discard A." **That rule was underspecified and
did not decide this case.** It assumed both estimators would be reliable and that the
only question was whether they agreed. What actually happened is a third outcome: A's
*value* agrees but A *refuses to certify it*.

Freezing A as production would mean adopting an estimator that declines to report in
precisely the regime the project is in. So B is frozen, on the rule's intent rather than
its letter. A is retained because it is one run rather than five and its refusal is
itself informative — it is a cheap probe of whether the signal is strong enough to be
worth a ladder.

A is **not** discarded: it agreed, which is real cross-validation between two estimators
sharing nothing but the definition of ln B.

## Estimator B diagnostics

```
  eps grid: 0.000, 0.250, 0.500, 0.750, 1.000
      eps     <dlogL/deps>     uncert       ESS
    0.000           0.2516     0.0419       327
    0.250           0.2359     0.0407       279
    0.500           0.1963     0.0347       376
    0.750           0.1262     0.0327       400
    1.000           0.0116     0.0357       376
  quadrature        : romberg (raw trapezoid 0.1725, correction +0.0029)
  residual after fix : 0.0000  (ceiling 0.1)
  min integrand ESS  : 279     (floor 50)
  covers endpoints   : True
```

The integrand declines smoothly and monotonically to ~0 at ε=1. That is a meaningful
shape, not noise: ∂lnZ/∂ε → 0 at ε=1 means the evidence is *stationary* at Hellings-Downs,
so HD is the best point on the path. The direction is correct. The total gain from CURN
to HD is simply small.

## The number fails the MDC2 gate

The `sgwb/model-selection` spec requires `lnB >= 3` on the injected-signal case. The
measured value is **0.175 ± 0.017** — odds 1.19 : 1. The gate fails, and per
`sgwb/array-analysis-procedure` a failed stage gate means the next stage is not launched
until the failure is diagnosed.

## Diagnosis: the amplitude itself is only marginally detected

From the ε=1 (pure HD) rung, the recovered band-referenced amplitude is

```
  pivot log-PSD  median -6.714   16-84% [-9.837, -6.083]   sd 1.760
  injected truth -6.319          (covered, consistent with M1's -0.35 sigma truth gate)
  27% of the posterior lies below -9   (the prior centre: GW-negligible)
  38% lies below -8
```

Over a quarter of the amplitude posterior is consistent with essentially no
gravitational-wave background. A common signal that marginal cannot support decisive
evidence for its *correlation pattern*, which is a strictly harder measurement: the
amplitude is carried by common auto-power, which CURN reproduces exactly, whereas
HD-vs-CURN rests only on the weaker cross-correlations.

So `lnB = 0.175` is the arithmetically correct answer given this amplitude posterior.
The estimators are not at fault. The question that remains open is whether the *amplitude
posterior itself* is as good as this dataset allows, or whether our procedure is losing
sensitivity — see "Open question" below.

## What this says about the old LHM number

The learned harmonic mean's uncalibrated matched-shrinkage values on this same Stage C
pair ranged **1.8 to 8.6**. The calibrated answer is 0.175. LHM was not merely noisy at
68-D, it was biased high by an order of magnitude — in the direction of a spurious
detection. This is exactly the failure the reliability gate was built to catch, and it
retrospectively justifies not having reported the M1 Stage D number.

## Open question, for the next decision

Is the marginal amplitude detection intrinsic to MDC2 dataset 2b at 33 pulsars, or is the
two-stage empirical-prior noise treatment absorbing the common signal into per-pulsar red
noise? The single-pulsar posteriors are known to be overconfident for exactly this reason
(each pulsar's fit absorbs common power into its own red noise), which is why
`empirical_prior_inflation = 2.0` exists. Whether 2.0 is enough has never been tested.

This must be settled before M3, and it is not settled by anything run so far.
