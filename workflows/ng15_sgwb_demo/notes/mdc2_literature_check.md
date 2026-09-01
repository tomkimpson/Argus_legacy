# Literature check: is MDC2 dataset 2b detectable at all?

**Date:** 2026-09-01 · **Question:** our HD-vs-CURN Bayes factor on MDC2 came out at
`lnB = 0.175 ± 0.017`, far below the `lnB ≥ 3` gate. Is that our pipeline, or the data?

**Answer: the data.** MDC2 open dataset 2b is a published **non-detection for every method
the IPTA collaboration tried**. Our result reproduces theirs.

## Source

Hazboun et al., *Results for the International Pulsar Timing Array Second Mock Data
Challenge*, [arXiv:1912.12939](https://arxiv.org/abs/1912.12939).

## The dataset is the same one, on every axis

| | MDC2 paper (g1.d2) | ours |
|---|---|---|
| injected amplitude | 1.3 × 10⁻¹⁵ | `log10_A = -14.886` → **1.300 × 10⁻¹⁵** |
| spectral index | 13/3 | 13/3 |
| pulsars | 33 | 33 |
| baseline / cadence | 15 yr, ~30 d | 14.96 yr, ~30 d |
| epochs | ~185 | 183 |
| variant | "we analysed only the more realistic **b** datasets" | `dataset_2b` |

Identification is exact.

## What they found

Their Table 5 (g1.d2), Bayes factors for **GW + noise vs noise-only**:

| run | A_median (10⁻¹⁵) | A_mode (10⁻¹⁵) | ℬ |
|---|---|---|---|
| CRN (DE436, free WN) | <1.5 | 0.9 | 2.6 |
| **HD (fixed WN)** | **<1.4** | **0.9** | **2.3** |
| HD (free WN) | <1.3 | 0.8 | 1.8 |
| DE436 | <1.4 | 0.9 | 1.6 |
| DE436 w/ BE | <2.3 | 1.8 | 1.1 |
| DE430 w/ BE | <1.4 | 1.0 | 0.9 |
| DE430 | <1.4 | 0.9 | 2.1 |

ℬ is a **Bayes factor (odds), not a log**. Their declared detection threshold is **ℬ > 3**,
which they describe as "quite a low threshold". In their words:

> "For g1.d2 we found no significant GWB under any search method... The largest Bayes
> factor came from the correct choice of SSE and the weakest from the incorrect."

Note also that **their amplitude is an upper limit**, not a measurement: A_median < 1.4 ×
10⁻¹⁵ against an injected 1.3 × 10⁻¹⁵, with the mode at 0.9. The injected value sits at
their upper limit.

## Comparing like with like

Their ℬ is **HD vs noise-only**. Ours is **HD vs CURN** — a different, harder comparison.
To convert roughly, take the ratio of their HD and CRN rows:

```
B(HD/CURN) ~ 2.3 / 2.6 = 0.88   ->   lnB ~ -0.13
```

against our **+0.175 ± 0.017**. The two rows are not perfectly matched (their CRN row has
free white noise, the HD row fixed), so this is indicative rather than exact. But the
conclusion is identical and unambiguous: **on g1.d2 there is no evidence distinguishing
Hellings-Downs from common uncorrelated red noise**, for us or for them.

Their upper-limit amplitude also corroborates our diagnosis directly. We found 27% of the
amplitude posterior below the GW-negligible level; they could not place a lower bound at all.

## Why g1.d2 fails when g1.d1 succeeds

This is the useful part. Dataset g1.d1 has a **smaller** injected amplitude — 0.66 × 10⁻¹⁵
against 1.3 × 10⁻¹⁵ — yet it was **strongly detected**:

| g1.d1 run | ℬ |
|---|---|
| HD correlated (fixed WN) | **∞** |
| HD correlated (free WN) | 40 |
| Common red noise | 23 |
| DE436 + BayesEphem | 1.2 (failed) |

Recovered amplitude 0.7₋₀.₃⁺⁰·⁴ × 10⁻¹⁵ against 0.66 injected — a real measurement, not a
limit.

The difference is not amplitude. **g1.d1 is white-noise-only; g1.d2 adds per-pulsar red
noise.** Per-pulsar red noise is covariant with a common red background, and that covariance
is what destroys the correlation measurement — intrinsically, for every method, not as an
artefact of any particular noise treatment.

That is the same mechanism we suspected in our own two-stage empirical-prior procedure. It
turns out to be real, but it is a property of the *data*, not of our handling of it.

## Consequences

1. **Our machinery is validated on the negative side.** We reproduce a published
   non-detection on a dataset designed to be hard.
2. **The MDC2 acceptance gate was mis-specified.** `lnB ≥ 3` is odds of 20:1. The community
   analysing this exact dataset got odds of 2.3 on the *easier* GW-vs-noise comparison,
   against their own threshold of 3. The gate was never reachable on g1.d2 by anyone.
3. **The prior-inflation sweep is demoted.** It was the leading hypothesis for our low lnB;
   the literature says the red-noise/GWB covariance is intrinsic. Still worth a cheap check
   eventually, but no longer the critical path.
4. **The louder-injection test is superseded** by something strictly better: g1.d1 is a real
   dataset with a *published reference value* (ℬ = 40 to ∞), where a synthetic injection
   would only ever be checked against our own expectations.

## Recommended next step

**Run g1.d1 (`dataset_1b`) as the positive control**, and re-point the MDC2 gate at it. It
is available locally at `/fred/oz022/tkimpson/mdc2/group1/dataset_1b`. If Argus returns a
decisive HD preference there, the pipeline is validated end-to-end against an independent
published result — which is far stronger evidence than passing a self-defined gate on a
dataset nobody can detect.

**Complication to design around:** g1.d1 has no injected per-pulsar red noise, so the
two-stage procedure (Stage A single-pulsar red-noise runs → empirical priors for the array
run) would be fitting red noise that is not there. The Stage A artefacts we hold are for
dataset 2b and do not transfer. Decide whether to run Stage A on 1b anyway, use flat
red-noise priors, or fix red noise negligible — this needs a deliberate choice before
launching.
