# The two-stage noise procedure absorbs the gravitational-wave background

**Status:** diagnosed, mechanism strongly indicated, not yet proven by removal.
**Blocks:** M3, and any detection claim from Argus's array pipeline.
**Date:** 2026-09-04 · **Branch:** `feat/sgwb-null-calibration` · **Issue:** #111

---

## One-paragraph summary

Argus's array analysis constrains per-pulsar red noise using priors derived from
single-pulsar fits (the "two-stage" or empirical-prior procedure). Those single-pulsar
fits are performed with the gravitational-wave amplitude **fixed at zero**, so by
construction each pulsar's red-noise posterior absorbs the GW background's contribution
to that pulsar. Stage C then uses those posteriors as priors, which tells the joint fit
that all the common red power is already accounted for as 33 independent noise processes.
The GW term has nothing left to explain, and its posterior collapses onto its prior. On
IPTA MDC2 dataset 1b — a dataset the IPTA collaboration detects strongly — Argus recovers
**no GW amplitude information at all**, and `lnB(HD/CURN) = 0.053 ± 0.004`.

We subtract the signal, then look for it in the residual.

---

## The measurement that exposed it

MDC2 open dataset 1b was run as a **positive control**: a dataset where the published
answer is a confident detection, so a working pipeline must return one.

| | published (arXiv:1912.12939) | Argus |
|---|---|---|
| Bayes factor | **ℬ = 40** (free WN) to **∞** (fixed WN), GW vs noise-only | — |
| ℬ(HD/CURN), inferred by ratioing their HD (40) and CRN (23) rows | ≈ 1.74, i.e. lnB ≈ 0.55 | **lnB = 0.0527 ± 0.0039** |
| amplitude | **0.7 ₋₀.₃⁺⁰·⁴ × 10⁻¹⁵** against 0.66 injected — a measurement | **posterior = prior** (see below) |

The Bayes-factor row carries a caveat: their headline ℬ is HD-vs-*noise-only*, a different
and easier comparison than ours, and the row-ratio is rough because the two rows may
differ in white-noise treatment. **The amplitude row carries no such caveat and is the
decisive evidence.**

### The amplitude posterior is the prior

Prior on the band-referenced pivot log-PSD is `N(-9.0, 1.333)`. The injected truth for 1b
is `-6.908`, which sits `+1.57` prior-σ from the centre.

| ε | posterior median | posterior sd ÷ prior sd | shift from prior centre |
|---|---|---|---|
| 0.00 | −9.613 | **0.950** | −0.46 σ |
| 0.25 | −9.561 | **0.954** | −0.42 σ |
| 0.50 | −9.560 | **0.975** | −0.42 σ |
| 0.75 | −9.554 | **0.984** | −0.42 σ |
| 1.00 | −9.507 | **0.987** | −0.38 σ |

The data remove 1–5% of the prior uncertainty and move the centre by less than half a
prior-σ, in the **wrong direction** (away from truth). This is not a weak measurement; it
is the absence of one.

### The path-sampling integrand never rises

```
      eps     <dlogL/deps>     uncert       ESS
    0.000           0.0437     0.0107       276
    0.250           0.0514     0.0093       400
    0.500           0.0446     0.0067       383
    0.750           0.0566     0.0080       347
    1.000           0.0645     0.0086       400
  -> ln B(HD/CURN) = 0.0527 +/- 0.0039   [reliable, all gates pass]
```

Flat at ~0.05 the whole way. Hellings-Downs correlation never does any work.

---

## The mechanism

**Stage A** fits each pulsar alone with `log10_ha = -20` (GW fixed negligible). The config
header states the intent without ambiguity:

> *"a single pulsar cannot separate GW auto-power from intrinsic red noise, so the OU
> process absorbs the **TOTAL** per-pulsar red power — exactly the quantity Stages B/C
> consume."*

This is correct as a description of what one pulsar can know. A single pulsar genuinely
cannot separate a common background from its own spin noise; only cross-correlations can.

**Stage C** then uses those posteriors as priors on per-pulsar red noise, with the GW
amplitude free. The intent was that the array's cross-correlations would pull the GW back
out.

**The flaw:** the Stage A posteriors already contain the GWB. Their centres say "your own
noise is this large", where "this large" includes the power the GWB deposited. Stage C
therefore begins from a prior state in which the common signal is fully explained by
independent noise. Adding GW amplitude would *over*-explain the data. The GW drifts back
to its prior.

`empirical_prior_inflation = 2.0` was the intended safeguard — widen the priors so the
array run has room to reattribute power. It widens the distributions but does not move
their **centres**, and the centres are what is wrong. A wide prior centred on "all this
power is yours" still concentrates its mass where the GW is absent. It makes the result
less certain, not more correct — which is exactly the observed posterior≈prior.

### The mechanism predicts the severity ordering, and the prediction holds

The effect should be worst when the GWB is a *larger fraction* of the red power Stage A
absorbs — i.e. on the cleanest datasets.

| dataset | injected red noise | GWB's share of red power | lnB |
|---|---|---|---|
| **1b** | none | all of it | **0.053 ± 0.004** |
| **2b** | yes, per-pulsar | diluted | **0.175 ± 0.017** |

Three times larger on the *dirtier* dataset. Recovered per-pulsar red noise confirms the
dilution: Stage A gives median `log10_σp` = −16.70 on 1b against −16.01 on 2b (2b higher
in 24 of 33 pulsars).

---

## What this invalidates

1. **The 2b result (lnB = 0.175) is no longer evidence of anything.** It agreed with the
   literature's non-detection, but plausibly for the wrong reason. That agreement cannot
   be used to validate the pipeline.
2. **M1's "Stage C truth gate PASS at −0.35σ" was passing by being uninformative.** The
   posterior was ±1.8 dex; a distribution that covers everything covers the truth. Breadth
   was read as success when it was the symptom.
3. **M1 had no positive control.** Nothing ever asked whether the pipeline could detect a
   signal known to be detectable. That single question is what exposed this, and it should
   be a standing requirement before any future gate is trusted.

## What this does *not* implicate

Bounded deliberately, because most of the machinery is fine:

- **Not the likelihood.** Goldens bit-identical (63618.93, 59420.06); masked and marginal
  filter paths agree to 6e-12.
- **Not the estimators.** Savage-Dickey and path sampling were validated against an
  analytic problem and agree with each other at 0.73σ on real data.
- **Not the sampler or geometry.** 44 of 44 SLURM runs converged, 0% divergences, r̂ ≤ 1.02.
  The ridge parameterization works.
- **Not the OU kernel mismatch.** That is a ~0.2 dex amplitude bias, not a factor of 100.

The defect is confined to the **noise-modelling procedure between the data and the
likelihood**.

---

## Why we took this route in the first place

The two-stage scheme is a workaround for a documented sampling failure. Issue #109 / T3.5
(2026-07-18) ruled full-joint NUTS **unsamplable** at 68 pulsars (~142-D). Three levers
triangulated it:

| configuration | outcome |
|---|---|
| diagonal mass matrix | r̂ ≈ 18, chains never mixed |
| dense GW block, `target_accept = 0.95` | 39% divergences (one chain 96%) — curvature overshoot |
| dense GW block, `target_accept = 0.99` | chains froze: within-chain sd = 0, r̂ ≈ 6e15 — step-size collapse |

Low accept → divergences; high accept → no mixing; no `target_accept` threads the needle.
Two pathologies were named: the curved `h_a↔γ_a` ridge, and a hierarchical funnel from the
short-baseline pulsars.

The scheme was always acknowledged as a deviation from field standard — NANOGrav fixes
**white** noise only and samples red noise jointly with the GWB under wide priors via
PTMCMC. It was adopted for tractability. We now know what the tractability cost.

## Why the original verdict deserves re-testing

1. **The ridge parameterization did not exist then.** It was built during M1 specifically to
   straighten the `h_a↔γ_a` ridge — one of the two named pathologies. Every run in this
   session used it and got **0% divergences**.
2. **Wide priors cost nothing in dimension at 33 pulsars.** Flat per-pulsar priors and
   empirical per-pulsar priors sample the *same* parameters: 2 GW + 2×33 = **68 either
   way** (`count_free_parameters`: empirical `+= 2*n_pulsars`; flat `+= n_pulsars` twice).
   The empirical scheme does not reduce dimensionality. It only narrows priors.
3. **The failure was at ~142-D, not 68-D.** We have just demonstrated clean sampling at 68-D.

The remaining genuine obstacle is that wide priors make the **GW↔red-noise degeneracy
explicit**: raise the GW amplitude, lower all 33 red-noise amplitudes, and the auto-power
is unchanged. That is a hard, near-flat direction. But it is the *real* inference problem,
and it is precisely what the Hellings-Downs cross-correlations exist to break. Hard
geometry is a sampling problem with known remedies; pre-subtracting the signal is a
correctness problem with none.

---

## Proposed way forward

### 1. Confirm the diagnosis by removal (next action, ~9 h GPU)

Run 1b at **ε = 0 and ε = 1 with flat red-noise priors** (`red_noise_prior = flat`, drop
`empirical_priors_path`). Same 68 dimensions as the runs that just sampled cleanly, so the
sampling risk is low.

- **If the amplitude moves from ≈ −9.5 toward −6.9 and the integrand rises**, both the
  diagnosis and the fix are confirmed at once.
- **If it does not**, the mechanism above is wrong and the problem lies elsewhere — in
  which case the next suspects are the GW prior parameterization and the Stage A white-noise
  treatment.

Configs to write: clone `configs/mdc2_d1_ladder_rung.ini.template`, set
`red_noise_prior = flat`, remove `empirical_priors_path` and `empirical_prior_inflation`,
keep everything else (ridge basis, fixed EFAC/EQUAD from truth, `orf_path = fixed`).

### 2. If confirmed, replace the noise treatment

In order of preference:

1. **Flat / weakly-informative per-pulsar red-noise priors, sampled jointly.** The honest
   formulation and the field standard. Costs sampling difficulty, which is now the problem
   to solve rather than avoid.
2. **Hierarchical population prior.** Per-pulsar amplitudes regularised toward a
   distribution *learned from the array itself* rather than pre-committed from
   single-pulsar fits, so it does not pre-subtract the GWB. The mode already exists in the
   library. Caveat: reintroduces the hierarchical funnel, the other T3.5 pathology.
3. **Re-derive Stage A with the GW free.** Probably a dead end — a single pulsar genuinely
   cannot separate the two, which is why it was fixed.
4. **Much larger inflation.** Expected to fail: the centre is wrong, not just the width.

### 3. Then confront 68 pulsars as its own problem

Already filed as **issue #115** (joint noise+GW formulation) and deferred as "real research
work". It is now on the critical path rather than off it, because the shortcut around it
does not work. Options recorded there: reparameterization, Gibbs, delayed acceptance.

### 4. Re-point the acceptance gate

`sgwb/model-selection` requires `lnB ≥ 3` on the injected-signal case, with MDC2 2b
implied. Two changes are needed regardless of the above:

- **The validation dataset should be 1b, not 2b.** 2b is a published non-detection; no
  method reaches the gate on it, so it can never validate anything.
- **A positive control must be a standing requirement.** The gate as written can be passed
  by an uninformative posterior, as M1's was.

---

## Reproducing the evidence

```bash
cd workflows/ng15_sgwb_demo

# the two Bayes factors
cat outputs/lnb_path_sampling_d1.json   # 1b: 0.0527 +/- 0.0039
cat outputs/lnb_path_sampling.json      # 2b: 0.1754 +/- 0.0171

# the amplitude-vs-prior table (prior is N(-9, 1.333))
python - <<'PY'
import arviz as az, numpy as np, glob
for t in ["eps000","eps025","eps050","eps075","eps100"]:
    p = np.asarray(az.from_netcdf(glob.glob(
        f"outputs/mdc2_d1_ladder_{t}/*_results.nc")[0]
        ).posterior["log10_pivot_psd"].values).ravel()
    print(t, round(np.median(p),3), round(p.std()/1.3333,3))
PY
```

Ladder posteriors, Stage A artefacts and the estimator outputs are all on disk; the
feathers are gitignored but regenerable via `scripts/ingest_par_tim.py` +
`scripts/stage_mdc2.py` (see `notes/mdc2_literature_check.md` for the dataset paths).
