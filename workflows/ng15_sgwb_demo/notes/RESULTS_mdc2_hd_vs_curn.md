# MDC2 HD-vs-CURN: results

**Date:** 2026-09-01 · **Branch:** `feat/sgwb-null-calibration` · **Issue:** #111
**Change:** `openspec/changes/sgwb-detection-route`

---

## Headline

On IPTA MDC2 dataset 2b (33 pulsars, injected `log10_A = -14.886`, γ = 13/3), analysed
with Argus's two-stage noise procedure and ridge GW parameterization:

> **ln B(HD/CURN) = 0.1754 ± 0.0171** — odds 1.19 : 1, reliable by every diagnostic.

That is **not** a detection of Hellings-Downs correlation, and it fails the change's own
MDC2 acceptance gate of `lnB ≥ 3`. The estimator is sound; the signal is not there to be
found in this configuration. Why, and what follows, is the substance of this note.

---

## What was built to get here

The learned harmonic mean (LHM) that M1 relied on goes degenerate above ~20 dimensions,
which is why M1 recovered an amplitude but never a calibrated Bayes factor. The
replacement is a single model extension plus two independent estimators over it.

**The correlation path.** Both hypotheses are embedded in one model by interpolating the
overlap reduction function,

```
C(ε) = (1 - ε)·I + ε·C_HD
```

so CURN is the point ε = 0 and HD the point ε = 1 of a single posterior, with every other
parameter, prior and data input shared. Both endpoints are pinned *exactly* by test:
ε = 1 reproduces the recorded goldens (63618.93 informative, 59420.06 diffuse)
bit-for-bit, and ε = 0 reproduces `run_curn.py`'s identity-ORF construction bit-for-bit.

**Estimator A — generalised Savage-Dickey.** One run. Reads lnB from the marginal density
of ε at its two endpoints. Uses two independent boundary-corrected density estimators
(reflection KDE, beta kernel) and extrapolates to zero bandwidth, because Silverman's rule
over-smooths and biases lnB low by ~0.16 nats — several times the sampling uncertainty.

**Estimator B — path sampling.** A ladder of runs at fixed ε, integrating
`E[∂lnL/∂ε]` from autodiff. No boundary-density step, so it survives where A cannot. Uses
Romberg extrapolation over the ladder, its half and its quarter.

Both refuse to report when their diagnostics fail, and both were validated against a
shared analytic problem (`ln Z = a·ε³`, so lnB = a exactly) before touching real data.

---

## What was run

Six A100 jobs on OzSTAR `milan-gpu`, 4 GPUs each.

| run | ε | wall | draws | max r̂ | min ESS | divergences |
|---|---|---|---|---|---|---|
| ladder ×5 | 0, 0.25, 0.5, 0.75, 1.0 | ~5h40m each | 1000 | ≤ 1.02 | 279–459 | **0%** |
| ε-sampled | sampled | 9h47m | 2000 | 1.010 | 600 | **0%** |

**Zero divergences in every run**, no stuck chains, all four chains tracking together
throughout. That is a materially better geometry than the direct-basis Stage B/C runs
(r̂ 1.5–2.3, one chain parked at high amplitude in every case). The ridge basis plus the
correlation path sample cleanly at 69 dimensions.

---

## The two estimators

| estimator | lnB | verdict |
|---|---|---|
| A — Savage-Dickey | 0.105 ± 0.095 | **NOT USABLE** — failed `fold_agreement` (gap 0.331 vs 0.095) |
| B — path sampling | **0.1754 ± 0.0171** | **reliable** |

They agree at 0.73σ. B is 5.5× more precise and passes every gate, so **B is frozen as the
production procedure**; A is kept as a cheap one-run first look, since its refusal is
itself a useful signal that a ladder is warranted.

A's failure mode was not the one predicted. The design expected ε to pile against 1 on a
signal-bearing dataset, emptying the CURN endpoint and tripping `endpoint_occupancy`.
Instead **the ε posterior came out essentially uniform** (mean 0.512, median 0.518, both
endpoints well populated at ~5% occupancy), and A failed on fold agreement — it cannot
resolve a quantity of size 0.1 to better than ~0.3.

That near-uniform ε posterior is the finding in its most direct form: **the array carries
almost no information about how HD-like the correlation is.**

### Estimator B integrand

```
      eps     <dlogL/deps>     uncert       ESS
    0.000           0.2516     0.0419       327
    0.250           0.2359     0.0407       279
    0.500           0.1963     0.0347       376
    0.750           0.1262     0.0327       400
    1.000           0.0116     0.0357       376

  quadrature         : romberg (raw trapezoid 0.1725, correction +0.0029)
  residual after fix : 0.0000  (ceiling 0.1)
  min integrand ESS  : 279     (floor 50)
  covers endpoints   : True
```

The integrand is positive throughout and declines monotonically to ~0 at ε = 1. The shape
is meaningful: `∂lnZ/∂ε → 0` at ε = 1 means the evidence is **stationary at
Hellings-Downs**, so HD really is the best point on the path. The direction is right. The
total gain from CURN to HD is simply small.

---

## Diagnosis: the amplitude itself is only marginally detected

From the pure-HD rung:

```
pivot log-PSD   median -6.714   16-84% [-9.837, -6.083]   sd 1.760
injected truth  -6.319          (covered)
27% of the posterior lies below -9   (prior centre: GW-negligible)
38% lies below -8
```

**Over a quarter of the amplitude posterior is consistent with no gravitational-wave
background at all.**

A common signal that marginal cannot support decisive evidence for its *correlation
pattern*, which is a strictly harder measurement. The amplitude is carried by common
auto-power, which CURN reproduces exactly; HD-vs-CURN rests only on the much weaker
cross-correlations. `lnB = 0.175` is the arithmetically correct answer given this
posterior.

This also reframes M1's headline. The Stage C "truth gate PASS at −0.35σ" was true but
weak: passing a coverage test with a ±1.76 dex error bar is not much of a test, and the
same breadth is what makes the Bayes factor negligible.

---

## The old LHM number was biased high

On this same Stage C pair, LHM's uncalibrated matched-shrinkage values ranged **1.8 to
8.6**. The calibrated answer is **0.175**.

LHM was not merely noisy at 68-D — it was biased high by an order of magnitude, in the
direction of a spurious detection. This retrospectively justifies M1's decision not to
report that number, and it is exactly the failure the new reliability gate exists to
catch. `logz_lhm.py` now refuses with named diagnostics instead of returning `nan`;
verified on the stored posteriors (reliable on the 2-D anchor, NOT USABLE on the 68-D
pair, naming four failing diagnostics).

---

## Defects found and fixed along the way

**1. The masked likelihood was wrong** — in the pre-existing sequential mask path from
PR #113, not in new code. Absent observations get a *unit* variance placeholder so the
innovation covariance stays invertible, but the positive-definiteness jitter was scaled as
`1e-9 · trace(S)/n`. PTA innovation variances are ~1e-12, so the placeholders — not the
data — set the jitter, inflating it ~1e12×.

| | before | after |
|---|---|---|
| MDC2 @ 41.6% occupancy, marginal vs sequential | 7e-4 relative (~23 nats) | 2.3e-6 |
| offset from masking one pulsar out entirely | −65.50 (theory: −11.03) | −11.0273, exact to 1e-14 |

Unmasked results and both goldens are bit-for-bit unchanged. Nothing merged had used
masked data, but the union-grid tooling was never run in anger — this is why it should not
have been trusted before M3.

**2. Savage-Dickey bandwidth bias.** Silverman's rule over-smooths and biases lnB low by
~0.16 nats; h→0 extrapolation cuts it ~4×.

**3. A vacuous diagnostic.** My first discretisation check for path sampling compared
Simpson against the trapezoid rule — but on a uniform grid Simpson *is* the Richardson
extrapolation of the trapezoid, so the difference is identically zero. Replaced with a
genuine Romberg estimate, which fires correctly on a deliberately under-resolved integrand.

**4. OOM in the integrand evaluation.** `evaluate_integrand` vmapped every posterior draw
at once, so peak memory scaled with run length: 200 draws worked, 400 was killed, and the
full 4000-draw posteriors would never have fitted. Now batched.

**5. The injector's OU branch recorded no pivot PSD** — only the power-law branch did, so
the two halves of a matched injection pair were literally not comparable.

---

## Also delivered (not yet exercised)

- **Masks on the marginalized filter**, replacing the silent fallback to the sequential
  path. 5.4× faster on MDC2; matters for M3's 41.6%-occupied union grid.
- **Checkpoint and resume** for long NUTS runs. A three-segment run reproduces an
  unsegmented one's draws to 1e-10 at the same seed; overhead ~0 s/segment. Exercised in
  every run above.
- **Sky-scramble null generator** and **warm starts**, for the false-alarm calibration.
- **Matched power-law/OU injection pair** at the MDC2 geometry, for the kernel systematic
  (both pinned to `log10 S(1/5yr) = -6.3194`, agreeing to 3e-7 dex).

---

## The open question

**Is the marginal amplitude intrinsic to MDC2 2b at 33 pulsars, or is our two-stage noise
treatment absorbing the common signal into per-pulsar red noise?**

Single-pulsar posteriors are known to be overconfident for exactly this reason — each
pulsar's fit absorbs common power into its own red noise — which is why
`empirical_prior_inflation = 2.0` exists. **Whether 2.0 is enough has never been tested.**

Per `sgwb/array-analysis-procedure`, a failed stage gate means M3 does not launch until
this is diagnosed. Three routes:

1. **Prior-inflation sweep** (~3 ladder-equivalents). Re-run the ε = 0 and ε = 1 rungs at
   inflation 1.0 / 2.0 / 4.0. If the amplitude sharpens and lnB climbs with inflation, the
   noise treatment is the culprit and it is fixable. Cheapest decisive test of the leading
   hypothesis.
2. **Louder-injection scaling** (~2 runs). Inject the same geometry at 5–10× amplitude and
   check lnB scales roughly as amplitude². If it does, the machinery is sound and MDC2 is
   simply below threshold; if it does not, something deeper is wrong with the correlation
   modelling. Uses the injector already built for task 5.1.
3. **External reference — start with the literature.** MDC2 is a public challenge and
   groups have published their results on it. If someone quotes a detection statistic or
   Bayes factor for dataset 2b at this amplitude, that answers "is it us or the data" for
   the cost of a reading session, with no pipeline to stand up. This should be tried
   *first*, before 1 and 2, precisely because it is nearly free.

If 1–3 all say "machinery fine, data quiet", the remaining decision is a scope one:
whether M3's claim rests on amplitude plus a scramble-calibrated significance rather than a
decisive Bayes factor — which would mean rewriting the acceptance gate in
`sgwb/model-selection`.

---

## Artefacts

| what | where |
|---|---|
| Bayes factor + diagnostics | `outputs/lnb_path_sampling.json` |
| Savage-Dickey report | `outputs/mdc2_stageC_path_sampled/lnb_savage_dickey.json` |
| Ladder posteriors | `outputs/mdc2_ladder_eps{000,025,050,075,100}/` |
| ε-sampled posterior | `outputs/mdc2_stageC_path_sampled/` |
| Bake-off decision record | `notes/evidence_bakeoff_decision.md` |
| Mask/jitter fix | `notes/masked_marginal_filter.md` |
| Checkpointing | `notes/checkpointing.md` |
| Injection pair | `notes/kernel_systematic_injection_pair.md` |
