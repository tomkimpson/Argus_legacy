# The matched injection pair (task 5.1)

## What it is for

Argus models the SGWB with a single-corner OU process. Well above its corner that PSD
falls as f⁻⁴, which is the steepest slope it can produce — short of the standard
power-law expectation f⁻¹³ᐟ³ ≈ f⁻⁴·³³. This is a deliberate modelling difference, not a
defect, and the change design commits to reporting the systematic it induces rather than
gating the project on it.

Measuring that systematic needs two datasets that differ **only** in spectral shape: the
same 33-pulsar MDC2 geometry, the same epochs, the same white-noise parameters, one
injected with a γ = 13/3 power law and the other with Argus's own OU process.

## What "matched" means

Matched *band-referenced PSD*, not matched parameters. The two shapes' native parameters
do not correspond at all — (log10_A, γ) against (log10_ha, log10_γ_a) — so the only
common ground is the PSD at a pivot frequency inside the band a PTA actually constrains.

Both injections are pinned to the same value at **f_pivot = 1/(5 yr) = 6.3376e-09 Hz**,
the pivot the M1 ridge parameterization already uses.

| | value |
|---|---|
| Power law | `log10_A = -14.886056647693163`, `γ = 13/3` (MDC2 dataset 2b truth) |
| OU (matched) | `log10_ha = -12.919767`, `log10_γ_a = -9.0` |
| Pivot PSD both | `log10 S(1/5yr) = -6.3194` (agree to 3e-7 dex) |

The power-law pivot reproduces the injected value of −6.32 recorded from M1, which
confirms the PSD conventions line up with the recovery code.

## Why the corner is at 10⁻⁹, not 10⁻⁸·⁵

Over the 14.96 yr baseline the lowest sampled frequency is 1/15yr = 2.11e-09 Hz. The
earlier T2.3/T2.4 control used `log10_γ_a = -8.5`, whose corner sits only ~4× below
that — so the bottom of the band is not yet in the pure f⁻⁴ regime and the spectrum
carries a 0.024 dex imprint of where the corner was placed. For a measurement whose
entire purpose is to isolate spectral *shape*, that is contamination.

At `log10_γ_a = -9.0` the corner is ~13× below the band and the imprint falls to
0.0025 dex. Asserted in `test/test_injection_psd.py` so the choice cannot be quietly
reverted.

## The systematic being measured

Matching at one frequency cannot match two different shapes everywhere. The residual
difference across the band **is** the systematic:

| frequency | power law | OU | OU − PL |
|---|---|---|---|
| 1/15 yr | −4.2519 | −4.43 | ≈ −0.18 dex |
| 1/5 yr (pivot) | −6.3194 | −6.3194 | 0.0000 |
| 1/yr | −9.3483 | −9.1150 | +0.2333 dex |

The OU is flatter: low at the bottom of the band, high at the top. Per-pulsar residual
RMS medians come out at 4.81 µs (power law) and 5.66 µs (OU), a 17% difference —
consistent with the OU carrying more high-frequency power.

## Caveat on the noise realisation

The injector draws from a single RNG stream seeded once, and the two modes consume
different amounts of it before reaching the white-noise draw. So the pair shares noise
*parameters* and geometry but **not** the white-noise realisation.

That contributes scatter to the measured difference, not bias, and it is averaged over
33 pulsars × 183 epochs. If it proves comparable to the systematic at analysis time
(task 5.3), the cheap fix is a second pair at another seed, or restructuring the
injector to draw noise from an independent stream.

## Reproducing

```bash
cd workflows/ng15_sgwb_demo
NOISE=../data/IPTA_MockDataChallenge2/group1_psr_noise.json

python scripts/inject_powerlaw_gwb.py --mode powerlaw \
  --aligned-dir data/mdc2_all --noise-json $NOISE \
  --out-dir data/mdc2_inject_powerlaw \
  --log10-A-gw -14.886056647693163 --gamma 4.3333333333333333 --seed 0 --overwrite

python scripts/inject_powerlaw_gwb.py --mode ou \
  --aligned-dir data/mdc2_all --noise-json $NOISE \
  --out-dir data/mdc2_inject_ou \
  --log10-ha -12.919767 --log10-gamma-a -9.0 --seed 0 --overwrite
```

Both verified through `get_processed_residuals(mode="gwb")`: residuals (183, 33), errors
(183, 33), HD (33, 33) with unit diagonal, all finite. The feathers are gitignored; the
`injection_truth.json` sidecars are tracked as the reproducible record.
