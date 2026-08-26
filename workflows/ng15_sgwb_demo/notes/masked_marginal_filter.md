# Masks on the marginalized filter

## Why

The 68-pulsar NG15 union grid is only ~41.6% occupied, so array runs carry a
missing-observation mask. Masks were wired into the sequential augmented-state filter
only (#113), so masked data fell back to it automatically — giving up the marginalized
filter's speedup on the single most expensive run of the project, and only ever on real
data, where it costs most.

## What changed

`_update_marginal` now takes the same per-epoch mask the sequential path uses, and
applies it the same way: zero the absent rows of `H_dyn` and `H_eps`, replace their
measurement variance with unity, and zero their innovation. Every timing-model
accumulator (`A`, `b`, `c`) is a quadratic form in `Psi` or `y0`, both of which vanish
on masked rows, so they need no separate treatment — the port turned out to be
mechanical, contrary to the expectation recorded in the change design.

The `use_marginal` fallback guard is gone: masked data now select the marginal filter,
and requesting it explicitly succeeds.

## The bug this exposed

Absent slots carry a **unit** variance so the innovation covariance stays invertible.
The positive-definiteness jitter was scaled as `1e-9 * trace(S) / n` — which, with
those unit placeholders in the trace, is set by the placeholders rather than by the
data. PTA innovation variances are ~1e-12, so on a masked epoch the jitter came out
~1e12 times too large.

This was not a rounding detail and it was **not** introduced by this work — it affected
the pre-existing sequential mask path from #113:

| symptom | before | after |
|---|---|---|
| MDC2, 41.6% occupancy: marginal vs sequential | 7e-4 relative (~23 nats) | 2.3e-6 relative |
| offset from masking one pulsar out entirely | −65.50 (predicted −11.03) | −11.0273 (exact to 1e-14) |

`_jitter_scale(cov, mask)` now averages over the observed diagonal only. With no mask
it is exactly `trace(cov)/n`, so the unmasked likelihood and both goldens are unchanged
bit-for-bit.

**Consequence:** any earlier result computed on masked data is suspect. Nothing in the
merged milestones used one — M1/MDC2 has no gaps and the 6-pulsar NG15 runs used the
intersection grid — but the union-grid tooling from #113 was never run in anger, and
this is why it should not have been trusted until now.

## Measured speedup

`scripts/benchmark_masked_filter.py`, MDC2 (32 pulsars x 183 epochs), CPU:

| occupancy | sequential | marginal | speedup |
|---|---|---|---|
| 0.416 | 2749 ms | 508 ms | 5.4x |
| 0.600 | 2745 ms | 510 ms | 5.4x |
| 1.000 | 2750 ms | 508 ms | 5.4x |

Occupancy does not change the cost — masking is by zeroing, not reshaping, so the array
shapes are identical either way. The speedup is larger than the ~1.4x quoted for A100 in
the filter docstring because this is CPU, where the O(d^3) update dominates more.

**Not yet measured on the real target.** The task called for the 68-pulsar union grid;
those feathers are gitignored and are built as part of the M3 data preparation. Rerun
with `--data` pointing at them, on A100, once they exist.
