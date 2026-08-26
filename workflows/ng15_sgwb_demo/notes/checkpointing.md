# Checkpointing long NUTS runs

## Why

Argus wrote results only after `mcmc.run` returned, so a run that hit its walltime,
was preempted, or crashed lost everything. That is not a rare event here: the M1 Stage C
array run took ~9.5 h, the full-array M3 run will be longer, and OzSTAR's `milan-gpu`
partition has had both long backlogs and outages. Losing a day of A100 time to a
timeout is a scheduling problem that should not also be a data-loss problem.

## How it works

Sampling runs as a sequence of segments. NumPyro exposes the sampler state between
runs, so segment *n+1* starts from segment *n*'s final state via `post_warmup_state`
instead of restarting: warmup is paid once, and the concatenated draws are the same
Markov chain a single uninterrupted run would have produced.

After each segment two files are written atomically (temp file then rename, so an
interrupted write cannot be loaded as a valid checkpoint):

- `<output_id>_checkpoint.pkl` — sampler state, run fingerprint, progress.
- `<output_id>_partial.nc` — the posterior accumulated so far, marked incomplete.

## Configuration

```ini
[Checkpointing]
enabled = true
interval = 250     ; draws per chain per segment
resume = false     ; set true to continue from an existing checkpoint
```

`directory` and `output_id` are injected by the run harness from the run's own output
paths, so an existing config gains checkpointing by setting `enabled` and `interval`
alone. With the section absent or `enabled = false`, behaviour is exactly as before.

To resume after a timeout, set `resume = true` and resubmit the same job.

## The fingerprint gate

A resume is refused unless the model, priors and sampler settings match those the
checkpoint was written under. The fingerprint covers the mode, pulsar count, draw and
chain counts, seed, NUTS settings, and a value-based summary of the prior
specification. It is deliberately insensitive to formatting and output paths, so
reformatting a config does not invalidate a checkpoint, but moving a prior bound does.

Without this, resuming after an edit would concatenate draws from two different
posteriors into one file that nothing downstream could detect as invalid.

## Partial results are labelled

An incomplete run's output carries `argus_sampling_complete = 0` plus the draw count
reached. `run_inference` logs a warning when it saves one, and
`checkpointing.is_complete` / `describe_progress` read it back. Results predating
checkpointing carry no marker and are treated as complete, which they are.

## Verified

- A three-segment run reproduces an unsegmented run's draws to 1e-10 (same seed) — the
  continuation is exact, not merely statistically equivalent.
- A simulated walltime kill after the first checkpoint leaves 8 of 12 draws on disk,
  marked partial; resuming completes the run and matches the unsegmented reference.
- A resume against a changed model is refused.
- Overhead: ~0.0 s per segment on the smoke run (24 draws, 3 segments). The kernel is
  reused, so JAX does not recompile between segments; the only cost is writing the
  accumulated posterior once per segment, which scales with posterior size rather than
  with sampling time. Re-measure on the first 68-pulsar run and pick `interval` so that
  a segment is ~1 h of sampling.
