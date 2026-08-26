## Purpose

Make long sampling runs survivable: a multi-day array run that hits its walltime, is preempted or
crashes must leave usable partial results and be resumable, instead of losing everything as it does
today.

## ADDED Requirements

### Requirement: Long runs persist state during sampling

A sampling run SHALL periodically persist enough state to disk during sampling that a run
interrupted at any point after the first checkpoint leaves usable output, rather than writing results
only on successful completion.

#### Scenario: Run interrupted mid-sampling

- **WHEN** a run is terminated after at least one checkpoint has been written
- **THEN** the samples drawn up to that checkpoint, and the diagnostics available for them, are
  present on disk

#### Scenario: Run completes normally

- **WHEN** a run finishes sampling normally
- **THEN** the final outputs are written as before and are unchanged relative to a run without
  checkpointing enabled

### Requirement: Interrupted runs can be resumed

A run SHALL be resumable from its most recent checkpoint, continuing sampling rather than restarting,
and the combined result SHALL be statistically equivalent to an uninterrupted run of the same length.

#### Scenario: Resume after timeout

- **WHEN** a run that was interrupted is relaunched in resume mode against its checkpoint
- **THEN** it continues from the recorded sampler state and produces the remaining samples

#### Scenario: Equivalence to an uninterrupted run

- **WHEN** a short run is executed uninterrupted, and the same run is executed with an induced
  interruption and resume, under the same seed and configuration
- **THEN** the resulting posteriors agree to within Monte Carlo error, and the diagnostics are
  computed over the combined chains

#### Scenario: Configuration changed between checkpoint and resume

- **WHEN** a resume is attempted against a checkpoint written under a different model or sampler
  configuration
- **THEN** the resume is refused, naming the mismatch, rather than silently mixing incompatible
  samples

### Requirement: Checkpointing does not materially slow sampling

Checkpointing SHALL be configurable and SHALL impose a small, measured overhead relative to the
run it protects.

#### Scenario: Overhead measured

- **WHEN** a benchmark run is executed with and without checkpointing at the configured interval
- **THEN** the wall-clock overhead is measured and recorded, and is within the documented tolerance

#### Scenario: Checkpointing disabled

- **WHEN** checkpointing is disabled in configuration
- **THEN** the run behaves exactly as it did before this capability existed

### Requirement: Partial results are labelled as partial

Output produced from an incomplete run SHALL be distinguishable from output of a completed run, and
SHALL NOT be used for a reported result without the convergence checks that apply to completed runs.

#### Scenario: Reading partial output

- **WHEN** outputs from an interrupted run are loaded
- **THEN** they carry a marker recording that sampling did not complete, along with how many samples
  of the target were drawn
