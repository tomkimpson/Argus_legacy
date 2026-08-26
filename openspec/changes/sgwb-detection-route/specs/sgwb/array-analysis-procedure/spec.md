## Purpose

Define the end-to-end procedure that takes a pulsar-timing dataset to an SGWB amplitude and a
calibrated HD-vs-CURN comparison, and the staged scale-up that carries it from a mock array with
known truth to the full real array without changing it along the way.

## ADDED Requirements

### Requirement: Noise is constrained in two stages

The array analysis SHALL constrain per-pulsar red-noise parameters from single-pulsar runs and carry
them into the array run as informative priors, with the per-pulsar parameters still sampled. Fixing
per-pulsar noise at point estimates SHALL be available only as a diagnostic pilot, not as the
procedure that produces reported results.

#### Scenario: Array run with empirical priors

- **WHEN** the array stage runs on a dataset whose single-pulsar stage has completed
- **THEN** each pulsar's red-noise parameters are sampled under priors derived from that pulsar's
  own single-pulsar posterior

#### Scenario: Fixed-noise pilot

- **WHEN** a fixed-noise array run is performed
- **THEN** its output is labelled as a pilot and is not used as the reported amplitude or evidence

#### Scenario: Single-pulsar stage incomplete

- **WHEN** the array stage is launched before every pulsar in the dataset has a converged
  single-pulsar posterior
- **THEN** the run is refused, naming the pulsars that are missing or unconverged

### Requirement: Every sampled run must pass convergence checks before use

A run's outputs SHALL NOT be used for a reported result until convergence diagnostics pass on the
sampled parameters. Diagnostics SHALL be computed on the parameters actually sampled, not only on
derived quantities.

#### Scenario: Converged run

- **WHEN** all sampled sites satisfy the documented r-hat and effective-sample-size thresholds with
  divergences below the documented tolerance
- **THEN** the run is marked usable and its diagnostics are stored with its outputs

#### Scenario: Stuck chain

- **WHEN** one chain's per-chain median for a sampled site is displaced from the others, or a chain
  shows zero within-chain variance
- **THEN** the run is marked failed with that chain identified, and is not rescued by discarding the
  chain for the reported result

### Requirement: The GW parameterization avoids the amplitude–corner ridge

Array-stage runs SHALL sample the GW background in a parameterization that decouples the
data-constrained band amplitude from the unconstrained along-ridge direction, and SHALL record which
parameterization was used.

#### Scenario: Array run launched

- **WHEN** an array-stage run is configured
- **THEN** the ridge parameterization is selected by default and recorded in the run metadata

#### Scenario: Direct parameterization requested

- **WHEN** the direct amplitude/corner parameterization is selected instead
- **THEN** the run is treated as a diagnostic comparison, and the known stalling behaviour is noted
  with its outputs

### Requirement: Scale-up proceeds in gated stages

The procedure SHALL be exercised on a mock array with known truth, then on an intermediate
long-baseline real subset, then on the full real array. Each stage SHALL pass its acceptance gate
before the next is launched.

#### Scenario: Mock stage gate

- **WHEN** the mock-array stage completes
- **THEN** it is accepted only if the injected amplitude is covered by the posterior and the frozen
  evidence procedure returns a decisively positive, reliable `lnB`

#### Scenario: Subset stage gate

- **WHEN** the intermediate real subset stage completes
- **THEN** it is accepted only if the run converges, the amplitude is consistent with the published
  result for comparable data, and the evidence procedure reports `reliable: true`

#### Scenario: Gate fails

- **WHEN** a stage fails its gate
- **THEN** the next stage is not launched, and the failure and its diagnosis are recorded before any
  remedy is attempted

### Requirement: The procedure is frozen after the intermediate stage

After the intermediate subset stage is accepted, the analysis configuration SHALL be recorded as
frozen, and the full-array stage SHALL apply it without modification other than the dataset and the
computational resources requested.

#### Scenario: Full-array stage launched

- **WHEN** the full-array stage is configured
- **THEN** its configuration differs from the frozen one only in dataset paths, pulsar list and
  resource requests, and the run records a diff confirming this

### Requirement: Real-array data preparation is explicit and checkable

Preparation of a real dataset for the array stage SHALL produce epoch-aligned per-pulsar data on a
common grid with an explicit per-epoch observation mask, and SHALL report the resulting coverage.

#### Scenario: Alignment produced

- **WHEN** a real dataset is prepared for the array stage
- **THEN** the number of joint epochs, per-pulsar retention and overall grid occupancy are reported,
  and each pulsar's mask marks its unobserved epochs

#### Scenario: Coverage too sparse

- **WHEN** the resulting grid occupancy or per-pulsar retention falls below the documented floor
- **THEN** preparation reports the shortfall and the dataset is not passed to the array stage until
  the grid choice is revisited
