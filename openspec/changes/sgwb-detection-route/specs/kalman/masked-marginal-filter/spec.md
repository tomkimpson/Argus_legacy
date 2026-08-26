## Purpose

Allow the marginalized (Rao-Blackwellized) Kalman filter — the default and faster likelihood path —
to run on array data where pulsars are observed on a sparsely occupied common grid, removing the
silent fallback to the slower sequential filter.

## ADDED Requirements

### Requirement: The marginalized filter accepts missing observations

The marginalized filter SHALL accept a per-epoch, per-pulsar observation mask and condition each
update on exactly the pulsars observed at that epoch, without truncating pulsars to a shared
baseline.

#### Scenario: Masked data on the marginal path

- **WHEN** the likelihood is evaluated on masked data with the marginalized filter selected
- **THEN** it returns a finite log-likelihood computed from only the observed entries, and does not
  fall back to the sequential filter

#### Scenario: Epoch with no observations

- **WHEN** an epoch has no observed pulsars
- **THEN** the filter propagates the state without an update at that epoch and the log-likelihood is
  unaffected by that epoch

### Requirement: Masked and unmasked results agree where they must

Mask support SHALL be exact, not approximate, and this SHALL be demonstrable against the existing
paths.

#### Scenario: All-ones mask

- **WHEN** the marginalized filter is run with an all-ones mask
- **THEN** the log-likelihood equals the unmasked marginalized result to numerical tolerance, and the
  existing golden values are unchanged

#### Scenario: Agreement with the sequential path

- **WHEN** the marginalized and sequential filters are run on the same masked dataset with matched
  timing-model prior settings
- **THEN** their log-likelihoods agree to the tolerance documented for the unmasked case

#### Scenario: Absent pulsar contributes nothing

- **WHEN** a pulsar's mask is all zeros
- **THEN** the log-likelihood is identical to the value obtained by removing that pulsar from the
  dataset, up to a constant that does not depend on the sampled parameters

### Requirement: The sequential fallback guard is removed once support lands

Once the marginalized filter supports masks, requesting it with masked data SHALL succeed rather
than raising, and masked data SHALL no longer be silently redirected to the sequential filter.

#### Scenario: Explicit marginal request with a mask

- **WHEN** the marginalized filter is explicitly requested together with masked data
- **THEN** the call succeeds

#### Scenario: Automatic backend selection with a mask

- **WHEN** the backend is left to automatic selection on masked data
- **THEN** the marginalized filter is selected and no fallback warning is emitted

### Requirement: Gradients remain usable for gradient-based sampling

The masked marginalized likelihood SHALL remain differentiable with finite gradients with respect to
every sampled parameter, so it can be used with gradient-based samplers.

#### Scenario: Gradient evaluation on masked data

- **WHEN** the gradient of the masked marginalized log-likelihood is taken with respect to the
  sampled parameters
- **THEN** every component is finite, including at epochs with partial or empty observation sets
