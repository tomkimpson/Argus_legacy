## Purpose

Characterise honestly what Argus's Ornstein–Uhlenbeck background model does and does not recover
when the true background is a γ = 13/3 power law, so that the published claim is bounded by measured
systematics rather than by assumption.

## ADDED Requirements

### Requirement: The kernel mismatch is quantified, not assumed

The systematic induced by fitting the OU background kernel to power-law-injected data SHALL be
measured at the scale at which results are claimed, and reported as a number, on both the
band-referenced amplitude and the Bayes factor.

#### Scenario: Amplitude systematic measured at array scale

- **WHEN** matched array-scale analyses are run on power-law-injected and OU-injected data with the
  same geometry and noise
- **THEN** the difference in recovered band-referenced amplitude is reported in dex and in units of
  the posterior width, at the documented pivot frequencies

#### Scenario: Evidence systematic measured

- **WHEN** the same matched pair of analyses is carried through the frozen evidence procedure
- **THEN** the difference in `lnB(HD/CURN)` attributable to the kernel choice is reported

### Requirement: Claims are bounded to what the kernel supports

Reported results SHALL be restricted to the background amplitude and the inter-pulsar correlation.
The system SHALL NOT present a recovered spectral index as a measurement of the background's
spectral index.

#### Scenario: Amplitude and correlation reported

- **WHEN** results are written up for a dataset
- **THEN** the band-referenced amplitude and the HD-vs-CURN comparison are reported, each with the
  measured kernel systematic attached

#### Scenario: Spectral index requested

- **WHEN** a spectral index or OU corner parameter is extracted from the posterior
- **THEN** it is presented as an internal model parameter with the f⁻⁴ ceiling stated, and not as a
  constraint on the astrophysical spectral index

### Requirement: The modelling difference is stated with every result

Any result artefact intended for external consumption SHALL state that Argus models the background
with an OU kernel whose steepest attainable residual slope is f⁻⁴, against a standard power-law
expectation of f⁻¹³ᐟ³, and SHALL cite the measured size of the resulting systematic.

#### Scenario: Result artefact produced

- **WHEN** a summary, plot set or write-up of an SGWB result is generated
- **THEN** it carries the kernel statement and the measured systematic, not a generic caveat

### Requirement: A richer kernel is considered only on measured evidence

Extending beyond the single-corner OU kernel SHALL be triggered by a measured systematic that
exceeds the reported uncertainty on the claim, not by the mismatch existing in principle.

#### Scenario: Systematic is subdominant

- **WHEN** the measured kernel systematic is small compared with the statistical uncertainty on the
  amplitude and does not change the sign or decisiveness of `lnB`
- **THEN** the single-corner OU kernel is retained and the systematic is reported

#### Scenario: Systematic dominates

- **WHEN** the measured systematic exceeds the statistical uncertainty, or flips the decisiveness of
  `lnB`
- **THEN** a richer kernel is proposed as its own change, and claims are held until it is resolved
