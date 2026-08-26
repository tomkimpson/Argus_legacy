## Purpose

Produce a trustworthy Bayes factor between the Hellings–Downs-correlated SGWB model and common
uncorrelated red noise, from Argus posteriors, at the dimensionality of a real pulsar-timing array
(68-D and above) where posterior-reuse estimators are known to break down.

## ADDED Requirements

### Requirement: Evidence estimator reports its own reliability

Every Bayes-factor estimate the system produces SHALL be accompanied by machine-checkable
reliability diagnostics, and the system SHALL refuse to report a point estimate as usable when
those diagnostics fail. A silent numerical result is not an acceptable output.

#### Scenario: Diagnostics pass

- **WHEN** the estimator runs on a posterior whose reliability diagnostics are within their
  documented tolerances
- **THEN** it emits `lnB` with an uncertainty and a `reliable: true` verdict together with the
  diagnostic values

#### Scenario: Diagnostics fail

- **WHEN** the estimator runs on a posterior where a diagnostic breaches tolerance (for example a
  degenerate importance-weight distribution, or no stable plateau across the estimator's tuning
  parameter)
- **THEN** it emits `reliable: false`, names the failing diagnostic, and does not present the
  numerical value as a result

#### Scenario: Reproducing the known LHM failure

- **WHEN** the learned-harmonic-mean estimator is run on the 68-D MDC2 Stage C posterior
- **THEN** it reports `reliable: false` rather than returning a bare `nan` or an uncalibrated number

### Requirement: Estimator selection is decided by a bake-off against a known answer

The production evidence procedure SHALL be selected by comparing candidate estimators on a case
whose answer is independently established, not by assertion. At least two independent candidate
estimators SHALL be evaluated.

#### Scenario: Candidates agree

- **WHEN** the candidate estimators are run on the same MDC2 Stage C posterior family
- **THEN** their `lnB` estimates are compared, and agreement within their combined stated
  uncertainties is recorded as the selection evidence

#### Scenario: Candidates disagree

- **WHEN** the candidates disagree beyond their combined stated uncertainties
- **THEN** neither is promoted to the production procedure until the disagreement is diagnosed and
  resolved in writing

### Requirement: Validation against a known-signal case gates production use

An evidence estimator SHALL NOT be used on real data until it returns a decisively positive
`lnB(HD/CURN)` on a dataset with a known injected Hellings–Downs signal, and a value consistent
with no correlated signal on a matched dataset without one.

#### Scenario: Injected-signal case

- **WHEN** the estimator is run on the MDC2 dataset 2b array posterior (known injected GWB)
- **THEN** it returns `lnB(HD/CURN) >= 3` with `reliable: true`

#### Scenario: Null case

- **WHEN** the estimator is run on a matched dataset with no injected correlated signal
- **THEN** it returns `lnB(HD/CURN)` consistent with zero within its stated uncertainty

### Requirement: The production procedure is frozen before real-data use

Once selected and validated, the evidence procedure SHALL be recorded as a fixed, versioned
configuration, and the same configuration SHALL be applied unchanged to every subsequent dataset in
the scale-up. Any change to it after freezing SHALL invalidate downstream results until they are
re-run.

#### Scenario: Applying the frozen procedure

- **WHEN** the array analysis advances to a new dataset in the scale-up
- **THEN** the recorded frozen configuration is used without modification, and the run records
  which frozen version it used

#### Scenario: Procedure changed after freezing

- **WHEN** the frozen configuration is modified
- **THEN** results previously produced under the old version are marked stale and are not combined
  with new ones

### Requirement: Bayes factors remain comparable across models

The HD and CURN models compared SHALL differ only in the overlap reduction function, with identical
data, noise treatment, priors and GW parameterization, so that the reported `lnB` isolates the
correlation hypothesis.

#### Scenario: CURN construction

- **WHEN** the CURN alternative is constructed from an HD analysis
- **THEN** the only substituted quantity is the inter-pulsar correlation matrix, replaced by the
  identity, and the run records a check confirming every other input is unchanged
