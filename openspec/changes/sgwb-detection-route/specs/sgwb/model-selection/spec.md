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
`lnB(HD/CURN)` on a dataset with a known injected Hellings–Downs signal, and SHALL additionally be
shown to respond to the inter-pulsar correlation pattern rather than to the flexibility of its
priors. A positive control on its own does not distinguish the two.

The injected-signal case is pinned to **MDC2 dataset 1b**, not 2b. 2b is a published non-detection
— Hazboun et al. (arXiv:1912.12939) report Bayes factors of 1.1–2.6 against their own threshold of
3 and an amplitude upper limit below the injection — so `lnB >= 3` was never reachable on it by any
method, and a gate demanding it tested the dataset rather than the estimator. 1b is strongly
detected in the same paper and is the dataset the gate can actually discriminate on.

#### Scenario: Injected-signal case

- **WHEN** the estimator is run on the MDC2 dataset 1b array posterior (known injected GWB)
- **THEN** it returns `lnB(HD/CURN) >= 3` with `reliable: true`

#### Scenario: Falsification by sky scramble

- **WHEN** the estimator is run on the same injected-signal data with the pulsar sky positions
  scrambled, so the correlation pattern is destroyed and every other input is unchanged
- **THEN** it returns a `lnB(HD/CURN)` decisively below the unscrambled value with `reliable: true`,
  demonstrating that the evidence tracks the correlation pattern and is not an artefact of the
  red-noise priors

  Note the scrambled value is expected to be **negative, not zero**: a wrong correlation pattern
  describes signal-bearing data worse than no correlation does. "Consistent with zero" is the
  criterion for the no-injection case below, where there is no correlation to mis-describe.

#### Scenario: Null case — no injected correlated signal

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
