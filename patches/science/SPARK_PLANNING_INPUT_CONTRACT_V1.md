# Spark Planning Input Contract v1 — staged qualification design

Status: **queue-neutral design artifact only**

Base: `main@2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`

Related issues:

- #857 — order-invariant / observation-profile-aware ambiguity relation
- #868 — missing prediction coverage is not a likelihood
- #885 — declared target scope, predictive coverage and discrimination are distinct

This file is not an apply-ready patch, not execution evidence, and not a scientific qualification result.

## Purpose

Define the minimum Spark-local planning-input boundary that should exist before `OutcomeClasses`, posterior update, EIG, ranking, or SCI-009 adapters consume an `ExperimentDesign`.

The target theorem is:

```text
raw ExperimentDesign
    !=
structurally admissible planning model
    !=
complete predictive model
    !=
calibrated likelihood
    !=
scientifically valid experiment
```

## Why this is required

Current public structures permit several categories of semantically different state to coexist without validation:

```text
ExperimentDesign::hypotheses_tested
ExperimentDesign::expected_outcomes
instrumentation / channel availability
cost / duration / priority metadata
```

Current planning code then derives outcome classes and numeric information quantities directly from these records.

A malformed or ambiguous record should become an explicit unavailable/invalid planning state, not a precise-looking score.

## 1. Structural prediction validity

For the current signature-model profile, each machine-readable `ExpectedOutcome` should satisfy at least:

```text
predicted_rate_range.lower is finite
predicted_rate_range.upper is finite
0 <= lower <= upper
predicted_energy_mev is finite
predicted_energy_mev >= 0
```

The v1 profile should permit **at most one** encoded outcome per `HypothesisType`.

Reason:

```text
same hypothesis + contradictory duplicate outcomes
    !=
one well-defined signature-level predictive model
```

If a richer future hypothesis intentionally has a multimodal prediction, it should use an explicit distribution/mixture representation rather than duplicate v1 rows.

## 2. Numeric planner metadata validity

Values used by selection/ranking should be structurally valid before arithmetic:

```text
estimated_cost_usd finite and >= 0
duration_months finite and >= 0
priority finite and within its declared 0..=1 semantics
```

Passing these checks establishes only numeric structural admissibility.

It does not establish that costs/durations are accurate or that priority is scientifically justified.

## 3. Separate the three hypothesis concepts

The planning boundary must preserve:

```text
DeclaredTargetScope
EncodedPredictionScope
DerivedDiscrimination
```

### DeclaredTargetScope

The experiment designer's scientific question/target.

Current compatibility source: `hypotheses_tested`, but the field should not be interpreted as evidence that discrimination exists.

### EncodedPredictionScope

Canonical set of hypotheses appearing in valid machine-readable `expected_outcomes`.

This is model support, not declared scientific intent.

### DerivedDiscrimination

A result of the exact observation/measurement model.

It is not copied from either of the two lists above.

## 4. Scope diagnostics are not all hard failures

A v1 assessment should distinguish hard-invalid states from scientifically meaningful mismatches.

Illustrative diagnostics:

```text
CompleteTargetPredictionCoverage
DeclaredTargetWithoutPrediction
PredictionOutsideDeclaredTarget
NoDeclaredTarget
CharacterizationOnly
NoDiscriminationEstablished
DuplicatePredictionForHypothesis   // hard invalid under v1
InvalidPredictionNumeric           // hard invalid
InvalidPlannerNumeric              // hard invalid for ranking
```

Do **not** force:

```text
DeclaredTargetScope == EncodedPredictionScope
```

The mismatch may be intentional and should remain visible.

## 5. Observation-profile binding

Prediction discrimination must bind an exact observation profile.

Minimum future Spark-local direction:

```text
SignatureObservationProfileV1 {
    neutron_rate_available,
    neutron_energy_available,
    neutron_energy_resolution_mev,
}
```

Then:

```text
pairwise ambiguity
outcome-class closure
observation matching
EIG
posterior update
```

must all declare which profile/model identity they use before planner/update coherence is claimed.

Missing energy is an unavailable channel, not automatic agreement.

## 6. Prediction coverage / likelihood boundary

After structural validation, compute predictive coverage explicitly:

```text
PredictionCoverageV1 {
    predicted_hypotheses,
    missing_hypotheses,
    predicted_prior_mass,
    complete_for_declared_scope,
    complete_for_full_model_scope,
}
```

Rules:

```text
missing prediction != uniform likelihood
coverage != uncertainty
coverage != discrimination
```

Full-scope Bayesian EIG is unavailable unless the exact planning profile supplies a normalized `p(y | H, E, M)` for every hypothesis in its declared probabilistic scope.

A domain may define an explicit completion/imputation model, but that model receives its own identity.

## 7. Conservative v1 planning flow

```text
RawExperimentDesign
  -> StructuralPlanningValidation
  -> DeclaredTarget / EncodedPrediction assessment
  -> ObservationProfile-bound ambiguity model          (#857)
  -> PredictionCoverage assessment                      (#868)
  -> planner coordinates / Pareto frontier              (SCI-009)
```

No stage grants experiment execution authority.

## Required negative controls for a future executable qualification

### A. Non-finite prediction

Inject `NaN` or infinity into a rate bound or predicted energy.

Current-main risk to demonstrate before repair: ordinary float comparisons can turn invalid values into apparent non-overlap/distinctness.

Post-repair requirement: fail closed before outcome-class/EIG evaluation.

### B. Reversed range

Use `(1000.0, 10.0)`.

Post-repair: structurally invalid, no EIG.

### C. Negative prediction

Use a negative predicted neutron-rate bound.

Post-repair: structurally invalid under signature-model v1.

### D. Duplicate hypothesis

Encode the same `HypothesisType` twice with contradictory ranges.

Current-main behavior must be characterized; post-repair the v1 signature model rejects it rather than selecting one class by incidental order.

### E. Existing target/prediction mismatches

Exercise at least:

```text
Hydrogen Control
Neutron Energy Spectroscopy
NASA LCF Replication
```

Post-repair they must preserve declared-target and encoded-prediction sets separately.

### F. Coverage without discrimination

Construct complete valid prediction coverage whose observations are all ambiguous under the declared observation profile.

Post-repair:

```text
coverage = complete
discrimination = zero/ambiguous
```

### G. Channel availability

Use predictions separated only by neutron energy.

With energy available: may discriminate.

With energy unavailable: must remain ambiguous.

## Implementation sequencing

Do not open another hosted workflow while the current focused qualification backlog remains starved.

When capacity is available:

1. implement a small Spark-local structural validator;
2. qualify hostile invalid-input controls;
3. add scope/coverage diagnostics without changing historical ranking behavior;
4. qualify #857 observation-profile-aware deterministic ambiguity;
5. migrate new planner APIs to explicit full/incomplete likelihood semantics;
6. retain legacy signature EIG as a clearly named compatibility baseline until consumers migrate.

## Deliberate non-claims

This contract does not establish:

- physical correctness of any prediction;
- calibrated probabilities;
- experimental validity;
- hypothesis truth;
- causal identification;
- safety, consent or ethical admissibility;
- universal experiment utility;
- action or execution authority.

It establishes only the design target for making Spark's machine-readable experiment planning inputs explicit, validated, and semantically non-laundering before richer SCI-008/SCI-009 integration.