# Spark Prediction Coverage Separation v1

Status: **staged design / not qualified / no product code materialized**

Base: `main@2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`

Tracks: #868

## Problem

Spark currently converts missing machine-readable predictions into a synthetic uniform likelihood over the outcome classes predicted by other hypotheses.

That crosses a scientific boundary:

```text
no encoded prediction
    !=
uniform predictive distribution
```

It also creates a planner/update coherence failure. A one-class experiment can have legacy `expected_information_gain() == 0.0` while `HypothesisBelief::update()` still changes belief because a matching tested hypothesis receives likelihood `0.9` and an untested hypothesis receives `1.0`.

## First additive product slice

Do not change historical Spark ranking behavior in the first tranche.

Add an explicitly non-authorizing assessment API alongside the legacy calculation:

```rust
pub struct PredictionCoverage {
    pub predicted_hypotheses: Vec<HypothesisType>,
    pub missing_hypotheses: Vec<HypothesisType>,
    pub predicted_prior_mass: f64,
}

pub struct ExperimentInformationAssessment {
    pub prediction_coverage: PredictionCoverage,
    pub conditional_eig_bits: Option<f64>,
    pub full_scope_eig_bits: Option<f64>,
}
```

### `PredictionCoverage`

Use `ALL_HYPOTHESES` as the canonical scope/order.

A hypothesis is `predicted` iff the design has at least one machine-readable `ExpectedOutcome` for it.

The coverage object records:

- exact predicted hypothesis set;
- exact missing hypothesis set;
- current prior mass represented by the predicted subset.

`complete_for_scope` is true only when `missing_hypotheses` is empty.

### Conditional EIG

Add a separately named calculation:

```text
conditional_expected_information_gain
```

This conditions the current belief on the subset with encoded predictions and computes mutual information only within that subset.

It must be documented as:

```text
EIG conditional on H belonging to the encoded-prediction subset
```

not:

```text
full experiment EIG
```

Do **not** multiply conditional EIG by `predicted_prior_mass` in the shared calculation. Such weighting is a planner policy and must remain explicit under SCI-009.

### Full-scope EIG

`full_scope_eig_bits` is available only if prediction coverage is complete.

Conceptually:

```text
if every declared hypothesis has an encoded prediction:
    full_scope_eig = Some(model_relative_eig)
else:
    full_scope_eig = None
```

`None` means **not established by this predictive model**, not zero information value.

## Compatibility boundary

The first tranche should preserve the existing:

```text
expected_information_gain()
rank_experiments()
greedy_sequence()
```

numerics as a legacy signature-model baseline.

Do not silently change historical reports in the same patch that introduces coverage semantics.

A later, independently qualified migration can move ranking/reporting onto the structured assessment.

## Required negative controls before candidate application

A future qualification lane must first establish current behavior on exact main:

1. **Single encoded class / incomplete coverage**
   - create a design with one predicted hypothesis and four missing hypotheses;
   - require legacy EIG to be `0.0`;
   - apply the sole matching observation through `HypothesisBelief::update()`;
   - require posterior belief to change.

2. **Missing prediction beats matching prediction**
   - under one encoded class, require a matching tested hypothesis likelihood to be `0.9`;
   - require an untested hypothesis likelihood to be `1.0`.

3. **Planner/update likelihood mismatch**
   - for a two-class design, reconstruct the EIG-side class likelihoods;
   - compare them with `HypothesisBelief::likelihoods()`;
   - require at least one hypothesis/outcome probability to differ.

These are negative controls only. They do not establish the replacement API as correct.

## Required post-candidate tests

After the additive assessment is applied:

1. incomplete coverage must produce `full_scope_eig_bits == None`;
2. no encoded predictions must produce `conditional_eig_bits == None` and full-scope EIG unavailable;
3. one encoded prediction may produce conditional EIG `0.0`, but still must not mint full-scope EIG;
4. complete five-hypothesis coverage must produce `full_scope_eig_bits == Some(...)`;
5. under complete coverage, the first-tranche full-scope value must equal the retained legacy signature-model EIG to numerical tolerance;
6. `predicted_prior_mass` must track the exact supplied `HypothesisBelief` rather than hypothesis count;
7. no new API may label conditional EIG as calibrated physical-world information gain.

## Relationship to #857

#857 should qualify first or in the same controlled integration lineage.

Current `OutcomeClasses` are order-dependent, so any coverage-aware EIG still inherits that ambiguity until the outcome-class representation becomes deterministic.

Preferred sequence:

```text
#857 order-invariant conservative ambiguity closure
    -> #868 explicit prediction coverage
    -> exact predictive/likelihood model identity
    -> SCI-008 uncertainty-bearing observation model
    -> SCI-009 richer planner adapter
```

## Longer-term model

The target scientific representation is not a hard class partition. It is an exact declared observation model:

```text
p(y | H, E, M)
```

where `M` may bind:

- detector resolution;
- background/noise;
- calibration state;
- measurement uncertainty;
- missing channels;
- censoring;
- OOD/applicability state;
- implementation and execution identity.

Only when those predictive distributions are normalized and complete over the declared hypothesis scope should the system issue a full Bayesian EIG coordinate.

## Non-claims

This design does not:

- calibrate Spark's hypothesis probabilities;
- choose a universal prior;
- define the final observation likelihood model;
- establish that conditional EIG should drive experiment ranking;
- establish any LCF anomaly hypothesis;
- authorize an experiment;
- establish safety.

## Queue policy

Do not open a hosted qualification PR for this tranche while the existing focused scientific lanes remain scheduler-starved.

The next transition requires an actual checkout/apply validation or an exact generated patch against the frozen base, followed by focused negative-control qualification.