# SCI-007 — Scientific Claim / Falsifier Contract v1 — Summary

SCI-007 defines the smallest generic falsification layer between the SCI-006 dependency graph and later Theory Atlas argument/disposition semantics.

## Central boundary

```text
scientific proposition
    != prediction
    != falsifier specification
    != experiment contract
    != observation
    != falsifier evaluation
    != scientific disposition
    != truth
    != action authority
```

The two most important negative rules are:

```text
NotTriggered != proposition true
Triggered    != universal refutation
```

## What SCI-007 owns

- exact falsifier target binding;
- prospective falsifier specification;
- applicability conditions;
- contradictory-observation policy identity;
- evaluator/analysis identity;
- uncertainty-handling policy reference;
- exact outcome classes;
- future falsifier-evaluation receipt shape;
- preservation of null/inconclusive/non-evaluable outcomes;
- SCI-006 dependency lineage for falsification attempts.

## What it deliberately does not own

- immutable proposition semantics (#729 direction);
- general support/opposition/defeater argument topology (#701 direction);
- universal refutation policy;
- uncertainty semantics (SCI-008);
- experiment selection (SCI-009);
- campaign orchestration (SCI-013);
- Theory Atlas disposition (SCI-014);
- scientific truth or action authority.

## Prospective chain

```text
target proposition
    -> frozen falsifier specification
    -> verified preregistration chronology
    -> prospective eligible evidence
    -> SCI-003 execution / observation
    -> falsifier evaluation
```

Post-hoc contradiction remains useful scientific evidence, but it does not receive preregistered falsifier status retroactively.

## Outcome vocabulary

At minimum:

```text
Triggered
NotTriggered
Inconclusive
NotEvaluable
NotApplicable
ExecutionFailed
MeasurementInvalid
ProtocolDeviation
```

Invalid or incomplete experiments must not be counted as theory survival.

## Applicability

A falsifier can bind requirements such as:

```text
measurement validity
instrument/calibration
population/domain/regime
input/state range
intervention fidelity
solver capability
boundary conditions
identification assumptions
observation completeness
uncertainty representation
```

Evidence outside those conditions does not automatically become `NotTriggered`.

## Falsifier vs defeater

A falsifier tests a claim/prediction.

A defeater attacks an evidence/inference path.

A triggered-looking result may therefore be undercut by bad calibration, provenance, execution, sampling, or identification without implying that the proposition is true.

## Dependencies

Multiple falsification attempts can still share the same:

```text
dataset
instrument
calibration
preprocessing
model / estimator
verifier
learned grammar
apparatus
hidden benchmark
```

Therefore multiple falsifier objects do not establish independent falsification or replication.

## First implementation slice

Keep it non-evaluating:

```text
FalsifierSpecificationV1
FalsifierApplicabilityProfileV1
FalsifierOutcomeClassV1
```

Then qualify one narrow evaluator in a domain with already-strong falsifier semantics.

## Dependency chain

```text
SCI-001
 -> SCI-002 artifact identity
 -> SCI-003 execution capsule
 -> SCI-004 experiment contract
 -> SCI-005 exploratory/confirmatory separation
 -> SCI-006 evidence dependency graph
 -> SCI-007 falsifier specification/outcome lineage
 -> SCI-008 uncertainty-bearing observations
 -> SCI-009 experiment design
 -> SCI-013 falsification campaign
 -> SCI-014 Theory Atlas integration
```

SCI-007's goal is to make scientific claims mechanically vulnerable to exact evidence without letting falsification collapse into confidence arithmetic or truth labels.
