# SCI-004 — Scientific Experiment Contract v1 — Summary

**Status:** architecture-only; non-authorizing; non-qualifying.

SCI-004 defines the prospective contract that later distinguishes exploratory analysis from honest confirmatory evidence.

## Core separation

```text
hypothesis
    != frozen experiment contract
    != verified preregistration chronology
    != executed experiment
    != valid measurement
    != successful outcome
    != scientific truth
    != action authority
```

The key rule is:

> Confirmatory authority requires decision-relevant semantics to have been frozen before the relevant outcome information became available to the decision process.

## Contract surface

A domain profile may compose exact identities/policies for:

```text
study class
target proposition / hypothesis set
input/data eligibility and roles
sampling/case selection
intervention/control policy
measurement specifications
outcome-observation policy
analysis implementation/model/estimator
uncertainty policy
metrics + decision criteria
multiplicity
stopping/sample-size
missingness/exclusions
blinding/custody/reveal
adaptation/active experiment selection
required SCI-003 execution profiles
reporting
```

The shared kernel does not impose one statistical design across sciences.

## Frozen contract is not chronology proof

```text
immutable contract identity
    != preregistered-before-outcome
```

A future `PreregistrationReceiptV1` must bind exact contract identity, chronology/custody evidence, and the relevant reveal boundary. A caller-supplied `preregistered=true` or a timestamp field is insufficient.

## Outcome selection is prospective

For mutable/revised external data, the contract may need to freeze:

```text
source
series/measure
release/vintage selector
transformation
missing-outcome disposition
```

This preserves the Futures theorem that raw resolution linkage is not preregistered outcome-observation semantics.

## Adaptive science remains allowed

Preregistration does not mean every future experiment must be known in advance.

A confirmatory adaptive campaign can prospectively freeze:

```text
candidate action space
selection/planner implementation
information/falsification objective
constraints/cost/risk
update rule
stopping rule
reveal policy
```

Then the selected experiments may depend on observed history without turning into unrestricted post-hoc discretion.

## Exposure matters

Prospective status depends on information available to the decision process, not only file creation time.

Potential contamination includes:

```text
human outcome exposure
model training on evaluation cases
revealed hidden benchmark reuse
post-cutoff retrieval/model knowledge
learned grammar carrying target information
confirmatory failures used for tuning
```

A later timestamp cannot erase prior exposure.

## Amendments are append-only

Pre-reveal amendment:

```text
old contract -> new immutable generation -> new chronology receipt
```

Decision-relevant post-reveal amendment cannot silently inherit the predecessor's confirmatory authority. Historical contracts are never edited in place.

## Failure/null evidence remains visible

The experiment lineage should retain:

```text
null outcomes
negative outcomes
indeterminate outcomes
missing outcomes
failed/incomplete runs
protocol deviations
predeclared exclusions
```

rather than letting unfavorable cases disappear.

## First implementation slice

Start non-executing and non-chronology-authorizing:

```text
StudyClassV1
ExperimentContractV1
ExperimentContractIdentityV1
```

The strongest first-tranche statement is only:

```text
contract semantics frozen
```

not:

```text
preregistration chronology verified
experiment valid
result passed
```

## Dependency order

```text
SCI-001 audit
    -> SCI-002 artifact identity
    -> SCI-003 execution capsule
    -> SCI-004 experiment contract
    -> SCI-005 exploratory/confirmatory separation
    -> measurement/adjudication/dependency layers
```

SCI-009 adaptive experiment design should later consume SCI-004 rather than bypass it: intelligent adaptive science and prospective rigor are compatible when the adaptation policy is itself frozen.
