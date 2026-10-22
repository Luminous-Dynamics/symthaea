# IFL-001 — Institutional Failure Learning

Status: design/contract tranche

## Purpose

Institutional Failure Learning (IFL) makes failure a typed, evidence-bearing learning event rather than an unstructured postmortem.

The system must be able to compare what was believed, what was predicted, what was authorized, what happened, and what was learned without collapsing those categories.

## Canonical loop

observation -> claim -> prediction -> decision -> action -> outcome -> prediction error -> mechanism hypothesis -> counterfactual -> constraint candidate -> future evaluation

Every transition preserves provenance, time cutoff/currentness, uncertainty, identity, authority class, and qualification ceiling.

## Typed objects

### FailureCaseV1

A bounded case-study envelope. It identifies the system, period, observed outcome, evidence set, affected domains, and explicit unknowns. It does not assert causality.

### PredictionV1

A time-bounded forecast with target, horizon, reference model/version, assumptions, uncertainty representation, and falsification condition.

### OutcomeV1

An observed result bound to an exact measurement definition, observation window, source/evidence identity, and currentness metadata.

### PredictionErrorV1

A deterministic comparison between a prediction and its corresponding outcome. It records the comparison method, scale/unit, tolerance, and error classification without inventing causal explanation.

### MechanismHypothesisV1

A falsifiable hypothesis about a mechanism that may explain an observed error or failure. It records supporting evidence, counterevidence, competing hypotheses, confidence/uncertainty, and a claim ceiling.

### ConstraintCandidateV1

A proposed change to a model, test, monitoring rule, or governance process derived from a failure analysis. It is a proposal, not authority. Adoption requires an explicit external authorization/governance event.

## Failure taxonomy v0

- CapitalMismatch
- LiquidityFragility
- FixedVariableMismatch
- Overexpansion
- GovernanceCapture
- ConflictOfInterest
- IncentiveMisalignment
- EvidenceFailure
- ForecastFailure
- ModelDrift
- DemandRegimeChange
- OperationalComplexity
- DependencyConcentration
- CoordinationFailure
- FeedbackFailure

Taxonomy members are hypotheses/labels for analysis, not conclusions about any historical organization.

## Core invariants

1. Missing knowledge is not false evidence.
2. Historical analogy is not causal proof.
3. Prediction is not observation.
4. Recommendation is not authorization.
5. Failure classification remains revisable.
6. Contradictory evidence remains explicit.
7. Attestation count cannot resolve contradiction.
8. A replay qualifies only the tested model and fixture boundary.
9. Postmortem output cannot silently escalate authority.
10. Every learned constraint must retain the failure/prediction/outcome lineage that motivated it.

## WeWork fixture

The motivating case should be represented as a bounded historical fixture using primary filings and dated evidence. The fixture must preserve the distinction between documented facts, management statements, analyst interpretation, and Symthaea-generated hypotheses.

The first fixture should test structural mismatch between long-lived fixed commitments and variable demand/revenue, liquidity sensitivity, and reversibility of obligations. It must not encode a universal claim that the same mechanism explains every organizational failure.

## Qualification ladder

G0 schema parses and canonicalizes.

G1 mutation tests reject identity/provenance/semantic tampering.

G2 permutation tests prove order-independent canonical results where ordering is semantically irrelevant.

G3 prediction/outcome pairing is exact and cannot silently compare incompatible measures.

G4 mechanism hypotheses require explicit supporting and counterevidence references.

G5 counterfactual envelopes are deterministic and authority-free.

G6 longitudinal evaluation measures whether the same failure mode recurs after a learned constraint is adopted.

G7 Mycelix receipt export preserves complete lineage without granting authority.

## Explicit non-goals

No autonomous governance, financial advice, physical actuation, authority escalation, or automatic conversion of historical cases into universal rules.
