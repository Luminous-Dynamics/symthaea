# Assurance Case V1

Status: review/composition metadata only  
Tracking issue: #5780  
Parent: Proof Challenge / Solution V1 / #5779

## Purpose

Give reviewers a compact engineering argument above individual proofs without collapsing proofs, tests, provenance, runtime qualification, vulnerability analysis, and operational controls into one vague `verified` label.

An assurance case answers:

```text
CLAIM          what are we asserting?
FORMAL         which theorem/refinement capsules support it?
OTHER EVIDENCE what tests, provenance, vulnerability and lifecycle evidence support it?
TRUST          what remains assumed?
GAPS           what remains unproved or unqualified?
REBUTTALS      what known evidence narrows or attacks the claim?
CHANGE         what changed since the last review?
RISK TIER      how much review is required?
```

## Core rule

```text
assurance case != theorem
more evidence != automatic stronger claim
formal theorem != lifecycle assurance
risk tier != confidence score
```

The assurance layer composes existing evidence identities; it does not mint stronger evidence classes.

## Assurance node

`AssuranceCaseNodeV1` binds:

- stable node ID;
- human claim and claim type;
- exact subject identity;
- supporting evidence references;
- trust inventory;
- explicit gaps;
- rebuttals/known counterexamples;
- claim ceiling;
- operational/lifecycle prerequisites;
- risk tier;
- status;
- previous node identity and semantic delta;
- human/independent review requirement.

## Evidence references

References are typed, for example:

- `FormalCapsule`
- `RuntimeQualification`
- `DifferentialTest`
- `Provenance`
- `VulnerabilityAnalysis`
- `OperationalControl`
- `IndependentReview`
- `ChallengeSolution`

A runtime qualification reference cannot be relabeled as a formal theorem merely because it supports the same high-level claim.

## Relations

Use a small argument vocabulary:

- `Supports`
- `DependsOn`
- `Refines`
- `Rebuts`
- `Gap`
- `Supersedes`

Known rebuttals and gaps are first-class edges, not prose hidden below the conclusion.

## Risk tiers

### R0 MetadataOnly

No semantic claim, assumption, dependency, trust, coverage, gap, or rebuttal change. Machine checks may be sufficient under repository policy.

### R1 LocalSemantic

Local statement/spec/assumption change with no composition or trust-boundary change. Focused semantic review required.

### R2 Composition

Supporting evidence, coverage, dependency, gap, or composition changes. Review the affected DAG frontier.

### R3 TrustBoundary

New axiom, solver, checker, extraction boundary, external provider, compiler assumption, or other trust root. Independent qualification/review required.

### R4 AuthorityCritical

Security/cryptography/safety or other authority-expanding claims. Require challenge/solution separation and an admitted independent checker profile where the lane supports one, plus explicit human signoff.

Risk tier determines review depth, not theorem truth or a confidence percentage.

## Trust inventory

Display trust concentration directly instead of inventing a numeric score. Categories may include:

- proof kernel/assistant;
- extraction/refinement bridge;
- compiler/toolchain;
- SMT solver/checker;
- axioms;
- external libraries/specifications;
- runtime/OS/hardware;
- operational assumptions;
- unverified implementation boundaries.

## Gaps and rebuttals

An unresolved gap must narrow status to `qualified-with-gaps` or `blocked`; it cannot remain `unconditional-current`.

An open rebuttal/counterexample likewise prevents an unconditional claim until resolved, narrowed, or superseded.

## Lifecycle composition

Formal evidence may be combined with exact-head build/qualification receipts, fuzzing, differential tests, configuration/provenance controls, vulnerability analysis, deployment controls, and independent review.

The evidence type remains visible at every join.

## Negative controls

Reject at least:

1. current parent claim when a required child is superseded/blocked;
2. a new trust root without R3/R4 escalation;
3. open rebuttal while status remains unconditional;
4. runtime evidence relabeled as formal theorem evidence;
5. R0 despite semantic changes;
6. R4 accepted without challenge/solution evidence and required independent/human review metadata;
7. unresolved gap hidden by unconditional status.

## V1 scope

V1 validates the argument metadata with synthetic fixtures. It does not claim an assurance standard certification, independent audit, or production UI. A future UI should render the same canonical node rather than inventing separate presentation semantics.
