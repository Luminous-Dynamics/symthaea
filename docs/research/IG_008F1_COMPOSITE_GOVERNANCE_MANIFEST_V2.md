# IG-008F1 — Composite governance manifest v2

Issue: #3297

Parent: IG-008S0 / draft #3296

## Purpose

IG-008F1 creates a monotonic successor to the v1 Mycelix composite manifest by adding the independently represented threshold-signing subsystem.

The v1 manifest is not edited.

## Core refinement: coverage is not assurance

For this manifest family:

```text
covered
= source-observed mechanism stage is represented by a content-bound component
```

and deliberately not:

```text
covered
= desired security/correctness property is satisfied
```

The frozen machine value is:

`SourceObservedMechanismRepresentedNotPropertySatisfied`.

Threshold signing is therefore covered in v2 precisely because Symthaea can reproduce the observed #959/#960 gap semantics. v2 does not describe threshold signing as secure.

## Identity

```text
manifest_id    mycelix-observed-composite-fca2c107-v2
revision       2
production     fca2c107a1ea5108823ce617ba4111b6f7f77230
authority      ObservedCompositeSlice
claim ceiling  ObservedCompositeSliceOnly
SHA-256        e3b42dbd7dacaa2f44de8bc330e85ff7d624f291d27c4a303fbaccf57091e541
```

Predecessor:

```text
mycelix-observed-composite-fca2c107-v1
366b87944e74fe01369b78a8a7ea1c34f6dee05237583a47d86e5999b93b150c
```

## Components

### Voting — retained exactly from v1

```text
profile SHA 680af4668889c299b7e0d74531f44894a64a384be71778bca54d3f21ca80ac01
corpus SHA  ee5e7649a773f564b443320689f465080d4641a0f4f09f13c0c49a7087d6dc10
Mycelix evidence head 4b4e27910a98ad393e5a508e54e0f73c48c19107
Symthaea conformance PR #3233
```

### Threshold signing — new in v2

```text
profile SHA c15dfd860b759747938af2a13129d729fa0af1e75284418c9ea6b9c172f643ac
corpus SHA  0f6532ae8e2c2e421da625592dbb3b38aa2b90c5342f46f3a305bdbec89b0269
Mycelix evidence head a580915d588338077ce6196514c43e24052f86cd
Symthaea conformance PR #3296
```

The component remains `ObservedSourceBound`. Its corpus preserves the absence of observed cryptographic/committee authority predicates rather than concealing them.

### Execution — retained exactly from v1

```text
profile SHA c977bdcef9e5faac83351050999451432b618d5cc523bece804eba5dd1ae81f6
corpus SHA  0c6669e44d6d18396ede43324f5cf3abbb25ddd3a2a9f59abb2c8a3699ba5fd4
Mycelix evidence head 197714209c60503f0fba4143409da383bc9cbf83
Symthaea conformance PR #3254
```

All three components describe the same production subject.

## Exact monotonic coverage change

v2 adds only:

```text
threshold_signing_producer_api_observed_semantics
threshold_signature_integrity_observed_semantics
```

to v1's covered stages.

v2 removes only:

`threshold_signing_as_independent_mechanism_profile`

from v1's uncovered stages.

Still uncovered:

- proposal creation/lifecycle as an independent mechanism profile;
- downstream constitutional parameter authorization;
- downstream treasury/credit authorization;
- deployment currentness.

The required-stage registry itself is unchanged from v1.

## Validator

`scripts/ig008f1_validate_composite_manifest_v2.py` first validates the exact v1 predecessor with the v1 validator and then enforces v2 lineage.

It rejects:

- a wrong predecessor commitment;
- any change to retained Voting or Execution component refs;
- any ThresholdSigning commitment/evidence-head drift;
- mixed production subjects;
- component authority promotion;
- conformance-class drift;
- extra or missing coverage changes;
- removing any uncovered stage besides threshold signing;
- required-stage registry drift;
- `ObservedEndToEnd` promotion;
- coverage semantics that imply security satisfaction;
- generic security/fairness/legitimacy verdicts;
- content-commitment drift.

## Qualification

The exact-head v2 qualifier must re-execute all three cross-repository evidence chains:

1. Voting — Mycelix A3/A4 + independent Symthaea A0;
2. ThresholdSigning — Mycelix S0/S1 + independent Symthaea S0;
3. Execution — Mycelix E0/E1 + independent Symthaea E0.

For each component the Mycelix and Symthaea canonical corpora must be byte-identical.

The workflow then validates v1, validates v2 twice byte-identically, verifies monotonic lineage, and checks all four repositories/checkouts remain immutable.

## Why v2 is still not end-to-end

Adding the signing producer closes a **coverage gap**, not every authority gap. It also does not repair #959/#960.

Therefore:

```text
Voting observed
+ ThresholdSigning observed
+ Execution observed
!= complete observed governance
!= secure governance
```

Proposal lifecycle and downstream mutation-authority stages are still outside the composite, and deployment currentness is explicitly unqualified.

## Non-claims

IG-008F1 establishes no secure threshold-signing theorem, cryptographic signature validity, complete governance coverage, deployment currentness, fairness, constitutional legitimacy, or governance safety.
