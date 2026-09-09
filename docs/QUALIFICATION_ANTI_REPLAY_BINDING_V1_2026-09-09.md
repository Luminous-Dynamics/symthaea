# Qualification Anti-Replay Binding v1

Date: 2026-09-09
Status: architecture / non-authorizing
Series: SCI-Q5

## Purpose

SCI-Q1, SCI-Q3, and SCI-Q4 deliberately separate three different questions:

1. Did the observed qualification checks justify a disposition under a trusted profile?
2. Which exact candidate artifact was under qualification?
3. Which exact execution occurrence produced the observations?

SCI-Q5 defines the integration invariant that prevents valid objects from those layers being replayed or spliced across different candidates, workflows, profiles, or executions.

This document does not mint materialization authority. It records the anti-replay conditions that must be qualified before such authority can exist.

## Central theorem

```text
valid qualification receipt
    + valid artifact identity
    + valid execution identity
    != valid qualified execution
```

unless all three are proven to describe the same qualification subject and the same execution lineage.

Equivalently:

```text
component validity != composite consistency
```

A green receipt for candidate A cannot authorize candidate B merely because both receipts and artifacts are individually valid.

## Current integration gap

The current SCI-Q1 candidate `QualificationReceiptV1` intentionally contains:

- the exact qualification profile structure;
- raw check observations;
- cached disposition claim;
- cached bounded reporting-authority claim.

It does not yet bind:

- repository identity;
- base commit;
- candidate patch bytes;
- result tree;
- workflow bytes;
- execution occurrence;
- execution environment;
- input/output artifact identities.

This was correct for Q1's scope: it proves interpretation semantics only.

SCI-Q3 separately binds candidate artifact coordinates such as repository, base commit, patch digest, result tree, workflow digest, profile-semantic digest, and changed paths.

SCI-Q4 separately defines execution profile, realized environment, and occurrence identity.

Therefore Q5 cannot safely be implemented as a wrapper that simply accepts three opaque witnesses. The witnesses must carry cross-references that can be checked for exact equality.

## Threat model

### 1. Receipt replay across candidates

```text
Receipt(A) = Qualified
Artifact(B) = valid
Execution(B) = valid
```

Naively combining these would let candidate B inherit candidate A's qualification disposition.

Required result: reject.

### 2. Artifact replay across executions

A valid artifact identity for candidate A is combined with an execution occurrence that actually ran candidate B or a different tree.

Required result: reject.

### 3. Workflow/instrument substitution

The candidate artifact binds workflow digest W1, but the execution occurrence ran W2.

Required result: reject even if both workflows have the same name or produce the same top-level conclusion.

### 4. Qualification-profile drift

The receipt is interpreted under profile semantics P1 while the artifact/execution lineage binds P2.

This includes silent role changes under the same human-readable profile id/version.

Required result: reject.

### 5. Observation splicing

Some observations come from execution occurrence E1 and others from E2, but the combined list is interpreted as one coherent qualification run.

Required result: reject unless the qualification profile explicitly defines a multi-occurrence aggregation protocol and every contribution is identity-bearing.

### 6. Step-result replay

A successful theorem step from an old run is combined with fresh formatting/check/lint observations from another run.

Required result: reject unless a qualified profile explicitly permits resumable multi-attempt evidence and binds the allowed attempt relationship.

### 7. Same run locator, different realized execution

A provider locator such as GitHub run/job id is reused or interpreted without verifying the bound workflow/artifact/environment evidence.

Required result: locator alone is insufficient.

### 8. Same patch digest, different base

Candidate patch bytes match, but the base commit differs.

Required result: reject because the resulting product candidate is different.

### 9. Same base+patch claim, different result tree

The bound result tree does not equal an independently derived tree after applying the candidate.

Required result: reject.

### 10. Valid receipt copied into another repository context

A receipt/artifact identity valid for `Luminous-Dynamics/symthaea` is replayed against another repository where an object id happens to have the same text representation.

Required result: reject.

## Required binding graph

A future qualified execution should have one explicit subject graph:

```text
QualificationSubjectV1
    repository
    base_commit
    candidate_patch
    result_tree
    changed_paths
    workflow
    qualification_profile_semantics
        |
        v
DeclaredExecutionProfileV1
        |
        v
ExecutionOccurrenceV1
    exact artifact subject
    exact command/instrument identity
    exact realized environment
    exact input identities
    exact output/evidence identities
        |
        v
QualificationObservationSetV1
    every observation produced by / attributed to that occurrence
        |
        v
QualificationReceiptV1
        |
        v
ValidatedQualification
```

No edge in this graph may be inferred from naming similarity or mutable external labels.

## Required additions before Q5 implementation

### A. Q1 receipt needs a subject/occurrence binding

Q1 should not be mutated while its exact candidate is under qualification.

After Q1 qualifies or fails and is repaired, a later additive qualification should introduce a binding reference equivalent to:

```text
receipt subject = exact QualificationSubjectV1
receipt occurrence = exact ExecutionOccurrenceV1
```

or an equivalent verifier-owned reference to those verified objects.

The important rule is not the exact field layout. It is that a Q1 receipt cannot remain context-free once it participates in cross-layer authority.

### B. Observations need provenance to an occurrence

Current Q1 `Observation { id, outcome }` is deliberately interpretation-only.

For Q5, either:

1. each observation carries a verified occurrence/evidence reference; or
2. the whole observation set is sealed as an output of one verified execution occurrence.

A caller-constructed vector of observation values is insufficient for action authority.

### C. Profile semantics must use one commitment protocol

Q1 currently validates the full profile structure directly. Q3 currently proposes a profile-semantic SHA-256 supplied by the qualification workflow.

Before Q5 can bind these, the system needs one qualified mapping:

```text
QualificationProfileV1 semantics
    -> canonical profile commitment
```

The commitment protocol must preserve semantic distinctions while making requirement ordering non-semantic where Q1 already treats it that way.

Do not bind Q1 and Q3 by hashing incidental serde JSON bytes without qualification.

### D. Q4 occurrence must bind the exact Q3 artifact subject

A verified execution identity must include the exact artifact identity it executed, not merely a source revision string or workflow run id.

### E. Q4 occurrence must bind the exact qualification instrument

The execution identity should bind the workflow/instrument content identity required by Q3 plus realized action/toolchain/environment identities required by Q4.

## Proposed non-authorizing Q5 object

Conceptually:

```text
QualificationBindingV1 {
    artifact_identity,
    execution_identity,
    receipt_identity,
    profile_semantics_identity,
    observation_set_identity,
}
```

A verifier checks all cross-links and returns:

```text
VerifiedQualificationBinding
```

This witness means only:

> these qualification, artifact, and execution objects form one internally coherent anti-replay lineage.

It does not mean the candidate is scientifically true, safe to deploy, or authorized for materialization.

## Receipt identity is a separate requirement

Q1 currently validates receipt semantics but does not define a canonical receipt commitment.

Q5 therefore requires a future receipt identity contract. It must bind at least:

- qualification profile semantics;
- exact observation set;
- derived disposition;
- receipt schema/version;
- qualification subject reference;
- execution occurrence reference.

The serialized cached authority claim should not be an independent source of identity or authority; validation recomputes it.

Do not use arbitrary JSON serialization as a public commitment protocol without an explicit canonicalization/version theorem.

## Observation-set identity and ordering

Qualification observations are logically keyed by requirement id. Their list order should normally not create distinct scientific meaning.

A future observation-set commitment should therefore define canonical ordering by requirement id and reject:

- duplicates;
- unknown ids;
- conflicting outcomes for the same required check;
- unbound observations from another occurrence.

This preserves Q1's existing fail-closed semantics.

## Execution-attempt semantics

Qualification workflows may be re-run. Q5 must distinguish:

```text
workflow run
    != workflow attempt
    != job
    != step
    != qualification evidence set
```

A re-run may be legitimate, but evidence from multiple attempts must not be silently spliced.

Default rule for v1:

> One qualified receipt consumes observations from one exact verified qualification occurrence/attempt.

A later profile may define a resumable protocol, but it must identify exactly which evidence can be inherited and why.

## Retrying infrastructure failures

If attempt E1 is interrupted before a scientific check runs and attempt E2 reruns the full profile, E2 can form a new qualification occurrence.

Do not mutate E1 into E2 or combine their results by default.

Historical E1 remains valid evidence of interruption.

## Anti-replay verifier checks

A future `verify_qualification_binding` should fail closed unless all required equalities hold.

At minimum:

1. receipt subject repository == artifact repository;
2. receipt subject base == artifact base;
3. receipt subject candidate patch identity == artifact patch identity;
4. receipt subject result tree == artifact result tree;
5. execution artifact identity == artifact identity;
6. execution workflow identity == artifact workflow identity;
7. receipt profile commitment == artifact profile commitment;
8. execution profile commitment == artifact/profile commitment where the profile says they are the same object;
9. receipt occurrence == execution occurrence;
10. every observation in the receipt belongs to that occurrence;
11. observation-set identity matches the execution output/evidence commitment;
12. changed-path scope matches the bound artifact scope;
13. no required coordinate is missing or unknown.

## Adversarial Q5 qualification cases

The eventual qualification-only implementation should construct valid independent Q1/Q3/Q4-style objects, then mutate one coordinate at a time.

Required negative controls:

1. Qualified receipt A + artifact B -> reject.
2. Artifact A + execution B -> reject.
3. Receipt A + execution B -> reject.
4. Same artifact except base commit -> reject.
5. Same base+patch except result tree -> reject.
6. Same artifact except workflow digest -> reject.
7. Same profile id/version but changed semantic role -> reject.
8. Same receipt except observation set from another occurrence -> reject.
9. Same run id locator but different occurrence identity -> reject.
10. Mixed observations from two attempts -> reject.
11. Receipt with duplicated/conflicting observation -> reject.
12. Changed-path scope mismatch -> reject.
13. Missing execution identity coordinate -> reject/incomplete, never silently pass.
14. A completely consistent but scientifically rejected receipt -> binding may verify, but no qualification-reporting/materialization authority follows.

The last case is essential:

```text
binding validity != qualified disposition
```

## Positive controls

The future implementation should prove at least:

- one exact same-subject/same-occurrence lineage verifies;
- a scientifically rejected receipt can still have valid anti-replay binding;
- an infrastructure-interrupted receipt can still have valid anti-replay binding;
- binding verification produces no materialization capability.

This keeps identity integrity separate from outcome interpretation.

## Cross-profile and cross-domain reuse

Q5 should be profile-driven rather than hard-coded to Rust CI.

Different qualification profiles may require different execution identities, but the anti-replay theorem is common:

```text
receipt subject == artifact subject == execution subject
```

and:

```text
receipt observations belong to execution occurrence
```

Domain-specific evidence semantics remain in the qualification profile or scientific subsystem.

## Relationship to provenance/authentication

Q5 establishes internal referential consistency, not authenticity.

An attacker who can fabricate every component consistently has not been defeated by anti-replay binding alone.

Therefore:

```text
anti-replay consistency != provenance/authentication
```

Trusted signatures, transparency logs, trusted repository channels, remote attestation, or operator authority remain later/orthogonal contracts.

## Relationship to materialization

Only after Q1-Q5 are qualified should SCI-Q6 consider a materialization capability.

Materialization should consume, not recreate:

- validated qualification disposition;
- verified artifact identity;
- verified execution identity;
- verified anti-replay binding;
- target repository/base/destination policy.

Even then, the capability must bind one exact target transition and fail after byte/base/tree drift.

## Materialization replay remains a separate problem

Q5 prevents replaying qualification evidence across subjects. It does not by itself prevent using one valid materialization authorization twice.

SCI-Q6 should therefore include one-shot/idempotent transition semantics or an explicit replay ledger/capability nonce appropriate to the repository action surface.

## Failure ontology

Q5 binding failures must not be collapsed into scientific rejection.

Suggested classes:

```text
ArtifactIdentityMismatch
ExecutionIdentityMismatch
ProfileIdentityMismatch
ReceiptIdentityMismatch
OccurrenceMismatch
ObservationProvenanceMismatch
ObservationSetMismatch
IncompleteBinding
```

These establish that evidence cannot be attributed to the proposed subject. They do not establish the candidate theorem false.

## Implementation dependency order

1. Let Q1/#1082 reach exact hosted disposition.
2. Let Q3/#1168 reach exact hosted disposition.
3. Keep Q4 architecture non-authorizing until its shared execution fields are justified against recurring domain implementations.
4. Qualify canonical profile commitment semantics.
5. Qualify execution occurrence identity.
6. Add Q1 receipt subject/occurrence binding without weakening Q1 interpretation semantics.
7. Qualify Q5 anti-replay verifier.
8. Only then design Q6 materialization authority.

## Non-goals

SCI-Q5 does not:

- prove scientific truth;
- prove independent replication;
- authenticate authors or runners;
- make a mutable CI locator immutable;
- define a universal serialization protocol;
- authorize merging or deployment;
- aggregate evidence across multiple attempts by default.

## Governing principle

> A valid statement about one artifact and one execution must never become authority over another merely because their surrounding metadata looks similar. Qualification evidence earns authority only over the exact subject and exact occurrence to which it is verifiably bound.
