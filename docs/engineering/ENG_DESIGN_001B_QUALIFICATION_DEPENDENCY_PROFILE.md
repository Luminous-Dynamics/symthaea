# ENG-DESIGN-001B — Immutable Qualification Dependencies and Receipt Admission

Status: Source/data contract  
Parent: ENG-DESIGN-001 #6005 / source PR #6006  
Authority: design-dependency composition only; no claim or physical execution authority

## 1. Purpose

ENG-DESIGN-001B defines how an engineering design packet names a qualification prerequisite without embedding live CI/provider state into the frozen design subject.

It is an adapter over existing Symthaea qualification/evidence semantics. It does **not** create another qualification engine, receipt grammar, freshness engine, provider-state model, or claim interpreter.

The core rule is:

```text
frozen dependency identity
!= hosted workflow/execution state
!= qualification receipt
```

and the downstream rule is:

```text
all prerequisite receipts admissible
!= consumer verification complete
!= consumer validation complete
!= release eligible
```

## 2. Canonical owners reused

This profile composes existing owners rather than duplicating them:

- QUAL-001 #3742 — subject, execution, conformance, and claim-interpretation separation;
- QUAL-INFRA-001 #4231 — hosted execution-state taxonomy and `QueuedInfrastructure != source failure`;
- SYM-FV-006 #5719 — machine-readable formal evidence receipt and one-primary-evidence-class rule;
- SYM-FV-007A #5962 — dependency composition cannot promote the root primary evidence class;
- SYM-FV-007B #5973 — currentness and blocking propagation.

The design profile only says what exact dependency a consumer requires and what receipt reference may later satisfy it.

## 3. Frozen dependency reference

A claim-bearing engineering dependency reference carries fields equivalent to:

```text
dependency_id
source_subject_id
source_head
qualifier_subject_id
qualifier_head
qualification_requirement
claim_scope
receipt_ref
change_impact_ref
```

The default qualification requirement in this V1 profile is:

`HostedExactHeadPassReceiptRequired`

The default frozen-source value is:

`receipt_ref = null`

That null means **required evidence not yet bound**. It does not mean failure.

## 4. Source immutability rule

The following belong to the design subject:

- dependency identity;
- exact source subject/head;
- exact qualifier subject/head;
- required qualification class;
- affected claim scope;
- change-impact reference.

The following do **not** belong in frozen design source:

- queued/running provider state;
- current run ID merely because it is latest;
- job scheduling status;
- transient PASS/FAIL status;
- retry count;
- “latest green” aliases.

Those are execution/evidence facts and belong to evidence receipts/provider envelopes.

Therefore:

```text
queue -> running -> terminal
```

must not require rewriting the design packet.

## 5. Qualification-result semantics

The profile consumes the canonical bounded vocabulary:

```text
Pass
Fail
Blocked
EnvironmentFailure
```

These remain distinct.

### Missing receipt

```text
required receipt absent
-> PendingNotFailure
```

### Pass

A positive dependency is admissible only when the receipt is current, matches the exact bound source/qualifier identity, and has result `Pass`.

Conceptually:

```text
Current(ref)
:= observed_generation = live_generation

Admissible(ref)
:= Current(ref)
 ∧ result = Pass
 ∧ exact subject identity matches
 ∧ exact qualifier identity matches
```

### Fail

`Fail` is a terminal negative qualification result for the exact subject/qualifier theorem. It blocks stronger dependent claims but does not automatically prove a broader scientific claim false.

### Blocked

`Blocked` means the required qualification theorem did not become admissibly positive. It remains distinct from scientific or product failure.

### EnvironmentFailure

`EnvironmentFailure` describes a failed qualification environment/execution prerequisite. It is not silently reclassified as product/source failure.

## 6. Staleness and drift

A historically valid PASS receipt becomes inadmissible for this dependency when its exact identity no longer matches the required dependency.

At minimum:

```text
old qualifier head + old PASS
!= PASS for new qualifier head

old source head + old PASS
!= PASS for changed source head
```

Historical evidence remains retained.

Requalification creates a new evidence lineage. It does not rewrite the old receipt.

## 7. Multiple dependencies

Dependencies are conjunctive when the design packet declares all of them required.

```text
dep A admissible
+ dep B receipt missing
!= prerequisites complete
```

No strongest-looking dependency may substitute for another independently required dependency.

Dependency composition also retains SYM-FV-007A's non-amplification boundary:

```text
dependency evidence
!= promotion of the consumer's primary evidence class
```

## 8. Consumer progression

Prerequisite qualification is only the first gate.

The generic progression is:

```text
required dependency receipts not all admissible
-> QualificationPending

all required dependency receipts admissible
+ consumer verification incomplete
-> VerificationPending

consumer verification complete
+ validation incomplete
-> VerifiedOnly

consumer verification complete
+ validation complete
-> ValidatedReleaseEligible
```

Even `ValidatedReleaseEligible` is an engineering-process disposition only. It grants no physical execution authority.

## 9. Verification and validation remain owned by the consumer

A dependency receipt can establish only the theorem stated by that dependency.

It cannot set the downstream consumer's `VER-*` cases to PASS.

It cannot set the downstream consumer's `VAL-*` case to PASS.

Therefore:

```text
dependency PASS
!= consumer verification PASS

consumer verification PASS
!= consumer validation PASS
```

A consumer must produce its own exact evidence for both.

## 10. Change impact

A dependency reference requires change-impact review when:

- bound source subject/head changes;
- bound qualifier subject/head changes;
- required qualification class changes;
- affected claim scope changes;
- receipt identity/currentness becomes stale or withdrawn under its canonical owner.

A new provider run against the **same frozen dependency** is evidence evolution, not source mutation.

## 11. Anti-forgery / anti-laundering rules

Reject at the profile boundary when:

- a source packet embeds provider queue/running status as qualification evidence;
- a caller supplies a bare `Pass` string without a receipt identity;
- a receipt refers to another source or qualifier head;
- a stale PASS is reused after subject drift;
- one dependency's receipt is reused for another dependency;
- dependency PASS is used to promote the consumer's evidence class;
- dependency PASS is used to grant actuation, procurement, allocation, deployment, or other physical authority.

## 12. Synthetic reference corpus

`eng-design-001b-qualification-dependency-profile-v1` freezes fifteen cases:

1. exact dependency with no receipt -> `QualificationPending`;
2. current exact-head PASS receipt with consumer verification unexecuted -> `VerificationPending`;
3. exact-head `Fail` -> `DependencyFailed`;
4. exact-head `Blocked` -> `DependencyBlocked`;
5. exact-head `EnvironmentFailure` -> `DependencyEnvironmentUnresolved`;
6. PASS for superseded qualifier head -> `DependencyStale`;
7. source-head drift with old PASS -> `DependencyStale`;
8. receipt subject mismatch -> `DependencyReceiptRejected`;
9. forged PASS field with no receipt identity -> `DependencyReceiptRejected`;
10. transient queue/running state embedded in frozen source -> `SourceContractRejected`;
11. two required dependencies with only one admissible -> `QualificationPending`;
12. all prerequisites admissible with consumer verification unexecuted -> `VerificationPending`;
13. verification complete while validation incomplete -> `VerifiedOnly`;
14. prerequisites + verification + validation complete -> `ValidatedReleaseEligible`;
15. dependency tries to promote evidence class or physical authority -> `AuthorityPromotionRejected`.

## 13. Intended integrations

Once independently qualified, this profile should become the common dependency-reference semantics for:

- WATER-MFG design packets;
- AGRI-MFG design packets;
- semiconductor/equipment qualification packets;
- robotics engineering design packets;
- other future ENG-DESIGN consumers.

Do not copy the receipt/currentness engine into each domain.

## 14. Claim ceiling

A source or qualifier PASS for ENG-DESIGN-001B may establish only faithful representation of these dependency/composition semantics.

It establishes no:

- truth of an underlying engineering claim;
- consumer verification completion;
- consumer validation completion;
- product safety or external certification;
- service sufficiency;
- economics;
- procurement/resource allocation;
- deployment or physical execution authority.

## 15. Next boundary

The correct next consumer topology is:

```text
frozen design packet
+ exact qualification dependency references
        ↓
externally produced exact receipts
        ↓
dependency admission
        ↓
consumer verification
        ↓
consumer validation
        ↓
bounded release-evidence review
```

No layer may silently stand in for the next.
