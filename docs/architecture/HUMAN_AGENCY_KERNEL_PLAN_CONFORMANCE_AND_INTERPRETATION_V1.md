# Human Agency Kernel — Plan Conformance & Bounded Interpretation v1

Status: HAK-008 architecture/tooling candidate.

Parent stack:

- HAK-001 — semantic separation;
- HAK-002 — authority provenance and lineage;
- HAK-003 — transformation validity and conservation;
- HAK-004 — agency/rights floor and constitutional continuity;
- HAK-005 — proof obligations and conformance profiles;
- HAK-006 — audit-only conformance-manifest linting;
- HAK-007 — qualification-plan, subject, execution, observation and receipt separation;
- HAK-008 — exact plan conformance and bounded evidence interpretation.

## 1. Core chain

HAK-007 establishes that execution evidence is not its interpretation. HAK-008 makes the missing joins explicit:

```text
QualificationPlan
+ Exact EvidenceSubject
        ↓
QualificationExecution
        ↓
TerminalQualificationReceipt
        ↓
PlanConformanceRecord
        ↓
EvidenceInterpretationRecord
        ↓
Bounded Claim State
```

The core separations are:

```text
ValidReceipt != ReceiptConformsToPlan
ReceiptConformsToPlan != ClaimQualified
ProviderSuccess != ClaimQualified
StoredConformanceLabel != RevalidatedConformance
CheckAssertion != VerifiedCheckEvidence
```

HAK-008 is audit/evidence infrastructure only. None of its artifacts grant runtime authority.

## 2. Exact plan identity

Plan identity has two parts:

```text
PlanLocationIdentity
+ PlanContentIdentity
```

The location identity is the immutable `plan_ref` (for example exact Git commit + repository path). The content identity is:

```text
plan_digest = SHA256(
  "hak.qualification-plan.v1\0"
  || canonical_json(plan)
)
```

A receipt interpreted under HAK-008 must bind the loaded plan's exact digest. A conformance record must bind the same `plan_ref` and `plan_digest`.

Therefore:

```text
SamePlanPath != SamePlanContent
```

and:

```text
LoadedPlanContentMismatch -> NoConformance
```

## 3. Plan conformance

A `PlanConformanceRecord` records whether one exact receipt satisfies one exact plan.

Conceptually:

```text
PlanConformanceRecordV1 {
  conformance_id
  subject
  plan { plan_ref, plan_digest }
  receipt { receipt_id, receipt_digest }
  evaluator
  status
  checks[]
  negative_cases[]
  limitations[]
  conformance_digest
}
```

The allowed statuses are:

```text
Satisfied
NotSatisfied
Indeterminate
```

`Satisfied` requires all required checks and required negative cases to be present and `Passed`, plus a successful terminal receipt.

A failed terminal receipt remains valid evidence and may support `NotSatisfied`.

Missing or unresolved required evidence cannot be converted into success by omission.

## 4. Stored status is not trusted

An interpretation must not trust `status: Satisfied` as an authority-bearing label.

Before a conformance record can support interpretation, HAK-008 revalidates:

```text
exact plan
exact plan digest
exact receipt
receipt digest
plan ↔ receipt join
subject equality
required checks
required negative cases
terminal conclusion
conformance digest
```

Therefore:

```text
StoredConformanceLabel != EstablishedConformance
```

A forged or stale `Satisfied` label cannot qualify a claim if the underlying record no longer establishes it.

## 5. Bounded evidence interpretation

An `EvidenceInterpretationRecord` maps revalidated evidence to claim states without strengthening the precommitted claim.

Candidate states:

```text
Qualified
NotSatisfied
InsufficientEvidence
BlockedBy
Revoked
```

For `Qualified`:

```text
PlanConformance == Satisfied
supporting receipt terminal == success
supported_tier <= claim.target_tier
supported_tier <= plan.evidence_target
```

This is the Evidence Ceiling Theorem:

```text
InterpretedTier <= PrecommittedEvidenceCeiling
```

A stronger tier requires separately identified stronger evidence and an explicit interpretation policy; interpretation cannot manufacture it.

## 6. Evidence identity, not mutable labels

Interpretation binds evidence by both logical identity and content digest:

```text
receipt_id + receipt_digest
conformance_id + conformance_digest
plan_ref + plan_digest
exact subject repository + commit
```

Therefore:

```text
EvidenceReference != EvidenceIdentity
```

Repointing an ID to changed content invalidates the join.

## 7. Responsible interpreter vs model assistance

The responsible interpreter kind is one of:

```text
DeterministicPolicy
HumanReviewer
ReviewCommittee
```

Model assistance is separate provenance:

```text
model_assisted: bool
model_ref: optional/required when assisted
```

There is deliberately no `ModelAssistedReview` authority kind.

Thus:

```text
ModelUsedDuringInterpretation != ModelIsInterpretationAuthority
```

and:

```text
ModelAssistance -/-> HigherEvidenceTier
```

## 8. Failure is evidence

A terminal execution failure can be authentic, well-bound evidence.

```text
ValidReceipt + ProviderFailure
-> ValidNegativeExecutionEvidence
```

Depending on plan policy it may support `NotSatisfied` or `InsufficientEvidence`.

HAK must not preserve only successful evidence.

## 9. Machine-readable schema parity

HAK-008 maintains JSON Schemas for `PlanConformanceRecordV1` and `EvidenceInterpretationRecordV1`.

Schema parity is itself a qualification obligation. The focused regression lane validates representative documents against Draft 2020-12 schemas and includes negative cases for stale authority kinds, missing plan digests, missing evidence digests, and missing model provenance.

Therefore:

```text
PythonSemanticsChanged
-> SchemaMustChangeOrRegressionFails
```

This prevents stale schemas from becoming a second, weaker contract.

## 10. Current limitation: check evidence is asserted, not authenticated

HAK-008 deliberately stops before treating a check's `status: Passed` and free-form `evidence_refs` as authenticated execution evidence.

This is the next boundary:

```text
CheckAssertion != VerifiedCheckEvidence
```

The natural HAK-009 handoff is an authenticated check-evidence artifact that binds a plan check to provider-owned job/step/artifact evidence from the exact execution attempt.

Until that exists, HAK-008 can prove completeness and consistency of the conformance bookkeeping, but not that every asserted check result was independently derived from provider evidence.

## 11. Non-claims

HAK-008 does not:

- grant runtime authority;
- make a CI provider a universal claim authority;
- create a central HAK certificate authority;
- claim independent certification from hosted CI;
- allow an AI model to become qualification authority merely by assisting review;
- prove the semantic adequacy of a self-declared qualification plan;
- authenticate free-form `evidence_refs` as provider evidence;
- convert a green workflow directly into a qualified human-safety, governance, scientific, or constitutional claim.

## 12. Qualification discipline

A hosted HAK-008 run is evidence only for the exact head it executed.

A later code, schema, plan, workflow, or test change creates a new subject and requires a new exact-head run.

Even a successful run still yields first an execution receipt. Plan conformance and interpretation remain explicit subsequent joins.
