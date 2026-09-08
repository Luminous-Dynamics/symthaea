# Human Agency Kernel — Plan Conformance & Bounded Interpretation v1

Status: HAK-008 design seed only. This file is intentionally introduced before HAK-008 receives its own branch so the parent HAK-007 lineage records the boundary being handed off.

## Missing stage

HAK-007 establishes:

```text
Subject != Plan != Execution != Receipt != Interpretation
```

HAK-008 makes the missing join explicit:

```text
QualificationPlan
+ TerminalQualificationReceipt
        ↓
PlanConformanceRecord
        ↓
EvidenceInterpretationRecord
```

A receipt can be authentic yet fail the plan. Therefore:

```text
ValidReceipt != ReceiptConformsToPlan
```

and:

```text
ReceiptConformsToPlan != ClaimQualified
```

## PlanConformanceRecord

The record should bind:

- exact plan identity;
- exact receipt identity/digest;
- exact subject identity;
- required-check results;
- required-negative-case results;
- provider/exact-head constraints;
- plan/receipt semantic join;
- status: `Satisfied`, `NotSatisfied`, or `Indeterminate`;
- limitations;
- conformance-record digest.

A missing required check cannot be converted into success:

```text
MissingCheck -> Indeterminate or NotSatisfied
```

according to plan policy, never `Satisfied` by omission.

## EvidenceInterpretationRecord

Only after plan conformance is established may an interpretation map evidence onto bounded claim states.

Candidate claim states:

```text
Qualified
NotSatisfied
InsufficientEvidence
BlockedBy
Revoked
```

A `Qualified` claim must obey:

```text
PlanConformance == Satisfied
supporting receipt(s) terminal and valid
claim exists in exact plan
supported_tier <= claim.target_tier
supported_tier <= plan.evidence_target
```

Provider success never bypasses this join.

## Evidence ceiling theorem

Evidence interpretation may attenuate a plan's target, but may not silently strengthen it:

```text
InterpretedTier <= PrecommittedPlanTier
```

unless a separately identified stronger evidence source and interpretation policy explicitly justify the higher tier.

## Failure is evidence

A valid failed receipt is not invalid evidence.

```text
ReceiptValid + ProviderFailure
-> valid negative execution evidence
```

which may support `NotSatisfied` or `InsufficientEvidence` depending on the plan and failure mode.

## Independence remains multidimensional

Interpretation must identify:

- conformance evaluator identity;
- interpretation authority identity;
- interpretation policy identity;
- whether either is independent of subject authorship;
- whether model assistance was used.

Model assistance never becomes authority merely because it helped summarize evidence.
