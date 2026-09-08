# Human Agency Kernel — Evidence Subjects & Qualification Receipts v1

Status: HAK-007 architecture/evidence/tooling candidate

Parent stack:

- HAK-001 — semantic separation;
- HAK-002 — authority provenance and lineage;
- HAK-003 — transformation validity and conservation;
- HAK-004 — agency/rights floor and meta-constitutional continuity;
- HAK-005 — falsifiable proof obligations and conformance profiles;
- HAK-006 — audit-only conformance manifest linting;
- HAK-007 — qualification plan, subject, execution, receipt, and interpretation separation.

## Core theorem

```text
Subject != Plan != Execution != Receipt != Interpretation
```

No artifact may silently stand in for another.

## Non-self-reference rule

A Git subject must not attempt to embed its own final commit SHA as authoritative self-identity. Writing that SHA changes the subject and immediately makes the embedded identity stale.

Therefore:

```text
SubjectGitIdentity
must be bound externally by execution/receipt metadata
```

not:

```text
subject file claims hash(subject commit)
```

The exact qualification subject is supplied by PR/provider metadata and later recorded in execution observations or terminal receipts.

## Qualification plan

A workflow is not a qualification plan.

```text
Workflow
= execution mechanism

QualificationPlan
= claims + subject requirements + checks + negative cases + provider policy + target evidence tier
```

A plan must be fixed before the execution it is later used to interpret. A plan change creates a new plan lineage and cannot retroactively change what an earlier run established.

A precommitted plan is not automatically independent. HAK records plan authority, execution-provider independence, and claim-interpretation independence separately.

## Evidence subject

Qualification binds an exact subject identity sufficient for the claim. Depending on the claim, this may include source commit/tree, lockfiles, features/configuration, toolchain, policy lineage, verifier set, or model identity.

```text
MaterialSemanticInputChanged
-> NewEvidenceSubject
```

unless an explicit equivalence/transfer proof exists.

## Execution observation versus terminal receipt

Mutable provider state may be represented as an `ExecutionObservation` while queued/running.

```text
QueuedObservation != TerminalQualificationReceipt
```

A terminal receipt may exist only after a terminal provider result. The receipt reports what happened to one exact execution attempt; it does not interpret the evidential meaning of that result.

## Provider success is not qualification

```text
ProviderSuccess != ClaimQualified
```

A bounded interpretation additionally requires the exact subject, exact plan, plan-execution conformance, required checks/jobs/artifacts, evidence-tier rules, and a separate interpretation policy.

## Plan artifact identity versus workflow identity

The qualification-plan document and the workflow that executes it are intentionally distinct artifacts.

For a self-declared plan, record-level validation may establish:

```text
plan repository == subject repository
```

It must not require:

```text
plan document path == execution workflow path
```

Instead the explicit plan-to-execution join establishes:

```text
plan_ref.path == supplied qualification plan document path
plan.scope.workflow_path == execution.workflow_path
execution.provider in plan.provider_policy.allowed_providers
```

Therefore:

```text
PlanArtifactIdentity != WorkflowArtifactIdentity
```

while `PlanExecutionConformance` remains a separately testable proof obligation.

## Receipt integrity

A terminal receipt carries a non-self-referential digest:

```text
receipt_digest = SHA256(
    "hak.qualification-receipt.v1\0"
    || canonical_json(receipt_without_receipt_digest)
)
```

This detects tested forms of post-materialization mutation without requiring the receipt to hash itself recursively.

## Evidence interpretation remains separate

A later interpretation record may map one or more receipts to bounded claim states. It must not rewrite the receipt or allow the provider to assign its own evidence tier.

```text
Receipt != Interpretation
AuthorityToAttestExecution != AuthorityToInterpretEveryClaim
```

HAK-008 owns the first executable interpretation semantics.

## Supersession and revocation

New evidence does not delete old evidence.

```text
Superseded != Deleted
Revoked != Superseded
```

A superseded record remains historical evidence of what was believed under an older subject/policy/evidence state. Revocation means the prior artifact may no longer be relied upon for current claims because its trust basis was invalidated.

## Current HAK-006 example

The materialized HAK-006 run is intentionally represented as a nonterminal execution observation. Its qualification-plan state is:

```text
plan_kind = UnspecifiedPlan
precommit_status = NotEstablished
```

because the workflow existed before execution but no complete HAK qualification-plan artifact had been established beforehand.

Thus it may later establish exact-head hosted execution facts, but HAK-007 forbids retroactively calling it `PlanQualifiedE5`.

## HAK-007 qualification rule

HAK-007 has a machine-readable self-declared E5-target plan. Its focused workflow resolves the exact PR head externally, checks out that exact subject, and asserts the checkout identity before executing the plan.

Any run on a superseded head remains historical evidence only. A green run can support only the claims explicitly named by the plan and only after a separate interpretation step confirms plan conformance.

## Non-claims

HAK-007 does not:

- create a central certificate authority;
- make hosted CI third-party certification;
- grant runtime authority;
- decide legal, scientific, ethical, or constitutional truth;
- treat AI/model output as source evidence;
- allow receipts to self-assign evidence tiers;
- allow historical evidence to be silently erased.
