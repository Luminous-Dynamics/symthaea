# Human Agency Kernel — Evidence Subjects & Qualification Receipts v1

Status: HAK-007 architecture candidate / documentation only

Parent stack:

- HAK-001 — semantic separation;
- HAK-002 — authority provenance and lineage;
- HAK-003 — transformation validity and conservation;
- HAK-004 — agency/rights floor and meta-constitutional continuity;
- HAK-005 — falsifiable proof obligations and conformance profiles;
- HAK-006 — audit-only conformance manifest linting;
- HAK-007 — separate the thing being qualified from the execution that tests it and the later record that interprets the evidence.

## 1. Purpose

HAK-005 and HAK-006 make evidence claims more precise, but they create a deeper provenance question:

> What exact thing was tested, under what precommitted qualification contract, by what execution, and what later artifact is allowed to interpret that result?

A conformance manifest must not become self-authenticating.

In particular, this shape is structurally problematic:

```text
artifact contains hash(artifact)
```

because changing the artifact changes its hash.

The same conceptual error appears when a source commit contains a claim that the same commit has already been qualified by a future execution.

HAK-007 therefore separates five lineages:

```text
QualificationPlan
+ EvidenceSubject
        ↓
QualificationExecution
        ↓
TerminalQualificationReceipt
        ↓
EvidenceInterpretationRecord
        ↓
Supersession / Revocation / Requalification
```

The core theorem is:

```text
Subject != Plan != Execution != Receipt != Interpretation
```

None of these artifacts may silently stand in for another.

## 2. Non-self-reference theorem

A subject under qualification must be identifiable without relying on an evidence record that can only exist after qualification.

For a Git subject, for example:

```text
EvidenceSubject.commit = exact immutable commit SHA
```

A later evidence record may refer backward to that commit.

The subject commit does not need to contain the later evidence record.

Therefore:

```text
EvidenceRecordMutation != SubjectMutation
```

and:

```text
QualificationOf(S)
can be recorded in R where R is created after S
without claiming QualificationOf(R)
```

This avoids recursive evidence identities.

## 3. Qualification plan must be precommitted

A test run has weak assurance if its acceptance criteria can be silently changed after seeing the subject or after seeing partial results.

HAK-007 introduces a `QualificationPlan` as a distinct artifact.

Conceptually:

```text
QualificationPlanV1 {
    plan_id
    plan_digest
    plan_lineage
    plan_authority

    applicable_claims[]
    required_subject_identity_fields[]
    required_environment_constraints[]
    required_commands_or_workflow_identity[]
    required_negative_cases[]
    required_artifacts[]
    required_evidence_tier
    allowed_provider_classes[]
    independence_requirement
    failure_semantics
    retry_semantics
    supersession_semantics
}
```

The exact representation is not frozen by HAK-007 v1.

The important invariant is temporal:

```text
PlanDigestFixedBeforeQualificationExecutionBegins
```

A later plan change creates a new plan lineage.

It cannot retroactively change what an earlier execution proved.

## 4. Plan independence is an explicit property

A precommitted plan is not automatically independent.

HAK-007 distinguishes at least:

```text
SelfDeclaredPlan
IndependentPlan
ProviderEnforcedPlan
```

### SelfDeclaredPlan

The subject repository/author defines the plan used to test itself.

This can provide useful reproducibility and exact-head hosted evidence, but it does not establish independent review merely because the runner is external.

### IndependentPlan

The plan is controlled by a distinct trusted authority or separately protected policy lineage.

### ProviderEnforcedPlan

The CI/provider or repository protection policy independently requires the plan/check before acceptance.

Candidate theorem:

```text
HostedExecution != IndependentQualification
```

and:

```text
IndependentQualification
requires explicit independence provenance
```

This prevents rhetoric from upgrading ordinary hosted CI into third-party certification.

## 5. Evidence subject

An `EvidenceSubject` identifies the exact implementation/policy/artifact being tested.

Conceptually:

```text
EvidenceSubjectV1 {
    subject_id
    subject_kind

    repository
    commit_sha
    tree_sha? 
    artifact_digests[]

    dependency_lock_digest?
    toolchain_digest_or_version?
    configuration_digest?
    feature_set?
    policy_digest?
    constitutional_lineage?
    external_model_digest?

    claim_scope[]
}
```

Not every field applies to every domain.

The subject identity must contain every input whose change could materially alter the claimed property.

Candidate theorem:

```text
MaterialSemanticInputChanged
-> NewEvidenceSubject
```

unless a separate equivalence proof allows qualification transfer.

## 6. Subject identity is claim-relative

There is no single universal subject identity sufficient for every claim.

For example:

```text
formatting claim
may depend only on source tree + formatter version
```

while:

```text
cryptographic wire-contract claim
may depend on source + Cargo.lock + features + target architecture + crypto backend
```

and:

```text
governance legitimacy claim
may depend on source + deployed policy + constitutional lineage + trusted verifier set
```

Therefore:

```text
SubjectIdentitySufficientForClaimA
-/-> SubjectIdentitySufficientForClaimB
```

The qualification plan owns the required subject identity fields for its claims.

## 7. Execution attempt identity

A provider run is not merely a boolean result.

HAK-007 models each execution attempt explicitly.

Conceptually:

```text
QualificationExecutionV1 {
    execution_id
    provider
    provider_run_id
    provider_run_attempt

    plan_digest
    subject_id
    resolved_subject_identity

    workflow_id
    workflow_path
    workflow_digest?
    event
    trigger_identity?

    requested_at
    started_at?
    completed_at?

    environment_identity
    runner_identity_or_class

    status
    conclusion?
}
```

A retry is a new execution attempt unless the provider exposes a stable run identity plus attempt number.

Do not overwrite history:

```text
Attempt1 Failed
Attempt2 Passed
```

is not equivalent to:

```text
Passed
```

without retaining both attempts.

## 8. Observation is not receipt

Provider state is mutable while an execution is queued or running.

HAK-007 therefore distinguishes:

```text
ExecutionObservation
```

from:

```text
TerminalQualificationReceipt
```

An observation may record:

```text
Queued
InProgress
Waiting
Requested
```

but it is a snapshot of mutable execution state.

A terminal receipt can only be created after the provider reports a terminal result such as:

```text
Success
Failure
Cancelled
TimedOut
ActionRequired
Neutral
Skipped
```

as appropriate to the provider.

Candidate theorem:

```text
QueuedObservation != QualificationReceipt
```

and:

```text
InProgressObservation != QualificationEvidenceForResult
```

## 9. Terminal qualification receipt

A terminal receipt records what the provider says happened to one exact execution attempt.

Conceptually:

```text
QualificationReceiptV1 {
    receipt_id
    receipt_schema

    subject_id
    exact_subject_identity
    plan_digest

    provider
    provider_run_id
    provider_run_attempt
    workflow_id
    workflow_path
    workflow_digest?

    event
    started_at
    completed_at
    terminal_conclusion

    job_receipts[]
    command_or_step_receipts[]
    artifact_refs[]
    log_refs[]

    provider_record_ref
    materialized_at
    receipt_digest
}
```

The receipt is an evidence artifact about execution.

It is not itself the claim interpretation.

## 10. Provider record vs materialized receipt

A provider API response is externally useful evidence, but providers may have retention limits, presentation changes, or mutable metadata.

HAK-007 therefore distinguishes:

```text
ProviderRecordReference
```

from:

```text
MaterializedReceipt
```

A strong receipt may retain:

- provider run ID;
- immutable/semistable provider URL or API identifier;
- exact subject SHA reported by provider;
- workflow ID/path;
- attempt number;
- terminal conclusion;
- job/step conclusions;
- timestamps;
- hashes of downloaded logs/artifacts where policy requires preservation.

Candidate theorem:

```text
ProviderPointer != PreservedEvidence
```

Provider retention loss should not silently rewrite history.

## 11. Provider success proves execution outcome, not claim adequacy

A green workflow proves only that the provider reports the configured workflow succeeded for the resolved subject under that execution.

It does not prove that the workflow was sufficient for a particular safety claim.

Therefore:

```text
ProviderSuccess
!= ClaimQualified
```

Qualification interpretation additionally requires:

```text
ExactSubjectMatch
+ ExactPlanMatch
+ PlanExecutionConformance
+ RequiredJob/StepSuccess
+ RequiredArtifactPresence
+ EvidenceTierRules
+ ClaimInterpretation
```

This preserves the boundary between CI truth and semantic assurance.

## 12. Workflow identity matters

Two executions of different workflow definitions are not equivalent merely because they have the same display name.

Where material, qualification should bind:

```text
workflow_id
workflow_path
workflow definition digest or exact source subject containing it
```

Candidate theorem:

```text
SameWorkflowName + DifferentWorkflowSemantics
!= SameQualificationPlanExecution
```

For workflows stored in the subject repository, the exact subject commit may transitively identify the workflow definition only if the plan explicitly allows that binding.

## 13. Plan conformance

A receipt is not valid evidence for a plan unless the actual execution conforms to the plan.

Candidate checks include:

```text
resolved subject == required subject
provider class allowed
workflow identity allowed
required jobs present
required commands/steps present where observable
required environment constraints satisfied
required artifacts produced
retry semantics allowed
terminal conclusion satisfies plan
```

This creates a separate proof obligation:

```text
ReceiptConformsToPlan
```

Do not infer it merely because the run URL appears in a manifest.

## 14. Evidence interpretation record

Only after a terminal receipt exists should a later artifact interpret what claims it supports.

Conceptually:

```text
EvidenceInterpretationRecordV1 {
    record_id
    record_digest

    subject_id
    plan_digest
    receipt_ids[]

    claims[] {
        claim_id
        evidence_tier
        status
        supporting_receipts[]
        supporting_artifacts[]
        limitations[]
    }

    interpreter_identity
    interpretation_policy
    created_at
}
```

The interpreter may be:

- a deterministic linter;
- a human reviewer;
- an authorized review committee;
- a domain-specific verifier;
- a later audited automation.

An AI model may assist interpretation, but model output alone must not silently become authoritative qualification unless an explicit policy gives it that role.

## 15. Interpretation authority is separate from execution authority

A CI provider has authority to report the execution state it controls.

It does not automatically have authority to define constitutional legitimacy, scientific truth, legal compliance, or human safety.

Likewise, a HAK conformance linter can reject inconsistent bookkeeping but cannot declare the runtime safe.

Candidate theorem:

```text
AuthorityToAttestExecution
!= AuthorityToInterpretEveryClaim
```

This is HAK-001/002 applied to evidence infrastructure itself.

## 16. Anti-circular evidence rule

Evidence for a claim must not depend circularly on that same claim already being accepted.

Forbidden conceptual pattern:

```text
Record A is trusted because Record B says A is trusted
Record B is trusted because Record A says B is trusted
```

without an external trust root or independently qualified base case.

Candidate theorem:

```text
EvidenceDependencyGraph
must be acyclic
```

except where a formally defined fixed-point protocol explicitly proves safe semantics.

HAK-007 v1 assumes acyclic evidence dependencies.

## 17. Evidence supersession is append-only

New evidence should not erase prior evidence.

A later record may supersede an earlier interpretation because:

- the subject changed;
- policy changed;
- a test was discovered to be inadequate;
- evidence was corrupted;
- provider attestation was revoked;
- a stronger run qualified the same subject;
- a later operational observation falsified the earlier claim;
- a constitutional interpretation changed.

Conceptually:

```text
EvidenceSupersessionV1 {
    prior_record_id
    new_record_id
    reason
    scope
    authority
    occurred_at
}
```

Candidate theorem:

```text
Superseded != Deleted
```

The old record remains historical evidence of what was believed and why.

## 18. Revocation vs supersession

These are distinct.

### Supersession

A newer record replaces an older interpretation for a scope while preserving the fact that the older record was once valid under its evidence.

### Revocation

A trusted authority declares that an earlier receipt/record must no longer be relied upon, for example because:

- signing key compromise;
- provider attestation compromise;
- fabricated artifact;
- discovered evidence corruption;
- invalidated qualification plan.

Candidate theorem:

```text
SupersededEvidence may remain historically valid
RevokedEvidence is no longer trusted for current claims
```

Both events must preserve provenance.

## 19. Requalification after subject change

If the subject changes materially:

```text
S1 -> S2
```

then by default:

```text
Qualification(S1) -/-> Qualification(S2)
```

A new execution is required unless an explicit qualification-transfer proof establishes preservation.

This includes seemingly small changes such as:

- test edits;
- workflow edits;
- dependency lock updates;
- feature/config changes;
- policy changes;
- trusted verifier changes;
- environment changes relevant to the claim.

## 20. Requalification after evidence-tool change

The evidence tooling itself can change without changing the original subject.

Example:

```text
subject S remains fixed
HAK linter v1 -> HAK linter v2
```

A new interpretation record may reassess the same receipts under the newer linter/policy.

Do not rewrite the original receipt.

Conceptually:

```text
Receipt(S, Execution)
        ↓
InterpretationPolicyV1 -> RecordR1
InterpretationPolicyV2 -> RecordR2
```

This allows audit methods to improve while preserving historical execution truth.

## 21. Evidence freshness

Different evidence kinds have different freshness semantics.

A deterministic unit test of immutable source may remain historically valid indefinitely.

A trust-list check may expire when the trust list changes.

An operational health receipt may be useful only for minutes.

A constitutional policy interpretation may remain current until amendment.

Therefore:

```text
EvidenceFreshness != UniversalTTL
```

Each claim/plan should define whether evidence is:

```text
immutable-historical
valid-until-subject-change
valid-until-policy-change
valid-until-trust-change
time-bounded
continuously-revalidated
```

## 22. Negative evidence and later falsification

Qualification evidence is not monotonic truth.

A later field failure can falsify a claim that had passed predeployment testing.

HAK-007 therefore permits negative evidence records:

```text
OperationalCounterexample
PropertyViolation
IncidentReceipt
PostmortemFinding
ScientificReplicationFailure
```

These should attach to the exact claim/subject lineage and can trigger supersession or revocation.

Candidate theorem:

```text
EarlierQualification
does not immunize a claim against later falsification
```

## 23. Current HAK-006 run as an example

At the time this HAK-007 candidate was authored, the focused HAK-006 execution had provider metadata equivalent to:

```text
provider = GitHub Actions
repository = Luminous-Dynamics/symthaea
workflow = HAK Conformance
workflow_path = .github/workflows/hak-conformance.yml
provider_run_id = 34212685766
run_attempt = 1
event = pull_request
head_branch = architecture/hak-conformance-linter-v1
subject_commit = 98883dfb03594f777390abe557e307af99ff4b5d
base_commit = ae07c516d06560e36206f61a9e8a1c5e366578de
status = queued
conclusion = none
```

This is an **ExecutionObservation**, not a terminal qualification receipt.

Therefore:

```text
HAK006FocusedRunObserved
AND status = Queued
-> no E5 qualification result yet
```

When the run becomes terminal, a later evidence record may materialize its terminal metadata without modifying the HAK-006 subject commit.

## 24. HAK evidence-level implications

HAK-005's evidence tiers remain useful, but HAK-007 tightens what they mean.

### E0 / E1

May be represented by specification/static-review records without provider execution receipts.

### E2–E4

Should bind exact subject identity plus local/adversarial/integration execution evidence as appropriate.

### E5

Requires at minimum:

```text
exact subject identity
+ hosted execution attempt
+ terminal hosted receipt
+ plan-conformance interpretation
```

A hosted green run with no subject binding is insufficient.

### E6

Additionally requires reproducible artifact/environment identity sufficient for the claim.

### E7 / E8

Require operational/empirical evidence lineages that cannot be reduced to CI receipts.

## 25. Independence dimensions

"Independent" is not one bit.

Potential dimensions include:

```text
independent plan author
independent execution provider
independent evidence custodian
independent claim interpreter
independent operational observer
```

A qualification record should name the dimension being claimed rather than saying simply:

```text
independently verified
```

Candidate theorem:

```text
IndependentExecutionProvider
!= IndependentClaimInterpretation
```

## 26. Anti-goal: universal certificate authority

HAK-007 must not create a central HAK certificate authority.

The architecture should support multiple evidence authorities and policies.

A scientific lab, civic institution, safety regulator, community, CI provider, or individual project may trust different evidence roots.

Therefore:

```text
HAKReceipt != UniversalTruthToken
```

Receipts are provenance-bearing evidence inputs to domain-owned interpretation.

## 27. Anti-goal: immutable mistakes

Cryptographically preserving an evidence record does not make its interpretation eternally correct.

The system must preserve both:

```text
historical integrity
```

and:

```text
corrigibility
```

Hence:

```text
TamperEvidentHistory + ExplicitSupersession
```

is preferred over either mutable history or permanently authoritative mistakes.

## 28. Candidate machine-checkable invariants

A future audit-only HAK evidence linter may enforce rules such as:

```text
terminal receipt -> terminal conclusion required
queued/running observation -> cannot claim receipt status
receipt subject -> exact match to execution resolved subject
receipt plan_digest -> known precommitted plan
interpretation -> all cited receipts exist
interpretation E5 -> at least one conforming hosted terminal receipt
supersession -> prior/new records exist
revocation -> authority + reason required
receipt dependency graph -> acyclic
plan-sensitive interpretation -> exact plan digest required
```

These validate evidence bookkeeping only.

They do not decide whether the underlying plan is ethically or scientifically adequate.

## 29. Candidate artifact layering

A practical repository layout might eventually look like:

```text
docs/architecture/hak/plans/
    <plan-id>.json

docs/architecture/hak/subjects/
    <subject-id>.json

docs/release/evidence/hak/executions/
    <execution-id>.observation.json

docs/release/evidence/hak/receipts/
    <receipt-id>.json

docs/release/evidence/hak/interpretations/
    <record-id>.json

docs/release/evidence/hak/supersessions/
    <event-id>.json
```

This is illustrative, not normative in v1.

Provider-generated artifacts may live elsewhere if their content-addressed identity and provenance are preserved.

## 30. Candidate state machine

Execution attempt:

```text
Planned
  -> Queued
  -> InProgress
  -> Terminal(Success | Failure | Cancelled | TimedOut | ...)
```

Evidence interpretation:

```text
Uninterpreted
  -> Interpreted
  -> Current
  -> Superseded
```

or:

```text
Current
  -> Revoked
```

Do not collapse execution state and interpretation state.

A failed execution can still have a perfectly valid terminal receipt.

## 31. Failure is evidence

HAK-007 explicitly treats failure records as useful evidence.

```text
FailedRun != MissingEvidence
```

A failed exact-head run can establish that a particular subject did not satisfy a plan at that execution attempt.

Likewise:

```text
InfrastructureFailure != PropertyViolation
```

The interpretation must preserve the provider's conclusion semantics.

## 32. Receipt authenticity and cryptographic attestation

HAK-007 v1 does not require one cryptographic receipt technology.

Potential future mechanisms include:

- provider-signed attestations;
- Sigstore-style provenance;
- in-toto/SLSA provenance;
- transparency logs;
- repository-signed evidence bundles;
- Xenia-backed authenticated receipts.

The semantic model should precede technology selection.

Candidate theorem:

```text
CryptographicallyAuthenticatedReceipt
!= SemanticallyAdequateQualificationPlan
```

Both layers matter.

## 33. Trust-root recursion must stop somewhere

A receipt verifier may itself depend on trusted keys, provider identities, or software.

HAK cannot eliminate trust roots by adding infinite metadata.

Instead it should make them explicit:

```text
TrustRoot {
    identity
    scope
    validity
    revocation
    provenance
}
```

Then reviewers can see where evidence assurance bottoms out.

Candidate theorem:

```text
ExplicitTrustRoot > HiddenTrustAssumption
```

## 34. Relationship to Xenia and supply-chain work

HAK-007 defines semantics, not a transport/security implementation.

Xenia may later provide:

- authenticated evidence envelopes;
- signature verification;
- revocation/currentness;
- transparency receipts;
- content-addressed evidence transport;
- threshold attestation.

Nix/Nixward may provide:

- reproducible environment identity;
- exact dependency/toolchain closure;
- build provenance.

HAK should consume those stronger primitives rather than duplicate them.

## 35. Relationship to Symthaea

Symthaea may help:

- generate candidate proof obligations;
- identify missing joins;
- compare plans to executions;
- summarize evidence;
- detect contradictions;
- propose falsification tests;
- surface stale/superseded evidence.

But by default:

```text
SymthaeaAssessment != QualificationAuthority
```

unless an explicit domain policy grants a bounded role.

The evidence should remain reconstructable independently of trusting the model's prose.

## 36. Relationship to Mycelix governance

The same evidence separation is useful for governance:

```text
Proposal/PolicySubject
+ VotingQualificationPlan
-> Tally/Transition Verification Execution
-> Receipt
-> Governance Evidence Interpretation
```

A status label such as `Approved` should not be its own proof that the qualifying process occurred.

This is the evidence analogue of:

```text
RepresentationOfAuthority != SourceOfAuthority
```

## 37. HAK-007 proof obligations

Candidate obligations:

### HAK-EVID-SUBJ-001

Every evidence interpretation binds an exact subject identity sufficient for the claim.

### HAK-EVID-PLAN-001

Qualification plan identity is fixed before the execution it governs begins.

### HAK-EVID-EXEC-001

Each execution attempt has a unique provider/attempt identity and exact resolved subject.

### HAK-EVID-OBS-001

Nonterminal execution observations cannot be represented as terminal qualification receipts.

### HAK-EVID-RECEIPT-001

A terminal receipt binds exact subject, plan, provider execution, attempt, workflow identity, and terminal outcome.

### HAK-EVID-PLAN-002

Receipt acceptance requires explicit conformance to the referenced qualification plan.

### HAK-EVID-INTERP-001

Evidence tier/claim interpretation is distinct from provider execution success.

### HAK-EVID-SUP-001

Supersession/revocation is append-only and preserves prior evidence lineage.

### HAK-EVID-CYCLE-001

Evidence trust dependencies are acyclic unless an explicitly proved fixed-point protocol applies.

### HAK-EVID-INDEP-001

Claims of independent verification name the independence dimension and its provenance.

## 38. Negative cases

HAK-007 should eventually be falsified/tested against cases including:

```text
receipt points to wrong subject SHA
receipt points to wrong run attempt
run is queued but record claims success
workflow display name matches but workflow identity changed
plan changed after execution began
receipt omitted required failed job
rerun overwrites failed first attempt
provider URL expired and no evidence preserved
interpretation claims E5 from local-only evidence
subject changed after receipt
policy changed after receipt
self-declared plan mislabeled independent
receipt says success but plan-required job absent
supersession deletes prior record
revocation has no authority/reason
circular evidence trust dependency
AI summary treated as source receipt
```

## 39. Minimal implementation direction

Do not build a universal evidence service next.

The safest next executable increment is an extension of the existing audit-only HAK conformance tooling that can validate **receipt bookkeeping** for one provider profile.

A good first target is GitHub Actions because the current HAK-006 workflow already exposes concrete provider fields.

Candidate first implementation:

```text
GitHubActionsExecutionObservationV1
GitHubActionsTerminalReceiptV1
```

with checks that:

- reported `head_sha` equals the evidence subject commit;
- run/attempt identity is explicit;
- queued/in-progress runs cannot become terminal receipts;
- terminal conclusion is preserved exactly;
- workflow identity/path is bound;
- plan digest is external to the receipt and must be known;
- later interpretation cannot upgrade the evidence tier beyond the plan.

This should remain audit-only.

## 40. Exit criteria for HAK-007 v1

HAK-007 should not be considered mature merely because this document exists.

Candidate exit criteria:

1. at least one real HAK-006 execution is represented as an observation and later terminal receipt without modifying the HAK-006 subject;
2. a failed/cancelled example is representable without collapsing to generic `FAIL`;
3. plan/subject/execution/receipt/interpretation identities remain distinct;
4. superseded evidence remains recoverable;
5. audit tooling rejects a queued run represented as E5-qualified;
6. audit tooling rejects subject/run mismatch;
7. independence language is explicit rather than inferred from hosted execution;
8. no runtime authority depends on the HAK receipt format.

## 41. Summary theorem

The HAK evidence architecture should converge toward:

```text
PrecommittedPlan
+ ExactSubject
        ↓
ExactExecutionAttempt
        ↓
TerminalProviderReceipt
        ↓
PlanConformanceCheck
        ↓
ClaimInterpretation
        ↓
CurrentEvidenceRecord
        ↓
AppendOnlySupersessionOrRevocation
```

with the invariants:

```text
Subject != EvidenceRecord
Queued != Qualified
GreenRun != AdequatePlan
Hosted != Independent
Receipt != Interpretation
Superseded != Deleted
Failure != MissingEvidence
```

HAK should make evidence stronger by making its assumptions, trust roots, exact subjects, and uncertainty more explicit—not by creating a new oracle.