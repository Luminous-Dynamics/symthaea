# Scientific Evidence Contribution Lifecycle v1

**Status:** architecture contract candidate only; non-authorizing; non-qualifying.

**Parent:** `#729@0c721de8e43f5fbef5d51c1cee66c6f6b4109973`

## 1. Purpose

SCI-001 defines the broad Scientific Method Kernel, #668 separates target compatibility from triangulation, #701 defines a defeater-aware non-monotonic argument graph, and #729 freezes immutable scientific proposition identity.

The next missing SCI-014 boundary is the lifecycle of an evidence contribution after it has been issued.

A scientific system must be able to preserve corrections, supersession, retraction, invalidation, and later re-evaluation without rewriting history or silently changing what an evidence object originally claimed.

The core requirement is append-only history plus derived current eligibility.

This document does not implement a shared kernel, a publication registry, a correction authority, a scientific disposition engine, canonical belief, governance authority, or execution authority.

---

## 2. Core theorem

The shared scientific layer must preserve:

```text
evidence artifact identity
    != evidence contribution identity
    != contribution relation to a proposition
    != lifecycle declaration
    != qualified lifecycle effect
    != current-use eligibility
    != current scientific disposition
    != historical existence
    != truth
```

and:

```text
correction != mutation of predecessor
supersession != deletion
retraction != proposition negation
external invalidation != source retraction
latest timestamp != canonical successor
current ineligibility != historical nonexistence
historical view != today's corrected view
```

No lifecycle event may rewrite the immutable predecessor contribution.

---

## 3. Evidence contribution is immutable

An `EvidenceContribution` should bind one exact evidence-bearing scientific assertion to one exact target and assessment lineage.

Conceptually:

```text
EvidenceContributionV1 {
    contribution_id,
    proposition_id,
    assessment_target_id,
    evidence_artifact_ids,
    contribution_relation,
    declared_scope,
    provenance,
    issued_at,
}
```

The exact Rust shape is not frozen here.

Once issued, the contribution body is immutable.

If scientific content must change, the system issues a new contribution and links it through a lifecycle event.

Therefore:

```text
edit contribution in place
    -> forbidden as scientific history
```

A database may cache a projection, but the scientific identity always points to the immutable issued contribution.

---

## 4. Contribution relation and lifecycle are orthogonal

What evidence says about a proposition is one axis.

What has happened to that evidence contribution since issuance is another.

Illustrative contribution relations may include:

```text
Supports
Opposes
FalsifiesWithinDeclaredScope
FailsToFalsify
ReplicatesWithinDeclaredScope
ChallengesGeneralization
InformsMechanism
InformsMeasurement
```

Illustrative lifecycle events may include:

```text
CorrectionDeclared
SupersessionDeclared
RetractionDeclared
WithdrawalDeclared
ExternalInvalidationDeclared
LifecycleChallengeDeclared
LifecycleResolutionDeclared
```

A lifecycle event never changes the predecessor's original relation.

Example:

```text
E1 --Supports--> P
E1 --RetractionDeclared--> source lifecycle
```

The historical statement remains:

```text
E1 originally supported P
```

while current eligibility may become:

```text
E1 ineligible for current support
```

under the exact lifecycle/admission policy.

---

## 5. Source-declared lifecycle and external adjudication are different

A generic scientific kernel must not collapse these two questions:

```text
what does the producer/publisher say happened to its contribution?
```

and:

```text
what does an independent scientific admission policy conclude about current eligibility?
```

### 5.1 Source-declared lifecycle

Examples:

```text
CorrectionDeclared
SupersessionDeclared
RetractionDeclared
WithdrawalDeclared
```

These require exact source/issuer identity and lifecycle authorization appropriate to the contribution lineage.

A source cannot anonymously retract some unrelated contribution merely by knowing its ID.

### 5.2 External adjudication

Examples:

```text
ExternalInvalidationDeclared
ProvenanceFailureDetected
ExecutionVerificationFailed
MeasurementAdmissionFailed
FraudOrIntegrityFinding
```

These are evidence/argument objects produced by a separately qualified evaluator or process.

They are not rewritten as if the original source had retracted the work.

Therefore:

```text
source retraction
    != independent invalidation
```

Both may make a contribution ineligible under some policy, but the historical reason remains distinct.

---

## 6. Lifecycle declaration is not lifecycle authority

A lifecycle-shaped message is evidence-shaped input only.

The shared kernel should preserve:

```text
LifecycleDeclaration
    != QualifiedLifecycleEvent
```

A qualified lifecycle event should bind at least:

```text
exact target contribution id
exact event kind
exact issuer / evaluator identity
exact authority or qualification artifact
exact event scope
reason / public notice identity when required
supporting evidence identities when applicable
successor contribution id when applicable
availability / registration evidence
event-profile identity
```

No public boolean such as:

```text
retracted = true
```

should itself mint current-use authority.

---

## 7. Correction creates a successor contribution

A correction is not an in-place patch to an evidence object.

The safe shape is:

```text
E1
    --CorrectionDeclared-->
E2
```

where `E2` is a newly issued contribution with its own immutable identity.

The correction event should retain at least:

```text
predecessor contribution id
successor contribution id
correction scope / changed fields or claims
reason
issuer authority / qualification
supporting evidence
public notice or publication reference when required
```

The predecessor remains recoverable exactly.

A correction may preserve much of the predecessor's scientific content, but compatibility between E1 and E2 is explicit rather than inferred from textual similarity.

---

## 8. Supersession is not invalidation

Supersession means a designated successor should normally be used for a particular current view.

It does not universally mean the predecessor was wrong.

Examples include:

```text
updated dataset release
new analysis version
new calibrated model estimate
revised publication
new standards-compliant artifact
```

Therefore:

```text
Superseded
    != Invalidated
    != Retracted
```

A superseded contribution remains useful for historical reconstruction and may remain scientifically informative in contexts where the predecessor itself is the object of study.

---

## 9. Retraction does not imply proposition negation

A retraction normally says that the source no longer stands behind a contribution for the declared use/scope.

It does not establish:

```text
opposite proposition is true
```

and does not erase:

```text
what was published
when it was published
what evidence it used
how it influenced later work
```

For a source publication lineage, a domain policy may define retraction as terminal for that exact lineage, following the strong existing Muse publication precedent.

The generic scientific kernel must not universalize that exact domain policy across every scientific object.

A later independent contribution may still investigate the same proposition.

A corrected successor may also be issued under a new contribution identity when the domain permits it.

---

## 10. Withdrawal is distinct from retraction

If a domain uses `WithdrawalDeclared`, it should preserve a semantic distinction from retraction.

For example, withdrawal may mean:

```text
producer requests removal from an active workflow before final publication/adjudication
```

while retraction may mean:

```text
an already issued/public scientific contribution is formally disavowed
```

The shared kernel should not assign universal meanings beyond the registered lifecycle profile.

If a domain cannot make the distinction precise, it should omit one of the event kinds rather than create decorative vocabulary.

---

## 11. External invalidation is an argument, not historical erasure

An external evaluator may discover that an evidence contribution fails a current requirement.

Examples:

```text
content digest no longer matches admitted bytes
execution receipt fails replay verification
measurement binding was invalid
historical cutoff leaked future information
causal assumption diagnostic detects a violation
source trust is revoked
```

The resulting object should attack eligibility or an inference through the #701 argument/defeater layer.

It should not mutate the original contribution into a different historical artifact.

Conceptually:

```text
E1 exists unchanged
D1 --Undercuts--> eligibility/inference from E1
```

A lifecycle projection may then classify E1 as ineligible under policy Q.

---

## 12. No implicit latest-wins rule

A naïve implementation might select the lifecycle event with the latest timestamp.

That is unsafe.

Two authorities may issue competing successor declarations:

```text
E1 -> E2
E1 -> E3
```

or conflicting lifecycle claims:

```text
issuer A: E1 superseded by E2
issuer B: E1 remains canonical
```

The shared kernel must retain the branch.

It may not silently choose:

```text
max(recorded_at)
```

or:

```text
last row in database
```

as scientific authority.

A registered domain/publisher policy may identify an authoritative lifecycle stream or resolve equivocation. Without such proof, the state remains branched, contested, or underdetermined.

---

## 13. Event chronology requires more than a timestamp

A human-readable timestamp is useful audit metadata but should not be the sole ordering authority.

For a single authorized lifecycle stream, the preferred structure is conceptually:

```text
stream identity
+ monotone sequence / generation
+ previous qualified event identity
+ exact target contribution
+ event body
```

This can detect:

```text
stale replay
fork/equivocation
missing predecessor
sequence regression
```

For events from independent authorities, the system should not manufacture a total order unless a qualified chronology source actually establishes one.

Therefore:

```text
recorded_at string
    != canonical lifecycle order
```

---

## 14. Current eligibility is a derived use-specific object

Lifecycle history itself should not expose a universal `active: bool`.

A future use-specific admission step should derive something closer to:

```text
ContributionEligibilityAssessmentV1 {
    contribution_id,
    requested_use,
    lifecycle_policy_id,
    lifecycle_generation_id,
    admissible,
    reasons,
    successor_or_branch_information,
}
```

Possible ineligibility reasons may include:

```text
SourceRetracted
SourceWithdrawn
SupersededForRequestedUse
ExternallyInvalidated
LifecycleAuthorityUnknown
LifecycleBranchUnresolved
QualificationExpired
TargetVersionMismatch
```

This object remains separate from proposition support/opposition.

A contribution can be historically valid and currently ineligible for one use while still inspectable for another.

---

## 15. Eligibility is use-specific

One lifecycle event need not have identical consequences for every scientific use.

For example:

```text
retracted clinical efficacy analysis
```

may be ineligible as current efficacy evidence but remain admissible as evidence in a meta-scientific study of publication correction behavior.

Likewise:

```text
superseded dataset release
```

may be ineligible for a present-day estimate while being exactly the correct historical vintage for a historical replay.

Therefore:

```text
contribution globally valid/invalid
```

is often too coarse.

Eligibility should bind the exact requested use and policy.

---

## 16. Historical views must respect lifecycle-event availability

Time-indexed Theory Atlas reconstruction must not apply future corrections to the past.

For an Atlas view at cutoff `t0`, a lifecycle event may influence that view only if its existence was admissibly available by `t0` under the selected historical-information policy.

Example:

```text
2028: E1 active
2030: E1 retracted
```

Then:

```text
AtlasView(as_of=2028)
    must not silently treat E1 as retracted
```

while:

```text
AtlasView(as_of=2031)
    may apply the qualified 2030 retraction
```

This reuses the broader Scientific Method/Futures theorem:

```text
current knowledge
    != historical information set
```

The lifecycle event therefore needs availability/registration provenance appropriate to the historical view, not merely an asserted effective date.

---

## 17. Effective date and known-at date are separate

Some lifecycle actions may claim an effective date earlier than the date on which the action became known.

The Atlas should preserve both where the domain requires them:

```text
event_effective_time
knowledge_available_time
```

Historical scientific-state reconstruction normally gates on `knowledge_available_time`.

Otherwise a later backdated correction could leak future knowledge into an earlier Atlas state.

---

## 18. Retraction of a lifecycle event must not erase the event

A lifecycle declaration or qualification may itself later be challenged, corrected, or invalidated.

The solution is not to delete it.

Instead the argument/lifecycle graph records another event or defeater targeting the earlier lifecycle object.

Example:

```text
R1: source retraction declaration
C1: qualified challenge showing R1 lacked issuer authority
```

The history remains:

```text
R1 existed
C1 later challenged R1
current policy treats R1 as ineligible/invalid
```

This permits non-monotonic current interpretation while keeping the archive append-only.

---

## 19. A successor does not inherit predecessor evidence authority automatically

If E2 corrects or supersedes E1, E2 must obtain its own evidence/provenance/admission identity.

The lifecycle relation may establish ancestry, but not automatic authority transfer.

Therefore:

```text
E1 qualified prospective evidence
    + E1 -> E2 correction
        != E2 qualified prospective evidence
```

unless E2 independently satisfies the required qualification or an explicit migration theorem exists for the requested use.

This prevents correction chains from becoming authority-laundering channels.

---

## 20. Corrections may change only part of a contribution

A correction can be scoped.

For example:

```text
E1 supports P1 and reports secondary measurement M2
E2 corrects only M2
```

The shared kernel should not automatically infer either:

```text
all of E1 invalid
```

or:

```text
all of E1 unchanged
```

A correction event should identify the exact corrected subclaims/artifacts or issue a complete replacement contribution whose compatibility is explicit.

Domain-owned adapters decide how contribution granularity maps to scientific claims.

---

## 21. No correction-chain authority from publication count

Multiple corrections or versions do not become multiple replications.

```text
E1 -> E2 -> E3
```

is one lifecycle lineage unless independent evidence lineage proves otherwise.

SCI-006 dependency analysis should therefore retain correction/supersession ancestry as a dependency.

A re-analysis of the same source material under a new estimator may be a distinct evidence contribution but still heavily dependent on the original data lineage.

---

## 22. Lifecycle generation is part of disposition identity

A cached scientific disposition must bind the exact lifecycle generation/view it was computed from.

Conceptually:

```text
ScientificDispositionAssessment {
    proposition_id,
    argument_graph_generation_id,
    lifecycle_view_generation_id,
    disposition_policy_id,
    ...
}
```

If a contribution is corrected, superseded, retracted, or invalidated, a previous disposition remains a valid historical artifact but cannot masquerade as the current view.

```text
same proposition + changed lifecycle generation
    -> new disposition assessment
```

---

## 23. Current projection should preserve the reason topology

The current lifecycle view should never reduce to only:

```text
status = inactive
```

It should preserve why:

```text
source retraction
external invalidation
supersession
unresolved branch
expired qualification
historical-only use
```

and the exact events/evidence that established that state.

That reason topology is required for audit, challenge, historical reconstruction, and future re-evaluation.

---

## 24. Muse precedent and generic generalization

Symthaea Muse already has a strong post-publication pattern:

```text
immutable original publication
+ forward-linked correction/addendum/retraction events
+ retained superseded text
+ public notice
+ terminal retraction for that publication lineage
```

The shared Scientific Method Kernel should reuse the structural lessons:

```text
immutability
append-only correction history
explicit authority/provenance
forward lineage
retained predecessor content
```

but should not steal Muse's exact domain policy.

In particular:

```text
Muse publication retraction terminality
    != universal lifecycle law for every scientific evidence object
```

---

## 25. Candidate lifecycle graph shape

A future shared representation might look conceptually like:

```text
EvidenceContribution E1
    |
    +-- CorrectionDeclared --> E2
    |
    +-- SupersessionDeclared --> E3
    |
    +-- RetractionDeclared R1
    |
    +-- ExternalInvalidation D1

R1
    +-- LifecycleChallenge --> C1
```

Then the current view is produced by:

```text
exact lifecycle graph
+ exact current qualifications
+ exact requested use
+ exact policy
+ exact historical cutoff
    -> ContributionEligibilityAssessment
```

not by mutating E1.

---

## 26. Suggested first implementation tranches

Do not begin with a mutable `status` column.

A narrow sequence is:

```text
SCI-014b.1  immutable EvidenceContributionV1 identity
SCI-014b.2  lifecycle event identity + append-only event store
SCI-014b.3  source-authority / external-adjudication separation
SCI-014b.4  correction/supersession successor binding
SCI-014b.5  historical-cutoff lifecycle view
SCI-014b.6  use-specific ContributionEligibilityAssessment
SCI-014b.7  lifecycle generation binding into disposition assessment
```

Each should receive independent qualification.

---

## 27. Required adversarial cases

Future qualification should prove at least:

1. correcting a contribution cannot mutate predecessor bytes or identity;
2. supersession does not erase or invalidate the predecessor automatically;
3. source retraction does not produce proposition negation;
4. external invalidation cannot masquerade as source retraction;
5. an unauthorized caller cannot mint a qualified retraction by naming a contribution ID;
6. two competing successors are retained as a branch rather than latest-wins;
7. timestamp ordering alone cannot resolve lifecycle authority;
8. stale lifecycle-event replay cannot overwrite a later qualified stream generation;
9. a successor does not inherit predecessor evidence qualification automatically;
10. a correction chain does not count as independent replication;
11. a 2030 retraction does not alter an Atlas view reconstructed as of 2028;
12. asserted effective time cannot substitute for known-at/availability time in historical views;
13. lifecycle events themselves can be challenged without deleting history;
14. current-use eligibility preserves exact reasons and event identities;
15. lifecycle state exposes no truth, belief, governance, or execution authority.

---

## 28. Authority boundary

The complete non-equivalence remains:

```text
EvidenceContributionV1
    != QualifiedLifecycleEvent
    != ContributionEligibilityAssessment
    != scientific disposition
    != canonical belief
    != recommendation
    != governance decision
    != execution authority
```

A scientific correction/retraction system manages the eligibility and interpretation of evidence.

It does not directly authorize action in the external world.

---

## 29. Refined SCI-014 path

The Theory Atlas path now becomes:

```text
SCI-006 evidence dependency graph
    -> #668 target compatibility / result comparison / triangulation
    -> #701 defeater-aware scientific argument graph
    -> #729 immutable scientific proposition identity
    -> evidence contribution lifecycle                    [this contract]
    -> disposition assessment + complete reason topology
    -> time-indexed SCI-014 storage/query projection
```

The lifecycle layer deliberately sits after proposition identity because every contribution and lifecycle event must target immutable scientific objects, and before final disposition because eligibility changes which historical contributions may influence a current view.

---

## 30. Important non-claims

This contract does not:

- implement evidence lifecycle Rust types;
- define one universal publisher/retraction authority model;
- declare retraction terminal for every scientific domain;
- define legal publication status;
- verify signatures, timestamps, or notices;
- define universal eligibility effects for every lifecycle event;
- decide whether a contribution was fraudulent;
- establish truth or falsity of any proposition;
- qualify any current draft evidence;
- transfer scientific state into action authority.

Review only whether the contract makes corrections, supersession, retraction, invalidation, and historical scientific-state reconstruction append-only, explicit, challengeable, and resistant to silent authority or history rewriting.
