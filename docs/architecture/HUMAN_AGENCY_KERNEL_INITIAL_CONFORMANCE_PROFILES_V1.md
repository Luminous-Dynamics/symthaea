# Human Agency Kernel — Initial Conformance Profiles v1

Status: HAK-005 architecture/evidence audit; documentation only

Parent: `HUMAN_AGENCY_KERNEL_PROOF_OBLIGATIONS_AND_CONFORMANCE_V1.md`

Source snapshots inspected for this audit:

- Symthaea `main@2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`
- Mycelix `main@db311c53c547c4dc35e6795ec9a9e2462f653e1c`

No qualification transfers from those source snapshots to the HAK documentation branch merely because the files are in ancestry or are semantically related.

## 1. Purpose

HAK-005 is useful only if it can distinguish:

```text
strong local mechanism
```

from:

```text
qualified end-to-end authority property
```

without either dismissing good work or overstating what has been established.

This document applies HAK-005 to three materially different domains:

1. Symthaea fabrication partition leases;
2. Symthaea subterranean rescue ethics;
3. Mycelix governance authority.

The goal is not to give each domain a score.

The goal is to test whether the HAK proof-obligation model produces useful, falsifiable distinctions across:

```text
non-human exclusive resource authority
human consent/emergency authority
collective civic/governance authority
```

If one generic runtime abstraction were required to express all three, that would be evidence that HAK had over-generalized.

## 2. Status vocabulary used in this audit

This document uses explicit states rather than `PASS` / `FAIL` alone.

```text
NotApplicable(reason)
Specified(E0)
SourceObserved(E1)
TestSourceObserved
Unknown
OpenFinding(issue/ref)
BlockedBy(obligation)
PropertyViolated(evidence)
QualifiedAt(E*)
```

Important distinction:

```text
TestSourceObserved != QualifiedAt(E2)
```

A unit test existing in source is useful evidence that the invariant was considered, but E2 requires actual execution evidence bound to the implementation/policy lineage being claimed.

Likewise:

```text
SourceObserved(E1) != IntegrationQualified(E4)
```

## 3. New HAK-005 theorem: qualification is not transitively compositional

Suppose:

```text
A -> B
B -> C
```

and both local edges have strong evidence.

It does not follow automatically that:

```text
A -> C
```

is qualified.

The join may introduce:

- subject mismatch;
- policy mismatch;
- stale data;
- TOCTOU windows;
- concurrent forks;
- replay;
- alternate entrypoints;
- partial effects;
- recovery/reset behavior;
- compatibility fallbacks;
- unqualified serialization/migration.

Candidate theorem:

```text
Qualified(A -> B)
AND Qualified(B -> C)
-/-> Qualified(A -> C)
```

without an explicit composition obligation.

For a consequential pipeline, each load-bearing join needs its own claim.

This document uses that rule throughout.

---

# Profile A — Symthaea Fabrication Partition Lease

## A.1 Domain identity

Profile candidate:

```text
profile_id = HAK-CONF-FAB-LEASE-V1
domain = symthaea-fabrication-kernel
authority_kind = ResourceLeaseAuthority
primary_conservation = ExclusiveLease + EvidenceBound + RestrictionMonotonic
human_standing = NotApplicable
```

Primary source:

```text
crates/domains/symthaea-fabrication-kernel/src/lease_authority.rs
```

Important consumer observed:

```text
crates/domains/symthaea-fabrication-kernel/src/release_promotion.rs
```

## A.2 Authority graph observed

Conceptually, the source implements:

```text
GatewayMembership
+ VerifiedGatewayConsensus
+ LeasePolicy
+ HolderIdentity
+ LeaseSequence
+ FencingToken
+ ValidityWindow
+ VerifiedThresholdCeremony(exact lease digest)
        ↓
AuthorizedPartitionLease
        ↓
LeaseAuthorityTracker::accept
        ↓
AcceptedPartitionLease
```

The lease is also consumed by release-promotion construction:

```text
CertifiedReleaseCandidate
+ ArtifactSet
+ GatewayReplayDigest
+ GatewayMembership
+ AuthorizedPartitionLease
+ Transparency Evidence
+ VerifiedTransparencyCheckpoint
        ↓
ReleasePromotionEvidence
```

This is important because it demonstrates an actual authority-bearing consumer rather than a completely isolated capability object.

## A.3 Strong local properties observed

### FAB-LEASE-PROV-001 — lease legitimacy inputs are explicit

Claim:

```text
An AuthorizedPartitionLease cannot be constructed solely from caller-declared holder/resource metadata.
```

Source observations:

- membership validates;
- membership must be active;
- holder must be a member;
- holder must participate in the verified consensus;
- lease window must be valid and policy-bounded;
- consensus voting weight must meet policy;
- failure-domain diversity must meet policy;
- ceremony purpose must equal `partition-lease-authority`;
- ceremony payload digest must equal the exact lease digest.

Status:

```text
SourceObserved(E1)
```

Required future evidence:

- exact negative tests for every authority input;
- exact-head execution;
- mutation/adversarial evidence for ceremony/subject binding.

### FAB-LEASE-LIN-001 — rollback barriers are explicit

Claim:

```text
Continuing lease authority cannot accept an older membership epoch,
older lease sequence, or non-increasing fencing token.
```

Source observations in `LeaseAuthorityTracker::accept`:

```text
membership_epoch < latest -> reject
lease_sequence < latest -> reject
fencing_token <= latest -> reject
```

Status:

```text
SourceObserved(E1)
```

### FAB-LEASE-CONS-001 — overlapping exclusive leases are rejected locally

Claim:

```text
While one different lease remains active, a new conflicting lease cannot become current.
```

Source observation:

```text
active_lease_expires_at_unix_ms > now
AND active_lease_digest != proposed_digest
-> ActiveLeaseConflict
```

The source also contains:

```text
tracker_rejects_overlapping_different_lease
```

Status:

```text
SourceObserved(E1)
TestSourceObserved
```

Do not call this E2 until the test is executed and attached to an exact implementation lineage.

### FAB-LEASE-EVID-001 — same-sequence substitution is rejected

Claim:

```text
Same lease sequence cannot identify a different lease artifact.
```

Observed source semantics:

```text
same sequence + same digest -> idempotent replay
same sequence + different digest -> reject
```

Status:

```text
SourceObserved(E1)
```

This is a strong candidate for reusable HAK idempotency vocabulary.

## A.4 Composition property observed

### FAB-PROMO-COMP-001 — release promotion consumes exact lease evidence

`build_release_promotion_evidence()` takes an `AuthorizedPartitionLease`, not an arbitrary lease-shaped struct.

It verifies that the lease:

- is unexpired at promotion authorization time;
- matches the current membership digest and epoch;
- matches the candidate gateway state digest;
- matches gateway generation;
- matches gateway consensus digest.

It then records:

```text
partition_lease_digest
fencing_token
```

inside release-promotion evidence.

Status:

```text
SourceObserved(E1)
```

This is evidence that HAK's proposed pattern:

```text
QualifiedArtifact -> ExactQualifiedConsumer
```

already exists in production-oriented domain design.

## A.5 Important unqualified claims

The current audit does **not** establish:

```text
all fabrication side effects require an accepted/current lease
```

or:

```text
lease tracker state is crash-safe / durably restored without authority widening
```

or:

```text
concurrent independent trackers cannot each accept conflicting authority
```

Those require broader call-path/distribution/recovery evidence.

Candidate obligations:

```text
FAB-LEASE-EFFECT-001
Every effect requiring exclusive partition authority consumes one exact current accepted lease/fence.

FAB-LEASE-REC-001
Continuing-lineage restart preserves the latest authority tracker rollback barriers.

FAB-LEASE-DIST-001
Partitioned/distributed acceptance cannot create two simultaneously effective holders for one fenced resource lineage.
```

Current status:

```text
Unknown
```

## A.6 Rights-floor exclusions

For the partition lease itself:

```text
HAK-FLOOR baseline civic standing = NotApplicable(non-human resource lease)
HAK consent floor = NotApplicable(non-human resource lease)
```

This is exactly why HAK should not require one universal profile.

However, if fabrication commands later act on humans or human-owned resources, those downstream boundaries may acquire separate HAK-FLOOR obligations.

## A.7 Profile conclusion

The fabrication lease domain is strong evidence that the HAK authority taxonomy and `ExclusiveLease` conservation family are useful.

It also demonstrates that HAK conformance must distinguish:

```text
lease construction
lease tracker acceptance
lease consumption
side-effect enforcement
recovery/distribution
```

rather than labeling the entire crate `secure`.

---

# Profile B — Symthaea Subterranean Rescue Ethics

## B.1 Domain identity

Profile candidate:

```text
profile_id = HAK-CONF-SUB-RESCUE-V1
domain = symthaea-subterranean
authority_kinds = ConsentAuthority + EmergencyExceptionAuthority + SafetyRestriction
primary_conservation = NonTransferable + RestrictionMonotonic + ThresholdQuorum
human_standing = directly applicable
```

Primary sources observed:

```text
crates/domains/symthaea-subterranean/src/rescue_ethics.rs
crates/domains/symthaea-subterranean/src/rescue_ethics_validation.rs
```

## B.2 Local ethics contracts observed

`RescueEthicsValidationReport` enumerates explicit contracts including:

```text
ConsentReplayRejected
WithdrawalStopsActiveRescue
EmergencyAuthorityRequiresIndependentRoles
ConflictingIdentityRequiresReconciliation
RefusalDominatesUrgency
TriageExcludesProtectedAttributes
RecoveryActuatorsSurviveEthicsHold
CheckpointPreservesConsentAuthority
```

This is unusually good prior art for HAK-005 because the domain already names claims rather than only tests.

Status of the contracts as observed by this audit:

```text
Specified(E0)
SourceObserved(E1)
TestSourceObserved
```

Not yet upgraded to E2 in this HAK evidence lineage.

## B.3 Consent non-transferability

### SUB-CONSENT-SEM-001

Claim:

```text
Distress, urgency, role, silence, or peer opinion do not become consent.
```

Source/protocol observations support the separation:

```text
Distress != Consent
EmergencyException != Consent
```

Status:

```text
SourceObserved(E1)
```

### SUB-CONSENT-LIN-001

Claim:

```text
A fresher refusal or withdrawal cannot be undone by replaying older acceptance in the same case/epoch lineage.
```

The domain has replay-resistant epoch/sequence semantics and explicit validation fixtures.

Status:

```text
SourceObserved(E1)
TestSourceObserved
```

### SUB-CONSENT-REC-001

Claim:

```text
Recovery checkpoint preserves current withdrawal/refusal authority rather than restoring earlier positive consent.
```

The validation harness explicitly checks checkpoint preservation of withdrawn consent.

Status:

```text
SourceObserved(E1)
TestSourceObserved
```

This is strong evidence for HAK's rule:

```text
storage lifecycle != normative authority lifecycle
```

## B.4 Emergency authority

### SUB-EMERG-CONS-001

Claim:

```text
Emergency rescue authority requires distinct qualified principals for required roles.
```

The validation harness constructs SafetyOfficer and IndependentWitness approvals and rejects a same-principal role collapse.

Status:

```text
SourceObserved(E1)
TestSourceObserved
```

This maps to HAK's `ThresholdQuorum` family, where distinct principals—not raw signature count—are the conserved unit.

## B.5 Safety restriction semantics

### SUB-RESCUE-FLOOR-001

Claim:

```text
Withdrawal/refusal can reduce subject-affecting rescue authority even under urgency.
```

The local ethics supervisor can produce `HoldForReview`, and `constrain_command()` removes prohibited motion while preserving selected recovery actuators.

Status:

```text
SourceObserved(E1)
TestSourceObserved
```

This demonstrates an important HAK-004 nuance:

```text
SafetyRestriction may reduce immediate action
while preserving future human agency/survival options.
```

## B.6 Critical composition finding

HAK-005 asks a stronger question than whether `constrain_command()` is correct:

```text
Does every live subject-affecting command path actually pass through it?
```

Current code search on the inspected source snapshot found `constrain_command()` in:

- its definition;
- the rescue ethics validation harness.

Likewise, the audit did not find an obvious live command-path consumer for `RescueEthicsSupervisor::assess()`.

This is not proof of absence.

It means the following stronger property is not established by the current audit:

```text
AllLiveRescueMotionIsEthicsConstrained
```

Tracked as:

```text
Symthaea #847
safety(subterranean): establish live rescue-ethics enforcement on every motion path
```

### SUB-RESCUE-EFFECT-001

Target claim:

```text
For every live command path capable of subject-affecting rescue motion,
a current subject/case-bound ethics assessment is consumed before the first applicable actuator effect.
```

Current status:

```text
OpenFinding(#847)
```

This is the first concrete demonstration of why:

```text
QualifiedLocalPolicy
-/-> QualifiedEnforcementReachability
```

## B.7 Additional open obligations

### SUB-RESCUE-ENTRY-001

All alternate motion paths—manual, autonomous, recovery, replay, admin, migration, compatibility—must either preserve the same minimum ethics obligations or declare different legitimate semantics.

Status:

```text
Unknown
```

### SUB-RESCUE-EFFECT-002

A valid emergency exception must widen only the explicitly permitted rescue scope and must not become generalized subject control.

Status:

```text
Specified(E0)
SourceObserved(E1) locally
Integration status Unknown
```

### SUB-RESCUE-REC-002

Restart/recovery must preserve refusal/withdrawal barriers across the real persisted operational lineage.

A deterministic checkpoint test is useful but does not by itself establish crash/restart persistence in a deployed runtime.

Status:

```text
TestSourceObserved
Operational composition Unknown
```

## B.8 Profile conclusion

The rescue domain strongly validates HAK's semantic distinctions:

```text
Consent
!= Distress
!= EmergencyException
!= TriagePriority
!= ExecutionAuthority
```

But it also demonstrates HAK-005's central reason for existence:

> a locally correct rights/consent mechanism is not enough unless every relevant effect path actually consumes it.

---

# Profile C — Mycelix Governance Authority

## C.1 Domain identity

Profile candidate:

```text
profile_id = HAK-CONF-MYC-GOV-V1
domain = mycelix-governance
authority_kinds = BaselineCivicStanding + DelegatedAuthority + ConsumableVotingBudget + GovernanceDecision + ExecutionAuthority + SafetyRestriction
human_standing = directly applicable
```

Source snapshot inspected:

```text
mycelix main@db311c53c547c4dc35e6795ec9a9e2462f653e1c
```

The purpose of this profile is **not** to claim current governance is conformant.

It is to show whether HAK-005 can organize already-separated findings into one dependency-aware evidence graph.

## C.2 Current provenance architecture candidate

Mycelix governance PR #329 now proposes the target chain:

```text
ProposalDraft
-> ProposalVotingPolicy
-> OpenVotingLineage
-> CanonicalBallots
-> QualifiedVotingClosure
-> QualifiedTally
-> EvidenceBoundProposalTransition
-> ApprovedProposalSubject
-> VerifiedProposalSignature
-> AuthorizedTimelock
-> AuthorizedExecution
-> ExternalSideEffects
-> ExecutionReceipt
-> OutcomeEvidence
```

That document is architecture, not runtime qualification.

Status:

```text
Specified(E0)
```

## C.3 Baseline civic standing

### MYC-GOV-FLOOR-001

Target claim:

```text
Change in model-derived consciousness/reputation assessment alone cannot erase baseline civic standing.
```

Tracked by:

```text
Mycelix #309
PR #312
```

Current status:

```text
OpenFinding(#309)
```

This remains constitutionally independent from proof-verification correctness.

## C.4 Privacy-preserving eligibility proof

### MYC-GOV-PROV-001

Target claim:

```text
A proof-container claim can affect governance eligibility only after cryptographic verification,
trusted-verifier qualification, freshness, and subject/policy binding are established.
```

Tracked by:

```text
Mycelix #292
```

Current status:

```text
OpenFinding(#292)
```

HAK distinction:

```text
StructureValid
!= ProofVerified
!= TrustedVerifier
!= PolicyQualifiedClaim
```

## C.5 Voice-credit mint and spend

### MYC-GOV-CONS-VC-001

Target claim:

```text
Quadratic voting credits originate only from an explicit qualified issuer/policy lineage.
```

Current source audit found public allocation semantics without a qualified issuer boundary.

Tracked by:

```text
Mycelix #330
```

Current status:

```text
OpenFinding(#330)
```

### MYC-GOV-CONS-VC-002

Target claim:

```text
For one grant lineage:
allocated = spent + remaining
spent' >= spent
remaining' <= remaining
allocated' = allocated
```

Current integrity update semantics do not yet establish all those immutable/monotonic fields.

Status:

```text
OpenFinding(#330)
```

### MYC-GOV-COMP-VC-003

Target claim:

```text
A quadratic ballot consumes/references one exact committed spend artifact from one qualified credit grant lineage.
```

Current composition creates the `QuadraticVote` entry before the separate credit-update operation and the ballot declares `credits_spent` rather than consuming an immutable spend receipt.

Status:

```text
OpenFinding(#330)
```

This is a canonical HAK-005 example:

```text
balance check valid
ballot shape valid
spend update valid
```

still does not automatically imply:

```text
atomic / fork-safe quadratic exercise
```

## C.6 Delegated civic mass

### MYC-GOV-CONS-DEL-001

Target claim:

```text
For one overlapping proposal scope:
retained source civic mass
+ all effective delegated children
<= original source civic budget.
```

Current source constrains each delegation independently to:

```text
0 < percentage <= 1
```

but does not establish overlap-aware conservation across multiple active outgoing delegations.

Tracked by:

```text
Mycelix #331
```

Current status:

```text
OpenFinding(#331)
```

### MYC-GOV-CONS-DEL-002

Target claim:

```text
Direct and delegated exercise cannot both consume the same civic source mass in one voting lineage.
```

Current status:

```text
OpenFinding(#331/#332)
```

## C.7 Delegated ballot admission

### MYC-GOV-ENTRY-DEL-001

Target claim:

```text
A delegated ballot is admitted only for the current proposal/version/voting lineage,
under the proposal's canonical policy, during the valid window, with one canonical exercise key.
```

Tracked by:

```text
Mycelix #332
```

Current status:

```text
OpenFinding(#332)
```

This profile treats author/voter binding as a separate positive property from uniqueness/window/policy binding.

```text
AuthenticatedCaster != UniqueQualifiedBallot
```

## C.8 Tally qualification

### MYC-GOV-PROV-TALLY-001

Target claim:

```text
A final tally is verifier-owned and reconstructed from a closed canonical ballot lineage under a precommitted policy.
```

Current source includes public `tally_votes()` inputs with caller-selectable tier and quorum/approval overrides, and creates a `final_tally: true` representation.

Tracked by:

```text
Mycelix #323
```

Current status:

```text
OpenFinding(#323)
```

HAK distinction:

```text
FinalTallyLabel != QualifiedVotingClosure + QualifiedTally
```

## C.9 Proposal state provenance

### MYC-GOV-EDGE-STATE-001

Target claim:

```text
Authority-bearing proposal states are projections of exact transition evidence,
not merely structurally allowed enum updates.
```

Tracked by:

```text
Mycelix #319
```

Current status:

```text
OpenFinding(#319)
```

## C.10 Ethics assessment boundary

### MYC-GOV-PROV-ETHICS-001

Target claim:

```text
Advisory ethics/model assessment cannot become a binding governance restriction
without authenticated assessment provenance and an explicit governance ethics policy.
```

Tracked by:

```text
Mycelix #325
```

Current status:

```text
OpenFinding(#325)
```

## C.11 Signing/timelock/execution

### MYC-GOV-EFFECT-001

Target claim:

```text
The first external side effect occurs only after current execution authority binds
the exact approved proposal subject/actions, required signature policy, timelock,
executor, resource scope, and current lineage.
```

Tracked by:

```text
Mycelix #317
```

Current status:

```text
OpenFinding(#317)
```

This is independent from whether voting itself is well formed.

## C.12 Governance profile dependency graph

The important result of HAK-005 is that these findings are not one blob called `governance security`.

They form a dependency graph:

```text
baseline legitimacy (#309)
proof qualification (#292)
voice-credit legitimacy/conservation (#330)
delegation conservation (#331)
ballot admission uniqueness (#332)
        ↓
canonical ballot lineage
        ↓
qualified closure/tally (#323)
        ↓
proposal transition provenance (#319)
        ↓
exact approved/signed subject (#317)
        ↓
authorized execution / effect ordering (#317)
```

with ethics provenance (#325) joining only where the precommitted policy gives it a binding role.

This is precisely the kind of structure a global `governance_passed = true` flag would destroy.

## C.13 Current profile conclusion

Current Mycelix governance should not receive one HAK conformance label.

A more accurate state is:

```text
architecture obligations: increasingly well specified
several local protections: present
multiple authority-bearing joins: open findings
exact end-to-end qualification: not established
operational/empirical governance quality: not claimed
```

That is not a negative judgment on the project.

It is the evidence accounting HAK-005 is designed to make possible.

---

# 4. Cross-domain comparison

The three profiles are materially different, but several reusable review patterns survive.

## 4.1 Qualified artifact must meet qualified consumer

Fabrication demonstrates a good pattern:

```text
AuthorizedPartitionLease
-> build_release_promotion_evidence(... lease ...)
```

HAK should look for this pattern elsewhere:

```text
QualifiedArtifact
-> consumer explicitly requires qualified type/artifact
```

rather than:

```text
qualified artifact exists somewhere
+ consumer accepts caller reconstruction
```

## 4.2 Local policy correctness and enforcement reachability are distinct

Rescue ethics demonstrates:

```text
CorrectConstraintFunction
!= AllRelevantEffectsConsumeConstraint
```

This should become a generic HAK composition review.

## 4.3 Conservation unit is domain-specific

Fabrication:

```text
exclusive holder count <= 1
```

Rescue emergency roles:

```text
required role principals are distinct
```

Governance delegation:

```text
retained + delegated civic budget <= source budget
```

Quadratic credits:

```text
allocated = spent + remaining
```

These cannot safely collapse into one scalar `authority_amount`.

## 4.4 Canonical exercise identity is widely reusable

Many domains need an answer to:

```text
what exact authority exercise is this?
```

Examples:

- lease sequence + digest + fence;
- consent case/epoch/sequence;
- governance proposal lineage + source civic unit + ballot exercise key;
- execution subject digest + execution generation/receipt.

Candidate generic HAK question:

```text
Can retries, alternate endpoints, forks, or recovery cause the same source authority
unit to be exercised more than once?
```

The exact key remains domain-owned.

## 4.5 Recovery is a proof boundary, not plumbing

Across all stateful domains, recovery must answer:

```text
which authority facts survive?
which must be revalidated?
which restrictions cannot disappear?
which counters/generations cannot roll back?
is this continuation or a new lineage?
```

This remains one of the highest-value HAK review questions.

## 4.6 Side-effect boundary is where abstractions become real

Authority reasoning matters only if it reaches the final effect boundary.

For each domain, HAK should eventually identify:

```text
last pure/verifier-owned authorization step
first irreversible/external effect
```

and require an explicit proof obligation on that edge.

---

# 5. Conformance matrix

| Domain | Semantic separation | Provenance | Lineage | Conservation | Rights/agency | Effect composition | Recovery | Current overall claim |
|---|---|---|---|---|---|---|---|---|
| Fabrication partition lease | SourceObserved | SourceObserved | SourceObserved | SourceObserved + test source | N/A at lease boundary | partial consumer evidence | Unknown beyond local tracker | no global conformance claim |
| Subterranean rescue ethics | SourceObserved | SourceObserved | SourceObserved | SourceObserved + test source | SourceObserved + test source | **OpenFinding #847** | test source + broader Unknown | no global conformance claim |
| Mycelix governance | partly specified/source-observed | multiple OpenFindings | multiple OpenFindings | multiple OpenFindings | **OpenFinding #309 + related** | multiple OpenFindings | architecture candidate | no global conformance claim |

This table intentionally refuses numeric grading.

## 5.1 Why no percentage

A profile with nine easy obligations and one unqualified side-effect boundary should not score `90% safe`.

Some obligations are load-bearing.

Therefore future profiles may mark obligations as:

```text
Informational
Supporting
Required
Critical
```

but should not turn those categories into one scalar assurance score without a domain-justified model.

---

# 6. HAK-005 refinement: critical-path closure

A domain should not claim an end-to-end property merely because most obligations are qualified.

For a claimed property `P`, identify the exact critical obligation set:

```text
Critical(P) = { O1, O2, ... On }
```

Then:

```text
EndToEndQualified(P)
requires every critical obligation and critical join for P
satisfy the required evidence tier.
```

A noncritical unknown does not necessarily block the claim.

A critical unknown does.

Candidate theorem:

```text
one unqualified critical join
-> end-to-end claim remains unqualified
```

This is stronger and more useful than test-count coverage.

---

# 7. HAK-005 refinement: obligation dependency graph

Proof obligations may depend on other obligations.

Example governance:

```text
QualifiedTally
depends_on:
    QualifiedBallotAdmission
    VotingClosure
    VotingPolicyFreeze
    Delegation/credit conservation where applicable
```

Example release promotion:

```text
AuthorizedReleasePromotion
uses:
    CertifiedReleaseCandidate
    AuthorizedPartitionLease
    VerifiedTransparencyCheckpoint
    VerifiedThresholdCeremony
```

A future audit manifest may therefore model:

```text
obligation_id
depends_on[]
consumes_artifacts[]
produces_artifacts[]
critical_for_claims[]
```

This remains audit metadata, not an authorization oracle.

---

# 8. HAK-005 refinement: evidence must be anti-circular

An authority claim must not be qualified only by evidence generated by the same unqualified authority path it is supposed to validate.

Candidate concern:

```text
System says action was authorized
because system emitted Authorized=true
```

is circular.

Prefer evidence that can reconstruct the premises independently:

```text
source artifact digests
policy digest
principal/subject binding
lineage/generation
verifier result
side-effect receipt
```

Candidate theorem:

```text
AuthorityLabel cannot be its own sole proof of authority
```

This extends:

```text
StateLabel != AuthorityEvidence
```

into HAK conformance itself.

---

# 9. What appears safely reusable

After these three profiles, the strongest candidates for shared HAK **audit vocabulary** are:

- exact claim identity;
- authority kind classification;
- implementation lineage;
- policy lineage;
- evidence tier;
- status state (`Unknown`, `Queued`, `Failed`, etc.);
- critical/noncritical designation;
- obligation dependencies;
- negative-case classes;
- canonical exercise identity question;
- last-authorization / first-effect boundary;
- recovery lineage question;
- explicit `NotApplicable(reason)`.

These are audit concepts.

They do not yet justify shared runtime types.

# 10. What remains domain-owned

The profiles reinforce that these should remain domain-owned unless equivalence is proven:

- consent semantics;
- emergency-role constitution;
- civic source-budget rules;
- voice-credit mint policy;
- lease fencing semantics;
- membership legitimacy;
- voting quorum/approval policy;
- subject-affecting actuator classification;
- resource ownership/property semantics;
- constitutional amendment authority;
- operational recovery rules.

# 11. First implementation candidate

If HAK-005 proceeds beyond documentation, the first implementation should **not** be an authorization library.

A safer first artifact is an audit-only manifest/schema plus validator that checks evidence bookkeeping.

For example, it could reject profiles where:

```text
qualification_status = ExactHeadCIQualified
but no exact commit/evidence ref exists
```

or:

```text
critical obligation = Unknown
while profile claims EndToEndQualified
```

or:

```text
NotApplicable
has no reason
```

or:

```text
policy-dependent claim
has no policy-lineage identifier
```

Such a tool would validate **claims about evidence**, not decide whether authority is legitimate at runtime.

# 12. Exit gate for HAK-005 architecture

Do not advance HAK-005 merely because these documents read well.

A reasonable architecture exit gate is:

1. at least two materially different domain profiles are reviewed;
2. each profile contains explicit negative/falsification cases;
3. at least one real composition gap is discovered or ruled out using the method;
4. at least one `NotApplicable` exclusion demonstrates domain-specificity;
5. evidence states do not collapse test source into executed qualification;
6. critical-path closure semantics are accepted;
7. obligation dependency semantics are accepted;
8. no shared runtime authority type is introduced prematurely;
9. exact-head CI qualifies the HAK documentation lineage independently.

This audit already satisfies the architecture-method requirement of producing a real new finding:

```text
Symthaea #847
```

but it does not qualify the affected runtime behavior.

# 13. Non-claims

This document does not claim:

- fabrication leases are globally safe;
- subterranean rescue ethics are currently wired into every live actuator path;
- Mycelix governance is deployable as public constitutional infrastructure;
- source-observed tests have executed in this evidence lineage;
- HAK is a certification standard;
- these three profiles exhaust relevant threats;
- HAK can determine universal ethics or legitimacy automatically.

The narrower result is:

> Boundary-scoped, evidence-tiered proof obligations produce materially useful distinctions across three very different authority domains, including at least one previously unqualified composition edge.
