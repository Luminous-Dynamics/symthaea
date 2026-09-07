# Human Agency Kernel — Authority Kind & Lineage Inventory v1

Status: HAK-003 architecture audit / documentation only

Parent: `HUMAN_AGENCY_KERNEL_AUTHORITY_AUGMENTATION_CONTRACT_V1.md` (HAK-002)

## 1. Purpose

HAK-001 separated assessment, recommendation, consent, delegation, execution, outcome, and human benefit.

HAK-002 added a lineage-aware authority monotonicity rule:

```text
SameAuthorityLineage(A, T(A))
    => Authority(T(A)) subset-of Authority(A)
```

and explicitly separated operational recovery from creation of a new simulation lineage.

HAK-003 asks the next prerequisite question:

> What kinds of "authority" actually exist in Symthaea/Mycelix, and which of them are comparable at all?

The repository currently uses authority-like vocabulary for several semantically different concepts. A premature shared `Authority` type would be dangerous because it could collapse epistemic strength, civic standing, delegation, resource leases, consent, and actuator permission into one abstraction.

This inventory therefore classifies **authority kind before implementation reuse**.

It does not introduce a common crate or normative cross-domain wire type.

## 2. Core anti-collapse theorem

```text
Same word != Same authority semantics
```

In particular:

```text
EpistemicAuthority      != ExecutionAuthority
EvidenceStrength        != Permission
Credential              != Authorization
BaselineCivicStanding   != ScopedQualification
Recommendation          != Consent
Consent                 != Delegation
Delegation              != Execution
SafetyRestriction       != PositiveGrant
ResourceLease           != HumanStanding
AdvisorySignal          != VoteWeight
```

A future common abstraction is justified only when source, subject, scope, lineage, widening transitions, narrowing transitions, expiry, revocation, and enforcement semantics are materially identical.

## 3. Authority-kind taxonomy candidate

### 3.1 Epistemic authority

Meaning: how strongly evidence supports a claim or inference.

Examples:

- causal / quasi-experimental / associational strength;
- calibrated confidence;
- scientific qualification state;
- verified observation provenance.

Epistemic authority may inform a policy decision but must not itself become permission to act.

```text
StrongEvidence != ExecutePermission
```

### 3.2 Baseline civic standing

Meaning: a person's/member's ordinary standing to participate in collective governance.

Candidate HAK default:

```text
eligible member -> equal baseline civic voice
```

Subject to domain-owned identity, membership, and Sybil-resistance policy.

It must not silently derive from model-assessed consciousness, reputation, wealth, behavioral engagement, or social approval.

### 3.3 Scoped role qualification

Meaning: evidence that an actor is qualified for a bounded specialized role.

Examples:

- safety officer;
- treasury operator;
- domain reviewer;
- infrastructure maintainer;
- incident responder;
- cryptographic ceremony participant.

Qualification can legitimately be evidence-sensitive because the role is scoped, functional, and reviewable.

```text
QualifiedForRole(R) != HigherHumanWorth
```

### 3.4 Delegated authority

Meaning: one principal explicitly grants a bounded subset of their authority to a delegate.

Required semantics normally include:

- principal;
- delegate;
- permitted actions;
- resources/subjects;
- purpose where material;
- temporal bounds;
- generation/sequence;
- revocation;
- integrity binding;
- audience/context binding.

### 3.5 Resource lease authority

Meaning: temporary exclusive or fenced authority over a concrete resource/service.

Examples include gateway or fabrication leases.

Lease authority is operational and resource-relative. It does not imply civic or epistemic standing.

### 3.6 Consent authority

Meaning: whether an affected human/subject has explicitly authorized a domain action concerning them.

Consent is subject- and purpose-specific.

```text
ConsentTo(A) != ConsentTo(B)
```

and:

```text
Distress != Consent
Silence  != Consent
Role     != Consent
```

### 3.7 Emergency exception authority

Meaning: a narrowly bounded exception used when ordinary consent/coordination cannot be obtained and a domain-specific emergency policy authorizes intervention.

This must remain distinguishable from ordinary consent.

```text
EmergencyException != RetroactiveConsent
```

### 3.8 Execution authority

Meaning: permission for an executor to perform an action on a resource/person/system.

Execution authority should be explicit and current:

```text
ExecutionPermission = CurrentExplicitGrant
```

not inferred from absence of known denial.

### 3.9 Safety restriction

Meaning: a negative constraint that reduces the executable action set.

Examples:

- emergency stop;
- hold for review;
- maintenance lock;
- degraded-operation restriction;
- causal/temporal hold;
- capability derating.

A safety restriction is not simply an inverse positive grant. Restriction removal is an authority-widening transition.

### 3.10 Advisory / deliberative signal

Meaning: information intended to improve human or collective reasoning without itself changing formal authority.

Examples:

- expertise indicators;
- Phi/consciousness research signals;
- prediction calibration;
- affected-party analysis;
- reputation context;
- argument/evidence quality;
- minority/dissent signals.

HAK recommends this category for many signals currently tempted to become person-level governance weight.

## 4. Lineage dimensions

Before comparing two authority artifacts, identify at least:

```text
principal
subject/resource
purpose
context/audience
lifecycle
policy domain
lineage identity
```

Two artifacts that differ materially in these dimensions may be incomparable rather than ordered.

### 4.1 Same-lineage examples

Potential same-lineage operations:

- delegation v1 -> delegation v2 migration;
- current policy -> compatibility representation;
- running system -> operational checkpoint restore;
- lease renewal within the same fenced resource lineage;
- explicit grant -> attenuated sub-grant.

These are candidates for monotonicity checks.

### 4.2 New-lineage examples

Potential lineage breaks:

- simulation scenario A -> freshly initialized scenario B;
- destroyed local authority identity -> newly provisioned deployment identity;
- new community constitution intentionally creating a distinct governance lineage.

Do not automatically compare authority sets across a deliberate lineage break.

## 5. Initial Symthaea inventory

### 5.1 Subterranean operator authority

Kind:

```text
ExecutionAuthority + SafetyRestriction
```

Properties:

- explicit operator command protocol;
- replay resistance;
- restrictive mission/operator states;
- independent recovery quorum for widening transitions;
- physical-hazard checks before resume.

Lineage boundary:

- operational checkpoint restoration is continuity;
- generic `EmbodimentBridge::reset()` is currently scenario/default-state reinitialization and must not be treated as the same operation.

HAK implication:

```text
operational recovery rules apply only after lifecycle/lineage identity is established
```

### 5.2 Subterranean rescue consent

Kind:

```text
ConsentAuthority
```

Properties:

- case-scoped;
- explicit acceptance/refusal/withdrawal semantics;
- replay/epoch sequencing;
- distress is not consent;
- withdrawal/refusal can constrain rescue action.

Open/active hardening established by the HAK audit:

```text
negative consent barrier must not expire and resurrect older positive authority
```

### 5.3 Subterranean emergency rescue authority

Kind:

```text
EmergencyExceptionAuthority
```

Properties:

- subject + rescue-case bound;
- expires;
- requires distinct hardware-backed SafetyOfficer and IndependentWitness roles;
- replay/epoch checks;
- not a substitute for ordinary consent.

### 5.4 Fabrication authority epoch

Kind:

```text
OperationalAuthorityLineageEvidence
```

The fabrication kernel tracks a vector of independent monotonic generations including trust, membership, gateway, resilience, containment, transparency, release-lineage, and incident history.

Important property:

```text
partial component rollback -> reject
```

This is useful prior art for HAK lineage continuity, but the exact vector is fabrication-domain-owned.

### 5.5 Fabrication partition lease

Kind:

```text
ResourceLeaseAuthority
```

Properties include:

- concrete holder identity;
- membership binding;
- consensus binding;
- threshold ceremony binding;
- lease sequence;
- fencing token;
- bounded validity interval;
- conflict detection;
- rollback rejection.

This is one of the clearest existing examples of explicit operational authority and should be studied before creating generic HAK execution types.

### 5.6 Scientific causal authority

Kind:

```text
EpistemicAuthority
```

Examples such as `CausalAuthority::{Causal, QuasiExperimental, Associational}` describe inference strength.

They must not be routed into `AuthorityEnvelopeV1` as executable permission merely because the type name contains `Authority`.

```text
CausalAuthority::Causal != PermissionToIntervene
```

### 5.7 Quantum adapter authority

Kind:

```text
CapabilityRestriction / AdapterBoundary
```

Example distinction:

- export artifacts only;
- observe external outputs;
- stronger adapter capabilities.

This is closer to execution/capability authority than epistemic causal authority, but still domain-owned.

## 6. Initial Mycelix inventory

### 6.1 AI SubPassport

Kind:

```text
DelegatedAuthority
```

HAK audit finding:

- the original signing transcript did not bind every authority-relevant semantic;
- variable-length identity fields were not safely framed;
- renewal/revocation semantics required hardening.

Tracked independently in the Mycelix delegation-transcript tranche.

HAK lesson:

```text
signed bytes must encode the exact grant semantics
```

### 6.2 8D civic requirement -> legacy civic requirement

Kind:

```text
GovernanceEligibilityPolicyTranslation
```

HAK audit finding:

- the native requirement can express dimensions the legacy policy cannot;
- legacy fallback must not silently erase unsupported constraints.

Required translation result:

```text
Exact | ConservativeProjection | NotRepresentable
```

not generic best effort.

### 6.3 Legacy ConsciousnessProfile / sovereign profile

Kind currently mixes:

```text
Assessment
+ Credential
+ GovernanceEligibility
+ VoteInfluence
```

This is precisely the kind of multi-role type HAK should decompose.

The legacy profile combines identity, reputation, community, and engagement and can derive vote weight. It also contains constructors that map Symthaea consciousness-related measurements into the engagement dimension.

HAK candidate decomposition:

```text
AssessmentCredential
DeliberativeSignal
ScopedRoleQualification
BaselineCivicStanding
```

with no implicit conversion between them.

### 6.4 Live Mycelix governance consciousness gate

Kind currently:

```text
Model/CredentialAssessment -> GovernanceEligibility
```

The governance bridge defines action-level consciousness thresholds for Basic participation, ProposalSubmission, Voting, and Constitutional actions.

The voting coordinator actively invokes this gate. Constitutional/Emergency votes use a fail-closed consciousness gate; other proposal types use a best-effort bridge path.

This demonstrates that the HAK concern is not limited to a dormant compatibility struct.

Tracked constitutionally in Mycelix issue #309.

### 6.5 Live Phi-weighted voting

Kind currently mixes:

```text
BaselineCivicInfluence
+ AssessmentSignals
+ Reputation
+ Stake
+ Participation
+ DomainReputation
```

Current vote-weight machinery uses `PhiWeight` and direct Phi-dependent weighting when data is available.

The integrity layer also defines tier-specific Phi thresholds and a Phi-weight composite.

HAK recommendation:

- create an equal baseline civic vote mode;
- move cognitive/reputation/expertise signals to explicit experimental or deliberative roles;
- preserve raw unweighted counts;
- preserve strong process safeguards for high-impact decisions.

### 6.6 Governance ZKP consciousness attestation

Kind intended:

```text
PrivacyPreservingEligibilityEvidence
```

Current security issue:

```text
StructureValid != ProofVerified != TrustedVerifier
```

The caller-provided ZKP path currently validates proof-container structure and tier/freshness but does not establish a cryptographically trusted verification result before using the claimed tier in a gate.

Tracked independently in Mycelix issue #292.

This is a security defect independent of the constitutional legitimacy question in #309.

## 7. Authority-kind matrix

| Artifact / subsystem | Kind | Positive authority? | Negative restriction? | Epistemic only? | Human standing? | Domain-owned? |
|---|---|---:|---:|---:|---:|---:|
| Subterranean operator constraint | execution / restriction | yes | yes | no | no | yes |
| Rescue consent ledger | consent | yes | yes | no | affected-subject authority | yes |
| Emergency rescue authorization | emergency exception | yes | bounded | no | no | yes |
| Fabrication authority epoch | lineage evidence | indirect | rollback barrier | no | no | yes |
| Partition lease | resource lease | yes | fencing | no | no | yes |
| Scientific causal authority | epistemic | no | no | yes | no | yes |
| Quantum adapter authority | capability boundary | yes-ish | yes | no | no | yes |
| Mycelix SubPassport | delegation | yes | revocable | no | no | yes |
| Civic requirement | eligibility policy | yes | yes | no | potentially | yes |
| Consciousness profile | assessment + influence | currently | no | partly | currently affects influence | yes |
| Phi-weighted vote | civic influence | yes | no | signal inputs | yes | yes |
| Consciousness ZKP | evidence container | no by itself | no | evidence | potentially feeds gate | yes |

The matrix is deliberately descriptive, not a claim that current policy is legitimate or qualified.

## 8. Candidate machine-readable inventory schema

Do not implement this until the classification is reviewed. If useful, a future audit manifest could resemble:

```text
AuthorityInventoryEntryV1 {
    id
    repository
    module
    type_or_boundary

    authority_kind
    principal_kind
    subject_kind
    resource_kind

    lineage_key
    source_of_authority
    evidence_dependencies

    widening_transitions
    narrowing_transitions
    expiry_semantics
    revocation_semantics

    enforcement_point
    fallback_behavior
    translation_boundaries

    human_standing_impact
    machine_execution_impact

    qualification_status
    known_findings
}
```

This should remain an **audit manifest**, not the runtime source of authority.

## 9. What may eventually be common

The inventory suggests some semantics may be safely reusable across domains:

- lineage identifiers;
- typed translation outcomes;
- explicit distinction between evidence and authority;
- subject/resource/audience binding vocabulary;
- expiry + generation vocabulary;
- audit-only authority-kind classification;
- property-test helpers for monotonic transformations.

Even these should not be extracted until at least two domains demonstrate identical operational semantics.

## 10. What should not be unified

Do not unify merely by name:

```text
CausalAuthority
OperatorAuthority
EmergencyRescueAuthorization
PartitionLease
CivicStanding
ConsciousnessGate
```

They have materially different principals, resources, failure modes, legitimacy sources, temporal models, and enforcement semantics.

Likewise do not create:

```text
AuthorityScore: f64
```

or:

```text
TrustScore -> GenericPermission
```

as a HAK primitive.

## 11. New HAK review theorem: authority provenance must name the legitimacy source

Security provenance answers questions such as:

```text
who signed this?
what bytes were bound?
is the credential current?
```

HAK also requires a distinct question:

```text
why is this actor legitimately entitled to grant this authority?
```

Examples:

- membership/constitution for baseline civic standing;
- affected subject for consent;
- delegator's own current grant for delegation;
- qualified threshold policy for emergency exception;
- membership + consensus + ceremony for a resource lease;
- scientific method/evidence policy for epistemic classification.

Cryptographic validity does not answer legitimacy by itself.

```text
ValidSignature != LegitimateAuthoritySource
```

## 12. New HAK review theorem: advisory signals need an explicit non-authority boundary

A signal that is intended only to help deliberation should be typed so consumers cannot silently reinterpret it as permission or standing.

Candidate invariant:

```text
DeliberativeSignal -> no direct execution/civic authority transition
```

without a separately named policy step.

This is especially important for:

- Phi/consciousness metrics;
- reputation;
- model confidence;
- expertise scores;
- forecasts;
- sentiment;
- alignment/value assessments.

The intended architecture is:

```text
signal
  -> human/collective deliberation
  -> explicit policy/decision
  -> authority
```

not:

```text
signal -> authority
```

## 13. HAK-003 gates for future shared code

A common HAK runtime type should not be created until all are true:

1. at least two domains implement materially identical authority semantics;
2. principal/subject/resource meanings align;
3. lineage identity is definable in both;
4. widening/narrowing transitions align;
5. expiry/revocation ordering aligns;
6. fallback behavior aligns;
7. enforcement semantics align;
8. the abstraction does not collapse civic, epistemic, consent, or execution authority;
9. domain-specific stronger guarantees remain expressible;
10. tests prove the common abstraction cannot widen authority during translation.

## 14. Proposed next tranches

```text
HAK-003A  complete authority-kind inventory across Symthaea
HAK-003B  complete authority-kind inventory across Mycelix
HAK-003C  machine-readable audit manifest candidate
HAK-003D  cross-domain transformation graph
HAK-003E  property-test plan for same-lineage monotonicity
```

Keep these audit-oriented.

Runtime extraction should remain blocked until the inventory demonstrates a real common denominator.

## 15. Cross-repository issues identified during HAK-003

Mycelix #292:

```text
security(governance): do not authorize voting from structurally-valid but unverified consciousness ZKPs
```

Mycelix #309:

```text
arch(governance): separate baseline civic standing from consciousness-derived influence
```

These issues are independent:

```text
#292 = evidence verification/security
#309 = legitimacy/constitutional architecture
```

Passing #292 does not resolve #309.

Resolving #309 does not remove the need to secure any retained proof-bearing eligibility or scoped-role path.

## 16. Non-claims

This inventory does not claim:

- every current authority type is unsafe;
- every weighted voting mechanism is illegitimate;
- all authority must use one cryptographic format;
- all consent domains have identical semantics;
- all revocations are permanent;
- current Mycelix governance is deployed as public-state infrastructure;
- current consciousness metrics are scientifically established measures of human worth;
- a generic HAK runtime crate is ready.

The narrower conclusion is:

> The ecosystem already contains several mature but different authority mechanisms. The safest next step is to classify and compose them explicitly rather than unifying them by vocabulary.

## 17. Review question

Before HAK-003 can advance, reviewers should answer:

```text
Does this taxonomy preserve the important semantic distinctions,
and are any two listed authority kinds actually the same enough
that keeping them separate would create needless duplication?
```

Until that question has evidence-backed answers:

```text
reuse theorem/shape cautiously
reuse runtime type only after semantic equivalence is proven
```
