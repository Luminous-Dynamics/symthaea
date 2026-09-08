# Human Agency Kernel — Authority Transformation Graph v1

Status: HAK-003 supporting architecture audit / documentation only

Parent: `HUMAN_AGENCY_KERNEL_AUTHORITY_LINEAGE_INVENTORY_V1.md`

## 1. Purpose

The HAK authority-kind inventory establishes that evidence, credentials, assessments, consent, civic standing, delegation, resource leases, epistemic strength, safety restrictions, and execution permission are different semantic objects.

This document makes the next rule explicit:

> A legitimate transformation between two authority kinds must be named, policy-owned, evidence-bearing, and lineage-aware.

The graph exists to detect category errors such as:

```text
model assessment -> civic authority
credential       -> execution
scientific confidence -> permission
emergency exception -> consent
cache eviction -> revocation removal
```

without claiming that every domain uses one universal authorization state machine.

## 2. Node vocabulary

Candidate semantic nodes:

```text
Observation
Evidence
EpistemicAssessment
Credential
DeliberativeSignal
MembershipEligibility
BaselineCivicStanding
ScopedRoleQualification
Consent
EmergencyExceptionAuthority
DelegatedAuthority
GovernanceDecision
ResourceLeaseAuthority
ExecutionAuthority
SafetyRestriction
ExecutedAction
OutcomeEvidence
```

These are conceptual categories, not a proposed Rust enum.

## 3. Graph principle

There are no "obvious" authority edges.

Every edge must answer:

```text
who owns the policy?
what source legitimizes the transition?
what subject/resource is bound?
what lineage is continued?
what evidence is required?
what can revoke/narrow it?
what is the enforcement point?
```

If those questions have no explicit answer, the edge is not qualified.

## 4. Observation -> Evidence

Candidate allowed edge:

```text
Observation
    -- validation/provenance -->
Evidence
```

Required properties may include:

- source identity;
- time;
- integrity;
- calibration;
- chain of custody;
- measurement policy;
- freshness.

This edge establishes evidence, not authority.

```text
Evidence != Permission
```

## 5. Evidence -> EpistemicAssessment

Candidate allowed edge:

```text
Evidence
    -- scientific/inference policy -->
EpistemicAssessment
```

Examples:

- causal;
- quasi-experimental;
- associational;
- calibrated confidence;
- verified/qualified experimental result.

The output remains epistemic.

Forbidden implicit edge:

```text
EpistemicAssessment -> ExecutionAuthority
```

A separate policy/decision is required.

## 6. Evidence -> Credential

Candidate allowed edge:

```text
Evidence
    -- issuer/credential policy -->
Credential
```

A credential states what was established under an issuer/domain policy.

It does not automatically answer what actions are permitted.

```text
Credential != Authorization
```

## 7. Credential -> ScopedRoleQualification

Candidate allowed edge:

```text
Credential
+ RolePolicy
+ CurrentContext
    -> ScopedRoleQualification
```

Example:

```text
licensed electrician credential
+ current site authorization policy
-> qualified electrical-maintenance role
```

This does not create baseline civic superiority.

Forbidden implicit edge:

```text
ScopedRoleQualification -> BaselineCivicStanding multiplier
```

unless a governance process explicitly and legitimately selected a role-weighted mechanism for a bounded decision lineage.

## 8. Identity/membership evidence -> MembershipEligibility

Candidate allowed edge:

```text
IdentityEvidence
+ MembershipPolicy
+ SybilPolicy
    -> MembershipEligibility
```

This answers whether an actor participates in a governance constituency/process.

It should not answer how worthy that actor is.

## 9. MembershipEligibility -> BaselineCivicStanding

Candidate HAK default:

```text
CurrentMembershipEligibility
+ Constitution
    -> BaselineCivicStanding
```

For equal-baseline governance:

```text
BaselineDirectVoteMass = 1.0
```

The legitimacy source is constitutional/membership semantics, not a cognitive model score.

Forbidden implicit edges:

```text
Phi                 -> BaselineCivicStanding
Reputation          -> BaselineCivicStanding
Engagement          -> BaselineCivicStanding
Wealth/Stake        -> BaselineCivicStanding
SocialApproval      -> BaselineCivicStanding
ModelConfidence     -> BaselineCivicStanding
```

These may become deliberative inputs or explicit experimental policy inputs, but not silent standing transformations.

## 10. Assessment -> DeliberativeSignal

Candidate allowed edge:

```text
Assessment
+ DisclosurePolicy
    -> DeliberativeSignal
```

Examples:

- domain expertise indicator;
- forecast calibration;
- Phi/consciousness research signal;
- affectedness estimate;
- argument-quality analysis;
- value-alignment assessment;
- minority-risk signal.

The key boundary is:

```text
DeliberativeSignal has no direct authority effect
```

unless a separately named governance policy intentionally consumes it.

## 11. DeliberativeSignal -> GovernanceDecision support

Candidate advisory edge:

```text
DeliberativeSignals
+ IndependentHumanJudgment
+ Evidence
    -> Deliberation
```

Then:

```text
Deliberation
+ GovernanceProcedure
+ Votes/consent/approval
    -> GovernanceDecision
```

Do not collapse this into:

```text
ModelSynthesis -> GovernanceDecision
```

without an explicit delegated decision policy.

## 12. BaselineCivicStanding -> DelegatedCivicAuthority

Candidate allowed edge:

```text
BaselineCivicStanding
+ ExplicitDelegation
+ Scope
+ Expiry
+ Revocation
    -> DelegatedCivicAuthority
```

Conservation rule under equal-baseline governance:

```text
sum(civic mass delegated from member M) <= civic mass owned by M
```

Delegate attributes do not create additional civic mass unless an explicit process policy says otherwise.

## 13. ScopedRoleQualification -> Operational role grant

Candidate allowed edge:

```text
ScopedRoleQualification
+ CurrentRoleAssignment
+ ResourceScope
+ Expiry
    -> Delegated/OperationalAuthority
```

This is a legitimate place for competence evidence to matter.

Example:

```text
qualified treasury signer
+ explicit committee appointment
-> bounded signing authority
```

It does not imply more baseline voting power.

## 14. Consent -> ExecutionAuthority

Candidate domain-owned edge:

```text
CurrentExplicitConsent
+ ActorAuthority
+ DomainPolicy
+ Resource/SubjectBinding
    -> ExecutionAuthority
```

Consent may be necessary but not sufficient.

Example:

```text
patient consent
```

may still require:

```text
qualified clinician + valid procedure authority
```

before execution.

Therefore:

```text
Consent != Capability
Consent != Competence
Consent != ExecutionByAnyone
```

## 15. EmergencyExceptionAuthority -> ExecutionAuthority

Candidate exceptional edge:

```text
EmergencyEvidence
+ EmergencyPolicy
+ RequiredIndependentApprovals
+ Subject/CaseBinding
+ Expiry
    -> EmergencyExceptionAuthority
    -> narrowly bounded ExecutionAuthority
```

Forbidden edge:

```text
EmergencyExceptionAuthority -> Consent
```

The exception authorizes a bounded emergency action; it does not rewrite the human's consent history.

## 16. GovernanceDecision -> ExecutionAuthority

Candidate allowed edge:

```text
QualifiedGovernanceDecision
+ ExecutionPolicy
+ CurrentSignatures/Capabilities
+ TimelockSatisfied
+ ResourceBinding
    -> ExecutionAuthority
```

A passed vote alone may not be sufficient for execution.

This separation enables:

- timelocks;
- threshold signing;
- execution review;
- rollback preparation;
- jurisdiction checks;
- resource-specific capability gates.

## 17. ResourceLeaseAuthority -> ExecutionAuthority

Candidate allowed edge:

```text
CurrentResourceLease
+ FencingToken
+ ResourceState
+ ExecutorIdentity
    -> ResourceScopedExecutionAuthority
```

A lease is not general authority.

```text
Lease(ResourceA) != Lease(ResourceB)
```

and:

```text
ExpiredLease -> NoExecutionAuthority
```

## 18. SafetyRestriction composition

Safety restrictions narrow an existing action set.

Conceptually:

```text
EffectiveAuthority
    = PositiveAuthority
      intersect SafetyConstraint1
      intersect SafetyConstraint2
      ...
```

Restrictions do not need to be modeled as negative numeric scores.

Critical rule:

```text
RestrictionRemoval == AuthorityWidening
```

within the same lineage.

Therefore a restriction cannot disappear from:

- cache eviction;
- representation fallback;
- expiry of the negative record when domain semantics require persistence;
- migration omission;
- same-lineage restart;
- default construction.

A new simulation lineage is a distinct case and must be identified as such.

## 19. ExecutionAuthority -> ExecutedAction

Candidate enforcement edge:

```text
ExecutionAuthority
+ PolicyEnforcementPoint
+ CurrentContext
    -> ExecutedAction
```

Important distinction:

```text
authorization decision != actuator action
```

The enforcement point must verify the authority artifact/current decision relevant to the concrete action.

## 20. ExecutedAction -> OutcomeEvidence

Candidate learning edge:

```text
ExecutedAction
+ Observation
+ Attribution
    -> OutcomeEvidence
```

Outcome evidence may update future models, policy, reputation, qualification, or scientific understanding.

It should not retroactively change whether the original action was authorized.

```text
GoodOutcome != RetroactiveAuthorization
BadOutcome  != ProofOfPriorUnauthorizedness
```

Authorization and outcome evaluation are distinct dimensions.

## 21. OutcomeEvidence -> future policy

Candidate learning edge:

```text
OutcomeEvidence
+ Review
+ Governance/ScientificProcess
    -> PolicyUpdate
```

This creates institutional learning without turning outcome metrics into automatic authority.

A policy update starts a new policy version/generation where appropriate.

## 22. Explicit experimental weighting edge

HAK does not prohibit weighted governance experiments.

It requires an explicit edge:

```text
Deliberative/AssessmentSignals
+ ExplicitWeightingPolicy
+ GovernanceLineage
    -> ProcessSpecificVoteWeight
```

Required safeguards:

- policy chosen before voting begins;
- policy identity/version recorded;
- inputs/provenance specified;
- raw equal-person tally retained;
- sunset/review semantics;
- no automatic transfer to baseline civic standing;
- no claim of universal human worth.

Thus:

```text
Phi -> VoteWeight
```

is not categorically forbidden.

The forbidden form is:

```text
Phi -> implicit universal civic authority
```

## 23. Prohibited implicit edges

HAK lint candidates:

### HAK-EDGE-001

```text
Assessment -> BaselineCivicStanding
```

without explicit constitutional/process policy.

### HAK-EDGE-002

```text
Credential -> ExecutionAuthority
```

without policy decision/delegation.

### HAK-EDGE-003

```text
EpistemicAuthority -> ExecutionAuthority
```

without decision policy.

### HAK-EDGE-004

```text
DeliberativeSignal -> ExecutionAuthority
```

without explicit delegation/policy.

### HAK-EDGE-005

```text
EmergencyException -> Consent
```

### HAK-EDGE-006

```text
Expired/evicted negative barrier -> prior grant resurrected
```

where domain semantics require negative continuity.

### HAK-EDGE-007

```text
CompatibilityFallback -> broader authority
```

### HAK-EDGE-008

```text
Unknown/Unavailable -> Healthy/Authorized
```

without evidence.

### HAK-EDGE-009

```text
NewSimulationLineage -> OperationalRecoveryLineage
```

without explicit lineage continuity.

### HAK-EDGE-010

```text
SignaturePresent -> SignatureValid -> SignerTrusted
```

collapsed into one boolean.

### HAK-EDGE-011

```text
ProofContainerValid -> ProofVerified -> PolicyTrustedClaim
```

collapsed into one boolean.

### HAK-EDGE-012

```text
GoodOutcome -> RetroactiveAuthorization
```

## 24. Required explicit bridge objects

Where a category transition is legitimate, prefer an explicit typed artifact owned by the policy domain.

Examples:

```text
EligibilityDecision
RoleQualification
ConsentReceipt
DelegationGrant
GovernanceDecision
AuthorizedLease
ExecutionCapability
EmergencyAuthorization
```

The exact names remain domain-owned.

The important property is that a semantic transition cannot occur invisibly inside an accessor such as:

```text
score() -> f64
```

## 25. Candidate transformation record

Audit-only design candidate:

```text
AuthorityTransformationRecordV1 {
    transformation_id
    lineage_id

    source_kind
    source_artifact_id
    destination_kind
    destination_artifact_id

    policy_id
    policy_version
    legitimacy_source

    principal
    subject_or_resource
    audience
    purpose

    disposition
    evidence_dependencies
    occurred_at
}
```

This is not proposed as a universal runtime authorization token.

It may become useful as audit/provenance metadata if multiple domains prove compatible semantics.

## 26. Translation dispositions

Candidate shared review vocabulary:

```text
Exact
Attenuated
PolicyQualified
AdvisoryOnly
Incomparable
NotRepresentable
Expired
Revoked
EvidenceStale
LineageMismatch
ContextMismatch
AudienceMismatch
ResourceMismatch
UntrustedVerifier
UnverifiedProof
```

Only domain-authorized dispositions may continue toward execution/standing.

## 27. Monotonicity property

For same-lineage authority transformations that are intended merely to re-express or attenuate authority:

```text
Authority(T(A)) subset-of Authority(A)
```

This theorem does **not** govern a legitimate new grant from an independent authority source.

Example:

```text
member receives new role appointment
```

may legitimately widen authority because the role assignment is itself a new authority source.

So every widening event should name the new legitimacy source rather than pretending it came from translation.

## 28. Legitimacy provenance theorem

Every authority-widening edge must identify a legitimacy source.

Examples:

```text
BaselineCivicStanding <- constitution + current membership
Consent               <- affected subject
Delegation            <- current principal grant + explicit delegation
EmergencyException    <- emergency policy + required approvals
ResourceLease         <- membership + consensus + threshold ceremony
ScopedRoleAuthority   <- qualification + appointment policy
ExecutionAuthority    <- current decision/grant + enforcement policy
```

If the implementation can name only:

```text
model score
credential hash
valid signature
```

but cannot name why that source is entitled to grant the destination authority kind, the transition is incomplete.

## 29. Cross-repository examples

### Mycelix consciousness gate

Current shape:

```text
ConsciousnessAssessment/Credential
    -> GovernanceEligibility
```

HAK-003 marks this as a legitimacy-review edge rather than assuming cryptographic verification resolves the question.

Security verification is separately tracked in Mycelix #292.

Baseline-standing redesign is tracked in Mycelix #309 / its RFC.

### Mycelix 8D -> legacy civic policy

Current repaired target:

```text
GovernanceEligibilityPolicy
    -> checked compatibility translation
    -> exact/conservative representation OR NotRepresentable
```

No silent constraint loss.

### Symthaea causal authority

Valid edge:

```text
Evidence -> CausalAuthority
```

Invalid implicit edge:

```text
CausalAuthority -> ActuatorPermission
```

### Fabrication partition lease

Valid edge:

```text
Verified membership/consensus/ceremony
    -> AuthorizedPartitionLease
    -> accepted fenced lease
    -> resource-scoped execution
```

This is strong prior art for explicit legitimacy provenance.

## 30. Candidate linter output

A future advisory linter could report:

```text
HAK-EDGE-001 high
Model-derived assessment reaches baseline civic authority without an explicit governance weighting policy.

HAK-EDGE-011 critical
Proof container structure is validated but proof verification/trusted verifier policy is not established before governance eligibility.

HAK-EDGE-007 high
Compatibility translation drops a source constraint before authorization.

HAK-EDGE-003 medium
Scientific causal classification is consumed by an execution boundary without a separately recorded decision policy.
```

Linter output remains advisory:

```text
LinterFinding != MoralVerdict != LegalConclusion != ExecutionDecision
```

## 31. Property-test candidates

### HAK-TG-P1 — re-expression non-expansion

For same-lineage representation transformations:

```text
permitted_after subset-of permitted_before
```

### HAK-TG-P2 — new grant names source

Any widening transition records a distinct legitimacy source.

### HAK-TG-P3 — advisory non-authority

Changing only a deliberative signal cannot change execution/civic authority unless an explicit policy consumes that signal.

### HAK-TG-P4 — evidence non-authority

Improving evidence strength alone cannot create an action grant.

### HAK-TG-P5 — credential non-authority

Possessing a valid credential alone cannot execute an action without policy authorization.

### HAK-TG-P6 — negative continuity

A same-lineage migration/cache/restart cannot remove a durable negative barrier merely because the representation changes.

### HAK-TG-P7 — lineage mismatch

Artifacts from unrelated authority lineages cannot be ordered/translated as though one were a continuation of the other.

### HAK-TG-P8 — verifier separation

```text
container_valid
signature_present
signature_valid
verifier_trusted
claim_policy_satisfied
```

remain independently testable states.

## 32. Non-claims

This graph does not claim:

- every listed node needs a shared Rust type;
- every edge is required in every domain;
- every authority system must be capability-based;
- all negative barriers persist forever;
- all governance must use equal voting;
- all weighted governance is a category error;
- scientific evidence can never justify action;
- emergency action is never legitimate without contemporaneous subject consent.

The narrow rule is:

> Meaningful authority transitions should be explicit enough that reviewers can identify the legitimacy source and prove that representation changes do not silently create authority.

## 33. Review gate

Before turning this graph into automated tooling, reviewers should answer:

1. Are the node categories distinct enough to avoid category collapse?
2. Which edges are genuinely common across two or more domains?
3. Which prohibited-edge rules need domain-specific exceptions?
4. What is the minimum representation of `legitimacy_source` that is useful without centralizing authority?
5. Which properties belong in static linting vs runtime enforcement vs experimental qualification?

Until then:

```text
transformation graph = architecture/audit model
not runtime oracle
```
