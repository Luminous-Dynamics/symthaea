# Human Agency Kernel — Authority Conservation Algebra v1

Status: architecture candidate / documentation only

HAK series:

- HAK-001 — human agency boundary
- HAK-002 — authority augmentation / lineage-aware monotonicity
- HAK-003 — authority kind, lineage, transformation, and conservation inventory

Cross-repository motivating findings include Mycelix #330, #331, and #332.

## 1. Purpose

The authority transformation graph establishes where authority may flow.

A second question is required:

```text
When authority flows, what quantity or invariant must be conserved?
```

There is no single universal numeric answer.

Different authority kinds obey different conservation algebras.

Examples:

```text
permission set
    -> conserve by subset / no unauthorized widening

consumable voice-credit budget
    -> conserve by balance / monotonic spend

exclusive resource lease
    -> conserve by non-overlap / single effective holder

consent
    -> not transferable at all

safety restriction
    -> conserve negatively until legitimate revocation/expiry

delegated civic mass
    -> conserve by scope-aware source budget
```

Attempting to collapse these into one scalar `authority_score` would destroy exactly the semantics HAK is meant to protect.

This document defines a candidate **authority conservation algebra family** rather than one generic conservation equation.

## 2. Meta-theorem

For an authority transformation `T` operating within one lineage and legitimacy source:

```text
ConservationKind(T(A))
    must not represent more legitimate authority
    than ConservationKind(A)
```

A legitimate widening requires a separately named authority source:

```text
Widen(A -> B)
    requires
NewLegitimacySource
+ ExplicitPolicy
+ ExactSubject/Scope
+ NewOrExplicitlyExtendedLineage
```

Therefore:

```text
Transformation != Mint
Representation != Mint
Retry != Mint
Fork != Mint
Fallback != Mint
CacheMiss != Mint
Migration != Mint
Delegation != Mint
```

unless the governing policy explicitly defines a legitimate issuance event.

## 3. Conservation algebra families

HAK should classify each authority kind by the algebra that governs safe transformation.

Candidate families:

```text
SetMonotonic
BudgetConserving
ConsumableMonotonic
ExclusiveLease
NonTransferable
RestrictionMonotonic
ThresholdQuorum
EvidenceBound
```

An authority type may combine more than one family.

For example, an execution capability can be both:

```text
SetMonotonic + ConsumableMonotonic + EvidenceBound
```

if it authorizes a bounded action set and is one-shot.

## 4. SetMonotonic authority

Examples:

- execution permissions;
- scoped role permissions;
- delegated operation permissions;
- API/tool capabilities;
- resource access scopes.

For same-lineage attenuation/re-expression:

```text
PermissionsOut subset-of PermissionsIn
```

A transformation may remove permissions freely if policy allows.

It may not add permissions merely because representation changed.

### 4.1 Examples

Valid:

```text
read+write+execute
-> read+write
-> read
```

Invalid without new legitimacy source:

```text
read
-> read+write
```

### 4.2 Candidate HAK rule

```text
HAK-CONS-SET-001
same-lineage permission transformation must be non-widening
```

## 5. BudgetConserving authority

Examples:

- delegated civic mass;
- bounded resource quotas;
- allocation shares;
- voting mass;
- fractional authority routing.

For source budget `B` and overlapping effective children `c_i`:

```text
Retained(B) + Sum(c_i) <= B
```

Scope matters.

Two child grants that can never apply to the same subject may each consume the full source budget without conflict.

Therefore the real rule is:

```text
For every effective scope intersection S:
Retained(B,S) + Sum(Children(B,S)) <= B(S)
```

### 5.1 Branching

Fan-out must not duplicate source budget.

```text
Sum(BranchBudgets) <= ParentAvailableBudget
```

### 5.2 Transitivity

For a transitive edge:

```text
ChildBudget <= ParentRoutedBudget
```

### 5.3 Decay

```text
BudgetAfterDecay <= BudgetBeforeDecay
```

### 5.4 Candidate HAK rules

```text
HAK-CONS-BUDGET-001
branch fan-out cannot exceed source budget

HAK-CONS-BUDGET-002
transitive routing cannot amplify parent budget

HAK-CONS-BUDGET-003
decay cannot increase authority budget
```

## 6. ConsumableMonotonic authority

Examples:

- voice credits;
- one-shot authorization tokens;
- execution nonce/capability use;
- rate/quota budgets;
- limited emergency actions;
- consumable resource claims.

For a grant:

```text
Allocated = Spent + Remaining
```

Across valid updates:

```text
Spent(t+1) >= Spent(t)
Remaining(t+1) <= Remaining(t)
Allocated(t+1) = Allocated(t)
```

unless a new issuance lineage is created.

### 6.1 Forks

Forks must not duplicate consumable value.

For concurrent children of source state `s`:

```text
Sum(CanonicalConsumption(children(s))) <= Remaining(s)
```

A distributed fork is a consistency event, not a mint.

### 6.2 Retry/idempotency

Retrying a one-shot action must not consume or exercise authority twice.

```text
SameExerciseId -> at-most-once authority effect
```

### 6.3 Candidate HAK rules

```text
HAK-CONS-CONSUME-001
spent is monotonic non-decreasing

HAK-CONS-CONSUME-002
remaining is monotonic non-increasing

HAK-CONS-CONSUME-003
fork/retry cannot duplicate consumable authority
```

## 7. ExclusiveLease authority

Examples:

- partition leases;
- exclusive fabrication ownership epochs;
- single-writer resource authority;
- device ownership/administrative control;
- exclusive leadership/coordination slots where policy requires uniqueness.

For an exclusive resource scope `R` and epoch/generation `G`:

```text
EffectiveExclusiveHolders(R,G) <= 1
```

unless the policy explicitly changes the authority kind from exclusive to shared.

### 7.1 Handoff

A safe handoff should ensure:

```text
OldLeaseRevokedOrExpired
before/with
NewLeaseEffective
```

according to the domain's atomicity model.

### 7.2 Partition

A parent exclusive lease may be divided into disjoint subresources:

```text
R = R1 union R2
R1 intersect R2 = empty
```

and then separately leased without violating exclusivity.

### 7.3 Candidate HAK rules

```text
HAK-CONS-LEASE-001
exclusive scope has at most one effective holder per epoch

HAK-CONS-LEASE-002
partitioned child scopes must not overlap unless policy permits shared authority
```

## 8. NonTransferable authority

Examples:

- personal consent;
- certain human rights/standing;
- subject-specific bodily authorization;
- attestations whose authority derives from being issued by one exact subject.

These are not budgets that may be delegated.

Core theorem:

```text
AuthoritySource = Subject
```

Therefore:

```text
Transfer(ConsentOfAlice -> Bob)
```

is not attenuation or delegation; it is a category error unless Alice separately authorizes Bob to act in a clearly distinct delegated role.

The consent itself remains Alice's.

### 8.1 Emergency exception

An emergency exception may override a normal consent requirement only if a separate legitimate policy source explicitly grants that exception.

It must not rewrite history as though consent existed.

```text
EmergencyException != Consent
```

### 8.2 Candidate HAK rule

```text
HAK-CONS-NONTRANSFER-001
non-transferable authority cannot become delegated authority by representation change
```

## 9. RestrictionMonotonic authority

Restrictions are negative authority.

Examples:

- revocation;
- veto;
- cooling period;
- safety block;
- expired credential;
- revoked delegation;
- quarantine;
- deny-list/resource restriction;
- consumed budget.

If a restriction removes allowed action set `R`:

```text
AllowedAfterRestriction
= AllowedBefore - R
```

A subsequent transformation must not silently re-add `R`.

The restriction remains effective until its legitimate expiry/revocation semantics are proven.

### 9.1 Missing data

```text
RestrictionStateUnknown
!= RestrictionAbsent
```

### 9.2 Cache eviction

```text
EvictedRestrictionCache
!= RevokedRestriction
```

### 9.3 Compatibility

```text
LegacyRepresentationMissingRestrictionField
!= PermissionToIgnoreRestriction
```

### 9.4 Candidate HAK rules

```text
HAK-CONS-RESTRICT-001
representation/migration/cache failure cannot resurrect restricted authority

HAK-CONS-RESTRICT-002
restriction removal requires exact legitimate expiry/revocation evidence
```

## 10. ThresholdQuorum authority

Examples:

- threshold signatures;
- multi-party approvals;
- committee grants;
- governance quorum;
- multi-sensor qualification where policy explicitly requires N-of-M evidence.

This authority is not conserved as scalar mass alone.

It depends on a cardinality/identity predicate.

Conceptually:

```text
Qualified = Predicate(UniqueQualifiedPrincipals, Policy)
```

### 10.1 Duplicate identities

Repeated evidence from one principal must not count as multiple principals unless the policy explicitly defines weighted multiplicity.

```text
SamePrincipalRepeated != AdditionalSigner
```

### 10.2 Scope

A threshold satisfied for subject A does not automatically satisfy subject B.

```text
ThresholdApproval(A) != ThresholdApproval(B)
```

### 10.3 Candidate HAK rules

```text
HAK-CONS-THRESHOLD-001
threshold multiplicity counts unique policy-qualified principals/claims

HAK-CONS-THRESHOLD-002
threshold satisfaction is bound to exact subject and policy lineage
```

## 11. EvidenceBound authority

Examples:

- scientific classification authority;
- verified eligibility;
- governance tally authority;
- signed approval;
- execution receipt;
- model-derived advisory claims when elevated by policy.

Evidence-bound authority does not obey a simple scalar conservation law.

Its safety property is **no unexplained epistemic/authority amplification**.

A derived artifact may summarize or combine evidence, but any increase in claim strength must be justified by an explicit inference/qualification rule.

```text
ClaimStrengthOut > ClaimStrengthIn
    requires
AdditionalEvidence or ExplicitValidInference
```

Examples of prohibited collapse:

```text
SignaturePresent -> SignatureValid
ProofContainerValid -> ProofVerified
ModelOutput -> TrustedPolicyClaim
Correlation -> Causation
StateLabel -> TransitionEvidence
```

### 11.1 Candidate HAK rule

```text
HAK-CONS-EVIDENCE-001
authority/evidence strength cannot increase without explicit new evidence or inference policy
```

## 12. Mixed algebras

Real authority artifacts often combine multiple conservation kinds.

Example: `AuthorizedExecution` may be:

```text
SetMonotonic
+ EvidenceBound
+ ConsumableMonotonic
```

because:

- its action/resource scope cannot widen;
- it must trace to exact authorization evidence;
- it may be one-shot.

Example: a device lease may be:

```text
ExclusiveLease
+ SetMonotonic
+ RestrictionMonotonic
```

Example: delegated governance authority may be:

```text
BudgetConserving
+ EvidenceBound
+ RestrictionMonotonic
```

HAK should therefore attach a set of conservation obligations to each authority kind rather than force one universal representation.

## 13. Conservation profile candidate

A future machine-readable architecture description might express:

```text
AuthorityConservationProfile {
    authority_kind
    lineage_kind
    conservation_families[]
    scope_identity
    source_budget_or_permission_subject
    widening_policy
    narrowing_policy
    fork_policy
    retry_policy
    expiry_policy
    revocation_policy
    enforcement_points[]
}
```

This is an architecture candidate only.

Do not create this shared runtime type until at least two domains demonstrate materially identical semantics.

## 14. Cross-repository finding: Mycelix governance

The Mycelix governance audit gives concrete examples of why this algebra is needed.

### 14.1 Voice credits — ConsumableMonotonic

Current finding #330 shows that quadratic voice-credit issuance/spend needs:

```text
QualifiedGrant
-> monotonic spend
-> exact spend receipt
-> ballot
```

not an open/public mint followed by structurally valid balance arithmetic.

### 14.2 Delegation — BudgetConserving

Current finding #331 shows that per-edge bounds are insufficient.

```text
60% to A + 60% to B
```

can violate a unit source budget when scopes overlap.

### 14.3 Delegated ballot admission — Threshold/uniqueness + EvidenceBound

Current finding #332 shows:

```text
AuthenticatedDelegate != OneQualifiedBallot
```

Ballot exercise must bind exact proposal/voting lineage and source delegation authority.

## 15. Human-agency implication

Conservation laws are not merely accounting constraints.

They protect agency.

Without conservation:

- one person's delegated voice can be multiplied;
- one consent can be reinterpreted into broader permission;
- one emergency exception can become permanent authority;
- one credential can become an execution grant;
- one resource lease can become overlapping ownership;
- one consumed/revoked capability can reappear through fallback;
- one advisory model output can become coercive authority.

The deeper HAK theorem is therefore:

```text
HumanAgencySafety
requires
AuthorityProvenance
+ AuthorityTransformationValidity
+ AuthorityConservation
```

## 16. Candidate conservation lint families

Future static/review tooling can flag patterns such as:

```text
HAK-CONS-LINT-001
multiple overlapping child grants with no source-budget accounting

HAK-CONS-LINT-002
consumable value update without original-state comparison

HAK-CONS-LINT-003
retry/fork path lacks idempotency or unique exercise identity

HAK-CONS-LINT-004
exclusive lease creation lacks overlap/epoch check

HAK-CONS-LINT-005
non-transferable authority appears in a delegation transform

HAK-CONS-LINT-006
restriction missing/unknown interpreted as unrestricted

HAK-CONS-LINT-007
threshold counts repeated claims without principal uniqueness

HAK-CONS-LINT-008
derived authority stronger than source evidence without explicit inference/policy step

HAK-CONS-LINT-009
compatibility/availability fallback yields larger authority set

HAK-CONS-LINT-010
same source authority is exercised through multiple endpoints without canonical exercise key
```

These remain lint candidates, not qualified automated verdicts.

## 17. Property-test families

Where runtime semantics become sufficiently explicit, domain property tests should include:

### Set monotonicity

```text
AuthorityOut subset-of AuthorityIn
```

for same-lineage attenuation transforms.

### Budget conservation

```text
Sum(EffectiveChildren(scope)) + Retained(scope) <= Source(scope)
```

### Consumable monotonicity

```text
Spent' >= Spent
Remaining' <= Remaining
Allocated' = Allocated
```

### Exclusive lease uniqueness

```text
EffectiveExclusiveHolders(resource, epoch) <= 1
```

### Restriction persistence

```text
RestrictionActive && !QualifiedRevocation
-> restriction remains effective
```

### Threshold uniqueness

```text
QualifiedPrincipalCount = count(unique qualified principals)
```

### Evidence-bound strength

```text
NoNewEvidence && NoExplicitInference
-> no stronger authority claim
```

## 18. Shared-code gate

This document strengthens the existing HAK-003 shared-code gate.

Do **not** create a universal authority-conservation runtime crate merely because the word `conservation` now appears across domains.

Before sharing an implementation primitive, require at least two real domains to match on:

```text
conservation algebra family
scope identity
source authority representation
fork semantics
retry semantics
expiry/revocation semantics
widening/narrowing semantics
enforcement point
failure behavior
```

Shared theory is useful before shared code.

## 19. End-state

For every authority-bearing transformation, HAK should eventually make three questions explicit:

```text
1. Provenance:
   Where did this authority legitimately come from?

2. Transformation:
   Is this edge semantically allowed?

3. Conservation:
   Did this edge preserve the authority invariant for its kind?
```

Together:

```text
LegitimateSource
+ AllowedTransformation
+ CorrectConservationAlgebra
-> QualifiedAuthorityTransition
```

This is a stronger foundation than a universal authority score because it preserves the semantic differences among consent, civic standing, delegation, evidence, leases, restrictions, and execution authority.