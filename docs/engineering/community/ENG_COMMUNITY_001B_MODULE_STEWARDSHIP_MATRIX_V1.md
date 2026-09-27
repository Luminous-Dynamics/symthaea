# ENG-COMMUNITY-001B — Module Rights, Obligations, Authority, and Expiry Reference v1

## Status

Reference/source subject for `ENG-COMMUNITY-001B` / issue #6161.

This document freezes a **synthetic, read-only engineering projection** of rights, obligations, external constraints, validity windows, delegation bounds, blockers, and currentness onto the exact synthetic campus source from `ENG-COMMUNITY-001A`.

It is not a legal instrument, licence, operating procedure, governance constitution, regulator decision, professional qualification, construction authorization, or actuation capability.

Upstream campus source:

`f3e87f5ff7ca823c0860de16b22eb4666189c2ee`

Campus generation:

`SYNTH_CAMPUS_G1`

Canonical JSON SHA-256:

`613fdd7a6e2962dbae6b0683ad7d08fd588a1cfdb843329e6e00266a457cd27a`

## Semantic boundary

Mycelix remains the owner of governance, constituency, capital, stewardship, voting, constitutional, and public-service semantics.

Symthaea projects exact/current upstream and external refs onto exact engineering modules.

Therefore:

```text
BenefitRight
!= GovernanceDecisionRight
!= TechnicalOperationRight

InspectionRight
!= ModificationAuthority

ContractualVetoRight
!= ActuationCapability

OwnershipInterest
!= EveryRight

OperatorQualification
!= EveryDecisionRight
```

No relation may be inferred solely from friendly labels such as `owner`, `operator`, `community`, `investor`, or `regulator`.

## `ModuleStewardshipMatrixV1`

Every claim-bearing row binds at least:

```text
module_subject
action_class
actor_ref
relation_kind
right_or_obligation_class
scope_ref
source_ref
source_owner
source_generation
currentness
valid_from
valid_until
prerequisite_refs[]
blocker_refs[]
delegability
revocability
claim_ceiling
```

## Relation kinds

The v1 profile distinguishes:

```text
Right
Obligation
ExternalConstraint
```

These are not interchangeable.

An external constraint that blocks an action does not establish asset ownership.

An obligation does not imply an economic interest.

A benefit right does not imply governance or technical authority.

## Currentness

The declared currentness vocabulary is:

```text
Current
Expired
Suspended
Revoked
Superseded
PendingPrerequisite
Blocked
Unknown
```

`Expired`, `Revoked`, and `Superseded` remain historical facts. They are not erased.

A renewed concession, licence, delegation, or constitutional authority creates a new generation; it does not silently extend the previous record.

`Unknown` fails closed for claim-relevant authority.

## Delegation

Delegation is valid only when explicitly bounded by the delegator's own authority, exact action/scope, validity interval, qualification prerequisites, external restrictions, and revocation path.

```text
delegation
cannot amplify
delegator authority
```

and:

```text
delegable governance authority
!= automatically delegable licensed technical authority
```

## Conflict

The projection may report multiple simultaneous rights, obligations, vetoes, qualification blockers, and external constraints.

Conflicting rows are preserved.

```text
ConflictPresent
```

is a valid result.

Symthaea does not invent a local winner unless an exact upstream or external precedence rule establishes one.

## Obligation continuity

Actor replacement cannot erase historical defects, evidence, findings, unresolved obligations, or handback requirements.

Examples:

```text
operator replaced
-> historical evidence preserved
-> unresolved outgoing obligations remain addressable
-> incoming duties begin only under exact accepted transfer/handback refs
```

and:

```text
ownership changes
!= regulator finding disappears
```

## Frozen synthetic rows

The v1 corpus contains six representative rows:

1. community/public-benefit governance over public-realm benefit allocation;
2. private-concession technical operation of the compute module;
3. thermal-network maintenance obligation;
4. energy-module technical-operation right deliberately blocked on a missing external licence;
5. an external energy-module constraint that creates no ownership claim;
6. a public-evidence inspection right that does not expose restricted security detail and does not create modification authority.

All identities and sources are synthetic placeholders.

## Public projection

A public-safe projection may show only publishable abstractions such as:

```text
module
actor class
relation class
scope class
currentness
validity class
public prerequisites/blockers
external-authority class
claim ceiling
```

Restricted details remain opaque `RestrictedPresent` dependencies under ENG-OPEN-SEC.

```text
RestrictedPresent
!= NotApplicable
!= Missing
```

## Corpus

The canonical JSON contains 24 hostile/reference cases covering:

- ownership does not imply operation;
- qualification does not imply asset-sale authority;
- inspection does not imply modification;
- proposal does not imply decision;
- governance does not imply actuation;
- contractual veto scope cannot expand into safety authority;
- expiry removes current authority while preserving history;
- renewal creates a new generation;
- delegation cannot exceed delegator scope;
- bounded delegation requires a validity window;
- licensed technical authority cannot be freely delegated without external support;
- external constraints can block action without changing ownership;
- operator replacement preserves historical evidence/defects;
- incomplete handback preserves transfer obligations;
- reserve blockers do not mutate title;
- conflicts remain conflicts rather than being silently resolved;
- public abstractions preserve `RestrictedPresent`;
- omission of a restricted dependency is structurally incomplete;
- stale source generation makes the relation stale;
- friendly-label changes do not change exact identity;
- exact identity changes create a new semantic subject even if labels stay constant;
- revocation preserves history but removes current authority;
- unknown applicability does not default to permission;
- complete synthetic closure cannot promote authority.

## Claim ceiling

A later qualification PASS may establish only deterministic composition of the exact frozen synthetic rights/obligations/external-constraint rows and the declared 24-case corpus.

It does not establish:

- legal enforceability;
- democratic legitimacy;
- professional qualification;
- licensing;
- engineering safety;
- regulatory approval;
- construction readiness;
- commissioning authorization;
- physical actuation authority.