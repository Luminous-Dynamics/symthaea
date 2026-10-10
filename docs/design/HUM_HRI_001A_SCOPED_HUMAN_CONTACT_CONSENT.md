# HUM-HRI-001A — Scoped Human-Contact Consent Contract

Status: source-design candidate
Issue: #4680
Authority: semantic consent scope only; **no motor authority**

## Purpose

Define a typed, fail-closed consent object for intentional human contact. The object is deliberately separate from preference, affect inference, operator authorization, physical qualification, motion intent, motor authority, and safety enforcement.

## Core theorem

```text
Preference / affect / physiology / operator intent
                cannot create consent

valid scoped consent
                cannot create motor authority

motor authority
                cannot broaden consent
```

All three boundaries are independent restrictions. Final physical eligibility requires the existing qualification, live-authority, whole-body, terminal-safety, and HAL chain in addition to valid consent.

## Semantic scope

A future `HumanContactConsentScopeV1` must bind at least:

- non-empty participant identity;
- non-empty session identity;
- monotonic consent epoch;
- explicit contact class;
- non-empty permitted human-region set;
- non-empty permitted robot-contact-site set;
- finite validity window with `valid_until > valid_from`;
- explicit revocation lineage/epoch.

No field may be inferred from preference, physiology, historical behavior, or operator approval.

## Contact classes

The semantic layer should use coarse, safety-relevant classes rather than technique catalogs. Initial vocabulary candidates:

- ordinary social contact;
- assistive/caregiving contact;
- therapeutic/manual contact;
- adult intimate contact.

A broader or more sensitive class is never inferred from a narrower class. Escalation requires fresh explicit authority.

## Human regions

The consent vocabulary must be explicit and non-graphic. Unspecified/unknown regions are not permitted. Adjacency never implies permission.

The exact region taxonomy is intentionally deferred to #4706 so this tranche does not accidentally freeze an immature ontology.

## Robot contact sites

Reachability is not eligibility. A robot surface must be explicitly allowed by the consent scope and separately qualified for human contact. Site eligibility/material/tactile/hygiene concerns remain separate evidence propositions (#4707, #4708).

## Narrowing algebra

Consent composition is intersection-only.

For any scopes `A` and `B`, a valid narrowed scope `C = A ∩ B` may only contain:

- the same participant/session;
- an epoch no older than the applicable live epoch;
- contact classes permitted by both inputs;
- human regions present in both inputs;
- robot sites present in both inputs;
- a validity interval contained by both inputs.

An empty intersection means **no contact authority**. There is no union/merge helper that broadens authority.

## Revocation

Revocation dominates all ordinary contact permission.

```text
Active(scope epoch N)
        |
        | revoke / stop / expiry / required-evidence loss
        v
Revoked(epoch N)
        |
        | disengagement/recovery
        v
Idle
        |
        | fresh explicit consent only
        v
Active(scope epoch > N)
```

There is no paused-consent state that can silently auto-resume. Reboot, relevant calibration/profile change, or authority-session reset requires fresh re-arming as tracked in #4699.

## Asymmetric composition

Granting contact is conjunctive: every required evidence source must be valid.

Stopping is disjunctive: any valid withdrawal, e-stop, permit expiry, safety fault, evidence-loss event, or authority invalidation may remove authority.

Conceptually:

```text
grant = AND(required evidence)
stop  = OR(valid stop channels)
```

## Preference separation

Preference evidence is non-authoritative. It may:

- rank already-permitted alternatives;
- reduce proposal confidence;
- suppress a proposal;
- trigger a clarifying question.

It may not:

- add a region or site;
- escalate a contact class;
- extend a validity interval;
- mint/refresh a consent epoch;
- restore revoked consent;
- increase motor authority.

This theorem is tracked for executable proof in #4684.

## Authenticity boundary

A deterministic scope identity is evidence identity only. It is not proof that the intended participant authentically granted consent. Authentication/verifier binding is a separate proposition (#4714).

## Privacy boundary

Durable audit should retain consent/policy/safety identities and bounded outcomes, not reconstructable intimate content by default. Raw sensitive observations remain governed by #4689 and #4697.

## Required adversarial tests before integration

- empty participant/session/region/site rejection;
- invalid or empty validity window rejection;
- stale epoch rejection;
- replay rejection after revocation;
- participant/session substitution rejection;
- human-region substitution rejection;
- robot-site substitution rejection;
- contact-class escalation rejection;
- expiry rejection;
- narrowing intersection cannot widen any dimension;
- empty intersection yields no authority;
- preference evidence cannot construct/refresh consent;
- operator approval cannot construct/refresh consent.

## Integration sequence

1. HUM-HRI-001A — typed scoped consent semantics (#4680).
2. HUM-HRI-001B — latching revocation/fresh epochs (#4681).
3. HUM-HRI-001C — bind `HumanContact` whole-body intent to exact scope (#4682).
4. HUM-HRI-001D — executable Preference != Consent theorem (#4684).
5. Tactile/contact-safety work begins only after those semantic boundaries are qualified (#4685, #4693, #4686, #4687).

## Nonclaims

This document does **not** establish:

- authentic consent evidence;
- age/eligibility assurance;
- human-contact qualification;
- tactile sensor accuracy;
- safe force/pressure/temperature limits;
- actuator authority;
- HIL/hardware qualification;
- human-trial authorization;
- legal or product-safety certification.

It freezes the intended semantic/authority boundary so later implementation can be reviewed against an explicit proposition.