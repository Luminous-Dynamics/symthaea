# Human Agency Kernel — Authority & Augmentation Contract v1

Status: architecture candidate / non-normative until independently reviewed and qualified

Parent: `HUMAN_AGENCY_KERNEL_AUDIT_V1.md` (HAK-001)

Supporting audit: `HUMAN_AGENCY_KERNEL_RESET_CONTINUITY_AUDIT_V1.md`

## 1. Purpose

HAK-001 separated assistance, scoring, consent, delegation, execution, outcome, and human benefit. HAK-002 makes one part of that separation more precise: how authority can be represented, translated, delegated, attenuated, revoked, recovered, and composed without silently becoming broader than the authority actually established.

Two design candidates are introduced:

- `AuthorityEnvelopeV1`: an explicit description of bounded authority and its lineage.
- `AugmentationContractV1`: an explicit description of how an AI system helps a human while preserving retained decisions, verification, dissent, interruption, and reliance calibration.

These are architecture shapes, not implementation APIs. HAK-002 does **not** create a shared Rust crate and transfers no qualification from existing domain code.

## 2. Constitutional distinctions

```text
ModelAssessmentOfPerson != AuthorityOverPerson
IdentityEvidence          != Authorization
Credential                != Authorization
Recommendation            != Consent
Consent                   != Delegation
Delegation                != Execution
Execution                 != HumanBenefit
```

A fact, score, credential, explanation, or model confidence may be relevant to a policy decision. It does not become execution authority merely by being verified.

For consequential execution, permission is established positively:

```text
ExecutionPermission = CurrentExplicitGrant
```

not negatively:

```text
ExecutionPermission = !KnownDenial
```

`Unknown`, `Unavailable`, `Refused`, `Revoked`, and `Expired` remain semantically distinct even where all fail closed for a particular action.

## 3. Authority monotonicity — with a lineage precondition

For a transformation `T` of an authority-bearing artifact `A` **within the same authority lineage**:

```text
Authority(T(A)) ⊆ Authority(A)
```

This is the central HAK-002 security theorem.

Examples include:

- new wire representation of the same grant;
- compatibility fallback for the same policy;
- child delegation from a parent grant;
- checkpoint/recovery of the same operational lineage;
- cache reconstruction of the same authorization;
- policy compilation of the same human commitment.

If exact preservation or a domain-proven conservative attenuation cannot be established:

```text
NotRepresentable -> NoAuthority
```

not:

```text
NotRepresentable -> BestEffortAuthorization
```

### 3.1 Lineage identity is load-bearing

The monotonicity theorem does **not** apply blindly across unrelated worlds or new scenarios.

A clean deterministic simulation reset may intentionally terminate simulation lineage A and construct simulation lineage B with configured nominal fixtures.

Therefore:

```text
same-lineage recovery -> monotonicity required
new simulation lineage -> old/new authority sets need not be comparable
```

Before applying an authority attenuation proof, HAK review must identify:

```text
principal
resource/subject
context
lifecycle operation
authority lineage
```

This correction emerged from the reset-continuity audit: the first implementation pass treated scenario `reset()` as if it were operational restart. Broader call-site evidence showed that interpretation was not established, and the resulting preservation PRs were closed unmerged.

## 4. Restriction removal is an authority transition — within a continuing lineage

Within the same operational lineage, removing a stop, hold, maintenance lock, refusal, revocation barrier, recovery latch, or review hold widens the action set.

```text
RestrictionRemoval == AuthorityWidening
```

That transition therefore needs explicit domain semantics.

For an operational lineage:

```text
restart     != ResumeNominal
cache miss  != ResumeNominal
expiry      != RevocationUndo
migration   != RevocationUndo
```

But a declared new simulation/scenario lineage is different: its configured initial authority is part of the new world, not a recovery claim about the old one.

## 5. Positive grants and negative barriers

Positive grants should normally be bounded by:

- action scope;
- resource/subject scope;
- context;
- audience/executor;
- time;
- evidence freshness;
- integrity binding.

Negative barriers such as refusal or revocation must not disappear merely because a record expires, is evicted, a cache misses, or an earlier positive grant remains in history when the domain defines the barrier as durable for that lineage.

Candidate generic ordering concept:

```text
Grant {
    generation = g
    valid_until = t
}

Revocation {
    revocation_generation >= g
}
```

A new explicit grant may supersede a revocation only according to domain-owned semantics and unambiguous ordering.

HAK does not assume every kind of refusal or revocation is permanent. It requires the domain to state how it can be superseded rather than letting storage behavior make that decision implicitly.

## 6. Authority is a partial order, not a scalar

HAK should not represent authority as one generalized score such as:

```text
trust = 0.82
```

A useful conceptual relation is:

```text
A <= B
```

when every action permitted by `A` is also permitted by `B` under the same principal, resources, context, audience, and lineage.

That relation may be undefined for arbitrary cross-domain grants.

If the domain cannot prove comparability, it should return a typed `Incomparable` / `NotRepresentable` result rather than fabricate an ordering.

This also reinforces HAK-001’s human-standing theorem: expertise, reputation, contribution, confidence, or civic measurements are not one universal currency of authority or human worth.

## 7. AuthorityEnvelopeV1 candidate

Conceptual shape:

```text
AuthorityEnvelopeV1 {
    schema_version
    envelope_id
    lineage_id

    principal
    delegate
    resource_subjects

    purpose
    permitted_actions
    explicit_denials

    source_authority
    evidence_dependencies

    valid_from
    valid_until
    grant_generation
    revocation_generation

    context_binding
    resource_binding
    audience_binding

    translation_lineage
    translation_disposition

    human_review_policy
    contestability_policy

    integrity_binding
}
```

### 7.1 Required semantics

`lineage_id`
: Identifies the continuity domain in which attenuation/revocation claims are meaningful. A new simulation or deliberate factory lineage must not masquerade as continuation of an old operational lineage.

`principal`
: The actor/entity whose legitimate authority is exercised or delegated. It is not inferred from model confidence.

`delegate`
: The actor/software receiving bounded authority. Re-expression cannot expand the grant.

`resource_subjects`
: Concrete resources/persons/objects to which the grant applies. Avoid ambient authority.

`purpose`
: Human-visible scope. If it constrains authority, it must be integrity-bound.

`permitted_actions`
: Explicit domain-owned action vocabulary.

`explicit_denials`
: Restrictions that remain visible rather than disappearing into a permit bit.

`source_authority`
: The artifact(s) that justify the grant, distinct from supporting evidence.

`evidence_dependencies`
: Evidence needed to keep authority current. Evidence can invalidate authority; evidence alone does not create it.

`valid_from` / `valid_until`
: Temporal bounds.

`grant_generation` / `revocation_generation`
: Ordering semantics for continuity.

`context_binding`
: Session/mission/case/environment constraints.

`resource_binding`
: Prevents replay against another object.

`audience_binding`
: Prevents a grant intended for one enforcement point from being reused elsewhere.

`translation_lineage`
: Records authority-relevant representation transitions.

`translation_disposition`
: Distinguishes exact preservation, attenuation, incompatibility, expiry, revocation, stale evidence, lineage mismatch, and context mismatch.

`human_review_policy`
: Identifies where human review is required and where it is advisory rather than authority-creating.

`contestability_policy`
: Defines inspect/contest/appeal semantics where relevant.

`integrity_binding`
: Strongly binds every authority-relevant field when the artifact crosses a trust boundary.

## 8. Typed authority transitions

Candidate vocabulary:

```text
AuthorityTranslationDisposition {
    Exact,
    Attenuated,
    Incomparable,
    NotRepresentable,
    Expired,
    Revoked,
    EvidenceStale,
    LineageMismatch,
    ContextMismatch,
    AudienceMismatch,
    ResourceMismatch,
}
```

Only `Exact` and a domain-proven `Attenuated` result may continue into authorization.

`Incomparable` is not logically identical to denial, but consequential execution fails closed because permission has not been established.

`LineageMismatch` is important: it prevents a proof about one scenario/session/deployment from being treated as authority continuity in another.

## 9. AugmentationContractV1 candidate

Formal authorization can be correct while human agency is still reduced through dependence, anchoring, opaque substitution, or excessive automation.

Conceptual shape:

```text
AugmentationContractV1 {
    human_goal
    assistance_mode
    retained_human_decisions
    delegable_decisions
    prohibited_substitutions

    explanation_mode
    verification_affordances
    uncertainty_policy

    independence_requirement
    dissent_preservation

    automation_budget
    interruption_policy
    rollback_policy

    capability_outcome_measurement
    reliance_measurement

    consent_scope
    data_scope
}
```

### 9.1 Assistance modes

At minimum distinguish:

```text
Inform
Explain
Critique
Scaffold
Recommend
Plan
PrepareAction
ExecuteDelegatedAction
```

Moving downward generally transfers more authority/responsibility to the system. A request for analysis must not silently become permission to execute.

### 9.2 Retained decisions

The contract should identify decisions intentionally retained by the human, especially values/ends, irreversible commitments, delegation changes, withdrawal/revocation, publication in the human’s name, and high-impact domain decisions where policy requires human authority.

These are domain-dependent boundaries, not one universal paternalistic policy.

### 9.3 Explanation as verification interface

The objective is calibrated reliance, not maximal acceptance.

Interfaces should help a human ask:

```text
What is claimed?
What evidence supports it?
What was inferred rather than observed?
What is uncertain?
What would falsify the claim?
What action would follow?
Who retains authority for that action?
```

### 9.4 Independence and dissent

Where common AI synthesis could anchor a deliberative group, HAK should support independent elicitation before synthesis:

```text
independent positions
-> provenance/evidence
-> comparison
-> shared synthesis
```

rather than assuming one shared AI answer is always the best first move.

## 10. Demonstrated repository findings

HAK-002 distinguishes demonstrated implementation findings from architectural hypotheses.

### 10.1 Rescue negative-consent continuity — demonstrated

A newer refusal/withdrawal could expire and allow an older accepted handoff to reappear as consent.

Invariant:

```text
storage expiry/eviction must not implicitly undo a domain-durable negative consent barrier
```

Tracked in the independent rescue-consent hardening tranche.

### 10.2 Mycelix SubPassport transcript — demonstrated

The legacy delegation transcript omitted authority-relevant semantics and concatenated variable-length identity strings without framing. Renewal/revocation semantics also needed tightening.

Invariant:

```text
integrity-bound delegation == exact grant semantics
```

Tracked independently in MYC-HAK-001.

### 10.3 Sovereign -> legacy governance fallback — demonstrated

A custom 8D `CivicRequirement` can contain dimensions that legacy governance requirements cannot enforce. Silent conversion could discard those constraints.

Invariant:

```text
compatibility fallback != authority to weaken policy
```

The checked fallback rejects nonrepresentable requirements instead of authorizing a weaker approximation.

### 10.4 Embodiment reset/recovery boundary — architectural ambiguity, not demonstrated bypass

The subterranean reset audit initially treated local reset helpers as operational recovery and opened three preservation patches.

A wider audit showed:

- the shared trait documents `reset()` as resetting the body to default state;
- tests and robotics bridge call sites use it as scenario/full-state reinitialization;
- subterranean operational checkpointing separately persists operator/degraded/partition/temporal state;
- checkpoint tests already preserve operator and temporal restrictions.

Therefore the first three reset-preservation drafts were closed unmerged.

Correct invariant:

```text
scenario reset may start a new clean lineage
operational recovery must preserve/revalidate same-lineage authority
```

The remaining work is to make those lifecycle semantics explicit enough that a caller cannot confuse them.

## 11. Reset/lifecycle classes

HAK recommends distinguishing conceptually:

```text
NewSimulationLineage
EphemeralRuntimeReset
OperationalRecovery
AdministrativeFactoryReset
```

The detailed evidence and revised sequence live in the reset-continuity supporting audit.

The key rule is not “reset must always preserve restrictions.” It is:

```text
new lineage must be explicit
same-lineage recovery must not invent authority
```

## 12. Evidence, credentials, policy and enforcement remain separate

Candidate pipeline:

```text
Observation / claim
      ↓
Evidence validation
      ↓
Credential / qualification
      ↓
Policy decision
      ↓
Explicit consent/delegation where required
      ↓
AuthorityEnvelope
      ↓
Policy enforcement point
      ↓
Bounded execution
      ↓
Outcome evidence
```

No arrow becomes equivalent to its neighbor merely because two components live in one process.

A credential can answer:

```text
what has been established about this subject?
```

while authority answers:

```text
what may this executor do,
on which resource,
for what purpose,
in which lineage/context,
until when?
```

## 13. Relevant external authorization architecture

HAK is not an OAuth replacement, GNAP profile, VC profile, or Zero Trust product. Several mature systems nevertheless reinforce useful narrower boundaries:

- OAuth Rich Authorization Requests (RFC 9396) provides fine-grained authorization details and fail-closed handling of unknown/invalid authorization details.
- GNAP (RFC 9635) treats authority delegated to software as an explicit negotiated grant.
- W3C Verifiable Credentials Data Model 2.0 notes that credentials are not by themselves a complete authorization framework.
- NIST SP 800-207 separates policy decision/administration/enforcement and emphasizes least privilege and continuing authorization evaluation.

References:

- https://www.rfc-editor.org/rfc/rfc9396.html
- https://www.rfc-editor.org/rfc/rfc9635.html
- https://www.w3.org/TR/vc-data-model/#authorization
- https://csrc.nist.gov/publications/detail/sp/800-207/final

These are prior art, not claims of HAK compliance or equivalence.

## 14. Candidate review checklist

For an authority-bearing change, reviewers should ask:

1. **Lineage** — Is this the same authority lineage or a declared new one?
2. **Source** — What artifact actually grants authority?
3. **Principal/delegate** — Who exercises whose authority?
4. **Action** — What exact action vocabulary is permitted?
5. **Resource** — What exact objects/persons/resources are bound?
6. **Purpose** — Is material scope integrity-bound?
7. **Time** — What establishes freshness and expiry?
8. **Revocation** — Can storage behavior, restart, migration, or cache loss undo it?
9. **Translation** — Can representation changes discard restrictions?
10. **Fallback** — Does incompatibility fail closed?
11. **Lifecycle** — Is reset/recovery/new-lineage semantics explicit?
12. **Evidence** — Can missing/stale evidence be confused with positive authorization?
13. **Human boundary** — Did recommendation/planning silently become execution?
14. **Contestability** — Can affected humans inspect/contest consequential decisions where required?
15. **Attenuation** — If authority is redelegated, is child scope provably no broader than parent scope?

## 15. Candidate property tests

### 15.1 Translation monotonicity

For every successful same-lineage translation:

```text
permitted(translated) ⊆ permitted(source)
```

### 15.2 Unknown-field fail closed

Adding an authority-relevant field unknown to an older representation must not leave authorization unchanged unless that field is explicitly declared non-authoritative.

### 15.3 Revocation non-resurrection

Once a grant generation is revoked, time passage, cache eviction, same-lineage restart, checkpoint restore, compatibility conversion, or missing data cannot make that generation executable again unless domain semantics explicitly establish a newer grant.

### 15.4 Delegation attenuation

For parent `P` and child `C` in the same delegation lineage:

```text
C <= P
```

must be mechanically established or issuance fails.

### 15.5 Evidence non-authority

Changing a model score or descriptive credential must not directly create executable permission without the relevant policy/grant transition.

### 15.6 Recovery continuity

A future operational recovery path must not gain authority merely from memory reinitialization or positive defaults.

### 15.7 Scenario lineage separation

A clean simulation reset may create configured nominal authority only if it creates/declares a new scenario lineage rather than claiming recovery of the prior one.

## 16. What remains domain-owned

HAK should not centralize every state machine.

Likely domain-owned semantics include:

- rescue/emergency exceptions;
- physical machinery recovery;
- civic membership and constitutional legitimacy;
- financial transaction authorization;
- medical/research consent;
- youth/guardian authority;
- publication/editorial approval;
- infrastructure operator qualification.

Common types should be extracted only after several domains demonstrate genuinely identical semantics.

## 17. Revised tranche sequence

```text
HAK-001  Human Agency Kernel audit
   ↓
HAK-002  Authority & Augmentation Contract v1   [this tranche]
   ↓
HAK-003  authority-bearing type + lineage inventory
   ↓
HAK-004  CapabilityOutcomeVector / AgencyImpactClaim
   ↓
HAK-005  reliance + unaided-transfer benchmarks
   ↓
HAK-006  DissentEnvelope / independent elicitation
   ↓
HAK-007  collective-intelligence experiments
   ↓
HAK-008  advisory institutional linter
```

Parallel implementation work remains independently qualified:

```text
rescue negative-consent continuity
Mycelix SubPassport transcript hardening
lossless civic compatibility fallback
embodiment lifecycle semantic split / operational recovery design
```

The earlier operator/degraded/partition reset-preservation drafts are intentionally **not** part of the active implementation stack after the lifecycle correction.

## 18. Exit criteria

HAK-002 should remain draft until review establishes that:

- authority monotonicity is scoped to a defined lineage;
- new-lineage vs same-lineage lifecycle semantics are explicit;
- human standing is not collapsed into scoped role authority;
- evidence and credentials remain non-self-authorizing;
- negative barriers have explicit supersession semantics;
- lossy translation fails closed;
- augmentation preserves a clear assistance/execution boundary;
- the architecture remains small enough to avoid premature centralization;
- the reset-audit correction is treated as evidence of successful falsification, not hidden as an inconvenience.

## 19. Non-claims

This document does not establish that:

- one authorization model fits every human institution;
- every action requires synchronous human approval;
- all delegation or automation is harmful;
- OAuth, GNAP, VC, or Zero Trust solve human agency;
- the sovereign-profile governance model is accepted or rejected;
- current HAK-related implementation PRs are qualified;
- generic scenario reset is presently an operational security bypass;
- a universal `AuthorityEnvelope` Rust type should already exist;
- Symthaea can measure human flourishing with one scalar.

The target is narrower and testable:

```text
Within a defined authority lineage,
representation change, inference, fallback, delegation, or recovery
must not silently become additional authority.

Across a deliberately new lineage,
the lineage break itself must be explicit enough that nobody mistakes
reinitialization for recovery evidence.
```
