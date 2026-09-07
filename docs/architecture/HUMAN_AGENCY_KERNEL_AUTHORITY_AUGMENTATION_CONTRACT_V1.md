# Human Agency Kernel — Authority & Augmentation Contract v1

Status: architecture candidate / non-normative until independently reviewed and qualified

Parent: `HUMAN_AGENCY_KERNEL_AUDIT_V1.md` (HAK-001)

## 1. Purpose

HAK-001 separated assistance, scoring, consent, delegation, execution, outcome, and human benefit. This document sharpens one part of that boundary into an auditable contract for future Symthaea/Mycelix work.

The central problem is not merely whether an actor is authenticated or whether evidence is valid. The problem is whether a chain of representations, translations, fallbacks, resets, caches, and delegations can accidentally produce *more authority* than its source established.

This document therefore defines two candidate architectural objects:

- `AugmentationContractV1`: what a human asks an augmenting system to do, what it must not substitute for the human, and how reliance/contestability are handled.
- `AuthorityEnvelopeV1`: the exact bounded authority, if any, that may cross from evidence/consent/delegation into an execution boundary.

This document does **not** create a shared Rust crate and does **not** transfer qualification from any existing domain implementation.

## 2. Constitutional invariants

### 2.1 Assessment is not authority

```text
ModelAssessmentOfPerson != AuthorityOverPerson
IdentityEvidence          != Authorization
Credential                != Authorization
Recommendation            != Consent
Consent                   != Delegation
Delegation                != Execution
Execution                 != HumanBenefit
```

A verified fact can be a prerequisite for an authority decision without itself becoming the authority decision.

### 2.2 No action without current explicit authority

For consequential execution, permission is established positively:

```text
ExecutionPermission = CurrentExplicitGrant
```

not negatively:

```text
ExecutionPermission = !KnownDenial
```

`Unknown`, `Unavailable`, `Refused`, `Revoked`, and `Expired` remain distinct states even when all fail closed for a particular action.

### 2.3 Authority monotonicity

For any transformation `T` applied to an authority-bearing artifact `A`:

```text
Authority(T(A)) ⊆ Authority(A)
```

A representation change, compatibility fallback, cache reconstruction, restart, migration, summary, or derived credential must not increase the set of actions that the source artifact permits.

If exact or conservative translation cannot be demonstrated:

```text
NotRepresentable -> NoAuthority
```

not:

```text
NotRepresentable -> BestEffortAuthorization
```

### 2.4 Restriction removal is an authority transition

Removing a hold, stop, maintenance lock, refusal, revocation barrier, degraded mode, or review latch is not ordinary cleanup.

```text
RestrictionRemoval == AuthorityWidening
```

Therefore it requires the same explicit semantics as any other authority-widening transition.

In particular:

```text
reset       != ResumeNominal
restart     != ResumeNominal
cache miss  != ResumeNominal
expiry      != RevocationUndo
migration   != RevocationUndo
```

### 2.5 Negative barriers and positive grants are temporally asymmetric

Positive grants should normally be bounded by time, scope, context, and evidence freshness.

Negative barriers such as refusal or revocation must not disappear merely because their record becomes old, is evicted from a cache, or a previous positive grant remains in history.

Candidate generic model:

```text
Grant:
  generation = g
  valid_until = t

Revocation:
  revocation_generation >= g

A grant is usable only if:
  current_time < valid_until
  and grant_generation > applicable_revocation_generation
```

A new explicit grant may supersede a revocation only according to domain-owned rules and an unambiguously newer generation/sequence.

## 3. Authority is a partial order, not a scalar

HAK must not represent authority as one number such as `trust = 0.82`.

Authority is multidimensional and action-relative. A useful conceptual relation is:

```text
A <= B
```

when every action permitted by `A` is also permitted by `B` under the same subjects, resources, contexts, and temporal bounds.

This relation is only defined when the relevant domain can compare the two grants safely.

If the implementation cannot determine whether one arbitrary authorization is a subset of another, it must report `Incomparable` / `NotRepresentable` rather than guess.

This mirrors an important limitation in fine-grained authorization systems: permission attenuation is meaningful only when the semantics of the authorization details are known to the relevant policy domain.

## 4. AuthorityEnvelopeV1 candidate

The following is a design shape, not yet a Rust API:

```text
AuthorityEnvelopeV1 {
    schema_version
    envelope_id

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

### 4.1 Required semantics

`principal`
: The actor/entity whose legitimate authority is being exercised or delegated. It is not inferred from model confidence.

`delegate`
: The actor/software receiving bounded authority. A delegate cannot expand the grant by re-expression.

`resource_subjects`
: The concrete resources/persons/objects to which the grant applies. Avoid ambient authority.

`purpose`
: Human-visible intent/scope. If purpose materially constrains the grant, it must be integrity-bound.

`permitted_actions`
: Explicit action vocabulary owned by the domain.

`explicit_denials`
: Restrictions that remain visible rather than disappearing into a single permit bit.

`source_authority`
: The artifact(s) that actually justify the grant, distinct from supporting evidence.

`evidence_dependencies`
: Evidence needed to keep the grant current. Evidence expiration may remove authority; evidence alone cannot create it.

`valid_from` / `valid_until`
: Explicit temporal bounds.

`grant_generation` / `revocation_generation`
: Durable ordering semantics for grant/revocation continuity.

`context_binding`
: Environment/session/mission/case constraints.

`resource_binding`
: Prevents a grant for resource A from being replayed against resource B.

`audience_binding`
: Prevents a grant intended for one executor/enforcement point from being reused elsewhere.

`translation_lineage`
: Records each authority-relevant representation transition.

`translation_disposition`
: Must distinguish exact preservation, conservative attenuation, incompatibility, expiry, revocation, stale evidence, and context mismatch.

`human_review_policy`
: Specifies where a human decision is required and, equally importantly, where review is advisory rather than authority-creating.

`contestability_policy`
: Defines inspect/contest/appeal semantics where the domain requires them.

`integrity_binding`
: Cryptographically or otherwise strongly binds every authority-relevant field for contexts that cross a trust boundary.

## 5. Typed authority transitions

HAK should prefer typed transition outcomes over booleans.

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
    ContextMismatch,
    AudienceMismatch,
    ResourceMismatch,
}
```

Only `Exact` and a domain-proven `Attenuated` result may continue into authorization.

`Incomparable` is not equivalent to denial as a semantic statement, but consequential execution should fail closed because permission has not been established.

## 6. AugmentationContractV1 candidate

Authority is only one half of HAK. A system can preserve formal permission while still reducing human agency through dependence, anchoring, hidden substitution, or opaque automation.

Candidate shape:

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

### 6.1 Assistance modes

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

Moving downward in this list generally increases the amount of authority and responsibility transferred to the system.

The transition must be explicit. A request for analysis must not silently become permission to execute.

### 6.2 Retained human decisions

An augmentation contract should identify decisions intentionally retained by the human, particularly:

- values and ends;
- acceptance of consequential trade-offs;
- irreversible commitments;
- delegation changes;
- withdrawal/revocation;
- publication or representation in the human's name;
- high-impact civic, medical, financial, legal, employment, or physical actions where domain policy requires human authority.

This list is domain-dependent; HAK should provide the shape, not one universal paternalistic policy.

### 6.3 Verification rather than persuasive explanation

The goal of explanation is not maximal user acceptance. It is calibrated reliance.

Useful interfaces should help the human answer:

```text
What is being claimed?
What evidence supports it?
What would falsify it?
What is uncertain?
What did the model infer rather than observe?
What action, if any, would follow?
Who retains authority for that action?
```

### 6.4 Independence and dissent

For deliberative or collective-intelligence contexts, HAK should support independent elicitation before common AI synthesis when anchoring would materially reduce epistemic diversity.

```text
independent positions
-> provenance/evidence
-> comparison
-> shared synthesis
```

not necessarily:

```text
one AI synthesis
-> everyone reacts to the same anchor
```

## 7. Existing implementation findings that motivate this contract

This architecture is grounded in concrete repository findings rather than hypothetical concerns.

### 7.1 Rescue consent continuity

Existing rescue semantics correctly distinguish explicit case consent from distress/silence, but the ledger previously allowed a later refusal/withdrawal to expire and fall back to an older accepted handoff state.

The HAK invariant is:

```text
negative consent barrier cannot disappear by expiry/cache behavior and resurrect older positive authority
```

Tracked independently in the rescue-consent hardening tranche.

### 7.2 Mycelix SubPassport transcript

The previous delegation transcript did not bind all authority-relevant semantics and concatenated variable-length identity strings without framing.

The HAK invariant is:

```text
signed delegation == exact integrity-bound grant semantics
```

A renewal that changes signed temporal bounds must require a new integrity binding; revocation cannot be ordinary renewal state.

### 7.3 Sovereign -> legacy governance fallback

An 8D civic requirement can contain constraints the legacy schema cannot represent.

The HAK invariant is:

```text
compatibility fallback != authority to weaken policy
```

Unsupported constraints must produce a typed incompatibility rather than being silently discarded.

### 7.4 Subterranean operator reset

The operator authority state machine requires quorum and hazard checks to clear restrictive authority. A generic runtime reset previously cleared the active constraint directly.

The HAK invariant is:

```text
reset/restart != recovery authorization
```

A reset may discard incomplete authority-accruing state (for example partial resume quorum) but must not itself widen authority.

### 7.5 Degraded-operation reset — open audit finding

`DegradedMode::RecoveryRequired` is intentionally latched: normal link restoration is insufficient, and clearing it requires an explicit external authorization plus healthy dwell at a safe/service location.

The current `reset_runtime()` nevertheless sets the mode directly to `Normal`.

This is an unresolved audit finding in this HAK-002 document, not a claim of qualification or a bundled code change.

Candidate repair theorem:

```text
runtime reset may clear ephemeral counters
but must preserve RecoveryRequired until the domain's authorized clear transition succeeds
```

### 7.6 Partition-recovery reset — open audit finding

Partition recovery explicitly states that restored connectivity is not restored operational truth and requires a reconciliation dwell before team state becomes authoritative.

The current `reset_runtime()` sets mode and cached assessment to `Connected`, with motion permitted and team state authoritative.

This is another unresolved audit finding.

Candidate repair theorem:

```text
reset cannot manufacture Connected / authoritative team state
```

A conservative reset may discard partial reconciliation progress, but it must require fresh observations to regain connected authority.

### 7.7 Temporal-assurance reset — open audit finding

Temporal assurance latches `HoldForReview` after broken clock/causal history and requires clean dwell at a safe service location to release the latch.

The embodiment reset currently replaces the temporal supervisor with `Default`, whose initial assessment is nominal.

Candidate theorem:

```text
reset cannot clear a causal/temporal review latch without the same evidence required by the normal recovery transition
```

## 8. Reset classes

One source of confusion is using one word, `reset`, for semantically different operations.

HAK recommends distinguishing:

### 8.1 Pure simulation reset

Purpose: deterministic test/scenario initialization.

May intentionally create a clean world *only when the simulated world itself is also reset and there is no claim of continuity with an operational authority lineage*.

It should be clearly named/scoped and must not be reused as a live recovery primitive by accident.

### 8.2 Ephemeral runtime reset

Purpose: discard caches, incomplete computations, partial handshakes, or partial quorum accumulation.

Rule:

```text
may remove provisional positive state
must not remove durable negative/restrictive state
must not manufacture positive evidence
```

### 8.3 Operational restart/recovery

Purpose: resume a real system after restart, fault, update, or checkpoint restoration.

Requires explicit continuity semantics for:

- active restrictions;
- revocation generations;
- replay barriers;
- evidence freshness;
- authority epochs;
- physical state;
- trust state;
- pending vs completed authorization transitions.

### 8.4 Administrative factory reset

Purpose: intentionally destroy an authority lineage/configuration.

This is a privileged destructive action, not a safety recovery shortcut. Any future implementation must specify who can authorize it, what physical state is required, and whether subsequent operation begins in an unqualified/hold state.

## 9. Evidence, credentials, policy and enforcement must remain separate

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

No arrow may be collapsed merely because two adjacent components currently live in one process or crate.

A credential can answer `what has been established about this subject?` while the authority layer answers `what action is this executor permitted to perform, on what resource, for what purpose, until when?`

## 10. Comparison with relevant external security architecture

HAK is not an OAuth replacement, GNAP profile, VC profile, or Zero Trust product. However, several mature authorization systems reinforce useful boundaries:

- OAuth Rich Authorization Requests (RFC 9396) represents fine-grained authorization details and requires unknown or invalid authorization-detail types/fields to fail rather than being interpreted approximately.
- GNAP (RFC 9635) treats authorization to software as an explicit negotiated grant rather than as a side effect of identity.
- W3C Verifiable Credentials Data Model 2.0 explicitly notes that verifiable credentials are not, by themselves, a complete authorization framework.
- NIST Zero Trust separates policy decision/administration/enforcement and emphasizes least privilege, deny-by-default policy, and continuing authorization evaluation.

References:

- https://www.rfc-editor.org/rfc/rfc9396.html
- https://www.rfc-editor.org/rfc/rfc9635.html
- https://www.w3.org/TR/vc-data-model/#authorization
- https://csrc.nist.gov/publications/detail/sp/800-207/final

The point of these references is not standards compliance. The point is that HAK should avoid reinventing known authorization mistakes while extending the model to human agency, augmentation, contestability, negative consent continuity, and cross-domain authority transformations.

## 11. Candidate review rules

Future code reviews involving authority should answer all of the following.

1. **Source** — What artifact actually grants authority?
2. **Subject** — Who/what is the principal and delegate?
3. **Action** — What exact action vocabulary is permitted?
4. **Resource** — What exact resources/persons/objects are bound?
5. **Purpose** — Is material scope integrity-bound?
6. **Time** — What establishes freshness and expiry?
7. **Revocation** — Can a negative barrier disappear through TTL, eviction, restart, or migration?
8. **Translation** — Can any representation change discard restrictions?
9. **Fallback** — Does incompatibility fail closed?
10. **Reset** — Can reset/restart widen authority or manufacture nominal state?
11. **Evidence** — Can stale/missing evidence be confused with positive authorization?
12. **Human boundary** — Did recommendation/planning silently become execution?
13. **Contestability** — Can affected humans inspect and contest consequential decisions where required?
14. **Audit** — Can the authority lineage be reconstructed without trusting mutable explanatory text?
15. **Attenuation** — If authority is delegated again, can the child grant be proven no broader than the parent?

## 12. Candidate property tests

Any future generic implementation should be tested with properties rather than only examples.

### 12.1 Translation monotonicity

For every successful translation:

```text
permitted(translated) ⊆ permitted(source)
```

### 12.2 Unknown-field fail closed

Adding an authority-relevant field unknown to an older representation must never leave authorization unchanged unless the field is explicitly declared non-authoritative.

### 12.3 Reset monotonicity

For every non-factory operational reset:

```text
permitted(after_reset) ⊆ permitted(before_reset)
```

until fresh evidence/authorization transitions occur.

### 12.4 Revocation non-resurrection

Once a grant generation is revoked, no combination of:

- time passage;
- cache eviction;
- restart;
- checkpoint restore;
- compatibility conversion;
- missing data;

may make that generation executable again.

### 12.5 Delegation attenuation

For parent grant `P` and child grant `C`:

```text
C <= P
```

must be mechanically established or child issuance fails.

### 12.6 Evidence non-authority

Mutation of a model score or descriptive credential must not directly create executable permission without the relevant policy/grant transition.

## 13. What should remain domain-owned

HAK should not centralize every authority state machine.

Likely domain-owned semantics include:

- emergency medical/rescue exceptions;
- physical machinery recovery;
- civic membership and constitutional legitimacy;
- financial transaction authorization;
- research consent;
- youth/guardian authority;
- publication/editorial approval;
- infrastructure operator qualifications.

HAK may eventually supply small common types for lineage, translation disposition, generation ordering, and envelope metadata, but only after at least several domains demonstrate genuinely identical semantics.

## 14. Proposed tranche sequence after HAK-002

```text
HAK-001  Human Agency Kernel audit
   ↓
HAK-002  Authority & Augmentation Contract v1   [this document]
   ↓
HAK-003  Authority-bearing type inventory + machine-readable audit manifest
   ↓
HAK-004  CapabilityOutcomeVectorV1 / AgencyImpactClaimV1
   ↓
HAK-005  reliance + unaided-transfer benchmark harness
   ↓
HAK-006  DissentEnvelopeV1 / independent elicitation
   ↓
HAK-007  collective-intelligence experiments
   ↓
HAK-008  advisory institutional linter
```

Parallel domain hardening remains independent:

```text
rescue negative-consent continuity
Mycelix SubPassport transcript hardening
lossless civic compatibility fallback
operator reset authority monotonicity
future degraded/partition/temporal reset continuity repairs
```

## 15. Exit criteria for HAK-002

HAK-002 should not be treated as ready merely because this document exists.

Review should establish that:

- the authority monotonicity principle is coherent across the audited domains;
- the contract does not confuse human standing with scoped role authority;
- the proposed envelope does not make evidence or credentials self-authorizing;
- negative barriers remain semantically distinct from expiring grants;
- reset/restart classes are sufficiently separated;
- translation outcomes are explicit enough to prevent lossy authorization fallbacks;
- augmentation semantics preserve a clear boundary between assistance and execution;
- the design remains small enough that domain-specific authority semantics are not prematurely centralized.

Only after that review should a machine-readable inventory or common implementation type be proposed.

## 16. Non-claims

This document does not establish that:

- one authorization model is appropriate for all human institutions;
- every action requires direct synchronous human approval;
- all delegation is harmful;
- all automated execution is illegitimate;
- OAuth, GNAP, VC, or Zero Trust solve human agency;
- the sovereign-profile governance model is accepted or rejected;
- current HAK-related code PRs are qualified;
- a universal `AuthorityEnvelope` Rust type should already be introduced;
- Symthaea can measure human flourishing with one scalar;
- safety restrictions should never be cleared; only that clearing them must be an explicit, authorized transition.

The target is narrower and more testable:

```text
A system may become increasingly capable and helpful
without allowing representation changes, inference, fallback,
reset, or delegation to silently become additional authority.
```
