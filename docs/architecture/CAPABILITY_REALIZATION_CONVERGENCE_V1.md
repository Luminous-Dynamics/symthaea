# Capability Realization Convergence V1

Status: architectural contract only; no runtime implementation or authority change.

This document freezes the trust boundary required to converge the typed continuity-subject lineage with the capability-definition / activation / counterfactual lineage without silently reinterpreting existing V1 evidence-bearing identities.

## 1. Core theorem

```text
CapabilityDefinition
    != ContinuitySubject
    != CapabilityRealization
    != Observation
    != AvailabilityAssessment
    != VerifiedSufficiency
    != Authority
```

Each layer answers a different question:

- **CapabilityDefinition** — what capability is modeled, and what modeled prerequisites does that exact definition revision declare?
- **ContinuitySubject** — what exact operational scope is being named?
- **CapabilityRealization** — which exact typed subject is declared to realize which exact capability-definition revision?
- **Observation** — what evidence-bearing statement was captured about an exact subject?
- **AvailabilityAssessment** — under explicit evidence/currentness rules and evaluation context, should a realization be treated as available, unavailable, or unknown for analysis?
- **VerifiedSufficiency** — does available evidence satisfy an exact continuity requirement / verifier policy for an exact target?
- **Authority** — may any actor or machine take an operational action?

No identity or result in an earlier layer grants a result in a later layer by itself.

## 2. Existing V1 values that must not be silently reused

### 2.1 `ObservationEnvelopeV1::subject` is free-form text

The existing observation envelope commits a `String` subject into `ObservationId`. It does not bind `ContinuitySubjectId` in its canonical payload.

Therefore:

```text
ObservationEnvelopeV1.subject == "cluster-a"
```

must **not** be interpreted as structural or cryptographic equality with any particular `ContinuitySubjectId`, even if labels happen to match.

A typed realization layer must not accept a legacy `ObservationId` as subject-bound evidence merely by parsing or comparing its subject string.

### 2.2 `TargetRealizationId` is a generic witness target digest

The existing witness-layer `TargetRealizationId::from_digest()` domain-separates an arbitrary nonzero digest. It does not structurally bind:

- `CapabilityId`;
- `CapabilityDefinitionId`;
- `ContinuitySubjectId`.

It remains valid for its existing witness purpose but must not be renamed, aliased, or silently reinterpreted as the new capability-realization identity.

Any future bridge between a typed `CapabilityRealizationId` and `TargetRealizationId` must be explicit, versioned, and reviewable.

## 3. Exact subject resolution is required before realization binding

An individually valid `ContinuitySubjectV1` is not enough to establish membership in the exact operational subject universe used by an analysis.

The subject lineage therefore requires a closed-world boundary equivalent to the `ValidatedContinuitySubjectSnapshotV1` introduced by the typed-subject snapshot child of #1092.

That boundary establishes only:

- canonical exact subject identities;
- exact `ContinuitySubjectSnapshotId`;
- closed parent references;
- bounded/acyclic containment structure;
- exact subject lookup.

It does not establish physical existence, ownership, currentness, availability, or authority.

A future realization validator must receive both:

```text
ValidatedCapabilityGraphV1
ValidatedContinuitySubjectSnapshotV1
```

and require its exact capability definition and exact subject to resolve in those inputs.

The realization identity itself should **not** include either whole snapshot identity. Unrelated graph or registry membership changes must not change the structural identity of the same exact `(CapabilityDefinitionId, ContinuitySubjectId)` binding.

The downstream validated realization wrapper or enclosing realization-snapshot artifact may retain the exact graph/subject-snapshot validation context as provenance.

## 4. Required new identity: capability realization

After subject and capability lineages independently qualify and converge, the first runtime convergence type should be conceptually equivalent to:

```text
CapabilityRealizationV1 {
    schema_version,
    capability_id: CapabilityId,
    capability_definition_id: CapabilityDefinitionId,
    subject_id: ContinuitySubjectId,
    realization_id: CapabilityRealizationId,
}
```

### 4.1 Identity semantics

`CapabilityRealizationId` must be a new domain-separated content identity over the exact canonical tuple above.

Validation must establish all of the following before returning a downstream-trusted wrapper:

1. `CapabilityDefinitionId` resolves in the supplied validated capability graph;
2. its stable `CapabilityId` equals the stored `capability_id`;
3. `ContinuitySubjectId` resolves in the supplied validated subject snapshot;
4. the stored `realization_id` equals the domain-separated hash of canonical fields.

The realization identity proves only structural binding.

It does **not** prove that:

- the subject exists in the physical world;
- the subject currently implements the capability;
- the capability is available now;
- the realization is safe or sufficient;
- the subject is owned by the caller;
- any actor may operate or modify the subject.

### 4.2 V1 multiplicity rule

V1 should deliberately define at most one structural realization binding for an exact:

```text
(CapabilityDefinitionId, ContinuitySubjectId)
```

pair.

If multiple independently meaningful implementations on one apparent operational object must later be distinguished, model them as distinct typed subjects where appropriate or introduce an explicitly versioned realization-instance model. V1 must not add a free-form instance label merely to manufacture uniqueness.

## 5. Required evidence boundary: typed subject observations

Typed availability must consume observations whose **own canonical identity payload** binds exact `ContinuitySubjectId`.

A future value should be conceptually equivalent to:

```text
SubjectObservationEnvelopeV2 {
    schema_version,
    subject_id: ContinuitySubjectId,
    observation_kind,
    provider_id,
    provider_version,
    captured_at_unix_ms,
    coverage,
    basis,
    evidence_digest,
    limitations,
    observation_id_v2,
}
```

The exact type name is not load-bearing; the typed-subject identity boundary is.

### 5.1 No string-match upgrade

There must be no implicit conversion of `ObservationEnvelopeV1.subject: String` into `ContinuitySubjectId` by:

- label equality;
- namespace convention;
- path convention;
- hostname or DNS name;
- serial number;
- provider-specific naming;
- adapter-local inference.

### 5.2 Legacy observation migration

If historical V1 observations need to participate in the typed system, conversion must produce a new evidence-bearing artifact that explicitly records:

- source legacy `ObservationId`;
- target `ContinuitySubjectId`;
- mapping method / mapper identity;
- mapping evidence digest/reference;
- limitations / ambiguity;
- a new domain-separated mapping-attestation or typed-observation identity.

The legacy observation remains unchanged. The adapter records a mapping assertion under provenance; it does not rewrite history.

## 6. Currentness must be explicit and deterministic

Observation capture time is not itself a currentness decision.

Currentness evaluation must distinguish at least:

```text
capture time
assessment time
evaluation time
```

and must not hide a wall-clock read inside identity construction or deterministic verification.

A future policy should have its own canonical identity and explicit finite bounds, conceptually:

```text
CapabilityCurrentnessPolicyV1 {
    schema_version,
    max_observation_age_ms,
    max_assessment_age_ms,
    max_future_skew_ms,
    policy_id,
}
```

Exact fields may be refined during implementation review, but these invariants are required:

- age/skew bounds are explicit and finite;
- zero/overflow/inert values are rejected where applicable;
- evaluation receives `evaluation_time_unix_ms` explicitly;
- implausibly future-dated evidence is never silently treated as fresh;
- policy identity changes when any bound changes.

Currentness policy must not absorb evidence-strength or operational-sufficiency policy. Those are separate trust decisions.

## 7. Required assessment boundary: availability

Capability realization existence must not imply availability.

A future assessment should be conceptually equivalent to:

```text
CapabilityAvailabilityAssessmentV1 {
    schema_version,
    realization_id: CapabilityRealizationId,
    disposition: Available | Unavailable | Unknown,
    source_observations: [TypedObservationId...],
    assessed_at_unix_ms,
    currentness_policy_id,
    assessment_id,
}
```

### 7.1 Canonical requirements

The assessment identity must commit to:

- exact realization identity;
- exact disposition;
- canonical sorted/deduplicated source evidence identities;
- explicit assessment time;
- exact currentness-policy identity.

An empty evidence set must not yield `Available` unless a separately specified explicit evidence policy permits a non-observational basis. V1 should prefer failing closed to inventing availability from structure alone.

### 7.2 Tri-state, not boolean

V1 must preserve at least:

- `Available`;
- `Unavailable`;
- `Unknown`.

Stale, incomplete, ambiguous, contradictory, or insufficiently scoped evidence must be representable without coercing uncertainty into `false` or `true`.

If `Degraded` is later needed, introduce it only with explicit capability/service semantics rather than as an unqualified confidence label.

## 8. Availability is not sufficiency

Even a structurally valid, current, `Available` realization may be insufficient for an exact continuity requirement.

Examples include:

- insufficient throughput/capacity;
- wrong failure domain;
- inadequate redundancy;
- incompatible protocol/version;
- unsafe operating envelope;
- insufficient evidence class;
- policy mismatch;
- human-rights / non-coercion constraints.

Therefore:

```text
Available(realization)
    !=
Satisfies(continuity_requirement)
```

Sufficiency belongs in an explicit verifier / obligation layer binding exact requirement, target, evidence, policy, and currentness context.

## 9. Activation assumptions remain assumptions

CC-03B intentionally calls its inputs `available` and `activatable` **assumptions**.

A typed availability assessment must not silently mutate those semantics.

The future bridge must be explicit, conceptually:

```text
AvailabilityAssessment
    -- explicit admission policy -->
CapabilityActivationAssumption
```

That admission boundary must state:

- accepted assessment dispositions;
- currentness requirements;
- exact graph / realization compatibility;
- required evidence or verifier class;
- contradictory-assessment handling;
- exact evaluation timestamp.

The resulting activation-assumption identity must remain reproducible from admitted inputs.

## 10. Counterfactual support is never realization evidence

CC-03C asks what would happen **if** modeled support capabilities were treated as available.

A counterfactual support set must never be fed directly into:

- realization registration;
- availability evidence;
- sufficiency claims;
- work orders;
- procurement;
- execution authority.

The safe direction is:

```text
counterfactual option
    -> question for evidence / realization discovery
```

not:

```text
counterfactual option
    -> fact about the world
```

## 11. Failure domains and resources come after realization/currentness

Ranking recovery options before typed realizations exist would erase correlated risk and physical constraints.

After realization/currentness qualify, a later layer may bind explicit properties such as:

- site / rack / power-domain membership;
- provider / region / network-path correlation;
- human skill / staffing dependency;
- consumables and spare parts;
- energy / water / bandwidth / storage requirements;
- restoration time windows;
- minimum capacity / redundancy obligations.

These properties must be evidence-bearing or policy-declared, not inferred from `CapabilityId` names.

Only after this layer qualifies should multi-objective decision support be considered.

## 12. Decision support still does not grant authority

Any future ranking must preserve:

```text
RecoveryCandidate
    != Recommendation
    != ApprovedPlan
    != WorkOrder
    != ProcurementAuthorization
    != HumanAssignment
    != MachineExecutionCapability
```

A ranking function may describe trade-offs. It must not become an implicit authority path.

## 13. Explicit bridge to existing witness `TargetRealizationId`

The witness subsystem already has `TargetRealizationId` for exact witness targets.

V1 convergence must preserve that existing identity contract.

If a capability realization becomes a legal witness target, introduce an explicit bridge/profile conceptually similar to:

```text
CapabilityRealizationWitnessTargetV1 {
    capability_realization_id,
    target_realization_id,
    bridge_schema,
    bridge_id,
}
```

Exact implementation may differ, but these rules are mandatory:

- no type alias;
- no semantic rename of existing `TargetRealizationId`;
- no hidden call making arbitrary target digests indistinguishable from typed capability realizations;
- canonical, versioned, domain-separated, testable bridge semantics.

## 14. Convergence / merge gates

Runtime implementation must not begin by mechanically mixing unqualified evidence lineages.

Required order:

1. typed continuity subjects qualify at exact head;
2. closed typed subject snapshot / registry boundary qualifies at exact head;
3. canonical capability identity + graph validation qualify;
4. assumption-scoped activation closure qualifies;
5. counterfactual frontier qualifies if used by downstream discovery;
6. converge the qualified subject-snapshot + capability heads into a new exact implementation parent;
7. implement `CapabilityRealizationV1` as the first convergence child;
8. implement typed subject observations as a separate child;
9. implement deterministic currentness policy/evaluation as a separate child;
10. implement availability assessment as a separate child;
11. only then bridge admitted availability into activation assumptions;
12. only after that add failure-domain/resource constraints;
13. defer ranking, planning, and authority integration until every earlier boundary has independent evidence.

Any environment/toolchain change after evidence begins follows the repository's reproducibility-lineage rule: do not mix incompatible evidence roots.

## 15. Required implementation tests

The convergence implementation must eventually prove at least:

- same exact capability definition + same exact subject -> deterministic same realization identity;
- capability revision change -> different realization identity;
- subject change -> different realization identity;
- unrelated capability-graph or subject-snapshot membership changes do not rewrite the structural realization identity;
- realization validation fails if exact definition is absent from supplied graph;
- realization validation fails if exact subject is absent from supplied subject snapshot;
- stable `CapabilityId` alone cannot substitute for exact `CapabilityDefinitionId`;
- free-form observation subject string cannot satisfy typed subject binding;
- legacy observation mapping requires explicit mapping evidence / identity;
- realization alone never produces `Available`;
- stale evidence never silently produces current availability;
- future-dated evidence beyond explicit skew policy is rejected/unknown;
- `Unknown` survives incomplete/ambiguous evidence;
- availability does not imply requirement sufficiency;
- typed realization cannot silently alias existing `TargetRealizationId`;
- counterfactual support cannot be admitted as observed availability without an independent evidence path.

## 16. Non-claims

This document adds no runtime realization type, observation adapter, realization registry, currentness evaluator, availability assessment, verifier, planner, work order, procurement action, human assignment, migration action, governance action, or execution authority.

It does not claim that any current capability, subject, realization, observation, or counterfactual option corresponds to a real operational resource.

Its purpose is narrower: prevent future convergence from collapsing distinct identities and trust decisions into one convenient but unsound object.
