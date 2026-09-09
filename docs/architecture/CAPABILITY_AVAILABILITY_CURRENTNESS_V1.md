# Capability Availability and Currentness V1

Status: architectural contract only; no runtime availability implementation or authority change.

This document refines `CAPABILITY_REALIZATION_CONVERGENCE_V1.md` by freezing the temporal and epistemic boundary between a typed capability realization and CC-03B activation assumptions.

## 1. Core theorem

```text
CapabilityRealization
    != Observation
    != CurrentObservation
    != AvailabilityAssessment
    != VerifiedSufficiency
    != ActivationAuthority
    != ExecutionAuthority
```

A realization is structural. An observation is evidence-bearing history. Currentness is a policy evaluation over time. Availability is a policy-scoped assessment. None of these independently proves that a continuity requirement is sufficient or that an action is authorized.

## 2. Availability is tri-state, not boolean

V1 must expose exactly three semantic outcomes:

```text
Available
Unavailable
Unknown
```

`Unknown` is load-bearing. It is not an error value and must not be silently coerced to `Unavailable` or `Available`.

Examples that must preserve `Unknown` unless an explicit policy says otherwise:

- no admissible typed evidence;
- stale evidence;
- evidence captured implausibly in the future;
- incomplete/unknown coverage where absence is the only signal;
- conflicting admissible evidence with no deterministic conflict rule;
- evidence whose subject mapping is ambiguous;
- provider semantics not admitted by the exact policy;
- evidence outside the policy's declared observation window;
- evaluation-time overflow or otherwise invalid temporal arithmetic.

## 3. Time has three distinct roles

The implementation must keep these values separate:

```text
captured_at
assessment_context_time
evaluation_time
```

### 3.1 Capture time

`captured_at` belongs to an exact typed subject observation. It states when the provider says the observation was captured.

It does not establish that the observation is current now.

### 3.2 Assessment context time

An availability assessment may name the operational point-in-time being assessed. This permits deterministic retrospective evaluation such as:

```text
"Was realization R available at 2026-09-09T12:00Z under policy P?"
```

This value belongs in the assessment identity when present.

### 3.3 Evaluation time

The deterministic verifier receives `evaluation_time` explicitly from its caller. It must not read the wall clock internally when constructing or validating an evidence-bearing assessment.

If a caller evaluates the same evidence at a later time and currentness changes, the resulting assessment identity must change.

## 4. Currentness policy must be exact and versioned

A future `CapabilityCurrentnessPolicyV1` should be a canonical serializable value with its own domain-separated `CapabilityCurrentnessPolicyId`.

At minimum, V1 policy semantics must bind:

- admitted typed observation kinds;
- maximum evidence age per admitted observation class/kind;
- maximum allowed future clock skew;
- whether capture time or an explicitly verified source timestamp is authoritative for age calculation;
- minimum acceptable `ObservationCoverage` where absence claims matter;
- admitted `EvidenceBasis` values for an availability conclusion;
- deterministic conflict behavior;
- deterministic multi-observation aggregation behavior;
- any required provider/profile allowlist identity;
- explicit policy schema/version.

Changing any load-bearing rule changes policy identity.

## 5. Currentness evaluation must be deterministic

For one exact typed observation and one exact policy/evaluation context, currentness should produce an explicit result rather than a bare boolean.

Conceptually:

```text
ObservationCurrentnessV1 {
    observation_id,
    currentness_policy_id,
    evaluation_time,
    disposition,
    reason,
}
```

Suggested dispositions:

```text
Current
Stale
FutureDated
PolicyExcluded
Indeterminate
```

The exact names are less important than preserving why evidence was admitted or rejected.

A stale observation remains historical evidence. It is not deleted or rewritten.

## 6. Typed evidence only

Capability availability must consume observations whose canonical identity binds exact `ContinuitySubjectId`.

Legacy `ObservationEnvelopeV1` values with free-form `subject: String` are not directly admissible.

A legacy observation may participate only through the explicit mapping/attestation boundary defined by CC-04A. Availability evaluation consumes the resulting typed evidence artifact, not a string match.

## 7. Availability assessment identity

After the realization and typed-observation lineages exist, a future value should be conceptually equivalent to:

```text
CapabilityAvailabilityAssessmentV1 {
    schema_version,
    realization_id: CapabilityRealizationId,
    subject_snapshot_id: ContinuitySubjectSnapshotId,
    capability_graph_snapshot_id: CapabilityGraphSnapshotId,
    currentness_policy_id: CapabilityCurrentnessPolicyId,
    assessment_context_time,
    evaluation_time,
    admitted_observation_ids: [...],
    excluded_observations: [... reason ...],
    disposition: Available | Unavailable | Unknown,
    assessment_id: CapabilityAvailabilityAssessmentId,
}
```

The whole graph/subject snapshot identities are provenance for validation context. Whether they belong directly inside `CapabilityAvailabilityAssessmentId` should follow one rule:

> include only context whose change can alter the exact assessment semantics or the resolution of referenced identities.

The structural `CapabilityRealizationId` itself must remain stable across unrelated registry membership changes, as frozen by CC-04A.

## 8. Evidence admission is policy-owned

The availability evaluator must not invent a universal confidence ordering across observation providers or evidence bases.

Examples:

- telemetry may establish process liveness but not service correctness;
- a hardware/BMC observation may establish machine power state but not application availability;
- a declaration may be the authoritative source for a contractual external dependency even when no telemetry exists;
- an inference must never silently become a direct observation.

Therefore the exact currentness/availability policy owns admission and aggregation rules.

## 9. Absence is not automatically unavailability

Negative reasoning must honor observation coverage.

```text
no positive observation + Complete relevant coverage
```

may support an unavailable conclusion if the exact policy explicitly defines that inference.

But:

```text
no positive observation + Incomplete/Unknown coverage
```

must remain `Unknown` unless stronger independent evidence establishes unavailability.

This preserves the existing continuity observation theorem that absence under incomplete coverage is not evidence of absence.

## 10. Conflict handling must be explicit

Conflicting current evidence is normal in distributed systems.

Examples:

- BMC says host powered on while agent heartbeat is absent;
- one replica reports healthy while another reports quorum loss;
- routing control plane reports a prefix while synthetic probes fail;
- storage membership exists while write quorum is unavailable.

V1 must not resolve such conflicts through collection order, provider order, majority count, or newest-timestamp-wins unless that exact rule is encoded in policy.

Default behavior for an unhandled admissible conflict is `Unknown`.

## 11. Currentness and availability are non-monotonic over time

CC-03B activation closure is mathematically monotone under one fixed assumption set.

Real-world availability is not.

A capability assessed `Available` at T1 may be `Unknown` or `Unavailable` at T2 without any model-definition change.

Therefore:

```text
AvailabilityAssessment(T1)
    must not be cached as an eternal activation fact.
```

Every admission of availability into a new operational analysis must bind an exact current assessment or an exact policy-approved freshness window.

## 12. Explicit bridge into CC-03B

CC-03B currently accepts caller-declared `available` and `activatable` capability sets. Those are deliberately assumptions, not observations.

The future typed bridge must therefore produce a separate admission artifact rather than changing CC-03B semantics.

Conceptually:

```text
CapabilityActivationAdmissionV1 {
    source_graph_snapshot_id,
    source_availability_assessment_ids,
    admitted_available_capability_ids,
    admission_policy_id,
    admission_id,
}
```

Only exact `Available` assessments admitted under the exact bridge policy may populate the derived available set.

`Unavailable` and `Unknown` must not be inserted into `available`.

`activatable` remains a separate planning/model assumption. A capability being unavailable does not by itself prove it is activatable.

## 13. Do not collapse realization multiplicity

A stable modeled capability may have multiple subject realizations.

Availability of one realization does not automatically imply availability of the capability at every operational scope.

Future aggregation needs an explicit realization policy, for example:

```text
AnyRealization
AllRealizations
AtLeastK
Quorum
FailureDomainDiverse(k)
Custom exact policy
```

These semantics belong in a later realization-aggregation/failure-domain layer. V1 availability assessment should remain realization-scoped unless an exact aggregation policy is present.

## 14. Counterfactual frontier integration

CC-03C counterfactual support sets remain questions, not facts.

When a counterfactual option is used to request new telemetry, field inspection, simulation, procurement research, or operator review, the resulting discovery request should carry:

- exact target `CapabilityId`;
- exact assumed-support set;
- exact CC-03C query/config provenance;
- exact counterfactual frontier receipt when the receipt layer qualifies;
- no execution authority.

This makes the path from model question to evidence collection auditable without promoting the model output into world state.

## 15. Error vs Unknown

Implementation errors and epistemic uncertainty must remain distinct.

Examples of **errors**:

- malformed canonical identity;
- graph/snapshot mismatch;
- invalid policy schema;
- arithmetic overflow;
- impossible internal invariant;
- evidence identity mismatch.

Examples of **Unknown**:

- stale evidence;
- incomplete coverage;
- conflicting admissible evidence;
- no admissible evidence;
- unsupported but well-formed provider semantics under this policy.

A malformed assessment must fail validation; it must not deserialize as `Unknown` and continue.

## 16. No hidden side effects

Currentness and availability evaluation must be pure with respect to operational systems.

It must not:

- poll devices;
- query Holochain;
- call cloud APIs;
- refresh credentials;
- mutate caches;
- initiate remediation;
- issue work orders;
- procure resources;
- invoke Nixward or Spore.

Observation collection is an adapter concern. Deterministic assessment consumes already-identified evidence and explicit time/policy inputs.

## 17. Evidence and reproducibility requirements

Once runtime types exist, focused tests should cover at least:

- identical exact evidence/policy/time => identical assessment identity;
- later evaluation time can change currentness and assessment identity;
- stale evidence => `Unknown`, not `Unavailable`, absent an explicit contrary policy;
- future-dated evidence outside skew => not silently current;
- incomplete coverage cannot establish absence;
- deterministic conflict -> exact configured result;
- unhandled conflict -> `Unknown`;
- legacy free-form observation cannot enter typed availability directly;
- wrong subject/realization binding fails closed;
- assessment for one realization cannot satisfy another realization by shared labels;
- `Unknown` cannot seed CC-03B `available`;
- `Unavailable` cannot seed CC-03B `available`;
- admission receipt binds exact source assessment identities;
- wall-clock reads are absent from deterministic assessment code paths.

## 18. Merge order

Recommended runtime sequence after qualification of the current draft lineages:

```text
subject identity (#1092)
    -> closed subject snapshot (#1154)
capability definition/graph (#1096/#1098)
    -> activation semantics (#1135)
    -> counterfactual analysis (#1144)
    -> exact analysis receipts (#1167)

then converge qualified heads
    -> CapabilityRealizationV1
    -> typed subject observations
    -> CapabilityCurrentnessPolicyV1
    -> realization-scoped AvailabilityAssessmentV1
    -> explicit CC-03B admission artifact
    -> failure-domain/resource aggregation
    -> decision support
    -> independently authenticated authority
```

Do not merge unqualified lineages merely to make types compile together.

## 19. Non-claims

This contract does not establish that any subject exists, any realization is available, any provider is trustworthy, any continuity requirement is satisfied, any recovery option is feasible, or any actor may take an action.

It defines only how future code must preserve uncertainty and temporal provenance when translating typed evidence into an availability assessment.
