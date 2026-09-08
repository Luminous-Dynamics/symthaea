# HAK-012 — Provider Source Observations & Projection Provenance v1

Status: architecture + audit-only tooling candidate. No runtime authority changes.

## Purpose

HAK-011 proved that a real provider execution can be materialized without turning cancellation into semantic claim falsification. HAK-012 closes the next provenance boundary:

```text
LocallyDigestBoundSnapshot != ProviderSourceObservation
```

A local snapshot can preserve internal consistency while still hiding where each field came from. HAK-012 separates:

```text
Provider source response
        ↓
ProviderSourceObservation
        ↓
Explicit field projection
        ↓
Materialized provider snapshot
```

The current tranche applies that model to historical GitHub Actions run `34225891059`.

## Non-goal: provider authentication

HAK-012 does **not** call its source observations cryptographically authenticated. The initial assurance class is `ProviderObserved`: the collector observed metadata through the configured provider integration and retained a normalized subset plus exact resource/source references.

The assurance vocabulary remains:

```text
ProviderObserved
ProviderBound
CryptographicallyAttested
```

but local digests or projection correctness cannot upgrade an observation into a stronger class.

```text
LocalDigest != ProviderBinding
ProviderBinding != CryptographicAttestation
CryptographicAttestation != SemanticTruth
```

## Provider source observation

`ProviderSourceObservationV1` carries independent identities for:

```text
resource identity
source reference
content identity
```

A source observation records:

- provider and repository;
- resource kind;
- exact run/attempt/job identity where applicable;
- canonical source reference;
- collector identity and method;
- a bounded observation window when the connector does not expose an exact retrieval timestamp;
- normalization method and replayability disclosure;
- normalized retained provider payload;
- canonical normalized-payload digest;
- retention disclosure for raw response bytes;
- assurance class;
- optional provider-binding / attestation references;
- supersession lineage;
- canonical observation digest.

The first three real observations are `WorkflowRunObservation`, `WorkflowJobsObservation`, and `WorkflowJobStepsObservation` for HAK Evidence run `34225891059`, attempt `1`.

The job-steps observation preserves the provider result:

```text
steps = []
```

without reconstructing steps from workflow source.

## Observation-time honesty

The connected GitHub read operation does not expose an exact retrieval timestamp to this collector. HAK-012 therefore must not invent one.

The historical source observations use the bounded window:

```text
after_or_at  = 2026-09-08T20:20:00Z
before_or_at = 2026-09-08T20:37:41Z
precision    = BoundedWindow
```

The lower bound is the start of the HAK-012 continuation task; the upper bound is creation of the first HAK-012 pull request, which occurred after the provider reads used to build the fixture.

```text
ExactCollectionInstantUnavailable
-> BoundedObservationWindow
```

The validator requires `after_or_at <= before_or_at`.

## Normalization and replayability

The historical observations retain a collector-selected normalized subset of the provider response. They do **not** retain the original raw HTTP response bytes.

They therefore state:

```text
normalization.method        = FieldSelectionNoSemanticTransform
normalization.replayability = NotReplayableWithoutRawResponse
raw_response_retained       = false
raw_response_digest         = null
```

The selected fields are copied without semantic transformation, but another verifier cannot independently replay the field selection against the original provider bytes because those bytes were not retained.

```text
RawResponseNotRetained
-> NormalizationNotIndependentlyReplayable
```

A future collector that retains raw bytes and their digest may claim `ReplayableFromRetainedRawResponse`, but the v1 validator rejects that claim when the raw response is absent.

## Explicit projection provenance

The HAK-011 provider snapshot is frozen at exact artifact identity:

```text
git:Luminous-Dynamics/symthaea@e305207ad39ca53146f87115938309d80e978336:
docs/architecture/hak/evidence/real/hak007-run-34225891059.capsule.json
```

HAK-012 adds a projection map whose source bindings are digest-bound and whose target bindings cover every leaf of `provider_snapshot` exactly once.

For every target field:

```text
TargetField
    -> SourceObservation
    -> SourcePath
    -> Transform
```

HAK-012 v1 permits only `Identity` transforms. A semantic transform belongs in a later explicitly versioned projection rule.

## Source-scope separation

Equivalent values do not make provider scopes interchangeable. The current projection policy requires:

```text
run-level provider fields
    <- WorkflowRunObservation

job metadata
    <- WorkflowJobsObservation

job.steps
    <- WorkflowJobStepsObservation
```

Therefore `provider = "github-actions"` cannot be silently repointed from the workflow-run observation to the jobs observation merely because the bytes are identical.

```text
EqualValue != EquivalentEvidenceSource
```

This protects source meaning, not only data equality.

## Exhaustive target coverage

Let `Leaves(S)` be every leaf field of the frozen `provider_snapshot`.

HAK-012 requires:

```text
MappedTargetPaths = Leaves(S)
MissingMappings = ∅
ExtraMappings = ∅
DuplicateTargetMappings = ∅
```

For every mapping `m`:

```text
Resolve(m.source_observation, m.source_path)
==
Resolve(provider_snapshot, m.target_path)
```

A snapshot field that cannot identify its provider source is outside HAK-012 conformance.

## Retention honesty

The current historical materialization retains a normalized projection, not the original response bytes:

```text
normalized_payload_retained = true
raw_response_retained = false
raw_response_digest = null
```

A later collector may retain a raw response and digest, but must say so explicitly.

```text
NormalizedPayloadDigest != RawProviderResponseDigest
```

## Re-observation and supersession

`supersedes_observation_id` is reserved for a later observation of the same provider resource. Re-observation appends lineage rather than silently mutating history:

```text
OldObservation
    -> NewObservation(supersedes = OldObservation.id)
```

A re-observation may reveal provider drift or corrected metadata. It does not rewrite the historical observation.

## HAK-011 compatibility

HAK-012 does not mutate HAK-011's historical capsule. It binds that frozen artifact by exact Git ref and capsule digest, then proves that HAK-012 source observations reconstruct its provider snapshot exactly.

```text
HAK-011 result semantics
+
HAK-012 stronger source provenance
```

without retroactively changing the HAK-011 subject.

## Self-qualification

HAK-012 has an E5-target qualification plan at `docs/architecture/hak/plans/hak012-provider-source-observations-e5-v1.plan.json` and a HAK-010 precommitted check-binding policy at `docs/architecture/hak/policies/hak012-provider-source-observation-binding-policy-v1.json`. The focused workflow is `.github/workflows/hak-provider-source.yml`.

The plan tests:

- source/resource identity joins;
- normalized payload and observation digests;
- bounded observation-window ordering;
- normalization replayability honesty;
- explicit assurance boundaries;
- raw-response retention honesty;
- exact frozen-target identity;
- exact source-observation bindings;
- exhaustive field projection;
- same-valued cross-scope substitution;
- target/source digest tampering;
- exact job binding for step observations.

Keep the tranche draft until the dedicated workflow passes on its exact head.

## Non-claims

HAK-012 does not:

- authenticate GitHub cryptographically;
- prove the connected collector itself is uncompromised;
- preserve raw response bytes for the historical fixture;
- claim an exact provider-retrieval instant when the collector cannot establish one;
- claim the normalized field selection is independently replayable without raw response bytes;
- make a provider observation equivalent to semantic claim truth;
- elevate `ProviderObserved` to `ProviderBound`;
- elevate `ProviderBound` to `CryptographicallyAttested`;
- qualify HAK-011 retroactively;
- grant runtime authority;
- certify governance, robotics, consent, or scientific truth.

## Next boundary

HAK-012 makes provenance queryable down to provider source fields, but a locally retained source observation can still be rewritten by an actor who can recompute its local digest.

The next stronger path should be optional and assurance-class explicit:

```text
ProviderSourceObservation
        ↓
Provider-bound or cryptographically attested evidence bundle
```

For providers that support signed provenance, that stronger layer should verify provider signature/transparency identity without changing the semantic meaning of the underlying test result.

Separately, receipt v2 should migrate ambiguous temporal terms such as `provider_completed_at` into exact scope names such as `provider_terminal_observed_at` rather than silently changing v1 semantics.
