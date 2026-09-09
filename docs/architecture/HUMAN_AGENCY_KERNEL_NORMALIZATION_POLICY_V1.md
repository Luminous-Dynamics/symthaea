# HAK-013 — Content-Bound Normalization Policy v1

Status: architecture + audit-only tooling candidate. No runtime authority changes.

## Purpose

HAK-012 proves how materialized snapshot fields map to normalized provider-source observations. HAK-013 closes the immediately prior boundary: what normalization policy selected those source fields in the first place?

Core distinctions:

```text
NormalizationProfileRef != NormalizationPolicyIdentity
ProfileName != PrecommittedSelectionPolicy
ProjectionCompleteness != SourceNormalizationCompleteness
ContentBoundPolicy != PrecommittedPolicy
```

A symbolic profile name is useful for human readability and compatibility, but it is not an immutable policy identity.

## NormalizationPolicyV1

A normalization policy now binds:

- provider and API family;
- symbolic profile reference;
- explicit allow-list inclusion rule;
- unknown-field handling;
- transform method/version;
- resource-specific required and optional provider paths;
- omission semantics;
- redaction semantics;
- policy supersession identifier;
- canonical domain-separated policy digest.

The initial prospective GitHub Actions policy covers:

```text
WorkflowRunObservation
WorkflowJobsObservation
WorkflowJobStepsObservation
```

Each resource profile has disjoint required and optional paths. Fields outside the explicit policy scope are intentionally omitted and are not evidence of absence.

## Policy identity

The policy digest is:

```text
SHA256(
  "hak.normalization-policy.v1\0"
  || canonical_json(policy_without_policy_digest)
)
```

Therefore:

```text
profile_ref
    = compatibility / human-facing name

policy_id + policy_digest
    = content identity
```

A future observation that claims to be bound to this policy must identify the exact policy content, not merely repeat the symbolic profile name.

## Historical HAK-007 observation

The existing HAK-007 source observation is intentionally **not** upgraded.

Its HAK-013 binding states:

```text
binding_status          = RetrospectiveUnbound
policy_identity         = null
commitment_relation     = NotEstablished
raw_response_retained   = false
normalization_replayability
    = NotReplayableWithoutRawResponse
omitted_field_completeness
    = NotEstablished
```

This preserves the historical truth:

```text
same symbolic profile name
!=
proof that an exact content-bound policy existed before collection
```

Because raw provider bytes were not retained, HAK-013 also refuses to establish omitted-field completeness for that historical observation.

## Prospective collection

For future observations, the stronger target is:

```text
ExactNormalizationPolicy
        +
RawProviderResponse (when retained)
        +
TemporalEvidenceOfPolicyAvailability
        ↓
PolicyBoundNormalizedObservation
```

HAK-013 does not yet claim that temporal relation automatically. A binding that says `PreObservationEstablished` must carry a separate temporal evidence reference. Static policy existence alone is insufficient.

```text
PolicyExistsNow != PolicyWasPrecommittedThen
```

## Replayability and completeness

HAK-013 keeps replayability and source completeness separate from policy identity.

```text
BoundToPolicy
!=
RawResponseRetained

RawResponseRetained
!=
ProviderAuthenticated
```

The v1 validator allows `EstablishedWithinPolicyScope` only when the observation is bound to a content-identified policy and the raw provider response was retained. Historical unbound evidence cannot make that claim.

## Adversarial checks

The focused suite rejects:

- symbolic profile names used as policy identity;
- policy digest tampering;
- duplicate resource profiles;
- required/optional path overlap;
- duplicate selected field paths;
- missing omission semantics;
- retrospective bindings upgraded without policy identity;
- precommit claims without temporal evidence;
- omitted-field completeness claims on the historical fixture;
- replayability claims without raw provider bytes;
- observation/profile mismatch;
- binding digest tampering.

## Self-qualification

HAK-013 has an E5-target qualification plan and a HAK-010 precommitted obligation-to-step binding policy. The dedicated workflow is:

`.github/workflows/hak-normalization-policy.yml`

A green focused run qualifies only HAK-013's normalization-policy bookkeeping claims. It does not authenticate GitHub, retroactively qualify HAK-012, establish the historical HAK-007 normalization as precommitted, or prove semantic truth.

## Non-claims

HAK-013 does not:

- authenticate provider responses;
- preserve historical raw response bytes that were never retained;
- prove historical omitted-field completeness;
- infer precommit timing from current Git history alone;
- make a policy digest equivalent to provider attestation;
- make projection completeness equivalent to source completeness;
- grant runtime authority;
- certify scientific, governance, consent, or robotics behavior.

## Next boundary

The next prospective collection layer should combine this content-bound policy with exact retrieval representation, collection-time bounds, raw-response retention where feasible, and typed re-observation lineage.

```text
CanonicalResourceIdentity
!= RetrievalRepresentation

NormalizationPolicyIdentity
!= ObservationLineage

NewerObservation
!= ErasureOfCounterevidence
```
