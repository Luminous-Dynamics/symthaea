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
PresenceRequirement != FieldSelection
ContainerExists != IncludeEntireSubtree
```

A symbolic profile name is useful for human readability and compatibility, but it is not an immutable policy identity.

## NormalizationPolicyV1

A normalization policy now binds:

- provider and API family;
- symbolic profile reference;
- explicit allow-list inclusion rule;
- unknown-field handling;
- transform method/version;
- explicit path-category semantics;
- resource-specific required container, required selected, and optional selected provider paths;
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

## Path semantics

HAK-013 separates three concepts that must not be collapsed:

```text
required_container_paths
    = RequireAndSelectContainerShapeOnly

required_paths
    = SelectValueAndRequirePresence

optional_paths
    = SelectValueIfPresent
```

A required container path proves that a container exists and permits the normalized representation to preserve its container shape, including an empty array/object, but it does **not** authorize copying arbitrary descendant fields.

This is important for the GitHub Actions steps source. The policy now says:

```text
required_container_paths = ["steps"]
required_paths           = []
optional_paths           = [
    "steps[*].name",
    "steps[*].status",
    "steps[*].conclusion",
    "steps[*].number",
    "steps[*].started_at",
    "steps[*].completed_at"
]
```

Therefore `steps = []` can be preserved faithfully, while `steps` does not mean copying every unknown field from every future GitHub step object.

Exact overlap between container/required/optional categories is rejected. A container path may intentionally be a prefix of selected descendants because those declarations have different semantics.

The precise selector grammar and execution behavior belong to the next prospective execution layer; HAK-013 only makes the categories explicit and content-bound.

## Policy identity

The policy digest is:

```text
SHA256(
  "hak.normalization-policy.v1\0"
  || canonical_json(policy_without_policy_digest)
)
```

Therefore `profile_ref` is a compatibility/human-facing name while `policy_id + policy_digest` is content identity. A future observation that claims to be bound to this policy must identify the exact policy content, not merely repeat the symbolic profile name.

## Historical HAK-007 observation

The existing HAK-007 source observation is intentionally **not** upgraded. Its HAK-013 binding remains `RetrospectiveUnbound`, has no policy identity, no established commitment relation, no retained raw response, no replayability claim, and no omitted-field completeness claim.

```text
same symbolic profile name
!=
proof that an exact content-bound policy existed before collection
```

## Prospective collection

For future observations, the stronger target is:

```text
ExactNormalizationPolicy
+
RawProviderResponse (when retained)
+
TemporalEvidenceOfPolicyAvailability
-> PolicyBoundNormalizedObservation
```

HAK-013 does not yet prove that temporal relation automatically.

```text
PolicyExistsNow != PolicyWasPrecommittedThen
NormalizationPolicyDefined != NormalizationPolicyExecutedByCollector
```

The latter composition boundary is tracked separately in #1035.

## Replayability and completeness

```text
BoundToPolicy != RawResponseRetained
RawResponseRetained != ProviderAuthenticated
```

The v1 validator allows `EstablishedWithinPolicyScope` only when the observation is bound to a content-identified policy and the raw provider response was retained. Historical unbound evidence cannot make that claim.

## Adversarial checks

The focused suite rejects symbolic profile names used as policy identity, policy digest tampering, duplicate resource profiles, exact overlap between required and optional selected paths, exact overlap between required container and selected paths, duplicate selected paths, missing required-container path semantics, missing omission semantics, retrospective binding upgrades, unsupported precommit claims, historical omitted-field completeness/replayability claims, observation/profile mismatch, and binding digest tampering.

It also positively checks that a container path may be a prefix of an explicitly selected child path without authorizing the entire subtree.

## Self-qualification

HAK-013 has an E5-target qualification plan and a HAK-010 precommitted obligation-to-step binding policy. The dedicated workflow is `.github/workflows/hak-normalization-policy.yml`.

A green focused run qualifies only HAK-013's normalization-policy bookkeeping claims. It does not authenticate GitHub, retroactively qualify HAK-012, establish the historical HAK-007 normalization as precommitted, prove selector execution semantics, or prove semantic truth.

Any hosted run on an earlier HAK-013 head is historical evidence only after this path-semantics correction.

## Non-claims

HAK-013 does not authenticate provider responses, execute selector/path semantics against raw provider data, preserve historical raw response bytes that were never retained, prove historical omitted-field completeness, infer precommit timing from current Git history alone, make a policy digest equivalent to provider attestation, make projection completeness equivalent to source completeness, grant runtime authority, or certify scientific/governance/consent/robotics behavior.

## Next boundary

Issue #1035 defines the next prospective collection layer:

```text
RawProviderResponse
+
ExactNormalizationPolicy
+
VersionedSelectorInterpreter
-> NormalizationExecutionReceipt
-> ProviderSourceObservation
```

That layer must define deterministic selector grammar semantics and establish that the exact content-bound policy was actually executed.

Separately, provider source lineage still needs to preserve:

```text
CanonicalResourceIdentity != RetrievalRepresentation
NormalizationPolicyIdentity != ObservationLineage
NewerObservation != ErasureOfCounterevidence
```
