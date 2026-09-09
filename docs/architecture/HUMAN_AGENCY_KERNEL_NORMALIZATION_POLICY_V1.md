# HAK-013 — Content-Bound Normalization Policy v1

Status: architecture + audit-only tooling candidate. No runtime authority changes.

## Purpose

HAK-012 proves how materialized snapshot fields map to normalized provider-source observations. HAK-013 closes the immediately prior boundary: what normalization policy selected those source fields in the first place?

```text
NormalizationProfileRef != NormalizationPolicyIdentity
ProfileName != PrecommittedSelectionPolicy
ProjectionCompleteness != SourceNormalizationCompleteness
ContentBoundPolicy != PrecommittedPolicy
PresenceRequirement != FieldSelection
ContainerPresence != ContainerType
ObservedShape != ExpectedShape
PolicyPathSyntax != PolicyPathSemantics
```

## Policy identity

The policy content is bound by a domain-separated SHA-256 digest over canonical JSON excluding the digest field. The symbolic `profile_ref` remains a compatibility/human-facing name; `policy_id + policy_digest` is the content identity.

HAK-013 also binds the selector grammar itself:

```text
selector_grammar = {
  id: "hak.selector-path",
  version: 1
}
```

A content-bound path string is not enough if two interpreters can assign different semantics to that string.

```text
PolicyPathBytes != PolicyPathSemantics
```

A future normalization execution receipt must therefore use the grammar identity required by the policy rather than selecting a grammar after collection.

## Typed path semantics

HAK-013 separates three path categories:

```text
required_containers
    = RequireTypedContainerAndSelectShapeOnly

required_paths
    = SelectValueAndRequirePresence

optional_paths
    = SelectValueIfPresent
```

A required container declaration carries both the provider path and expected container type:

```text
{ "path": "steps", "container_type": "array" }
```

This proves more than presence. A provider schema drift from array to object must not be silently accepted by a future executor.

`required_containers` permits preserving the container's shape, including an empty array/object, but does **not** authorize copying arbitrary descendants. Descendants remain governed by required/optional selected paths.

For workflow-job steps the policy therefore says conceptually:

```text
required_containers = [
  { path: "steps", container_type: "array" }
]
required_paths = []
optional_paths = [
  "steps[*].name",
  "steps[*].status",
  "steps[*].conclusion",
  "steps[*].number",
  "steps[*].started_at",
  "steps[*].completed_at"
]
```

Thus `steps=[]` can be retained faithfully without turning `steps` into permission to copy every future nested step property.

Exact overlap across container/required/optional categories is forbidden. A typed container path may intentionally be a prefix of selected descendants.

The selector grammar is now content-bound, but actual selector execution remains deferred to #1035:

```text
NormalizationPolicyDefined != NormalizationPolicyExecutedByCollector
```

## Historical HAK-007 boundary

The historical HAK-007 source observation remains `RetrospectiveUnbound` with no policy identity, no established commitment relation, no retained raw provider response, no replayability claim, and no omitted-field completeness claim.

```text
PolicyExistsNow != PolicyWasPrecommittedThen
```

No later HAK-013 policy may retroactively upgrade that historical evidence.

## Replayability and completeness

```text
BoundToPolicy != RawResponseRetained
RawResponseRetained != ProviderAuthenticated
```

`EstablishedWithinPolicyScope` requires a content-bound policy plus retained raw source evidence; the historical fixture cannot make that claim.

## Adversarial qualification contract

The focused HAK-013 suite covers policy/profile identity separation, policy digest tampering, selector-grammar substitution, duplicate resource profiles, selected-path overlap, typed-container/selected-path overlap, invalid container types, missing typed-container semantics, duplicate selected paths, omission-semantics loss, retrospective binding upgrades, unsupported precommit claims, historical completeness/replayability inflation, profile mismatch, and binding digest tampering.

It also positively checks that typed container prefixes may coexist with explicit descendant selectors.

The E5-target qualification plan and HAK-010 obligation-to-step policy precommit these checks before hosted execution. Any hosted result from an earlier HAK-013 head is historical only after a semantic correction.

## Non-claims

HAK-013 does not execute provider selectors, authenticate GitHub, retain historical raw bytes that were never captured, prove historical omitted-field completeness, infer historical precommit timing, establish semantic truth, or grant runtime authority.

## Next boundary

Issue #1035 defines the prospective execution chain:

```text
RawProviderResponse
+ ExactNormalizationPolicy
+ VersionedSelectorInterpreter
-> NormalizationExecutionReceipt
-> ProviderSourceObservation
```

That layer must prove both expected container types and selected-field semantics were actually enforced under `hak.selector-path` v1.
