# HAK-018 — Retained Traversal Verification v1

Status: candidate audit/evidence tooling. No runtime authority changes.

## Purpose

HAK-017 makes collection-completeness declarations internally precise while deliberately refusing to call referenced provider exhaustion evidence verified.

HAK-018 v1 takes the next bounded step: it deterministically verifies the structure and deterministic transforms of a retained traversal transcript and binds that transcript to the exact loaded HAK-017 `DeclaredComplete` witness.

```text
VerifiedAgainstRetainedTranscript
!=
ProviderAuthenticatedTraversal
```

The verifier proves facts about the retained transcript it was given. It does not prove that the transcript faithfully reproduces provider response bytes.

## Artifact

The verification record is:

```text
hak.collection-traversal-verification.v1
```

implemented by:

```text
scripts/hak_verified_traversal.py
```

with schema:

```text
docs/architecture/hak/collection-traversal-verification-v1.schema.json
```

The record carries:

- exact inherited HAK-015 canonicalization profile identity;
- retained traversal transcript + transcript digest;
- exact HAK-017 declaration binding;
- deterministic dedup execution receipt;
- page-count replay result;
- explicit anti-oracle boundary fields;
- HAK-018 traversal digest.

## Mandatory external HAK-017 witness

The embedded `hak017_binding` is not allowed to prove itself.

Validation requires the exact HAK-017 record as an external witness, and that record must itself validate as:

```text
collection_status = DeclaredComplete
```

Therefore:

```text
EmbeddedUpstreamBinding
!=
ExactUpstreamWitness

SameLookingRefs
!=
ExactUpstreamDeclaration
```

The witness digest and collection/query/scope/pagination fields must all match. A second valid HAK-017 record with the same human-readable refs but a different digest is rejected.

Binding an exact HAK-017 declaration still does not turn that declaration into provider-verified completeness.

## Retained traversal transcript

The transcript binds:

```text
provider
resource_kind
collection_ref
query_scope_ref
scope_ref
pagination_model
pagination_contract_ref
temporal declaration
ordered pages
```

Each page retains:

```text
ordinal
page_id
request_ref
continuation_in
response_ref
raw_response_digest
continuation_out
observation bounds
entity_ids
```

These are retained evidence fields. Their presence does not prove provider authentication or source-content correspondence.

## Page-chain verification

HAK-018 v1 verifies these deterministic graph properties:

```text
ordinals == 1..N
page_id unique
request_ref unique
response_ref unique
first continuation_in == null
page[i+1].continuation_in == page[i].continuation_out
non-final continuation_out != null
final continuation_out == null
continuation_out tokens do not repeat
```

For `NonPaginated`, exactly one page is allowed and both continuation directions are null.

This establishes:

```text
ContinuationEdgeConsistency
```

but not:

```text
ContinuationSourceVerification
```

A producer capable of coherently rewriting both ends of an edge can preserve local equality. HAK-018 v1 does not replay provider response bytes to prove where a continuation token originated.

Therefore:

```text
continuation_source_verification = NotEstablished
```

## Source-content anti-oracle boundary

The record fixes all of these to `NotEstablished`:

```text
provider_authentication
raw_response_content_verification
entity_projection_verification
continuation_source_verification
temporal_snapshot_verification
```

This preserves:

```text
RawResponseDigestPresent
!=
RawResponseBytesReplayed

EntityIdsRetained
!=
EntityProjectionVerified

ContinuationTokenRetained
!=
ContinuationSourceVerified

ProviderIdentityNamed
!=
ProviderAuthenticated

SequentialObservationBounds
!=
TemporalSnapshotConsistency
```

A future provider-source layer may strengthen those joins. HAK-018 v1 must not pre-claim them.

## Observation bounds

Each page carries:

```text
observed_after_or_at
observed_before_or_at
```

HAK-018 verifies:

```text
start <= end
next.start >= previous.end
```

This proves internal chronology of the retained transcript only. It does not establish clock authenticity or population stability between requests.

```text
SequentialTraversal
!=
OneConsistentSnapshot
```

## Deterministic dedup execution

HAK-017 only names dedup provenance. HAK-018 v1 executes one deliberately narrow rule:

```text
identity_ref = hak.entity-id.literal-string.v1
rule_ref     = hak.dedup.stable-first-occurrence.v1
```

All retained entity identifiers are flattened in traversal order. The first exact string occurrence is retained and later exact matches are counted as duplicates.

The result records:

```text
input_count
unique_count
duplicate_count
retained_unique_ids
receipt_digest
```

Validation recomputes that result rather than trusting it.

Thus:

```text
DedupRuleNamed
!=
DedupRuleExecuted
```

is closed for this exact rule over the retained entity-id transcript.

It does not prove the entity IDs were correctly projected from provider bytes.

## HAK-017 count/scope join

Strong validation checks the exact HAK-017 witness for:

```text
collection_digest
provider
resource_kind
collection_ref
query_scope_ref
scope_ref
pagination_model
pagination_contract_ref
pages_observed
raw/unique/duplicate counts
dedup identity/rule
```

The replayed page count and dedup counts must equal the HAK-017 declaration.

```text
HumanReadableScopeEquality
!=
ExactSourceScopeJoin
```

## Deterministic replay

Validation recomputes:

```text
transcript structure
transcript digest
dedup execution
page count
HAK-017 join
traversal digest
```

This preserves:

```text
InternallyRedigestedArtifact
!=
DeterministicallyReplayEquivalentArtifact
```

A coherently redigested but false dedup result is still rejected.

## Retry boundary

HAK-018 v1 models accepted logical pages only and rejects duplicate page/request identities.

It does not yet model retry/attempt lineage.

```text
DuplicatePage
!=
QualifiedRetry
```

A retry-aware version must represent attempts explicitly rather than flattening retries into repeated pages.

## Composition with HAK-016

HAK-018 v1 does not produce population-qualified selector coverage.

HAK-016 binds coverage to an exact HAK-014 normalization receipt. A future composition layer must reconstruct:

```text
HAK016 coverage
-> exact HAK014 receipt/input lineage
-> exact HAK018 traversal source
-> exact HAK017 scope
```

and prove coverage-set conservation across every population-contributing traversal source.

The critical theorem remains:

```text
CoverageSourceIncludedInTraversal
!=
CoverageExhaustsTraversalPopulation
```

One page with `Present` coverage plus a verified retained traversal of many pages is not population-wide `Present`.

## Qualification boundary

The E5-target qualification plan is:

```text
hak018-retained-traversal-e5-v1
```

with plan digest:

```text
sha256:a20bc7e7e6a50b4915cb4579f95f4ef912656b894c398a5512393d16bc26a4dd
```

and HAK-010 binding-policy digest:

```text
sha256:c826571fec10d1fc3bdbb4d32c381353ad10cda85a57d26a7f0f3ed633c75ed1
```

The dedicated workflow binds:

```text
3  Assert exact-head checkout
6  Compile HAK-018 tooling
7  Validate HAK-018 schema syntax
8  Lint HAK-018 qualification plan
9  Lint HAK-018 check-binding policy
10 Run HAK-018 regressions
```

A passing hosted run may qualify only the exact committed retained-transcript verification contract. It does not upgrade HAK-017 or any earlier layer.

## Non-claims

HAK-018 v1 does not:

- authenticate provider responses;
- replay raw response bytes;
- prove continuation tokens came from provider response content;
- prove entity IDs were correctly projected from provider responses;
- resolve HAK-017 exhaustion evidence refs;
- establish actual provider exhaustion;
- establish temporal snapshot stability;
- model retry lineage;
- establish HAK-016 coverage-set completeness;
- establish population-qualified field coverage;
- establish semantic/scientific truth;
- establish governance/legal legitimacy;
- grant runtime authority.

```text
RetainedTranscriptVerification
!=
ProviderTruth
!=
SemanticTruth
!=
Authority
```
