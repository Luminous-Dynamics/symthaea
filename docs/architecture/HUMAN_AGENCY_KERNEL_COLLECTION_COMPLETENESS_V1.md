# HAK-017 — Collection / Population Completeness v1

Status: candidate audit/evidence tooling. No runtime authority changes.

## Purpose

HAK-016 makes selector coverage precise across contexts retained in a HAK-014 receipt. That does not establish whether the retained contexts exhaust the population the collector intended to observe.

```text
CompleteAcrossRetainedContexts
!=
DeclaredCompleteWithinScope
!=
VerifiedCompleteWithinScope
!=
VerifiedCompleteProviderPopulation
```

HAK-017 adds a separate **completeness declaration** record so these claims cannot collapse.

## Anti-oracle boundary

HAK-017 does not resolve provider evidence refs or replay a provider pagination transcript. Therefore it must not call its own declarations verified completeness.

```text
ExhaustionKindLabel
+
NonEmptyEvidenceRef
!=
VerifiedCollectionExhaustion
```

and:

```text
DeclaredComplete
!=
VerifiedComplete
```

The v1 validator proves only that the declaration is internally consistent with its declared scope, pagination model, exhaustion classification, count provenance, canonical profile identity, and digest.

A later traversal/provider-evidence layer must resolve and verify those evidence refs before any stronger completeness claim is made.

## Provider motivation

GitHub REST list endpoints may paginate responses. A response can expose a provider-reported total while still requiring continuation traversal under the provider pagination contract.

Therefore:

```text
ProviderTotalCountObserved
!=
AllPagesRetrieved

CountEquality
!=
VerifiedCollectionExhaustion
```

Counts reconcile a traversal declaration; they do not prove traversal exhaustion.

## Artifact

The v1 artifact is:

```text
hak.collection-completeness.v1
```

implemented by:

```text
scripts/hak_collection_completeness.py
```

with schema:

```text
docs/architecture/hak/collection-completeness-v1.schema.json
```

The record binds:

- exact HAK-015 canonicalization profile identity;
- provider/resource/collection identity;
- query and declared-scope identity;
- population extent;
- narrowing constraints;
- pagination model and contract reference;
- observed page count and page size where known;
- declared exhaustion kind and evidence references;
- retained raw/unique/duplicate counts;
- named deduplication identity and rule;
- provider-reported total value plus exact denominator scope, where available;
- collection declaration state and reasons;
- HAK canonical digest.

## Scope definition versus scope breadth

```text
ScopeDefinitionCompleteness
!=
ScopeBreadth
```

A collection can be declared complete over a precisely defined subset without representing the provider's full population.

HAK-017 separates:

```text
population_extent =
    FullProviderPopulation
    | DefinedSubset
    | Unknown
```

from independent constraints:

```text
filtered
sampled
time_windowed
permission_limited
```

A `DefinedSubset` may carry narrowing constraints and can still be `DeclaredComplete` when the record declares that exact subset exhausted.

For example:

```text
population_extent = DefinedSubset
time_windowed = true
collection_status = DeclaredComplete
```

means:

> the record declares the exact time-windowed scope complete under its bound pagination/exhaustion metadata.

It does **not** mean the provider population outside that window was observed, nor that provider exhaustion has been independently verified.

Conversely, `FullProviderPopulation` must not carry narrowing constraints.

```text
DeclaredCompleteDefinedSubset
!=
DeclaredCompleteProviderPopulation
```

## Declared completeness within the scope

`collection_status = DeclaredComplete` requires:

```text
population_extent != Unknown
pagination.model != Unknown
model-compatible non-Unknown exhaustion declaration
non-empty exhaustion evidence refs
count reconciliation where a scope-compatible provider total is available
```

Conceptually:

```text
InternallyConsistentDeclaredComplete
=
DefinedScope
+
KnownPaginationSemantics
+
CompatiblePositiveExhaustionDeclaration
+
ReferencedEvidenceMetadata
+
CountConsistency
```

This is still weaker than:

```text
VerifiedCollectionComplete
```

because HAK-017 does not resolve the evidence refs or validate a page-by-page provider transcript.

## Pagination and exhaustion compatibility

The initial pagination vocabulary is:

```text
LinkHeader
PageNumber
Cursor
NonPaginated
Unknown
```

Positive exhaustion declarations are:

```text
NoContinuationByProviderContract
CursorExhausted
NonPaginatedEndpoint
ProviderExplicitComplete
```

plus `Unknown` for no positive exhaustion claim.

The v1 compatibility matrix is intentionally narrow:

```text
LinkHeader | PageNumber
    -> NoContinuationByProviderContract
    -> ProviderExplicitComplete

Cursor
    -> CursorExhausted
    -> ProviderExplicitComplete

NonPaginated
    -> NonPaginatedEndpoint
    -> ProviderExplicitComplete
```

`Unknown` pagination cannot support `DeclaredComplete`.

A known pagination model requires a non-empty contract ref. `Unknown` must not pretend to carry one.

Positive exhaustion declarations require non-empty evidence refs. `Unknown` exhaustion carries no positive evidence refs.

```text
EvidenceRefPresent
!=
EvidenceRefVerified
```

## Provider-population declared completeness

The helper:

```text
provider_population_declared_complete(record)
```

returns true only for a valid record satisfying:

```text
collection_status == DeclaredComplete
population_extent == FullProviderPopulation
all narrowing constraints == false
```

A declared-complete filtered, sampled, time-windowed, or permission-limited subset remains explicitly non-provider-wide.

Again:

```text
ProviderPopulationDeclaredComplete
!=
ProviderPopulationVerifiedComplete
```

## Count provenance and conservation

The count model distinguishes values from the semantics needed to interpret them.

```text
ProviderTotalCountValue
!=
ProviderTotalCountScope

UniqueCount
!=
DeduplicationRuleIdentity
```

HAK-017 records:

```text
retained_raw_count
retained_unique_count
duplicate_count

deduplication.identity_ref
deduplication.rule_ref

provider_reported_total = null
| {
    value,
    scope_ref
  }
```

It requires arithmetic conservation:

```text
retained_raw_count
==
retained_unique_count + duplicate_count
```

The deduplication identity/rule must be named so `unique` is not treated as a self-explanatory number. V1 does **not** prove the named deduplication rule was actually executed correctly.

When a provider total is present, its `scope_ref` must equal the exact declared HAK-017 `scope_ref` before the total may participate in reconciliation.

```text
ProviderTotalValueWithoutScopeBinding
!=
QualifiedDenominator
```

For a scope-compatible provider total:

```text
retained_unique_count <= provider_reported_total.value
```

and a `DeclaredComplete` record additionally requires equality.

But equality never proves exhaustion.

```text
CountEquality
!=
CollectionExhaustion
```

## Honest non-green states

The status vocabulary is:

```text
DeclaredComplete
Partial
Unknown
Failed
NotApplicable
```

The old unqualified label:

```text
Complete
```

is intentionally invalid in v1.

`Partial`, `Unknown`, `Failed`, and `NotApplicable` require explicit reasons.

```text
NotApplicable
-> reason required
```

and:

```text
IncompleteCollectionEvidence
!=
InvalidEvidence
```

## Composition with HAK-016

HAK-016 coverage remains coverage across retained terminal contexts only.

A stronger claim over the exact declared scope would require at minimum:

```text
HAK016Coverage = Present
+
HAK017CollectionStatus = DeclaredComplete
+
ExactSourceScopeCompatibility
+
VerifiedCollectionTraversal
```

A provider-wide claim would additionally require:

```text
HAK017ProviderPopulationDeclaredComplete = true
```

plus the same traversal verification.

HAK-017 v1 does **not** qualify either:

```text
ExactSourceScopeCompatibility
```

or:

```text
VerifiedCollectionTraversal
```

Those belong to the next layer.

## Canonicalization identity

HAK-017 binds the inherited HAK-015 profile with:

```text
profile_id
exact Git artifact ref
raw profile-file SHA-256
```

This preserves:

```text
ProfileName
!=
ProfileArtifactIdentity
```

The collection digest uses domain:

```text
hak.collection-completeness.v1
```

through HAK-015 canonical evidence encoding.

## Qualification boundary

The E5-target qualification plan covers only:

- internally consistent `DeclaredComplete` semantics;
- scope/population distinction;
- rejection of the old unqualified `Complete` label;
- rejection of unknown population extent for `DeclaredComplete`;
- rejection of `FullProviderPopulation` with narrowing constraints;
- pagination/exhaustion model compatibility;
- presence and internal consistency of exhaustion evidence refs, not their external verification;
- arithmetic raw/unique/duplicate conservation;
- named deduplication identity/rule provenance;
- exact scope binding for provider-reported totals;
- exact HAK-015 profile binding and digest integrity;
- explicit reason requirement for `NotApplicable`;
- schema/linter behavior for committed cases.

The qualification claim explicitly does **not** establish actual provider exhaustion.

## Historical boundary

HAK-012/014/016 artifacts that did not retain pagination/query/scope provenance are not retroactively upgraded.

```text
FutureCompletenessModel
!=
RetroactiveHistoricalCompleteness
```

Likewise, earlier HAK-017 staging heads that used the word `Complete` do not establish the final `DeclaredComplete` contract.

## Non-claims

HAK-017 does not:

- resolve or verify exhaustion evidence refs;
- authenticate provider data;
- prove provider-reported totals are truthful;
- prove the named deduplication rule executed correctly;
- prove snapshot stability while multiple pages are traversed;
- qualify the HAK-016/017 source-scope join;
- establish verified collection completeness;
- establish semantic significance of a field;
- establish scientific truth;
- establish governance/legal legitimacy;
- grant runtime authority;
- upgrade HAK-014/015/016 evidence tiers.

```text
DeclaredCompleteness
!=
VerifiedCompleteness
!=
ProviderTruth
!=
SemanticTruth
!=
Authority
```
