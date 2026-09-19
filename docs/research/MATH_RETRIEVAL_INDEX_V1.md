# MATH-RET-001A — Mathematical Retrieval Index Contract v1

Status: preregistration / qualification contract for representation-specific retrieval indices.

Authority: `MeasurementOnly`.

This contract exists because equal context/compute budgets are necessary but not sufficient for a fair retrieval comparison. Two arms can obey the same item/byte ceiling and still receive materially different information through candidate eligibility, duplicate handling, truncation, approximate search, or tie-breaking.

## Core law

```text
same budget != same information access

index score != mathematical truth
nearest neighbor != proof
normal-form neighbor != theorem equivalence authority
retrieval refusal != mathematical falsehood
```

## Candidate universe

Every index binds:

- corpus snapshot digest;
- knowledge-boundary digest;
- candidate-eligibility policy digest;
- exact candidate-set digest;
- exact candidate count.

The v1 candidate universe requires:

```text
exclude_query_source_item = true
exclude_solution_artifacts = true
source_identity_kind = SourceObjectDigest
dedup_policy = OneEntryPerSourceObject
canonical_candidate_order = SourceObjectDigestAscending
```

The candidate-set digest is taken after eligibility filtering and source-object deduplication.

Representation-specific indices intended for the same experiment should therefore share the same candidate-set digest and count. A later cross-index validator should enforce that equality against the experiment manifest.

## Why source-object dedup happens before indexing

The same theorem/problem/artifact may appear through aliases, paths, generated views, or multiple representations. If one arm indexes duplicates while another does not, repeated copies can inflate nearest-neighbor probability.

v1 therefore permits one candidate entry per source object.

Different representations of that object belong in different representation-specific indices or fusion channels, not as duplicate source candidates inside one index.

## Leakage exclusions

The exact query source is excluded from its own candidate universe.

For blind solved-problem transfer, solution artifacts are also excluded. A theorem/problem statement may be eligible under the frozen knowledge-boundary policy, but its hidden proof/solution artifact cannot leak into retrieval.

The candidate-eligibility policy digest remains authoritative for finer domain-specific exclusions.

## Representation identity

An index binds one channel:

```text
None
Syntax
ExactNormalForm
```

and one family:

```text
None
Lexical
CanonicalSparse
HDC
```

`None` is reserved for deterministic random-retrieval controls.

Lexical representations are syntax-only.

Exact-normal-form indices additionally bind the exact normalization-contract and normalization-implementation digests. Syntax indices are forbidden from carrying those fields so an index manifest cannot imply that normalization was consumed when it was not.

## Item serialization

Every candidate representation binds:

- item-serialization digest;
- UTF-8 byte encoding;
- maximum serialized item size;
- oversize policy.

v1 allows:

```text
RejectItem
DeterministicTruncate
```

If truncation is used, a truncation-policy digest is mandatory.

If rejection is used, a truncation digest is forbidden.

This makes byte accounting reproducible and prevents one arm from receiving larger candidate payloads through a different serializer.

## Negative-control transformations

The same index contract represents control indices with:

```text
None
RandomRetrieval
ShuffledHdcVectors
PermutedChallengeAssociations
```

Every non-None control transformation binds:

- deterministic control seed;
- control-artifact digest.

Examples:

- shuffled HDC binds the exact shuffled-vector artifact;
- permuted associations bind the exact permutation artifact;
- deterministic random retrieval binds its generated control artifact.

Random retrieval consumes no representation channel and must use `RandomDeterministic` scoring.

## Index construction

Every index binds:

- exact index-build-policy digest;
- exact built-index artifact digest;
- index seed;
- maximum supported top-k;
- scoring metric;
- scoring-policy digest;
- score-precision policy digest;
- query-normalization policy digest.

Supported scoring labels are:

```text
Cosine
HammingSimilarity
Jaccard
BM25
ExactMatch
RandomDeterministic
Custom
```

The label is descriptive only; the scoring-policy digest binds the exact implementation/parameters.

## Determinism

v1 requires:

```text
deterministic = true
tie_break = SourceObjectDigestAscending
```

Final retrieval order is:

```text
ScoreDescendingThenSourceObjectDigest
```

This matters because tied floating-point or binary-HV similarities are common. Insertion-order tie-breaking would make corpus/index build order an unregistered intervention.

## Exact versus approximate search

Two search modes are allowed:

```text
ExactDeterministic
ApproximateDeterministic
```

Exact search may not carry an ANN validation field.

Approximate search must preregister:

- approximation-validation policy digest;
- minimum exact Recall@k floor.

An approximate index is therefore not silently treated as equivalent to exact retrieval. The actual ANN-vs-exact audit remains evidence to be executed later.

## Provenance and byte accounting

Retrieved items must retain source provenance.

Byte accounting is over canonical UTF-8 representation bytes.

Partial items are forbidden:

```text
partial_item_policy = RejectWholeItem
```

An experiment-level packer may stop when adding the next complete item would exceed the shared byte ceiling. It may not slice one arm's final candidate in a way another arm does not.

## Relationship to MATH-EXP-001 v2

MATH-EXP-001 v2 arm entries carry an `index_manifest_sha256`.

That digest should bind an instance of this contract.

The experiment manifest holds shared global budgets and causal contrasts. This contract holds representation-index mechanics.

The intended chain is:

```text
experiment shared candidate/corpus boundary
                  |
        MATH-RET-001 index manifest
                  |
       representation-specific index
                  |
            ranked candidates
                  |
         shared experiment packer
                  |
              retrieval arm
```

## Required next cross-index gate

Before causal evaluation, add a small cross-index validator that loads every arm index manifest and verifies that comparable arms share:

- corpus snapshot;
- knowledge boundary;
- candidate eligibility policy;
- candidate-set digest;
- candidate count;
- query/self/solution exclusion;
- source identity/dedup/canonical ordering.

Representation/scoring/index identities are allowed to differ because those are the intervention.

That cross-manifest gate is intentionally separate from this single-index contract.

## Adversarial validator tests

The stdlib validator rejects:

- solution leakage;
- duplicate-preserving candidate policy;
- syntax indices smuggling normalizer identity;
- insertion-order tie breaks;
- top-k greater than candidate count;
- deterministic truncation without a policy digest;
- approximate search without a recall-audit policy/floor.

It additionally accepts explicit exact-normal-form and properly audited approximate-index fixtures.

## Nonclaims

A valid index manifest does not establish that:

- the index artifact was built correctly;
- an ANN index meets its promised recall floor;
- one representation retrieves better neighbors;
- a retrieved neighbor is mathematically equivalent;
- HDC is useful;
- normalization is useful;
- any theorem is true or novel.

It freezes the information-access mechanics required to interpret later retrieval experiments.
