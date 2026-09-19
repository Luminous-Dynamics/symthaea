# MATH-RET-RANK-REPLAY-001A — Exact Independent Ranking Replay

Status: draft mechanical qualification contract.

## Purpose

Prove that a runtime retrieval ranking is exactly the deterministic result of the frozen metric over the exact content-addressed query and index representation bytes.

This closes the gap between:

```text
"the backend returned these source IDs"
```

and:

```text
"an independent scorer recomputed the same top-k from exact frozen bytes"
```

## Inputs

Replay consumes:

1. a replay receipt;
2. a scoring-policy bundle;
3. a query-representation receipt + PASS report;
4. exact canonical query-wire bytes;
5. a qualified exact-index-build PASS report;
6. the exact retrieval-index manifest;
7. the exact transparent index artifact;
8. the runtime retrieval trace;
9. the trace-validation PASS report.

The replay receipt binds the exact bytes of every input.

## Executable policy digests

The index manifest already contains:

```text
scoring_policy_sha256
score_precision_policy_sha256
query_normalization_policy_sha256
```

v1 turns these from opaque hashes into executable commitments.

The policy bundle has three canonical closed subobjects. Replay canonicalizes each subobject independently and requires its SHA-256 to equal the corresponding manifest digest.

### Scoring

Common invariants:

```text
candidate_population = AllEligibleFromExactIndex
query_source_exclusion = Required
threshold_policy = None
tie_break = SourceObjectDigestAscending
top_k_policy = MinRequestedKEligibleCount
```

There is intentionally no `similarity > 0.1` or other threshold.

### Precision

Ordering may not use floating point.

For canonical sparse cosine, candidate ordering is exact integer comparison. With query norm constant across candidates:

```text
candidate A outranks B iff

dot(A)^2 * norm_sq(B)
>
dot(B)^2 * norm_sq(A)
```

Equal values tie by `SourceObjectDigestAscending`.

For BinaryHV, replay computes raw Hamming distance:

```text
Σ popcount(query_byte XOR candidate_byte)
```

and orders by minimum distance, then source digest.

This avoids ambiguity among displayed Hamming-similarity conventions.

### Query normalization

v1 allows only:

```text
CanonicalWireValidationOnly
transform = None
```

No query-side rescaling, pruning, thresholding, feature filtering, or hidden normalization is allowed.

## Query provenance

Replay requires MATH-RET-QUERY-REP-001A.

The trace query ID/source digest must equal the query receipt, and the query representation/serialization identities must equal the selected index.

For an `ExactNormalForm` index, the query receipt must also bind the exact same normalization contract and implementation as the index manifest.

## Trace selection

One replay receipt qualifies exactly one ranking:

```text
SingleIndex
Syntax
ExactNormalForm
```

For fusion traces, Syntax and ExactNormalForm input rankings are replayed independently. Fusion ordering remains qualified separately by the existing trace/fusion replay theorem.

## Exact return-count law

For every selected index:

```text
eligible =
    all exact-index items
    minus query source if present

returned_count =
    min(requested_k, len(eligible))
```

Returning fewer items because similarity is "too low" is a qualification failure.

## Replay witness

The validator emits a content-addressable witness for every eligible candidate.

Sparse witness:

```text
source_object_sha256
dot
candidate_norm_sq
```

HDC witness:

```text
source_object_sha256
hamming_distance
```

The PASS report binds the witness digest and the digest of the complete expected ranking, while also recording expected/runtime top-k explicitly.

## Authority boundary

A replay PASS establishes deterministic scoring/ranking equivalence only.

It does not establish:

- retrieval usefulness;
- semantic or mathematical equivalence;
- HDC benefit;
- fusion benefit;
- proof success;
- theorem truth;
- novelty.

Authority remains `MeasurementOnly`.

## Adversarial self-test

The local integration fixture:

1. constructs exact sparse query/index bytes;
2. produces a correct runtime ranking and replays it;
3. rebinds all outer hashes around a deliberately permuted runtime ranking and requires rejection;
4. rebinds all outer hashes around a threshold-like early-truncated ranking and requires rejection;
5. separately checks exact Hamming popcount and rejects any non-`None` threshold or floating-point ordering policy.

## Next step

After this contract qualifies, the remaining representation/runtime convergence work under #4367 can connect the real canonical-sparse S backend first.

That backend must then produce a runtime trace whose S ranking passes this independent replay without changing:

- candidate population;
- query/index representation bytes;
- scoring policy;
- budgets;
- materializer/packer path;
- legacy memory isolation.

Only after real S qualifies should the real structural-HDC H backend be introduced as a one-variable representation/scoring successor.
