# MATH-RET-RUNTIME-001C — Candidate Membership Guard

Status: isolated executable candidate stacked on MATH-RET-CANDIDATE-001A.

This package closes a runtime integrity gap between a content-addressed candidate-set commitment and the actual source identities returned by a representation-specific retrieval backend.

## Law

A qualified backend may choose only among identities in the exact frozen eligible universe:

```text
backend result source ID
        ↓
request candidate_set_sha256 == guard candidate_set_sha256
request candidate_count      == guard cardinality
        ↓
source ID ∈ frozen candidate set
        ↓
ONLY THEN RetrievalExecutor
        ↓
materialization / packing / evidence
```

The guard checks raw single-index results or both raw fusion-channel rankings. This happens before deterministic fusion, so duplicate removal cannot hide an illegal upstream identity.

Query-specific exclusions remain a separate invariant in the parent retrieval seam. The candidate set is the static eligible universe; `query_source_object_sha256` must still be excluded from each runtime ranking by the parent executor.

## Package boundary

`MembershipGuardBackend<B>` implements the existing `QualifiedRetrievalBackend` trait and wraps any backend without changing its implementation.

The guard does not know about:

- HDC similarity;
- lexical scoring;
- normal-form scoring;
- Phi;
- mathematical truth;
- evidence score;
- memory retention.

It proves only candidate-universe membership and exact universe identity.

## Qualification

```bash
cargo fmt \
  --manifest-path tools/math-retrieval-runtime-guard/Cargo.toml \
  -- --check

cargo check \
  --manifest-path tools/math-retrieval-runtime-guard/Cargo.toml \
  --locked

cargo test \
  --manifest-path tools/math-retrieval-runtime-guard/Cargo.toml \
  --locked
```

## Mechanical canaries

The tests require:

1. candidate-set digest mismatch rejects before the inner backend runs;
2. candidate-count mismatch rejects before the inner backend runs;
3. a legal single-index ranking passes unchanged;
4. an illegal single-index identity is rejected;
5. an illegal identity in either fusion channel is rejected before fusion;
6. an illegal identity reaches neither canonical materialization nor evidence sinks;
7. duplicate identities in the frozen universe are rejected instead of silently deduplicated.

## Authority boundary

Passing this package establishes only that source identities are members of the frozen eligible universe before downstream retrieval execution proceeds. It does not establish relevance, mathematical equivalence, theorem truth, proof success, HDC benefit, normal-form benefit, or novelty.
