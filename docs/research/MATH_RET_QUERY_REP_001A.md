# MATH-RET-QUERY-REP-001A — Content-Addressed Query Representation Receipt

Status: draft prerequisite for independent ranking replay.

## Purpose

A ranking replay is only meaningful if the bytes being scored are demonstrably the representation of the same query named by the runtime trace.

This receipt binds:

```text
query identity
    +
source-object identity
    +
representation input-stage identity
    +
representation/serialization identity
    +
producer/toolchain identity
    +
exact canonical query-wire bytes
```

without claiming that the representation is mathematically correct or useful.

## Receipt

`math-retrieval-query-representation-v1` records:

- `query_id`;
- `query_source_object_sha256`;
- `source_object_contract_sha256`;
- `input_stage`;
- `input_object_sha256`;
- optional exact-normal-form contract/implementation identities;
- `representation_sha256`;
- `item_serialization_sha256`;
- `wire_kind`;
- producer implementation and toolchain identities;
- exact query-wire SHA-256 and byte length.

Allowed input stages:

```text
SourceObject
ParsedFolFormulaExt
ExactNormalForm
```

An `ExactNormalForm` query must bind both the normalization contract and implementation. Syntax/source queries are forbidden from carrying those fields.

## Wire validation

The receipt does not define another representation serialization.

It delegates exact bytes to MATH-RET-REP-WIRE-001A:

```text
CanonicalSparseV1 -> math-canonical-sparse-wire-v1
BinaryHV16K       -> math-binary-hdc-wire-v1
```

The query-wire SHA and exact byte length are recomputed from the supplied canonical wire file.

## Why this is needed

Without this receipt an independent scorer could accidentally or maliciously score:

```text
trace says query = Q1
replay bytes = representation(Q2)
```

and still reproduce a backend ranking.

The replay chain therefore becomes:

```text
trace query identity
      ||
query-representation receipt
      ||
exact query-wire bytes
      ↓
independent scorer
```

## Authority boundary

A valid receipt proves provenance/binding only.

It does not establish:

- encoder correctness;
- algebraic equivalence;
- retrieval relevance;
- HDC benefit;
- proof success;
- theorem truth;
- novelty.

Authority remains `MeasurementOnly`.

## Qualification

The lightweight workflow compiles the existing wire validator and this validator, runs both adversarial self-tests, and does not invoke monorepo Rust CI.

The self-test rejects wrong wire hashes/lengths, invalid wire kinds, malformed producer digests, normalization-field leakage into syntax queries, and missing normalizer identity for exact-normal-form inputs.

## Next gate

MATH-RET-RANK-REPLAY-001A consumes a valid query-representation report together with:

- qualified exact-index-build report;
- valid retrieval-trace report;
- exact index manifest/artifact;
- exact query wire;
- frozen scorer policy.

It then independently recomputes the complete eligible ordering and requires exact top-k equality with the runtime trace.
