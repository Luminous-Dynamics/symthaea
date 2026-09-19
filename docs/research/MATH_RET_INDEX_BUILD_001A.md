# MATH-RET-INDEX-BUILD-001A — Transparent Exact Index Build Receipt

Status: draft qualification contract. Authority: `MeasurementOnly`.

## Purpose

MATH-RET-SOURCE-COVERAGE-001A proves that every frozen candidate can produce every required representation object. It deliberately does not prove that a retrieval index actually contains those objects.

This tranche closes that gap for the first scientific retrieval campaigns by standardizing a transparent exact-scan artifact whose content can be independently audited byte-for-byte.

## Why exact + transparent first

For Q0/Q1 representation experiments, hidden ANN/index-building behavior is an unnecessary causal variable.

The initial research artifact therefore uses:

```text
search_mode = ExactDeterministic
build_mode  = ExactDeterministicMaterializedScan
```

and stores one canonical representation payload for every frozen source object.

Optimized or approximate indices remain possible later, but they require a separate qualification lineage demonstrating equivalence/recall properties. They must not silently replace this exact research baseline.

## Exact index artifact

`math-retrieval-exact-index-artifact-v1` contains:

```text
index_id
candidate_set_sha256
target_id
representation_sha256
item_serialization_sha256
payload_encoding = Base64
item_order = SourceObjectDigestAscending
item_count
items[]
```

Each item contains:

```text
source_object_sha256
representation_object_sha256
serialized_bytes
payload_base64
```

Base64 is only the JSON transport. The decoded bytes are the exact canonical representation serialization.

The validator requires:

```text
len(decoded payload) == serialized_bytes
SHA256(decoded payload) == representation_object_sha256
decoded payload is valid UTF-8
```

and then requires the digest/size pair to equal the corresponding representation entry in the qualified coverage artifact.

## Build receipt

`math-retrieval-index-build-receipt-v1` binds:

```text
coverage_sha256
candidate_set_sha256
index_manifest_sha256
index_artifact_sha256
target_id
index_build_policy_sha256
builder_implementation_sha256
toolchain_manifest_sha256
index_seed
input_set_sha256
build_mode
```

`input_set_sha256` is SHA-256 over canonical JSON bytes of the exact ordered list:

```text
[
  {
    source_object_sha256,
    representation_object_sha256,
    serialized_bytes
  },
  ...
]
```

with object keys sorted, compact separators, source rows already in ascending digest order, and one final newline.

This creates a compact identity for the exact logical index inputs independently of the larger artifact bytes.

## Qualification chain

The validator consumes:

1. build receipt;
2. source-coverage artifact;
3. candidate-set artifact;
4. experiment manifest;
5. retrieval-index manifest;
6. exact index artifact.

It first re-runs source-coverage qualification and index-manifest validation. It then requires:

```text
SHA256(coverage bytes) == receipt.coverage_sha256
SHA256(index manifest bytes) == receipt.index_manifest_sha256
SHA256(index artifact bytes) == receipt.index_artifact_sha256
index.index_artifact_sha256 == SHA256(index artifact bytes)
```

The coverage target must match the index manifest for:

```text
channel
representation_family
representation_sha256
item_serialization_sha256
max_serialized_item_bytes
```

The artifact source list must exactly equal the candidate-set source list.

Every artifact representation digest and byte count must exactly equal the corresponding target entry in MATH-RET-SOURCE-COVERAGE-001A.

## Causal discipline

This makes the first real S/H/N experiment deliberately boring at the index layer:

```text
same candidate set
same exact-scan artifact structure
same deterministic source order
same retrieval evidence path

only representation bytes/scoring metric differ as preregistered
```

That is preferable to introducing HNSW/IVF/ANN behavior while trying to measure whether the mathematical representation itself helped.

## Required adversarial canaries

The validator self-test rejects at least:

- payload-byte mutation;
- missing index item;
- reordered index items;
- substituted representation digest;
- substituted coverage identity;
- substituted target ID;
- substituted build policy;
- changing the index to approximate search;
- substituted input-set identity.

## What a PASS means

A PASS establishes:

```text
this exact deterministic index artifact
contains these exact canonical representation bytes
for exactly this frozen candidate population
under these exact representation/build identities
```

It does not establish:

- that the representation is useful;
- that rankings are relevant;
- HDC advantage;
- normal-form advantage;
- proof success;
- theorem truth or novelty.

Runtime ranking/fusion/packing remains governed by the guarded retrieval seam. Scientific usefulness remains an experimental result.

## Next tranche

Once the frozen encoders converge under #4367, the first real index implementation should materialize the S target into this exact artifact format and qualify it. H should then reuse the same builder/artifact structure with only the preregistered representation/scoring intervention changed.

Approximate indices should not be introduced until exact S/H results justify the engineering need and an ANN qualification protocol freezes recall and determinism requirements.
