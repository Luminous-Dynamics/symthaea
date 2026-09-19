# MATH-RET-REP-WIRE-001A — Deterministic Representation Wire Formats

Status: draft qualification contract. Authority: `MeasurementOnly`.

## Purpose

The retrieval contracts already freeze representation identities and exact index artifact bytes, but `item_serialization_sha256` is intentionally opaque. Independent scoring replay cannot decode an opaque representation without importing the encoder implementation, which would weaken causal independence.

This tranche freezes two representation-neutral wire formats for the first structural S/H/N campaigns. It does **not** change either frozen encoder.

## Canonical sparse integer wire

`math-canonical-sparse-wire-v1` contains:

```text
version = math-canonical-sparse-wire-v1
feature_order = FeatureIdUtf8Ascending
features = [{ feature_id, count }, ...]
```

Rules:

- at least one feature;
- feature IDs are canonical printable ASCII identifiers matching `[A-Za-z0-9_.:/-]+`;
- IDs are strictly ascending and unique;
- counts are positive integers in `1..=2^31-1`;
- exact wire bytes are canonical JSON with sorted object keys, compact separators, UTF-8, and one final newline.

This matches the current v1 canonical-AST baseline's count semantics without embedding any AST traversal rule in the wire contract.

## BinaryHV 16K wire

`math-binary-hdc-wire-v1` contains:

```text
version = math-binary-hdc-wire-v1
dimension_bits = 16384
byte_count = 2048
bit_semantics = BinaryHVRawByteArray
payload_encoding = StandardBase64
bytes_base64 = <canonical padded Base64 of exactly 2048 bytes>
```

The 2,048-byte shape is the existing `BinaryHV` memory representation: 16,384 binary dimensions packed into `[u8; 2048]`.

No lossy truncation, tiling, quantization, textual bit expansion, or alternate dimension is permitted in this wire version.

## Canonical-byte rule

Semantic JSON equivalence is not enough. The validator requires the exact input bytes to equal:

```text
json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
```

conceptually, expressed identically in other languages.

This gives representation objects a stable content digest independent of whitespace, map iteration order, or pretty-printing defaults.

## Why integer/exact representations matter

These formats allow the next scorer-replay tranche to avoid floating-point authority entirely.

For canonical sparse vectors, feature counts are non-negative integers. Since the query norm is common across candidates, cosine ranking can be compared exactly using integer products rather than rounded `f32/f64` scores.

For BinaryHV, ranking can use exact Hamming distance:

```text
popcount(query_bytes XOR candidate_bytes)
```

with source digest as the frozen tie break. Reporting a conventional normalized similarity is optional measurement; it need not determine ordering.

## Adversarial canaries

The validator rejects:

- sparse feature reorder;
- duplicate sparse feature IDs;
- zero/negative sparse counts;
- noncanonical feature IDs;
- wrong HDC dimension/byte count;
- non-2048-byte HDC payload;
- non-standard Base64 mode;
- noncanonical JSON byte serialization.

## Integration

After #4367 converges the frozen example-local encoders into reusable library code, compatibility canaries should require:

```text
encoder output -> wire serializer -> exact bytes
```

and preserve representation behavior from the qualified Q0 lineage.

The resulting wire contract identity is then bound through #4382 source coverage and #4388 exact index build receipts.

## Nonclaims

A valid wire object says only that representation bytes have the declared canonical shape. It does not prove the encoder is correct, that two formulas are equivalent, that retrieval is relevant, that HDC is advantageous, or that any theorem is true/proved/novel.
