# MATH-RET-CONVERGENCE-001B — Structural Compatibility Wire v1

Status: frozen compatibility-wire contract  
Authority: `MeasurementOnly`

## Why this exists

`MATH-RET-CONVERGENCE-001A` freezes what a relocation receipt must prove. This
tranche freezes **how the compared values become bytes** before they are hashed.

That distinction matters because the predecessor Q0 harness prints similarities
with decimal formatting. Text such as `0.812345` is useful for a human report,
but it is not a durable identity for an IEEE-754 result. Compiler/library
formatting, rounding, and display precision must not determine whether an
extraction counts as bit-for-bit compatible.

The compatibility profile is:

`math-structural-compat-wire-v1`

All commitments use SHA-256, matching Symthaea's existing evidence convention.

## General primitives

All integer lengths are unsigned 32-bit little-endian.

A UTF-8 string is encoded as:

```text
u32_le(byte_length) || utf8_bytes
```

Every wire type begins with a domain separator ending in a NUL byte. Domain
separation prevents a byte sequence valid for one evidence type from being
reinterpreted as another.

No digest in this profile is computed from Rust `Debug`, CSV, pretty JSON, or
human-formatted floating-point text.

## HDC vector wire

Domain:

```text
MATH-HDC-V1\0
```

Payload:

```text
exact BinaryHV.0 bytes in array order
```

The payload must contain exactly 2,048 bytes. `BinaryHV` already exposes its
backing `[u8; 2048]`, so no endian conversion or bit reordering is permitted.

Commitment:

```text
SHA256(domain || raw_2048_bytes)
```

## Canonical sparse map wire

Domain:

```text
MATH-AST-SPARSE-V1\0
```

The predecessor is a `BTreeMap<String, f64>`. Serialize:

```text
domain
u32_le(entry_count)
for key in lexicographic UTF-8 key order:
    lp_utf8(key)
    f64::to_bits(value).to_le_bytes()
```

V1 sparse values must be finite, non-negative, integer-valued `f64` counts no
larger than `2^53`. This matches the predecessor feature construction, where
`bump` increments counts by exactly `1.0`, and rejects accidental semantic drift
into weighted/fractional features under the same representation identity.

Map insertion order is deliberately irrelevant. Key/value content is not.

## Candidate ranking wire

Domain:

```text
MATH-RANKING-V1\0
```

Serialize:

```text
domain
u32_le(candidate_count)
lp_utf8(candidate_id_0)
...
lp_utf8(candidate_id_n)
```

Candidate IDs must be unique. **Order is committed.** This makes deterministic
tie order part of the compatibility theorem rather than an incidental display
choice.

## HDC similarity transcript wire

Domain:

```text
MATH-HDC-SIM-TRANSCRIPT-V1\0
```

For records in frozen evaluation order:

```text
domain
u32_le(record_count)
for record:
    lp_utf8(case_id)
    lp_utf8(candidate_id)
    f32::to_bits(similarity).to_le_bytes()
```

Similarity must be finite and in `[0, 1]`.

The exact `f32` bits are committed. Decimal output is not an input to this
commitment.

## Canonical-AST cosine transcript wire

Domain:

```text
MATH-AST-COSINE-TRANSCRIPT-V1\0
```

For records in frozen evaluation order:

```text
domain
u32_le(record_count)
for record:
    lp_utf8(case_id)
    lp_utf8(candidate_id)
    f64::to_bits(cosine).to_le_bytes()
```

Cosine must be finite and in `[0, 1]` for the non-negative predecessor feature
vectors.

Again, exact `f64` bits are committed instead of formatted decimal strings.

## Blind holdout representation aggregate

Domain:

```text
MATH-HOLDOUT-REPR-COMPAT-V1\0
```

The holdout compatibility artifact intentionally excludes case IDs, candidate
IDs, positive labels, similarities, rankings, margins, and scores.

For cases in frozen source order, with query first and candidates following in
frozen source order, serialize only representation commitments:

```text
domain
u32_le(case_count)
for case:
    u32_le(representation_count)
    for representation:
        raw_32_bytes(hdc_sha256)
        raw_32_bytes(sparse_sha256)
```

This proves that predecessor and extracted representations produce the same
bytes over the held-out source without producing a scientific result from the
holdout.

It is a compatibility check, not evaluation.

## Outer receipts

The extraction receipt itself may continue to use canonical JSON + SHA-256 in
the existing Symthaea evidence style, because its compatibility digests are
hex strings/booleans/integers rather than raw floating-point measurements.

The *measurement identities inside that receipt* must come from this wire
profile, not from JSON float rendering.

## Reference vectors

`data/benchmarks/math_structural_compat_wire_vectors_v1.json` freezes independent
reference commitments for:

- all-zero 2,048-byte HDC;
- all-`ff` 2,048-byte HDC;
- a two-feature sparse map supplied in deliberately noncanonical input order;
- ranking `[a1, a2, a3]`;
- two exact `f32` HDC similarities;
- two exact `f64` AST cosines;
- a label-free two-case holdout aggregate.

The reference implementation
`.github/scripts/math_structural_compat_wire_v1.py` verifies these commitments
and negative controls without third-party dependencies.

## Negative controls

The reference implementation rejects:

- HDC vectors that are not exactly 2,048 bytes;
- fractional, negative, non-finite, or inexact-large sparse counts;
- duplicate ranking IDs;
- non-finite/out-of-range similarities;
- duplicate transcript coordinates;
- holdout records containing labels or IDs;
- malformed/non-lowercase fixed-width bit/digest hex.

It also verifies that reordering sparse-map input does not change its digest,
while reordering a candidate ranking **does** change the ranking digest.

## Rust implementation requirement

The extraction tranche should implement these bytes directly in Rust and use
`f32::to_bits()` / `f64::to_bits()`; it should not shell out to this Python
reference implementation in production or runtime code.

The Python implementation exists as an independent qualification oracle. A
Rust implementation must reproduce every frozen reference digest exactly before
it can be used to populate a `math-structural-extraction-receipt-v1` receipt.

## Nonclaims

A matching wire digest establishes only equality under this serialization
contract. It does not establish theorem truth, proof validity, representation
quality, HDC advantage, generalization, production readiness, or novelty.
