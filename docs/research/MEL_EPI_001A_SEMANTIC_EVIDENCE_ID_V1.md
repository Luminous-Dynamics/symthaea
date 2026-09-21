# MEL-EPI-001A — Semantic Evidence Identity V1

Status: source declaration only. This document defines the first tranche of the shared Melothaea epistemic kernel tracked by #5275. It grants no executable qualification by itself.

## Purpose

Melothaea already distinguishes source-native evidence, structural validation, storage admission, and profile-relative evidence. This tranche adds one narrow common primitive: an exact semantic identity contract for evidence-like artifacts that can be referenced across provenance, attestation, registration, and analysis layers without promoting serialization shape into source authority.

## Core theorem

```text
runtime/trial ID
!= serialized bytes
!= structural validation
!= semantic evidence identity
!= source authority
!= producer authentication
!= scientific truth
```

A semantic evidence identity names one exact typed semantic statement under one exact namespace/schema/profile/record identity. It does not decide whether that statement is true, admissible, consented, causal, generalizable, or musically good.

## Types/crypto separation

The shared `symthaea-epistemic-types` crate freezes the exact canonical SHA-256 **preimage** and the identifier shape. It deliberately does not add a cryptographic dependency merely to hash that preimage.

The source-authoritative adapter must:

```text
native evidence
 -> native validation / canonical rederivation
 -> source-specific semantic payload
 -> semantic_sha256_preimage(...)
 -> reviewed SHA-256 provider in the owning layer
 -> EvidenceSemanticIdV1::from_sha256_digest(...)
```

Therefore:

```text
EvidenceSemanticIdV1 parses/validates
!= digest recomputed
!= source evidence validated
```

A source-specific adapter that grants authority must recompute both the native evidence and the digest.

## V1 identity shape

```text
EvidenceSemanticIdV1 {
    digest_algorithm = "sha256"
    transcript_version = "melothaea-semantic-transcript-v1"
    namespace
    schema_version
    profile_id
    record_id
    digest_hex
}
```

Evidence with no source profile uses the explicit profile identity `none`; an empty profile is invalid.

## Canonical SHA-256 preimage

V1 does not hash arbitrary Serde JSON and does not depend on object/map iteration order.

```text
DOMAIN = UTF8("melothaea:semantic-evidence:v1")

identity_transcript =
    field_utf8(1, namespace)
 || field_utf8(2, schema_version)
 || field_utf8(3, profile_id)
 || field_utf8(4, record_id)
 || field_bytes(5, semantic_payload)

sha256_preimage = DOMAIN || identity_transcript
```

Each field is:

```text
tag:u16-be || kind:u8 || len:u64-be || payload:len-bytes
```

V1 kinds:

```text
0x01 = UTF-8 text
0x02 = opaque bytes
0x03 = unsigned integer, minimal big-endian magnitude
0x04 = signed integer, sign byte + minimal big-endian magnitude
0x05 = finite f64 bits, canonicalized below
0x06 = ordered sequence: count:u64-be || (len:u64-be || item)*
0x07 = boolean (exact byte 0 or 1)
0x08 = 32-byte digest reference
```

Maps are deliberately not a generic primitive. Domain adapters project unordered structures into canonically sorted sequences under their own schema semantics.

## Numeric rules

- NaN and infinities are rejected.
- `-0.0` normalizes to `+0.0`.
- finite f64 is encoded as canonical IEEE-754 binary64 big-endian bits after zero normalization.
- unsigned/signed integer magnitude is minimal; redundant leading zero bytes are forbidden by construction.
- exact-rational domains encode exact numerator/denominator integer fields rather than converting through floating point.

## Text and collection rules

- text is exact UTF-8 bytes; V1 performs no silent Unicode normalization.
- a source schema may define normalization, but that choice belongs to the schema/profile identity.
- sequences preserve order.
- unordered semantic sets must be sorted by the source adapter using a schema-defined comparator.
- duplicate set members must be rejected unless the source schema explicitly defines multiset semantics.

## Why the shared crate freezes preimages instead of implementing SHA-256

The workspace already contains multiple reviewed crypto-bearing boundaries, including the formal-safety receipt attestation contract. Making the shared *types* crate own yet another crypto implementation/provider would create unnecessary dependency and authority coupling.

The preimage contract gives every source adapter the same bytes while allowing the owning domain/Xenia/formal-safety layer to perform cryptographic work with its reviewed provider.

This is also deliberately distinct from `symthaea-evidence-plane::config_hash`, which is documented as a non-cryptographic `DefaultHasher`/Debug fingerprint and must never be promoted into semantic evidence identity.

## Source-level adversarial controls

The implementation specifies tests for:

1. equivalent semantic payloads -> identical SHA-256 preimage;
2. namespace or record substitution -> changed preimage;
3. semantic payload change -> changed preimage;
4. duplicate/non-increasing field tags rejected;
5. JSON whitespace/key ordering cannot affect identity because JSON is not directly hashed;
6. NaN/inf rejected;
7. `-0.0` and `+0.0` canonicalize identically;
8. exact numerator/denominator can remain integer semantics without float conversion;
9. malformed/uppercase SHA-256 text rejected;
10. unsupported transcript version rejected;
11. supplied digest equality can be checked exactly;
12. semantic-ID shape can never substitute for native validation/rederivation.

## Explicit nonclaims

A valid `EvidenceSemanticIdV1` does not establish:

- that its digest was honestly computed unless the source adapter independently recomputes it;
- persistence or durability;
- producer identity or authentication;
- chronology or trusted time;
- preregistration;
- participant consent or export authority;
- randomization quality or concealment;
- listener attention;
- analysis eligibility;
- statistical significance;
- causal inference;
- population/generalization claims;
- listener preference;
- artistic or musical quality;
- product authority.

## Successor boundary

MEL-EPI-001B (#5284) may place this semantic ID inside source references and exact/projected preservation receipts. MEL-EPI-001C (#5285) may then add positive closed-world claim scopes. Neither successor may treat a generic semantic ID or projection as source evidence on shape alone.
