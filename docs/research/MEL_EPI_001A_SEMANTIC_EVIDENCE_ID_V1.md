# MEL-EPI-001A — Semantic Evidence Identity V1

Status: source declaration only. This document defines the first tranche of the shared Melothaea epistemic kernel tracked by #5275. It grants no executable qualification by itself.

## Purpose

Melothaea already distinguishes source-native evidence, structural validation, storage admission, and profile-relative evidence. This tranche adds one narrow common primitive: an exact semantic identity for evidence-like artifacts that can be referenced across provenance, attestation, registration, and analysis layers without promoting serialization shape into source authority.

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

A semantic evidence identity identifies one exact typed semantic statement under one exact namespace and schema. It does not decide whether that statement is true, admissible, consented, causal, generalizable, or musically good.

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

The namespace, schema version, profile ID, and record ID are part of the semantic identity rather than display-only metadata.

## Canonical transcript

V1 does not hash arbitrary Serde JSON and does not depend on object/map iteration order.

The authoritative transcript is a sequence of typed, length-delimited fields:

```text
DOMAIN = UTF8("melothaea:semantic-evidence:v1")

transcript =
    field_utf8(1, namespace)
 || field_utf8(2, schema_version)
 || field_utf8(3, profile_id)
 || field_utf8(4, record_id)
 || field_bytes(5, semantic_payload)

digest = SHA-256(DOMAIN || transcript)
```

Each field is encoded as:

```text
tag:u16-be || kind:u8 || len:u64-be || payload:len-bytes
```

V1 kinds:

```text
0x01 = UTF-8 text
0x02 = opaque bytes
0x03 = unsigned integer, minimal big-endian magnitude
0x04 = signed integer, sign byte + minimal big-endian magnitude
0x05 = finite f64 bits, canonicalized as specified below
0x06 = sequence of complete encoded fields/items
0x07 = boolean (exact byte 0 or 1)
0x08 = semantic digest reference
```

Maps are deliberately not a primitive in V1. Domain adapters must project maps into a canonically sorted sequence under their own schema semantics.

## Numeric rules

- NaN and infinities are rejected before transcript construction.
- `-0.0` is normalized to `+0.0` before f64 encoding.
- finite f64 values are encoded as canonical IEEE-754 binary64 big-endian bits after zero normalization.
- unsigned and signed integer magnitudes are minimal: no redundant leading zero bytes.
- a domain using exact rational semantics must encode numerator/denominator as exact integer fields rather than convert through floating point.

## Text and collection rules

- text is exact UTF-8 bytes; V1 performs no Unicode normalization silently.
- domain schemas may require a normalization profile explicitly, but that profile must be part of schema/profile identity.
- sequences preserve order.
- unordered semantic sets must be sorted by the domain adapter before transcript construction according to a schema-defined comparator.
- duplicate members in a semantic set must be rejected unless the source schema explicitly defines multiset semantics.

## Construction boundary

Generic transcript helpers may construct bytes and digests, but source authority remains with source-specific adapters.

```text
native evidence
    -> source-native validation/rederivation
    -> source-specific semantic projection
    -> canonical transcript
    -> EvidenceSemanticIdV1
```

A caller must not be able to manufacture source authority merely by constructing a shape-valid generic transcript.

## Required adversarial controls

The implementation tranche should include regressions for:

1. same semantic fields -> identical digest;
2. changed namespace -> different digest;
3. changed schema/profile/record identity -> different digest;
4. changed field order -> different digest where order is semantic;
5. unordered-domain adapter canonical sorting -> same digest for equivalent source maps/sets;
6. arbitrary JSON whitespace/key order cannot affect the digest because JSON is not hashed directly;
7. NaN/inf rejected;
8. `-0.0` and `+0.0` normalize to one semantic identity;
9. exact rational adapter identity is stable without f64 conversion;
10. malformed digest text / wrong digest length rejected on deserialization/validation;
11. unknown transcript version rejected;
12. semantic ID equality cannot substitute for source-native validation.

## Reuse boundary

This tranche should reuse existing reviewed SHA-256 machinery where doing so does not import unrelated domain authority. It should not move fabrication, recruitment, comparison-storage, or FORM-specific semantics into a generic crate merely to deduplicate names.

## Explicit nonclaims

A valid `EvidenceSemanticIdV1` does not establish:

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

MEL-EPI-001B may use this semantic ID inside generic source references and exact/projected preservation receipts. MEL-EPI-001C may then add a positive closed-world claim scope. Neither successor may treat semantic identity as source evidence on shape alone.
