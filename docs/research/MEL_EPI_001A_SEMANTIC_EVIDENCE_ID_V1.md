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

A source-specific adapter that grants authority must recompute both the native evidence commitment and the semantic digest.

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
0x08 = SHA-256 digest reference, exactly 32 digest bytes
```

Field kind `0x08` is **not** a generic 32-byte digest slot. It must never be reinterpreted as BLAKE3-256 or another algorithm merely because that algorithm also emits 32 bytes. A future digest algorithm requires a distinct schema-defined field kind or a later transcript version.

Maps are deliberately not a generic primitive. Domain adapters project unordered structures into canonically sorted sequences under their own schema semantics.

## Numeric rules

- NaN and infinities are rejected.
- `-0.0` normalizes to `+0.0`.
- finite f64 is encoded as canonical IEEE-754 binary64 big-endian bits after zero normalization.
- unsigned/signed integer magnitude is minimal; redundant leading zero bytes are forbidden by construction.
- exact-rational domains encode exact numerator/denominator integer fields rather than converting through floating point.

## Text and collection rules

- text is the exact supplied UTF-8 byte sequence.
- V1 performs no Unicode normalization, whitespace trimming, or case folding.
- therefore canonically equivalent Unicode spellings remain distinct unless the frozen source schema explicitly normalizes them before transcript construction.
- sequences preserve order.
- unordered semantic sets must be sorted by the source adapter using a schema-defined comparator.
- duplicate set members must be rejected unless the source schema explicitly defines multiset semantics.

## Frozen cross-implementation vector

The following vector covers every V1 field kind and the complete identity wrapper. It exists so independent Rust, Xenia, or other adapters can prove byte-for-byte agreement rather than merely agree on relational properties.

Identity:

```text
namespace      = melothaea.vector
schema_version = vector-v1
profile_id     = source-native
record_id      = record-0001
```

Semantic payload:

```text
tag 1 UTF-8                  = "melody"
tag 2 bytes                  = 00 ff
tag 3 unsigned integer       = 0x0102
tag 4 signed integer         = -258
tag 5 finite f64             = 1.5
tag 6 ordered sequence       = ["a", "bc"]
tag 7 boolean                = true
tag 8 SHA-256 digest ref     = 11 repeated 32 times
```

Frozen lengths:

```text
semantic_payload_len = 169 bytes
full_preimage_len    = 303 bytes
```

Frozen full preimage hex:

```text
6d656c6f74686165613a73656d616e7469632d65766964656e63653a763100010100000000000000106d656c6f74686165612e766563746f720002010000000000000009766563746f722d7631000301000000000000000d736f757263652d6e6174697665000401000000000000000b7265636f72642d3030303100050200000000000000a900010100000000000000066d656c6f6479000202000000000000000200ff00030300000000000000020102000404000000000000000301010200050500000000000000083ff8000000000000000606000000000000001b00000000000000020000000000000001610000000000000002626300070700000000000000010100080800000000000000201111111111111111111111111111111111111111111111111111111111111111
```

Independent reference SHA-256 over those exact 303 bytes:

```text
39fcacd18445633886e551976663f0acf9dcc0f8dd50a69a2fbdefeef315b651
```

The shared types crate need not compute that digest at runtime. The digest is a cross-implementation reference for reviewed crypto-bearing adapters.

## Why the shared crate freezes preimages instead of implementing SHA-256

The workspace already contains reviewed crypto-bearing boundaries, including Muse's RustCrypto SHA-256 evidence layer and the formal-safety receipt attestation contract. Making the shared *types* crate own another crypto implementation/provider would create unnecessary dependency and authority coupling.

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
12. semantic-ID shape can never substitute for native validation/rederivation;
13. composed/decomposed Unicode remains byte-distinct without source-defined normalization;
14. one complete 303-byte vector freezes every V1 field kind and identity wrapper;
15. field kind `0x08` is SHA-256-specific and cannot silently mean another 32-byte digest algorithm.

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

MEL-EPI-001B (#5284) may bind this semantic ID to an independently recomputed source-native commitment and an explicit preservation profile. MEL-EPI-001C (#5285) may then add positive closed-world claim scopes bound to that exact admitted source reference. Neither successor may treat a generic semantic ID or projection as source evidence on shape alone.
