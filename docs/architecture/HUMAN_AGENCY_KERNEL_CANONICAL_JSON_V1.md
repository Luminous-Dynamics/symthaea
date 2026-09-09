# HAK-015 — Cross-Runtime Canonical Evidence Encoding v1

Status: audit/evidence tooling candidate. No runtime authority changes.

## Purpose

HAK uses digest-bound evidence artifacts across qualification plans, receipts, provider observations, conformance records, and interpretations. Earlier HAK tranches deliberately used the repository's existing Python convention:

```text
json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

That convention is deterministic inside Python, but it is not a named cross-runtime wire contract.

```text
PythonDeterministicSerialization
!=
CrossImplementationCanonicalization
```

HAK-015 defines a narrow canonical JSON profile for future HAK evidence metadata and tests it independently in Python and Rust.

## Base standard

`hak.canonical-json.v1` is an RFC-8785-compatible subset. RFC 8785/JCS supplies the important cross-platform rules: recursive object sorting by UTF-16 code units, array-order preservation, strict JSON/I-JSON input discipline, primitive serialization without extra whitespace, and UTF-8 output.

HAK v1 deliberately narrows the numeric surface rather than implementing ECMAScript floating-point serialization in every validator.

## Profile identity is executable content

The profile is represented by the machine-readable artifact:

```text
docs/architecture/hak/canonical-json-v1.profile.json
```

The string `hak.canonical-json.v1` is only a profile identifier. It is not sufficient evidence that an implementation follows the profile.

```text
ProfileName
!=
ProfileContent
```

Both runtime lanes must validate the committed profile manifest against the semantics they implement. A profile-content change under the same symbolic identifier must therefore fail the focused contract rather than silently acquiring new meaning.

Python requires full equality between the loaded profile object and its implementation contract. The Rust lane independently checks all required semantic values **and exact object shape** through `tools/hak_canonical_profile_contract_rust.rs`; unknown root or nested semantic fields are rejected rather than ignored.

```text
KnownFieldsMatch
!=
ExactProfileContract
```

The machine-readable contract binds, at minimum:

- base standard;
- UTF-8 input/output;
- duplicate-name rejection;
- Unicode-scalar string domain;
- safe-integer bounds;
- floating-point prohibition;
- negative-zero behavior;
- UTF-16 object-key ordering;
- recursive object sorting;
- array-order preservation;
- whitespace and escaping rules;
- no Unicode normalization;
- SHA-256 digest algorithm;
- profile/domain-separated digest preimage;
- historical migration semantics.

## HAK v1 value domain

Allowed values:

```text
null
boolean
Unicode scalar string
safe integer
array
object with unique string keys
```

Integers are restricted to:

```text
-9007199254740991 .. 9007199254740991
```

Floating-point values are not permitted in HAK canonical evidence metadata v1. Exponent-form numbers such as `1e3` are also outside the v1 representation even when mathematically integral.

```text
NumericallyEquivalent
!=
EncodingEquivalent
```

This is not a claim that scientific measurements should avoid floating point. Scientific/domain payloads should carry an explicit numeric representation and units contract rather than silently inheriting evidence-metadata digest semantics.

## Strict input rules

Before canonicalization:

- raw bytes must be valid UTF-8;
- duplicate object names are rejected;
- lone Unicode surrogates are rejected;
- floating-point and exponent-form numbers are rejected;
- `NaN`, `Infinity`, and `-Infinity` are rejected;
- integers outside the safe interoperable range are rejected.

No decoder replacement-character repair is permitted for invalid UTF-8 evidence bytes.

```text
ParserAccepted
!=
CanonicalProfileValid
```

## Canonical serialization

Objects are sorted recursively by the raw property-name UTF-16 code-unit sequence, matching RFC 8785 rather than host-language code-point ordering.

Arrays retain exact element order.

Strings use JSON/JCS-compatible escaping and are emitted as UTF-8 **without Unicode normalization**. Canonically equivalent-looking Unicode sequences therefore remain byte-distinct when their scalar sequences differ.

```text
VisualOrCanonicalUnicodeEquivalence
!=
EvidenceByteIdentity
```

No insignificant whitespace is emitted.

Negative integer zero canonicalizes to `0`.

## Digest contract

HAK-015 separates canonical profile identity from hash algorithm identity.

For the initial SHA-256 binding:

```text
preimage =
UTF8("hak.canonical-json.v1")
|| 0x00
|| UTF8(domain)
|| 0x00
|| canonical_bytes
```

The domain must be non-empty and contain no NUL byte.

```text
CanonicalProfileIdentity
!=
HashAlgorithmIdentity
```

An evidence digest therefore needs both the canonicalization profile/version and the hash/domain semantics to be interpretable.

## Cross-runtime golden corpus

A single shared golden corpus is consumed by both reference implementations:

```text
docs/architecture/hak/golden/hak-canonical-json-v1.vectors.json
```

Positive vectors cover:

- ASCII object reordering;
- recursive nested sorting;
- the RFC-8785-relevant UTF-16 ordering case where an astral emoji sorts before U+FB33;
- control/string escaping;
- safe integer boundaries;
- `-0 -> 0`;
- array-order preservation;
- preservation of composed and decomposed Unicode as distinct byte sequences.

Negative/adversarial coverage includes:

- duplicate names;
- floating-point input;
- exponent-form numeric input;
- unsafe positive/negative integers;
- `NaN`/`Infinity`;
- lone-surrogate input;
- invalid UTF-8 bytes;
- accidental Unicode normalization;
- machine-readable profile drift;
- unknown root/nested profile fields accepted by the Rust lane;
- code-point ordering substituted for UTF-16 ordering;
- profile identity omitted from the digest preimage.

The Python reference and dependency-free Rust reference consume the same corpus but do not share a parser or canonicalizer implementation. Both must produce the same canonical UTF-8 bytes and profile-bound SHA-256 digests for positive vectors and reject the committed negative corpus. The Rust step additionally executes the exact-shape profile validator.

```text
SemanticValueSameAcrossSupportedRuntimes
-> CanonicalBytesEqual
-> DigestEqual
```

The implication is only tested for the committed HAK v1 profile and golden/adversarial corpus. It is not a universal proof over every possible Unicode/JSON input or every future implementation language.

```text
GoldenCorpusAgreement
!=
UniversalCanonicalizationProof
```

## Migration rule

Existing HAK and Symthaea digests produced using historical Python `sort_keys=True` serialization remain historically valid under their original domain/version semantics.

They MUST NOT be silently reinterpreted as HAK-015 canonical digests.

```text
OldDigestValidHistorically
!=
OldDigestUsesNewCanonicalProfile
```

Future migration should introduce explicit new digest/profile fields or new artifact schema versions. A migration itself is a provenance-bearing transformation and must not erase the original digest semantics.

## Qualification contract

HAK-015 has a self-declared E5-target qualification plan and a HAK-010 obligation-to-step binding policy. Its focused workflow checks the exact PR head, compiles the Python reference, validates the machine-readable profile/vector syntax and Python profile contract, executes Python golden/adversarial vectors, compiles and executes the independent Rust reference plus exact-shape profile validator against the same profile/corpus, and validates the plan/binding bookkeeping.

A passing hosted run may qualify only the tested HAK-015 canonicalization/tooling contract on that exact head.

In particular:

```text
PythonPass
!=
CrossRuntimeQualified
```

and:

```text
CrossRuntimeGoldenAgreement
!=
SemanticClaimTrue
```

## Non-claims

HAK-015 does not:

- prove semantic truth;
- authenticate an evidence provider;
- establish that a digest-bearing artifact is safe or legitimate;
- prove portability beyond the named implementations/profile/corpus actually exercised;
- migrate historical digests automatically;
- define scientific floating-point representation;
- grant runtime authority.

```text
CanonicalBytesEqual
!=
SemanticClaimTrue
!=
Authority
```
