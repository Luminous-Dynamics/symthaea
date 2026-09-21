# MEL-EPI-001B — Source Reference and Semantic Preservation V1

Status: source-staged, non-authority-bearing substrate only. Depends on the complete MEL-EPI-001A code+spec subject #5319 and does not assume that subject has executable qualification.

Tracker: #5284

## Purpose

MEL-EPI-001B binds one exact source-native commitment to one exact MEL-EPI semantic identity while making semantic preservation/loss explicit.

It does **not** replace source-native validators and does not allow a generic parsed object to become source evidence.

Core theorem:

```text
shape-valid EvidenceSourceRefV1
!= source-native evidence
!= source-native commitment recomputed
!= MEL-EPI semantic digest recomputed
!= admission into any positive claim scope
```

A source-specific adapter must independently establish both commitment paths before the reference can participate in stronger authority.

## Canonical machine codes

001B introduces `NamespacedCodeV1` as a shared identity primitive for commitment profiles, semantic loss codes, and later positive claim codes.

V1 grammar:

```text
1..128 bytes
lower-case ASCII only
alphanumeric tokens separated by '.', '-' or '_'
no leading/trailing separator
no adjacent separators
```

Human-readable labels/descriptions are intentionally not part of the machine identity.

Examples:

```text
muse.methodology.commitment-v1
muse.loss.display-metadata
qualification.engineering.rust-tests-pass
```

## Source-native commitment

```text
SourceNativeCommitmentV1 {
    scheme_id: NamespacedCodeV1,
    sha256_hex: 64 lower-case hex chars,
}
```

`scheme_id` identifies the source-native commitment theorem/profile, not merely the cryptographic algorithm.

Therefore:

```text
same SHA-256 algorithm
+ different source commitment theorem
!= same SourceNativeCommitmentV1
```

For example, a canonical JSON commitment over a methodology record and a canonical JSON commitment over an unblinding receipt must use distinct scheme identities even when both are implemented with the same `canonical_json_sha256` helper.

## Evidence source reference

```text
EvidenceSourceRefV1 {
    source_ref_version,
    semantic_id: EvidenceSemanticIdV1,
    native_commitment: SourceNativeCommitmentV1,
    preservation: SemanticPreservationV1,
}
```

The source identity fields are **not duplicated** beside `semantic_id`; namespace/schema/profile/record already belong to the exact semantic identity. Avoiding duplicated truths removes substitution/mismatch states.

## Preservation ceiling

V1 defines:

```text
SemanticPreservationV1::LosslessUnderProfile

SemanticPreservationV1::ProjectedWithLoss {
    loss_codes: sorted unique non-empty Vec<NamespacedCodeV1>
}
```

`LosslessUnderProfile` means the frozen source adapter/profile declares no known semantic loss for the semantics it is designed to carry.

It does **not** mean:

```text
byte-identical serialization
all conceivable semantics preserved
future profiles must make the same judgment
```

`ProjectedWithLoss` requires at least one explicit loss code. Loss codes are canonical, strictly sorted, and unique so ordering cannot become accidental semantic entropy.

## Two commitments remain independent

The intended source-specific admission path is:

```text
native artifact
   ├─ native validator / canonical rederivation
   │       ↓
   │  native commitment theorem
   │       ↓
   │  SourceNativeCommitmentV1
   │
   └─ source-specific semantic projection
           ↓
      MEL-EPI 001A preimage
           ↓
      independently computed SHA-256
           ↓
      EvidenceSemanticIdV1

both verified independently
           ↓
EvidenceSourceRefV1 may be admitted by a source-specific profile
```

Neither commitment substitutes for the other.

## Deterministic generic payload

`EvidenceSourceRefV1::semantic_payload()` freezes a deterministic generic transcript over:

1. source-reference version;
2. the complete validated semantic ID;
3. source-native commitment scheme;
4. source-native SHA-256 digest;
5. preservation mode;
6. explicit semantic loss codes.

This enables later provenance/claim-scope objects to bind the exact source reference without hashing arbitrary Serde JSON.

Important:

```text
semantic_payload() deterministic
!= source-native validator executed
!= either digest recomputed
!= source authority granted
```

## Required adversarial controls

The source implementation specifies tests for:

- noncanonical machine codes rejected;
- upper-case/native malformed SHA-256 rejected;
- `ProjectedWithLoss` with zero loss codes rejected;
- duplicate or unsorted loss codes rejected;
- source-native commitment scheme substitution changes the generic payload;
- source-native digest substitution changes the generic payload;
- preservation ceiling change changes the generic payload;
- semantic identity substitution changes the generic payload;
- generic JSON round-trip still requires full source-reference validation.

Source-specific adapter qualification must additionally prove:

- forged generic reference cannot bypass native validation;
- native commitment is recomputed from the exact source artifact/profile;
- semantic ID is recomputed from the exact source projection;
- source-native digest equality alone cannot imply semantic-profile equality;
- `LosslessUnderProfile` cannot be emitted when the frozen adapter knows it drops semantics.

## Authority ceiling

This tranche establishes no producer/signature authentication, trusted chronology, preregistration validity, consent/export authority, runtime protocol adherence, randomization quality, statistical significance, causal inference, preference, artistic quality, or product authority.

MEL-EPI-001C may later bind a validated/admitted `EvidenceSourceRefV1` into a positive closed-world claim scope. Generic 001B shape validation alone must never mint such claims.
