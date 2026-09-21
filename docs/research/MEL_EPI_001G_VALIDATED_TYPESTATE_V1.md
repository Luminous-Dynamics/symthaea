# MEL-EPI-001G — Raw / Validated / Source-Admitted Typestate V1

Tracker: #5356
Parent architecture: #5275
Source parent: MEL-EPI-001C draft #5354 / `3f3a6b3bf4728879688eb53961236b0756180c47`

Status: source-staged only. This document defines an interface-hardening layer; it establishes no executable qualification and no source-native admission.

## Purpose

MEL-EPI raw DTOs are intentionally serializable ordinary data. They may be loaded from JSON, storage, or another process and may therefore be forged, stale, malformed, or mutated after deserialization.

001G separates that state from immutable structural validation:

```text
raw/deserialized DTO
    != structurally validated wrapper
    != source-native admitted evidence
```

This prevents a common temporal-validity error:

```text
validate(raw)
mutate(raw)
continue as though validation still applies
```

## V1 wrappers

```text
ValidatedEvidenceSourceRefV1
ValidatedClaimScopeProjectionV1
```

Both wrappers:

- own the validated raw DTO privately;
- are constructed only through fallible `TryFrom`;
- expose read-only accessors only;
- cache the canonical semantic payload at construction;
- do not implement unconstrained `Deserialize`;
- require explicit `into_raw()` demotion before mutation;
- do not imply source-native admission.

## Source-reference transition

```text
EvidenceSourceRefV1
        |
        | TryFrom / full generic validation
        v
ValidatedEvidenceSourceRefV1
        |
        | future source-specific verifier
        v
MuseAdmittedCollectionCloseRef   # example domain type
```

The shared crate stops at the middle state.

A validated source reference proves only:

- source-reference version is supported;
- embedded semantic ID is structurally valid;
- source-native commitment scheme/digest shape is canonical;
- preservation metadata is canonical;
- canonical generic source-reference bytes were frozen at validation time.

It does **not** prove:

- the source-native artifact exists;
- the native commitment was honestly recomputed;
- the MEL-EPI semantic digest was honestly recomputed from source evidence;
- the source projection profile is scientifically appropriate;
- any positive claim has been admitted.

## Claim-scope transition

```text
ClaimScopeProjectionV1
        |
        | TryFrom / full generic validation
        v
ValidatedClaimScopeProjectionV1
        |
        | future exact source/profile admission
        v
DomainAdmittedClaimScope
```

The validated wrapper preserves the existing non-credential query vocabulary:

```text
DeclaredEstablished
ExplicitNonclaim
NotDeclared
```

It deliberately does not add `is_established()`.

## Cached semantic payloads

V1 caches each object's canonical semantic transcript during wrapper construction.

This means later read-only use does not depend on a caller remembering to re-run validation after every query and cannot observe a mutated invariant-bearing field because no mutable access is exposed.

```text
validated wrapper created
    -> canonical payload frozen
    -> read-only access

semantic mutation desired
    -> into_raw()
    -> validated state lost
    -> mutate raw DTO
    -> TryFrom again
```

The cached payload is a structural/canonicality result only. It is not a signature, credential, trusted timestamp, or source verification receipt.

## Serialization boundary

Raw DTOs remain the interchange/storage form.

Validated wrappers intentionally do not implement direct Serde deserialization. A consumer must deserialize to raw data and cross the validation boundary explicitly:

```text
JSON / storage
    -> EvidenceSourceRefV1
    -> ValidatedEvidenceSourceRefV1::try_from(...)
```

This prevents serialized bytes from selecting a stronger typestate merely by choosing a target Rust type.

## Admission remains domain-specific

Do not add a universal shared:

```text
AdmittedEvidence
TrustedClaimScope
ValidScientificEvidence
```

Source admission remains tied to the exact native theorem/profile. Examples may eventually include:

```text
MuseAdmittedCollectionCloseRef
TonalQualifierAdmittedReceipt
RecruitmentAdmittedGrant
```

Those types must retain the exact native verification evidence needed by their domain.

## Required controls

The source tests cover:

1. invalid source-reference raw DTO cannot enter validated state;
2. valid source reference freezes the same canonical payload as the raw validated form;
3. mutation requires explicit demotion and fresh validation;
4. Serde round-trip produces raw data, followed by explicit validation;
5. invalid claim-scope ordering cannot enter validated state;
6. validated claim queries preserve declaration-only semantics;
7. claim-scope mutation after demotion cannot silently retain validated status.

Future qualification should also enforce API-surface expectations, including absence of unconstrained `Deserialize` and mutable invariant access on validated wrappers.

## Relationship to provenance

001D provenance should prefer validated references/projections as construction inputs where practical, but provenance graph shape still grants no source authority.

```text
validated node reference
!= source-admitted node

provenance edge
!= claim propagation
```

001F remains responsible for explicit fail-closed claim derivation rules.

## Explicit nonclaims

001G does not establish:

- source-native artifact validity;
- honest digest computation;
- signer authentication or authorization;
- trusted chronology;
- preregistration authenticity;
- consent/export authority;
- randomization quality;
- protocol adherence;
- statistical significance;
- causal validity;
- listener preference;
- artistic or musical quality;
- product authority.

It establishes only a safer in-process distinction between raw serialized data and immutable structurally validated generic projections once executable qualification eventually confirms the exact implementation subject.
