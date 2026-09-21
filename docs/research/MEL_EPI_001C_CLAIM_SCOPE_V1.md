# MEL-EPI-001C — Positive Closed-World Claim-Scope Projection V1

Status: source-staged, non-authority-bearing substrate only. Depends on MEL-EPI-001B (#5350) and ultimately on executable-clean MEL-EPI-001A.

Tracker: #5285

## Purpose

001C defines a canonical, source-bound declaration of which namespaced claims one exact profile says are positively established, plus explicit nonclaims.

The serialized/shared type is intentionally named:

```text
ClaimScopeProjectionV1
```

not a bare `ClaimScopeV1` or credential, because generic shape validation does not itself grant trustworthy authority.

Core theorem:

```text
ClaimScopeProjectionV1 parses/validates
!= source-native verifier executed
!= source ref independently re-established
!= declared positive claims admitted as trustworthy
```

Source-specific admission remains mandatory.

## Contract

```text
ClaimScopeProjectionV1 {
    claim_scope_version,
    source_ref: EvidenceSourceRefV1,
    scope_profile_id: NamespacedCodeV1,
    establishes: sorted unique Vec<NamespacedCodeV1>,
    explicit_nonclaims: sorted unique Vec<NamespacedCodeV1>,
}
```

`scope_profile_id` identifies the exact interpretation/admission profile. It is deliberately distinct from the source semantic projection profile inside the source ref.

## Closed-world semantics

Within a source-specific profile that has actually been verified/admitted:

```text
claim in establishes -> declared positive authority under that exact profile
claim absent          -> NOT ESTABLISHED
```

`explicit_nonclaims` are useful for auditability, documentation, and regression protection, but they are never an inverse allowlist. A claim absent from both sets remains not established.

The same canonical claim code may not appear in both `establishes` and `explicit_nonclaims`.

## Query API

The generic type intentionally does not expose `is_established()`.

Instead:

```text
ClaimDeclarationStatusV1 {
    DeclaredEstablished,
    ExplicitNonclaim,
    NotDeclared,
}
```

and:

```text
projection.declaration_status(claim)
```

first validates the complete projection before returning a declaration status.

This language preserves the distinction between:

```text
projection says profile establishes X
!= source-specific verifier actually admitted X
```

## Canonical codes

001C reuses `NamespacedCodeV1` from 001B for:

- scope profile IDs;
- established claim identities;
- explicit nonclaim identities.

No second claim-code grammar is introduced.

Examples:

```text
muse.collection-close.protocol-bound
muse.collection-close.participant-evidence-rederived
muse.collection-close.signers-authorized
qualification.engineering.rust-tests-pass
```

Human descriptions remain separate non-authority metadata.

## Deterministic typed payload

`ClaimScopeProjectionV1::semantic_payload()` canonically binds:

1. claim-scope projection version;
2. complete validated 001B source-reference semantic payload;
3. exact scope profile ID;
4. ordered established-claim set;
5. ordered explicit-nonclaim set.

This uses `SemanticTranscriptV1` rather than arbitrary Serde JSON.

Changing any of those changes the generic semantic payload.

Important:

```text
semantic_payload() deterministic
!= source ref independently verified
!= source-specific admission profile executed
!= positive claims trustworthy
```

## Why sorted vectors instead of sets

Serialized authority identity must be deterministic. V1 therefore requires both claim vectors to be strictly increasing and unique rather than relying on container iteration order.

No silent sorting occurs during validation. A producer that emits a noncanonical ordering is rejected rather than normalized into a different byte identity after the fact.

## Legacy E/N/M isolation

No conversion API is provided from legacy empirical ordinal levels to namespaced claims.

Especially:

```text
large z-score
many observations
many runtime cycles
high empirical ordinal

!= cryptographic verification
!= trusted chronology
!= public reproducibility
```

Those authorities require independent evidence profiles.

## Required adversarial controls

Source tests cover:

- valid canonical projection;
- positive, explicit-nonclaim, and absent claim lookup;
- absent claim returns `NotDeclared`;
- duplicate/unsorted positive claims reject;
- duplicate/unsorted explicit nonclaims reject;
- positive/nonclaim overlap rejects;
- source-reference substitution changes semantic payload;
- scope-profile substitution changes semantic payload;
- positive claim set change changes semantic payload;
- explicit nonclaim set change changes semantic payload;
- JSON round-trip still requires complete projection validation.

Source-specific adapter qualification must additionally prove:

- generic projection cannot bypass native source validation;
- source-native commitment is recomputed;
- MEL-EPI semantic identity is recomputed;
- only claims allowed by the exact source/profile are emitted;
- absent claim is never inferred from neighboring/ordinal claims;
- cryptographic/reproducibility claims cannot be minted from digest-shaped strings, statistical significance, sample count, or runtime count.

## Authority ceiling

A shape-valid `ClaimScopeProjectionV1` establishes no source authenticity, signer authorization, trusted chronology, preregistration validity, runtime protocol compliance, causal inference, listener preference, artistic quality, or product authority by itself.

The first planned Muse adapter (#5351) should use this model to keep rederived facts, policy-admitted assertions, signoff presence, and authenticated signoff authority as separate claim classes rather than one `valid` bit.
