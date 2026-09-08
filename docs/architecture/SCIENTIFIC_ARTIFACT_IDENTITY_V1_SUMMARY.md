# SCI-002 — Canonical Scientific Artifact Identity v1 — Summary

**Status:** architecture-only; non-authorizing; non-qualifying.

SCI-002 freezes the identity substrate required by later Scientific Method Kernel receipts.

## Core rule

```text
label
    != digest syntax
    != verified bytes
    != semantic identity
    != provenance
    != scientific validity
    != independent replication
    != action authority
```

## Four identity layers

```text
1. locator / human label
2. raw byte content identity
3. canonical semantic artifact identity
4. composite / manifest identity
```

Raw content identity answers which exact bytes were observed.

Semantic identity additionally binds a domain-owned semantic schema and canonicalization profile.

Composite identity binds exact child identities plus relation/order/graph semantics.

## Strict digest boundary

The first implementation should support exact stable digest algorithms such as SHA-256 and BLAKE3-256, with one canonical text form:

```text
sha256:<64 lowercase hex>
blake3:<64 lowercase hex>
```

Parsing a digest does not verify bytes. Positive verified-content state requires recomputing the digest over the exact observed bytes.

## No accidental serializer identity

Scientific identity must not be defined by:

```text
DefaultHasher
Rust Hash
Debug formatting
incidental serde output
```

unless a deliberately versioned canonical serialization profile explicitly defines those bytes.

## Domain-owned semantic canonicalization

The shared kernel does not decide universal rules for:

- scientific payload fields;
- float canonicalization;
- unit semantics;
- coordinate systems;
- theorem languages;
- causal estimands;
- raster interpretation;
- domain-specific graph semantics.

Those belong to exact versioned domain semantic/canonicalization profiles.

## Transformation boundary

```text
source artifact
    -> transformation
    -> target artifact
```

never silently implies semantic equivalence.

A transformation/equivalence receipt must bind source, target, transform implementation/configuration, execution/provenance, and the exact claimed relation.

## Persistence boundary

Positive verified-content wrappers should be private-fielded and not directly deserializable into authority. Serialized artifacts are audit records; current trusted state requires revalidation or a separately qualified persistence boundary.

## Existing work remains domain-native

SCI-002 does not replace or rewrite:

- RCA canonical lineage identities;
- NeuroBridge raw/normalized/execution/scientific roots;
- Matter reference-only SHA-256 evidence references;
- Earth-observation fixture/content identities;
- proposition semantic identities.

Future migration requires explicit adapters/correspondence receipts.

## First implementation slice

```text
ContentDigestV1
RawArtifactContentIdentityV1
VerifiedArtifactContentV1
```

Only after that should one narrow domain semantic-canonicalization pilot be attempted.

## Dependency order

```text
SCI-001 audit
    -> SCI-002 strict content identity
    -> SCI-002 verified bytes boundary
    -> one semantic identity pilot
    -> SCI-003 execution capsule
    -> #786 receipt-derived verification authority
```

#786 also remains separately gated by verifier-soundness/target-binding/admission-completeness prerequisites.
