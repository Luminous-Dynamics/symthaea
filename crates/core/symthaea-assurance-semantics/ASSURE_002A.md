# ASSURE-002A — Self-Describing Semantic Definition Commitments

ASSURE-002A is a reusable shared-core hardening tranche based directly on the qualified ASSURE-000 core. It deliberately does not depend on ASSURE-001 subject identity or ASSURE-002 campaign mechanics.

## Governing theorem

```text
semantic id + definition digest
    != self-describing semantic commitment

schema id
    != immutable schema specification
```

A conforming commitment therefore binds:

```text
semantic id
+ definition schema id
+ exact schema-specification digest
+ exact definition digest
```

Likewise:

```text
definition/schema digest known
    != bytes available
    != semantics adequate
    != semantics trusted
```

Availability/replayability remains outside this tranche.

## Reusable primitive

`symthaea-assurance-semantics` defines:

```text
DefinitionSchemaV1 {
    schema_id,
    specification_digest,
}

SemanticCommitmentV1 {
    semantic_id,
    definition_schema,
    definition_digest,
}
```

The semantic commitment wire form is domain-separated and orders the commitment schema/version, semantic ID, definition-schema ID, schema-specification digest, and definition digest. Changing any one changes identity.

`canonical_semantic_set()` sorts only by semantic identifier and rejects duplicate semantic IDs even if callers attach different schemas, schema specifications, or definition digests. One semantic ID names one slot in one declared set.

Canonical string framing uses UTF-8 byte lengths. Unicode normalization is deliberately not implicit: byte-distinct NFC/NFD inputs remain distinct commitments unless a committed higher-level schema explicitly requires normalization before these bytes reach this primitive.

## Conformance corpus

The focused corpus proves:

1. definition-schema ID drift changes identity;
2. schema-specification drift changes identity;
3. definition drift changes identity;
4. exact semantic/schema/specification/definition components are preserved;
5. duplicate semantic IDs fail closed across schema differences;
6. semantic sets canonicalize by semantic ID only;
7. canonical lengths are UTF-8 byte lengths;
8. Unicode normalization is not implicit;
9. an independently calculated primitive golden vector is stable.

Primitive golden fixture:

```text
semantic-id: matched-sham
definition-schema-id: symthaea.assurance.semantic-definition.canonical-text-v1
definition-schema-specification-digest: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
definition-digest: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
commitment-digest: 3a510b215853c77285e39c95b6ab45c0d816370e17e14ccda8337110a790f1ba
```

This freezes only the generic semantic-commitment wire format. Full campaign/registration/ordering/admission vectors belong to ASSURE-002B integration.

## Lineage and qualification

The source was previously exercised successfully through Rust 1.96 format/check, 9/9 tests, and strict Clippy while stacked on ASSURE-002, but that lineage also inherited unrelated campaign and lockfile state.

This branch intentionally re-homes the identical semantic source on the exact qualified ASSURE-000 head `a39e9f1692570b7731c416cc98b4f875a199b3a1`. Its dedicated qualification lane is read-only and independently proves exact head, Rust 1.96 format/check/tests/strict Clippy, generated root-lock evidence, committed lock parity, and final tracked-checkout immutability.

Historical stacked runs remain historical evidence; this cleaner shared-core lineage must earn its own exact-head PASS.

## Integration sequence

```text
qualified ASSURE-000
        ↓
qualified ASSURE-002A shared semantic primitive
        +
qualified ASSURE-002 campaign mechanics
        ↓
ASSURE-002B integration
        ↓
ASSURE-002C temporal provenance correction
        ↓
ASSURE-003 resolver
```

ASSURE-002B must consume this crate as the single semantic-definition authority rather than independently re-encoding the schema/specification/definition tuple.

## Deliberate nonclaims

ASSURE-002A does not establish definition-byte availability or authenticity, semantic adequacy, criterion satisfaction, ordering-provider trust, claim support, replication, verifier independence, compliance/certification, deployment authority, or production-time provenance.
