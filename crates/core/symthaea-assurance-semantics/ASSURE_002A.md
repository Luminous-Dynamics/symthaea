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

The focused Rust corpus proves:

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

## Independent cross-implementation vectors

The qualification lane also runs a standard-library Python implementation over language-neutral JSON fixtures. It reconstructs the canonical framing and SHA-256 independently of Rust for:

```text
matched-sham-ascii
unicode-nfc
unicode-nfd
```

The independent implementation deliberately exercises UTF-8 byte lengths and proves that canonically equivalent-looking NFC/NFD text remains byte-distinct when no higher-level normalization rule has been committed.

The current theorem is intentionally narrow:

```text
independent canonical-encoding agreement
for declared valid vectors
```

It is not yet:

```text
complete cross-language parser / constructor equivalence
```

Rust `StableId` and `DigestSha256` enforce their own accepted-value domains. A future general-purpose non-Rust parser claiming constructor equivalence must mirror those validity predicates rather than coercing arbitrary JSON values or accepting malformed identifiers/digests.

## Independent lineage and qualification

This active lineage is based directly on exact qualified ASSURE-000 head:

`a39e9f1692570b7731c416cc98b4f875a199b3a1`

It supersedes the earlier stacked semantic experiment only as the active qualification lineage. Historical stacked runs remain historical evidence and are not rewritten.

Read-only ASSURE-002A qualification run `34828725215` executed exact semantic source head:

`b1e3e5540ad3b4516d931dcc1453f5d6fb5fb3c0`

and passed:

```text
exact candidate checkout
Rust 1.96 formatting
independent Python valid-vector oracle
cargo check
9/9 Rust conformance tests
doc tests
strict Clippy -D warnings
```

That run failed only committed root-lock parity. Cargo 1.96 produced exactly one missing workspace package stanza for `symthaea-assurance-semantics`; no dependency upgrades or unrelated lock changes were observed.

The generated target root-lock SHA-256 was:

`391b766f3f812f2d4c1a5b6a377ce1bcdeb5b7e2f88213078c88dab21846274a`

A temporary one-purpose repair workflow then proved the exact starting candidate head and starting lock blob, applied only that reviewed stanza, required the exact target SHA-256 above, committed exactly `Cargo.lock`, and pushed repair commit:

`f172621...`

The temporary workflow executed no Symthaea build/test/candidate code with write authority, was removed immediately after use, and its PR was closed without merge. That repair process is not product qualification.

The exact post-repair head containing this document must independently pass the permanent read-only qualification lane before ASSURE-002A is called qualified.

## Permanent qualification contract

Final PASS requires all of:

```text
exact PR-head checkout
Rust 1.96 format
independent Python valid-vector oracle
cargo check
9/9 Rust conformance tests
strict Clippy -D warnings
generated root-lock evidence
committed root-lock parity
tracked-checkout immutability
```

A failed, canceled, action-required, or superseded run remains evidence of that actual state; it is never rewritten as PASS by a later run.

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
base claim-strength convergence
        ├─ ASSURE-002C: commitment ordering != witnessed production
        ├─ ASSURE-002D: terminal in supplied view != checkpoint current
        └─ ASSURE-002E: declared principal != authenticated authority
        ↓
ASSURE-003 resolver
```

ASSURE-002B must consume this crate as the single semantic-definition authority rather than independently re-encoding the schema/specification/definition tuple.

The stronger optional production-witness, external-currentness, and authority-verification adapters are separate theorems. They do not change this primitive and should be implemented only where a campaign/customer actually requires those stronger claims.

## Deliberate nonclaims

ASSURE-002A does not establish definition-byte availability or authenticity, semantic adequacy, criterion satisfaction, complete cross-language parser equivalence, ordering-provider trust, registration authority, external registration-view completeness/currentness, claim support, replication, verifier independence, compliance/certification, deployment authority, or production-time provenance.
