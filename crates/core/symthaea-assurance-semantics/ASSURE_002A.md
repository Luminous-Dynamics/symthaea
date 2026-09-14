# ASSURE-002A — Self-Describing Semantic Definition Commitments

ASSURE-002A is a narrow hardening tranche stacked on ASSURE-002. It extracts semantic-definition identity into a reusable core crate because campaign planning, evidence resolution, imported evidence, and future standards projections may all need the same theorem.

## Governing theorem

```text
semantic id + definition digest
    != self-describing semantic commitment

schema id
    != immutable schema specification
```

A conforming commitment therefore requires all of:

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

The schema specification digest is SHA-256 over the exact specification artifact bytes under this v1 theorem. This stops a schema from retaining the same identifier while its own definition changes unnoticed.

The semantic commitment's canonical wire form is domain separated and explicitly orders:

1. commitment schema/version;
2. semantic ID;
3. definition-schema ID;
4. definition-schema specification digest;
5. definition digest.

Therefore all of these are identity-bearing:

```text
semantic-id
schema-id
schema-specification-digest
definition-digest
```

Changing any one changes the commitment identity.

## Why schema specification is committed

A bare schema label would merely move the ambiguity up one level:

```text
canonical-text-v1
```

would still rely on an external mutable explanation of what `canonical-text-v1` means.

Binding an exact schema-specification artifact commitment creates a finite trust boundary:

```text
schema identifier
+ immutable specification bytes commitment
```

The specification may still be bad, ambiguous, unavailable, or untrusted; those are different theorems. But its bytes cannot change while retaining the same semantic commitment unnoticed.

## Canonical semantic-set semantics

`canonical_semantic_set()` sorts only by semantic identifier and rejects duplicate semantic IDs even when callers attach different schema IDs, schema-specification digests, or definition digests.

One semantic ID names one slot in one declared set. Schema and digest are identity-bearing contents of that slot, not a mechanism for creating multiple meanings under the same label.

Canonical string framing uses UTF-8 byte lengths. Unicode normalization is deliberately not implicit: byte-distinct NFC/NFD inputs remain distinct commitments unless a committed higher-level definition schema explicitly specifies normalization before the bytes reach this primitive.

## Conformance corpus

The reusable-crate corpus currently proves:

- definition-schema ID changes commitment identity;
- definition-schema specification digest changes commitment identity;
- definition digest changes commitment identity;
- exact semantic/schema/specification/definition components are preserved;
- duplicate semantic IDs fail even when schema differs;
- semantic sets canonicalize by semantic ID only;
- length framing counts UTF-8 bytes rather than Unicode scalar values;
- Unicode normalization is not implicit;
- an independently calculated canonical commitment golden vector is stable.

The primitive-level golden fixture is:

```text
semantic-id: matched-sham
definition-schema-id: symthaea.assurance.semantic-definition.canonical-text-v1
definition-schema-specification-digest: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
definition-digest: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
commitment-digest: 3a510b215853c77285e39c95b6ab45c0d816370e17e14ccda8337110a790f1ba
```

This freezes only the generic semantic-commitment wire format. Full campaign/registration/ordering/admission golden vectors remain deferred until integration into ASSURE-002.

## Executable evidence

Exact semantic source head `41b77d97c9617b7ba726e71b79d13737777f4146` executed under Rust 1.96.0 in ASSURE-002 qualification run `34795741819`, dedicated job `qualify-semantic-schema`.

That exact source established:

```text
exact checkout                 PASS
cargo fmt --check              PASS
cargo check                    PASS
9/9 conformance tests          PASS
strict Clippy -D warnings      PASS
root-lock evidence capture     PASS
committed root-lock parity     FAIL
```

The failure was limited to missing workspace package entries in committed `Cargo.lock`; no semantic-source failure was observed. The read-only job produced exact reconciled root-lock SHA-256:

```text
c8393c2ac181cb86151aa82bfd4ac406cf7f7eb8f9da4242b02c3712997f0cd8
```

A narrowly constrained repair subsequently committed only that previously observed lock transformation. Because the repair push was authored through GitHub Actions, GitHub did not create executable qualification jobs for the resulting commit; a fresh ordinary repository commit therefore starts a new exact-head qualification rather than treating the repair run as product evidence.

Historical failed/action-required runs remain their actual evidence states.

## Integration sequence

ASSURE-002A is qualified as a separate core primitive before integration. The parent campaign kernel is qualified independently.

Then integrate from frozen inputs:

```text
qualified ASSURE-002 campaign kernel
        +
qualified semantic commitment primitive
        ↓
replace campaign-local label+digest semantics
with reusable SemanticCommitmentV1
        ↓
make full schema identity-bearing in support criteria,
controls, custom evidence, conditions, and
ordering-validation profiles
        ↓
prove validation-profile schema/spec drift
makes ordering lineages incomparable
        ↓
recompute full campaign golden vectors independently
        ↓
qualify integrated exact head
```

The integration contract is tracked separately in ASSURE-002B (#2874). Final campaign-level golden vectors are not frozen before that integration.

## Deliberate nonclaims

ASSURE-002A does not establish:

- authenticity of schema or definition bytes;
- availability of schema or definition bytes;
- adequacy/correctness of a schema, criterion, or control;
- criterion satisfaction;
- ordering-provider authenticity;
- experimental quality;
- claim support;
- successful replication;
- verifier independence;
- compliance/certification;
- deployment or action authority.
