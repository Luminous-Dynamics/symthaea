# RCA Relation Declaration Provenance v1

A structurally valid epistemic relation is still only a **declaration**. Before a later RCA disposition policy can rely on that declaration, the architecture must know who or what declared it and what immutable artifact records how the declaration was produced.

## Identity boundary

`EvidenceRelationV1::relation_id` remains a producer-supplied reference. It is useful for discussion and audit, but it is not the canonical governance identity of the declaration.

RCA therefore derives:

```text
declaration_id = H(
    identity-profile contract,
    provenance schema,
    declarer id,
    optional declarer version,
    declaration method,
    immutable provenance artifact digest,
    relation schema,
    producer relation reference,
    evidence id,
    relation kind,
    target kind + target id,
    declared strength
)
```

The encoding is explicit and domain-separated. JSON bytes, debug formatting, Rust `Hash`, and enum discriminant layout do not define identity.

## Declaration methods

V1 records one of:

- human annotation;
- deterministic rule;
- model inference;
- formal procedure;
- imported assertion.

These categories record **how a relation was declared**. They are not ranks of truth or authority.

## Persistence

`BoundEvidenceRelationDeclarationV1` is persistable because deserialization revalidates the provenance and relation and recomputes both the profile digest and canonical declaration id. Changing the declarer, provenance artifact, relation body, target, kind, or strength without recomputing the exact derived identity fails closed.

## Core theorem

```text
producer relation reference
        !=
canonical relation-declaration identity
```

and:

```text
relation declaration provenance
        != truth
        != independence
        != currentness
        != canonical evidence admission
        != belief/workspace authority
        != action authority
        != self-improvement promotion
```

A later RCA case/disposition layer may inspect the provenance, but it must establish its own policy for whether a declarer/method/provenance artifact is qualified for a particular use.
