# RCA Canonical Evidence Lineage Identity v1

Status: **generic epistemic-governance identity contract**

`EvidenceLineageGraphV1::graph_id` is a legacy producer-supplied wire label. It is retained for compatibility and audit, but it is not trusted as governance identity.

The canonical identity is derived only after the graph has passed the existing closed-DAG lineage validation.

## Core theorem

```text
producer graph label
        !=
canonical evidence-lineage generation identity
```

`canonical_evidence_lineage_graph_id_v1(...)` commits the exact validated semantic graph:

- lineage graph schema version;
- every node schema version;
- every evidence id;
- every node's complete parent-id set;
- every node's explicit derivation-kind tag.

Node order and parent order are canonicalized before hashing.

The producer `graph_id` field is explicitly omitted.

## Identity consequences

Two validated graphs with identical semantic contents but different producer graph labels receive the same canonical identity.

Any semantic graph change receives a different identity, including:

- adding or removing an unrelated node;
- changing a parent edge;
- changing a node's derivation kind;
- changing an evidence id;
- changing a schema version.

This matters because evidence-set independence can otherwise be replayed across lineage generations whose selected local subset looks identical.

## Typed semantic encoding

The implementation does **not** derive governance identity from JSON, serde field names, serde enum tags, debug formatting, Rust `Hash`, or raw serialized bytes.

The input is already a `ValidatedEvidenceLineageGraphV1`. Canonical identity reads its typed, read-only semantic node view directly. Each validated node supplies its evidence id, complete parent-id set, and typed `CognitiveDerivationKindV1`. Parent ids and nodes are canonicalized, the derivation enum is mapped through an exhaustive v1 semantic-tag match, schema versions are bound from the validated v1 contract, and those exact values are fed into a domain-separated, length-prefixed BLAKE3 encoding.

The path is therefore:

```text
validated typed lineage semantics
        -> explicit canonical semantic tree
        -> domain-separated BLAKE3
```

not:

```text
validated object
        -> serializer/wire projection
        -> identity
```

Because validation has already established the supported schema, canonical lowercase digest shapes, closed ancestry, unique ids/parents, and acyclicity, canonical identity derivation is infallible after validation. Persistence may continue to use serde, but persistence representation does not define governance identity.

## Interoperability known-answer vectors

The exact v1 byte contract is pinned outside the implementation module in `tests/lineage_identity_vectors.rs`. A non-Rust verifier should reproduce these values from the semantic encoding above rather than copying serialized Rust objects.

Profile contract digest:

```text
blake3:ff23c2ba0796d14eefd10c327542e8f0287bab3d09cb3bb2515c86fd72803d4a
```

Minimal graph:

```text
A = sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
B = sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb

A: RootObservation, parents=[]
B: Inference,       parents=[A]
```

Canonical graph identity:

```text
blake3:916114cbe9c4ff598097fd2702e965a4b3a763bfcc92e569fb986397371363d6
```

Complex ordering graph:

```text
A = sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
B = sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
C = sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
D = sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
E = sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee

input node order = [E, C, B, D, A]
E: RootObservation, parents=[]
C: Inference,       input parents=[B,A]
B: RootObservation, parents=[]
D: RootObservation, parents=[]
A: RootObservation, parents=[]
```

Canonicalization sorts nodes by evidence id and `C`'s parents to `[A,B]`. The resulting graph identity is:

```text
blake3:9115d0bacd4e4a9f461d2803ad660ff47bf2e0828fae21021add160e6ec936a4
```

Changing either pinned value without an explicit v1 profile/version transition is a compatibility break.

## Authority boundary

```text
canonical lineage identity
        !=
evidence independence
        !=
relation correctness
        !=
shadow disposition
        !=
canonical belief
        !=
action authority
```

The identity says only: these exact validated lineage semantics belong to one content-addressed generation.

## Required qualification

Qualification must establish:

- producer graph-label changes do not affect canonical identity;
- node order does not affect identity;
- parent order does not affect identity;
- adding an unrelated node changes identity;
- changing derivation kind changes identity;
- changing a parent edge changes identity;
- every v1 derivation variant has one explicit stable semantic tag;
- no serde/wire projection participates in canonical identity production;
- canonical identity derivation cannot fail after graph validation;
- simple and complex known-answer vectors reproduce exactly through the public API;
- the canonical identity remains a generic governance dependency with no RCA runtime dependency.
