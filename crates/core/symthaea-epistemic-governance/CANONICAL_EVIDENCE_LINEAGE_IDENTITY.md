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

## Encoding

The implementation does **not** hash raw JSON bytes.

The validated graph is projected to its serde semantic value tree only so private validated-node contents can be read without widening the lineage API. The identity routine then explicitly extracts the known v1 fields, sorts nodes and parents, maps derivation kinds through their explicit snake-case wire tags, and feeds those values into a domain-separated, length-prefixed BLAKE3 encoding.

JSON object order, whitespace, debug formatting, Rust `Hash`, and the legacy graph label therefore do not define identity.

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
- only known v1 derivation tags are admitted into the identity encoding;
- the canonical identity remains a generic governance dependency with no RCA runtime dependency.
