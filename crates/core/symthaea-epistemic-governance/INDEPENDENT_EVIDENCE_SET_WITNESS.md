# RCA Independent Evidence Set Witness v1

Status: **shadow-only epistemic-governance contract**

This contract closes two ambiguities before RCA shadow disposition exists:

```text
distinct evidence-root count
        !=
independent evidence-item count
```

and:

```text
selected local lineage shape
        !=
complete lineage generation identity
```

One derived evidence item may inherit several ancestry roots. Those roots describe provenance; they do not turn one item into several independent confirmations.

Example:

```text
root A ─┐
        ├── derived item C
root B ─┘

root D ───── item D
```

For selected items `{C, D}`:

```text
independent evidence items = 2
complete distinct ancestry roots = 3
```

A later policy may require two independent evidence items, but it must not reinterpret this as three independent confirmations.

## Canonical lineage-generation binding

`EvidenceLineageGraphV1::graph_id` is retained as a legacy producer wire reference. It is **not** the governance identity used by this witness.

Before issuance, the witness derives:

```text
canonical_evidence_lineage_graph_id_v1(validated_graph)
```

from the exact validated graph semantics:

- graph schema version;
- every node schema version;
- every evidence id;
- every complete parent-id set, sorted canonically;
- every explicit derivation-kind tag;
- the complete node set, sorted canonically.

The legacy `graph_id` field is explicitly excluded.

Therefore:

```text
same validated graph + different legacy graph labels
        -> same canonical lineage generation
```

while:

```text
same selected evidence subset
+ same selected local roots
+ any additional/removed/rewired lineage node
        -> different canonical lineage generation
        -> different witness identity
```

This prevents a witness issued under one full lineage generation from being replayed under another generation merely because the selected subset appears identical.

## Issuance

`issue_independent_evidence_set_witness_v1(...)` accepts only:

```text
ValidatedEvidenceLineageGraphV1
+
selected evidence item ids
```

It does not accept caller-supplied graph identity, root sets, independence labels, pair counts, or witness identity.

The issuer:

1. derives the canonical complete lineage-generation identity;
2. canonicalizes selected item ids;
3. rejects duplicate selection;
4. recomputes each item's complete ancestry-root set from the closed validated graph;
5. assesses every selected evidence-item pair through `ValidatedEvidenceLineageGraphV1::assess_independence`;
6. requires every pair to be exactly `EvidenceIndependenceV1::Independent`;
7. independently verifies that the complete ancestry-root sets are disjoint;
8. constructs the full selected-item pair topology;
9. retains the union of ancestry roots for provenance/audit;
10. derives a domain-separated BLAKE3 witness id over the complete canonical artifact, including the canonical lineage generation.

## Pairwise-independent evidence items

A witness for N selected items proves the selected set itself is pairwise independent under the exact lineage generation:

```text
for every distinct i,j in S:
    assess_independence(i,j) == Independent
```

Because the lineage contract defines `Independent` only for complete disjoint root sets and no ancestor relation, sibling/ancestor/partially shared selections fail closed.

The witness is therefore an explicit set witness rather than an edge-count shortcut.

## Multi-root items

`IndependentEvidenceItemV1` retains all complete root ids for one item.

The public API exposes:

```text
lineage_graph_id()
items()
pairs()
distinct_root_ids()
```

but it does not expose a method that converts distinct-root cardinality into independent-item cardinality.

The later disposition policy must compare its evidence requirement against **selected evidence items in an issued witness**, not against `distinct_root_ids().len()`.

## Persistence

`IndependentEvidenceSetWitnessV1` has private fields and derives `Serialize` only.

It deliberately does not implement `Deserialize`. Archived bytes are audit material. Current trust requires recomputation from a currently validated closed evidence-lineage graph.

## Identity

Witness identity explicitly binds:

- witness profile contract;
- schema version;
- canonical complete lineage-generation id;
- canonical selected evidence item ids;
- every selected item's complete canonical root set;
- every selected-item pair;
- the complete canonical distinct-root union.

Identity does not depend on JSON byte order, Rust `Hash`, debug formatting, the legacy producer graph label, or caller input order.

## Authority separation

```text
IndependentEvidenceSetWitnessV1
        !=
relation declaration
        !=
shadow disposition
        !=
canonical belief
        !=
workspace/GWT authority
        !=
action authority
        !=
self-improvement promotion
```

The witness says only that the exact selected evidence items satisfy the lineage layer's pairwise-independence criterion under one exact canonical lineage generation.

It does not say the evidence is current, relevant, correctly interpreted, sufficient, or true.

## Required adversarial cases

Qualification must establish at least:

- a multi-root derived item remains one selected evidence item;
- the distinct-root union may be larger than selected-item cardinality without changing item cardinality;
- shared-root siblings cannot receive a witness;
- ancestor/descendant pairs cannot receive a witness;
- partially shared root sets cannot receive a witness;
- duplicate selected item ids fail closed;
- unknown evidence ids fail through lineage validation;
- selected-item input order does not change witness identity;
- changing the selected item set changes witness identity;
- adding an unrelated graph node changes the canonical lineage generation and witness identity even when the selected subset is unchanged;
- changing only the legacy wire graph label does not change canonical lineage generation or witness identity;
- the issued witness cannot deserialize into trusted state;
- no caller-supplied graph identity, root sets, pair statuses, or independence counts exist in the issuance API;
- no downstream belief/workspace/action/promotion authority enters this layer.
