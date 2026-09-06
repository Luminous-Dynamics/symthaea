# RCA Independent Evidence Set Witness v1

Status: **shadow-only epistemic-governance contract**

This contract closes a counting ambiguity before RCA shadow disposition exists.

## Core theorem

```text
distinct evidence-root count
        !=
independent evidence-item count
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

## Issuance

`issue_independent_evidence_set_witness_v1(...)` accepts only:

```text
ValidatedEvidenceLineageGraphV1
+
selected evidence item ids
```

It does not accept caller-supplied root sets, independence labels, pair counts, or witness identity.

The issuer:

1. canonicalizes selected item ids;
2. rejects duplicate selection;
3. recomputes each item's complete ancestry-root set from the closed validated graph;
4. assesses every selected evidence-item pair through `ValidatedEvidenceLineageGraphV1::assess_independence`;
5. requires every pair to be exactly `EvidenceIndependenceV1::Independent`;
6. independently verifies that the complete ancestry-root sets are disjoint;
7. constructs the full selected-item pair topology;
8. retains the union of ancestry roots for provenance/audit;
9. derives a domain-separated BLAKE3 witness id over the complete canonical artifact.

## Pairwise-independent evidence items

A witness for N selected items proves the selected set itself is pairwise independent under the current lineage contract:

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
- canonical selected evidence item ids;
- every selected item's complete canonical root set;
- every selected-item pair;
- the complete canonical distinct-root union.

Identity does not depend on JSON, Rust `Hash`, debug formatting, or caller input order.

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

The witness says only that the exact selected evidence items satisfy the lineage layer's pairwise-independence criterion.

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
- the issued witness cannot deserialize into trusted state;
- no caller-supplied root sets, pair statuses, or independence counts exist in the issuance API;
- no downstream belief/workspace/action/promotion authority enters this layer.
