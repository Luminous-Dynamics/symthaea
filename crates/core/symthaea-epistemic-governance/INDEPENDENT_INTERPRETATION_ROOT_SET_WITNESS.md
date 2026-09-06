# RCA Independent Interpretation Root Set Witness v1

Status: **shadow-only epistemic-governance contract**

This layer turns the fail-closed interpretation-lineage graph into an explicit witness for one selected pairwise-independent interpretation-root set.

## Core theorem

```text
root count
    !=
qualified independent-root-set witness
```

A later shadow disposition engine must not discover or infer an independent interpretation set from identifiers or edge counts. The selection must already have been checked against one exact issued `InterpretationLineageV1`.

## Issuance

`issue_independent_interpretation_root_set_witness_v1(...)` accepts only:

```text
InterpretationLineageV1
+
selected interpretation-root ids
```

Issuance recomputes the selected set against the exact lineage:

1. every selected root must exist in the lineage;
2. duplicate selected roots fail closed;
3. every distinct selected pair must have one root-pair assessment;
4. every selected pair must be `IndependenceQualified`;
5. every qualified pair must retain its exact independence-qualification id.

Distinct roots whose independence is unknown cannot enter a witness.

## Single-root case

A one-root witness is valid without pair edges. This does not assert corroboration; it represents an exact set of cardinality one. Any policy threshold above one still requires the corresponding fully qualified clique.

## Canonical identity

The issued witness receives a domain-separated BLAKE3 id over:

- witness profile/schema;
- exact proposition id;
- exact interpretation-lineage id;
- canonical selected root ids;
- every selected root pair;
- every accepted pair qualification id.

Caller input order does not change witness identity.

## Persistence

`IndependentInterpretationRootSetWitnessV1` has private fields and deliberately does not implement `Deserialize`.

Archived bytes are audit material only. Current trust requires reissuing from a currently issued interpretation lineage.

## No count shortcuts

The public artifact exposes:

```text
root_ids()
pairs()
```

There is no qualified-edge-count or independent-root-count convenience method. Policy comparison may use the selected root-set cardinality directly; pair-edge cardinality is never a substitute.

## Authority separation

```text
IndependentInterpretationRootSetWitnessV1
        !=
relation truth
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

## Required adversarial cases

Qualification must preserve at least:

- selected-root input order does not change the normalized set;
- unknown independence fails closed;
- a qualified pair without a qualification id fails closed;
- duplicate selected roots fail closed;
- unknown selected roots fail closed;
- a single known root can be witnessed without pair edges;
- issued witness remains non-deserializable;
- no count shortcut or downstream authority path is introduced.
