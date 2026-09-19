# MATH-EXP-001 Manifest v2.1 — Explicit Retrieval Binding

Status: narrow successor to MATH-EXP-001 v2.

Predecessor v2 exact head: `02ec910d458447e2ff9f020bdada2bc14245683b`.

Authority: `MeasurementOnly`.

## Purpose

v2 correctly separates shared experimental invariants from arm-specific representation and retrieval interventions. During the MATH-RET-001 fusion design, one remaining ambiguity became visible:

```text
single-channel arm -> one index manifest
fusion arm         -> two index manifests + one fusion policy
```

but v2 names the arm edge:

`index_manifest_sha256`

singular.

That is unambiguous for single-channel arms and ambiguous for fusion arms.

v2.1 preserves every v2 causal rule and changes only that reference edge to:

`retrieval_binding_sha256`

## Core graph

```text
experiment arm
      |
      v
retrieval binding
      |
      +-- single-index mode
      |      +-- one MATH-RET-001A index
      |
      +-- fusion mode
             +-- syntax MATH-RET-001A index
             +-- exact-normal-form MATH-RET-001A index
             +-- one MATH-RET-001B fusion policy

      +-- shared downstream context packer
```

The retrieval binding is a content-addressed dependency graph, not an extra experimental intervention.

## Narrow change from v2

For every arm with:

`retriever_family != None`

v2.1 requires:

`retrieval_binding_sha256 = sha256:<...>`

For the no-retrieval baseline, the field is forbidden.

The legacy arm field:

`index_manifest_sha256`

is forbidden in v2.1.

Everything else remains governed by v2:

- one shared source/corpus/knowledge boundary;
- one exact normalization contract and implementation;
- one root resource budget;
- no per-arm budget overrides;
- representation/retriever identities;
- fusion-policy identity;
- explicit controls;
- stage-specific endpoints;
- manipulation checks;
- preregistered contrasts;
- negative-search-memory parent binding;
- evolutionary-search exclusion.

## Why not reinterpret the old field?

Reinterpreting `index_manifest_sha256` to sometimes mean “one index” and sometimes mean “a bundle containing two indices” would make the exact same field name change semantics by arm type.

That would complicate auditing and make older v2 manifests harder to interpret.

v2.1 instead creates an explicit successor identity and leaves v2 unchanged.

## Validator reuse

`.github/scripts/validate-math-search-experiment-v2.1.py` is deliberately a compatibility layer rather than a copied validator.

It performs two stages:

1. validate v2.1 retrieval-binding shape;
2. project `retrieval_binding_sha256` to the predecessor field in-memory and call the exact sibling v2 semantic validator.

Thus inherited budget/control/contrast logic has one implementation.

The wrapper rejects:

- retrieval arms with no binding;
- no-retrieval baseline carrying a binding;
- the legacy singular index field;
- malformed binding digests;
- inherited v2 arm-local budget smuggling;
- inherited fusion-policy removal.

## Next contract

MATH-RET-001C should define the retrieval-binding object itself.

Single-index binding:

```text
mode = SingleIndex
index_manifest_sha256
context_packer_sha256
```

Fusion binding:

```text
mode = Fusion
syntax_index_manifest_sha256
normal_form_index_manifest_sha256
fusion_policy_sha256
context_packer_sha256
```

The binding validator should require channel/index consistency and should ensure the same downstream context-packer digest can be compared across all retrieval arms.

## Downstream payload principle

Representation is allowed to change **ranking**.

It should not silently change the content object passed downstream.

The common context packer should therefore render selected source-object identities using one canonical source-object serialization across lexical, sparse, HDC, exact-normal-form and fusion retrieval arms.

A future experiment may deliberately compare alternative downstream representations, but that must be a new intervention axis rather than an accidental consequence of the retriever.

## Nonclaims

A valid v2.1 manifest does not establish that:

- any retrieval binding exists or authenticates itself;
- any index was built correctly;
- fusion is fair at runtime;
- normalization improves retrieval;
- HDC improves retrieval;
- any theorem is true or novel.

It removes an ambiguous dependency edge so the later cross-contract validator can bind the experiment to exact retrieval artifacts without guessing.
