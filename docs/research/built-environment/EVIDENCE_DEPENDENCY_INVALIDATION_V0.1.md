# Evidence dependency invalidation v0.1

**Implementation:** `symthaea-built-environment`  
**Tracking:** [DESIGN-CHANGE-INVALIDATION-001 #12780](https://github.com/Luminous-Dynamics/symthaea/issues/12780)  
**Parent:** [DESIGN-INTENT-GRAMMAR-001 #12765](https://github.com/Luminous-Dynamics/symthaea/issues/12765)

## Purpose

A resident or authority may revise one input after a design has accumulated geometry, simulation, procurement or approval evidence. The system must not blindly rerun everything, and must never carry forward evidence whose assumptions changed.

The v0.1 dependency graph provides a deterministic *invalidation calculation*. It does not execute a solver or discover hidden causal relationships.

## Direction and semantics

Each node has a stable ID, a BLAKE3 content digest, direct dependency IDs, and optionally an evidence-receipt digest. Edges are oriented from prerequisite to dependent:

`A → B` means **B depends on A**.

Illustrative graph:

```text
                  ┌─ style intent ───────┐
resident brief ───┤                      ├─ realized geometry ── energy evaluation
                  └─ typology intent ────┘

water policy ───────────────────────────────────────────────── water evaluation
```

Changing style intent invalidates style intent, realized geometry and energy evaluation. Typology evidence and the independent water chain remain preserved in this declared graph. Changing the resident brief invalidates both design branches, geometry and energy; the independent water chain remains preserved.

The actual unit tests build this graph with generated BLAKE3 digests, avoiding placeholder hashes in fixtures.

## Validation contract

- Graph ID and graph revision are mandatory.
- Node IDs are unique and non-empty.
- Node content digests and optional evidence receipt digests are lowercase 64-character BLAKE3 hex.
- Every dependency must exist; self-edges and duplicate edges are rejected.
- Kahn topological validation rejects directed cycles.
- The graph's declared revision must equal the requested prior revision.
- Changed-node set must be non-empty, unique and entirely present in the graph.
- New revision must be non-empty and different from the prior revision.

## Receipt contract

`compute_invalidation_receipt` creates a deterministic receipt containing:
- graph ID, prior revision and new revision;
- digest of the canonical graph (nodes and dependency lists sorted);
- sorted changed-node IDs;
- the transitive invalidated-node closure;
- invalidated evidence receipt references; and
- preserved evidence receipt references.

The receipt digest is BLAKE3 over all receipt fields other than the digest itself. `verify_invalidation_receipt` recomputes the result against the current graph and change set; graph drift or changed inputs cause rejection.

No solver is run by this crate. An owner system receives the invalidation set and decides what to re-evaluate. Symtropy retains geometry and physical-solver authority. Mycelix retains decision authority, signatures, revocation and supersession. Sol-Atlas retains resident intent and decision receipts.

## Critical limitation: completeness of the dependency graph

This mechanism proves only what the declared graph encodes. It cannot infer that a wind load depends on a geometry feature omitted from the graph, or that a weather-file revision affects a model that failed to declare that dependency.

Therefore:
- an incomplete/unknown dependency graph is not evidence of independence;
- missing dependency declarations must block preservation for the affected claim domain;
- a preserved receipt means **unaffected under this exact declared graph**, not universally still true;
- owner adapters must add dependencies whenever their model inputs, solver settings, external sources, applicability domain or authority requirements can change the result.

## Metrics and targets

Proposed, not yet measured:
- 100% correct invalidation closure on frozen positive/negative DAG fixtures;
- zero stale evidence retained as active after a declared dependency changes;
- zero unrelated evidence invalidated when independence is explicit and complete;
- 100% deterministic replay for pinned graph/revision/change inputs;
- report invalidation precision, invalidation recall, graph coverage/completeness, and avoided re-evaluation cost separately.

## Current implementation state

Authored in Rust with typed nodes, graph validation, canonical graph hashing, transitive invalidation, deterministic receipt digest and replay verification. Synthetic unit tests cover:
- branch-local invalidation in a diamond graph;
- common-root invalidation;
- cycles and missing dependencies;
- duplicate edges and self-edges;
- malformed hashes, stale prior revision and repeated changed IDs;
- order-independent receipt generation and graph-drift rejection.

**Exact-head CI and Rust compilation are pending.** These tests are authored, not claimed passed. This software contract does not establish physical, legal, professional or resident approval.
