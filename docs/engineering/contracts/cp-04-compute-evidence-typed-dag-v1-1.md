# CP-04 Compute Evidence Typed DAG Contract v1.1

Status: architecture follow-up / synthetic-reference qualification target.

## Claim ceiling

This contract establishes only deterministic software semantics for typed dependency and invalidation relationships among compute-evidence references over synthetic/reference workflows. It establishes no physical execution, hardware performance, accelerator correctness, compiler correctness, model validity, benchmark generalization, production availability, safety/security/capacity guarantee, deployment authorization, or operational authority.

## Purpose

CP-04 v1 used a deliberately minimal linear evidence thread:

`requirement -> representation -> model -> runtime -> accelerator -> deployment -> execution -> observation -> statistics -> disposition`

v1.1 makes the dependency structure explicit as a typed directed acyclic graph (DAG). The graph is an invalidation oracle, not a replacement ontology or evidence database.

## Canonical node distinctions

The following node classes remain independently addressable:

`requirement != representation != model != model parameters != runtime != toolchain != accelerator != deployment artifact != execution context != observation != uncertainty != statistics != provenance reference != currentness != applicability != disposition`

A derived disposition is never an immutable evidence identity.

## Typed edges

Every dependency edge MUST carry one declared semantic type. v1.1 recognizes:

- `requires`
- `implements`
- `parameterizes`
- `executes_with`
- `compiled_by`
- `runs_on`
- `deploys`
- `executes`
- `observes`
- `quantifies`
- `summarizes`
- `traces_to`
- `currentness_for`
- `applicable_to`
- `derives`

An edge type is part of graph identity. Replacing an edge type with another type is therefore an evidence-dependency mutation.


## Edge endpoint semantics

The edge vocabulary is not merely a label set. Each edge type has an explicit endpoint contract: the source and target node classes it is permitted to connect. The qualifier MUST validate these endpoint pairs independently before applying the exact fixture edge oracle. This prevents a graph from relying on a semantically incorrect endpoint pair merely because the edge type itself is recognized.

The v1.1 endpoint oracle is:

- `requires`: requirement -> representation
- `implements`: representation -> model
- `parameterizes`: model -> model_parameters
- `executes_with`: model -> runtime
- `compiled_by`: runtime -> toolchain
- `runs_on`: runtime -> accelerator
- `deploys`: accelerator -> deployment artifact
- `executes`: deployment artifact -> execution context
- `observes`: execution context -> observation
- `quantifies`: observation -> uncertainty
- `summarizes`: observation -> statistics
- `traces_to`: observation -> provenance reference
- `currentness_for`: currentness -> runtime
- `applicable_to`: applicability -> execution context
- `derives`: statistics -> disposition

Endpoint mutations are qualified directly against this semantic oracle and are bound into the persisted mutation manifest.

## Invalidation semantics

If an evidence-relevant node changes, its direct and transitive downstream dependants become candidates for requalification.

The oracle MUST compute downstream closure from the typed graph rather than trusting a hand-maintained list.
The qualifier MUST compare the computed closure for every declared node against an explicit semantic closure oracle. Coverage of all node classes is required; checking only selected critical paths is insufficient. A graph mutation that preserves schema shape but changes any node's invalidation closure MUST be detected.

Changing a node does not rewrite historical nodes. Requalification produces a new derived disposition for the affected context.

Currentness and applicability are dependency boundaries; they are not evidence of performance or execution.



## Invalidation algebra invariants

The qualifier MUST also preserve three closure-algebra properties over representative edge removals:

- **Locality:** removing a dependency cannot unexpectedly alter an unrelated node's closure.
- **Monotonicity:** removing dependencies cannot introduce new reachable downstream nodes.
- **Compositionality:** applying two dependency removals sequentially MUST produce the same closure as applying the two removals together, independent of removal order.

These are deterministic graph semantics only; they do not imply physical execution, performance, availability, or operational authority.

## Edge-level closure sensitivity

The qualifier additionally verifies representative dependency edges at the closure boundary. Removing a dependency edge MUST remove exactly the downstream nodes reachable only through that edge from the edge's source closure; unrelated closure members MUST remain unchanged. This is a semantic sensitivity check, not a physical execution claim.

## Provenance boundary

`provenance_reference` is a reference/derivation boundary. It may identify custody, generation, derivation or provenance context, but its presence cannot create an observation, benchmark result, performance claim, or physical execution authority.

This follows the general provenance separation between entities, activities and derivations used by W3C PROV; CP-04 does not implement or replace PROV.

## Cross-contract boundary

CP-04 references canonical owners for:
- requirements and verification;
- representation/model semantics;
- runtime/toolchain/accelerator/deployment;
- physical observation and metrology;
- provenance/custody;
- operational availability and authority.

CP-04 MUST NOT create duplicate canonical identity stores for these concerns.

## Mutation manifest

The executable mutation suite is bound to `docs/engineering/data/cp-04-compute-evidence-dag-mutation-manifest-v1-1.json`. The qualifier MUST reject any difference in mutation order, mutation identifier, or expected stable guard between the persisted manifest and executable suite. The qualification receipt records the canonical manifest digest.

## Qualification target

The independent qualifier MUST verify:
1. exact node and edge schemas;
2. exact typed edge vocabulary;
3. DAG acyclicity;
4. no duplicate nodes or edges;
5. no dangling references;
6. every node has a deterministic downstream closure;
7. edge-type mutation changes graph identity and is rejected;
8. reversed, removed, duplicated, malformed, dangling and cyclic mutations are rejected;
9. provenance cannot promote a synthetic/reference claim into physical execution authority;
10. derived disposition changes do not alter immutable graph identity.

Synthetic qualification remains below physical execution authority.
