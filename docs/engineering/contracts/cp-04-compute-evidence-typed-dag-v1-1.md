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

## Invalidation semantics

If an evidence-relevant node changes, its direct and transitive downstream dependants become candidates for requalification.

The oracle MUST compute downstream closure from the typed graph rather than trusting a hand-maintained list.

Changing a node does not rewrite historical nodes. Requalification produces a new derived disposition for the affected context.

Currentness and applicability are dependency boundaries; they are not evidence of performance or execution.

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
