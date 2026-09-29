# Mycelix DKG ↔ Symthaea Engineering Evidence Boundary v1

Status: architecture contract / implementation roadmap

## Purpose

Symthaea and Mycelix already contain overlapping knowledge, epistemic, provenance, and integration concepts. This document defines the boundary before adding a second graph implementation.

The target architecture is:

`Mycelix DKG -> typed engineering projection -> CP-04 evidence/replay DAG -> qualification`

The DKG is the knowledge substrate. The CP-04 DAG is a bounded qualification projection. They are complementary and MUST NOT be collapsed into one graph semantics.

## Existing Mycelix capabilities

The current Mycelix DKG architecture already provides:

- Holochain DHT-backed claims as a decentralized source of truth;
- link-based graph indexing and traversal;
- RDF-like subject/predicate/object claim representation;
- attestations, endorsements, challenges, and disputes;
- epistemic classification and confidence calculation;
- contradiction detection and knowledge discovery/query paths;
- belief-graph relationships including support, contradiction, derivation, generalization, specialization, composition, causality, and equivalence;
- optional query acceleration that can be rebuilt from DHT data.

Symthaea already has Mycelix integration seams, shared epistemic types, provenance-aware reasoning, and local knowledge-graph structures.

The engineering task is therefore **boundary design and typed composition**, not creation of an independent competing DKG.

## Ownership boundaries

| Concern | Canonical owner | Rule |
|---|---|---|
| Distributed claim storage | Mycelix DKG | Symthaea does not duplicate the DHT claim store. |
| Claim relationships | Mycelix DKG | DKG relations remain knowledge semantics. |
| Epistemic belief/confidence | Mycelix epistemic/DKG layer | Confidence is not deterministic engineering validity. |
| Engineering object identity | Shared integration contract | Identity MUST be canonical, stable, and compositional. |
| Engineering dependency semantics | Symthaea/Mycelix integration contract | Deterministic dependency edges MUST be explicit and typed. |
| Evidence qualification projection | CP-04 | Projection is bounded, typed, and acyclic. |
| Replay identity | CP-04 / integration layer | Replay identity binds the exact projected graph and relevant inputs. |
| Physical observation/metrology | Existing observation authority | A DKG claim does not manufacture physical evidence. |
| Operational authority | Existing governance/authorization systems | Graph connectivity, confidence, provenance, or qualification MUST NOT silently grant authority. |

## DKG and evidence DAG are intentionally different

A DKG is allowed to represent relationships that are cyclic or recursively self-referential when those relationships are semantically meaningful.

Examples:

- `equivalent` can be bidirectional;
- `supports` and `contradicts` can participate in competing evidence networks;
- `part-of` supports arbitrary-depth composition;
- a claim can describe another claim or a relationship among claims.

CP-04 has a stricter requirement: its qualification dependency projection is a DAG because downstream invalidation and replay require deterministic acyclic dependency semantics.

Therefore:

`DKG = rich knowledge relation space`

`Evidence DAG = bounded acyclic dependency projection`

A projection MUST NOT pretend that the entire DKG is acyclic.

## Epistemic propagation is not engineering invalidation

The two propagation systems must remain distinct.

### Epistemic propagation

A change in one claim can alter belief, confidence, support, contradiction, or discovery outcomes in related claims.

This is an epistemic computation over knowledge relationships.

### Engineering invalidation

A change in an engineering dependency invalidates or requalifies dependent artifacts.

For example:

`toolchain_v1 -> runtime_build_v1 -> deployment_v1 -> execution_context_v1 -> observation_v1`

If `toolchain_v1` changes, the engineering dependency closure identifies affected downstream artifacts.

This is not equivalent to saying that a claim about the toolchain became less believable.

**Rule:** epistemic confidence MUST NOT be substituted for engineering dependency validity, and engineering dependency validity MUST NOT be substituted for epistemic truth.

## Recursive engineering object model

Engineering objects MUST support arbitrary recursive composition rather than a fixed hierarchy.

A conceptual object can contain:

- requirements;
- subsystems;
- components;
- implementations;
- models;
- model parameters;
- runtimes;
- toolchains;
- accelerators;
- deployment artifacts;
- execution contexts;
- observations;
- uncertainty/statistics;
- provenance;
- currentness;
- applicability;
- derived dispositions.

Each object can itself have requirements, dependencies, evidence, provenance, currentness, applicability, and child objects.

This gives a recursive lifecycle:

`requirement -> system -> subsystem -> component -> implementation -> runtime -> execution -> observation -> qualification`

with the same contract reusable at every level.

## Projection invariants

The DKG-to-CP-04 projection MUST satisfy all of the following:

1. **No fact creation** — projection may select, normalize, and bind existing knowledge; it MUST NOT invent evidence.
2. **No authority amplification** — projection cannot raise operational authority.
3. **Type preservation** — every projected dependency has an explicit engineering edge type.
4. **Identity preservation** — projected nodes retain stable references to their canonical DKG/engineering identities.
5. **Provenance preservation** — every projected claim remains traceable to its source.
6. **Currentness separation** — currentness is temporal applicability metadata, not truth.
7. **Applicability separation** — applicability constrains context; it is not evidence of execution.
8. **Contradiction preservation** — conflicting DKG claims MUST remain visible or be represented by an explicit alternative/contradiction boundary.
9. **Acyclic projection** — only the selected qualification dependency projection is required to be acyclic.
10. **Deterministic replay** — the same canonical inputs MUST produce the same projected graph identity and closure results.

## Identity model

Do not use a graph-local string as the sole identity of an engineering object.

The integration contract should converge on a canonical identity tuple conceptually equivalent to:

`namespace + object_kind + canonical_identifier + version + content_digest`

The exact serialization belongs in the implementation PR.

Important distinctions:

- identity is not belief;
- identity is not provenance;
- version is not currentness;
- content digest is not physical execution evidence;
- a graph edge is not an authorization.

## Currentness and applicability

Currentness answers:

> Which version/state is considered current for this scope and time?

Applicability answers:

> Under which context/constraints does this object or claim apply?

Neither answers:

> Did this system physically execute correctly?

CP-04 therefore treats currentness and applicability as dependency boundaries while preserving the physical-execution claim ceiling.

## Contradictions and alternatives

The DKG can contain competing or contradictory claims. The projection layer MUST NOT silently collapse them into a single engineering fact.

A future typed projection should be able to represent:

- selected source claim;
- competing claim;
- contradiction relationship;
- selection rationale or policy;
- qualification impact;
- unresolved status.

The presence of a higher-confidence claim is not, by itself, permission to delete or hide a contradictory claim.

## Proposed integration sequence

1. **DKG archaeology + boundary contract** — this document and a source inventory.
2. **Canonical engineering object identity** — stable recursive identifiers and version/digest semantics.
3. **Recursive engineering relations** — composition and dependency edges without fixed depth.
4. **Temporal/currentness + applicability** — explicit time/context semantics.
5. **Recursive impact/invalidation closure** — deterministic engineering propagation, distinct from belief propagation.
6. **DKG -> CP-04 projection** — bounded typed DAG with endpoint/type validation.
7. **Replay identity + replay bundle** — exact projection/input binding.
8. **Cross-contract integration** — CP-01/CP-03 and existing authority owners.
9. **Contradiction/alternative projection** — preserve epistemic disagreement without authority escalation.
10. **Full recursive E2E** — requirement through requalification after a dependency mutation.

## First end-to-end acceptance scenario

Use one synthetic engineering scenario and mutate one dependency:

1. create a requirement;
2. compose a system containing a subsystem and component;
3. bind an implementation and model parameters;
4. bind runtime and toolchain;
5. bind accelerator and deployment artifact;
6. create an execution context;
7. attach observation, uncertainty, statistics, and provenance;
8. project the relevant DKG relations into a CP-04 evidence DAG;
9. compute deterministic closure and replay identity;
10. mutate the toolchain version/digest;
11. recompute impact;
12. verify only the dependent qualification lineage becomes stale/requires requalification;
13. preserve the historical record;
14. verify no provenance/confidence/graph connectivity change creates physical execution or operational authority.

## Non-goals

This contract does not:

- replace the Mycelix DKG;
- require all DKG relationships to be acyclic;
- claim that confidence propagation proves engineering correctness;
- claim that a synthetic qualification proves physical performance;
- create a new operational authorization system;
- duplicate canonical hardware, deployment, metrology, provenance, or governance stores.

## Relationship to CP-04

CP-04 remains deliberately narrow:

> deterministic typed dependency and invalidation semantics over synthetic/reference workflows.

The DKG can be richer than CP-04. A richer DKG MUST NOT make CP-04's claim ceiling wider.

The safe direction is therefore:

`richer DKG knowledge -> explicit projection -> bounded evidence qualification`

and never:

`richer graph connectivity -> implicit authority`
