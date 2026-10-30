# Proof-carrying derived-state witnesses

Status: research implementation on `research/broca-proof-carrying-derivation-witnesses-2026-10-30`. Source tests are authored but not compiled or executed.

## Problem

MemDerive demonstrates that memory systems can contain the right intermediate state yet fail to gather it for a question requiring a derived answer. Its benchmark exposes the intermediate derivation DAG and supporting-event witnesses precisely because final-answer accuracy alone cannot locate this failure.

Broca's `DerivationGraph` already stores the dependency graph needed to explain a current fact, but until this change it exposed a fact or state digest—not a minimal, queryable proof witness.

## API added

`DerivationGraph::supporting_witness(fact_id)` returns `DerivationWitness`:

- `root_fact_id`: the requested current derived fact.
- `nodes_dependency_first`: each transitive derived premise exactly once, ending with the root. Every node includes the exact fact payload, dependencies, valid-time interval, rule ID, and rule version.
- `source_event_ids`: the unique terminal source-event IDs that must be resolved into source-bound evidence by the retrieval layer.
- `witness_sha256`: deterministic digest over root, ordered nodes, and source IDs. It is a parity identifier, not an authenticity signature.

The algorithm calls `verify_consistency` first; unknown roots, invalidated roots, missing/invalidated derived dependencies, and unavailable source events fail closed. It uses an explicit stack, so witness traversal does not recurse on the Rust call stack. BTree ordering plus dependency-first traversal makes node order and the digest deterministic.

## How this changes query execution

A memory query can now be planned as a proof-obligation workflow:

1. A query planner identifies a candidate derived fact or rule goal.
2. The derivation graph returns the minimal witness for that fact.
3. The evidence layer resolves the witness's source-event IDs through its source-pointer and content-digest checks.
4. Role/scope authorization is applied before source content is materialized.
5. The answerer receives the required leaf evidence and intermediate rule/version state, not merely the nearest lexical/HDC neighbours.
6. The final answer carries a structured derivation reference that can be audited independently.

This provides the retrieval side of query-driven premise gathering. It does **not** yet implement natural-language query decomposition or a complete Datalog evaluator.

## Tests authored

- A hand-built multi-level graph returns only the transitive supporting events and intermediate derived states, in dependency-first order.
- A diamond-shaped graph deduplicates the shared ancestor and produces the same witness regardless of insertion order for independent branches.
- Unknown and invalidated roots are rejected; source retraction invalidates the witness.
- Twelve deterministic generated DAGs are compared against a separately computed fixed-point closure for both derived-node and source-event sets. Every parent must appear before its dependent node, and repeated extraction must preserve the digest.

These tests have not been compiled or run in this environment.

## Scope limits

- The current `DerivationGraph` represents one conjunction of dependencies per fact. Alternative proof support is handled by the separate `memory_supports` prototype; these models are not yet unified into one derivation algebra.
- The witness lists source-event identities, not evidence payloads. The caller must resolve them from an authenticated evidence index and preserve exact source/payload digests.
- This is not a rule engine. It does not implement stratified negation, aggregation, closed-world/completeness semantics, or the complete MemDerive Datalog language.
- The graph is an in-memory reference model using clone-validate-swap updates; durable transactional persistence, crash/replay parity, and distributed invalidation remain future gates.
- The witness digest is not signed. Authenticity and anti-rollback must come from the attestation and persistence layers.

## Evaluation gates

On MemDerive, report more than final answer accuracy:

- supporting-event retrieval recall/precision and witness F1;
- intermediate-node answer accuracy;
- derivation-DAG exact coverage;
- implicit-change refresh and unchanged-answer preservation;
- token, query-latency, ingest-latency, and witness-size costs;
- 100% agreement with an independently implemented full-closure oracle on finite generated graphs.

A fair system comparison must pin the official dataset/evaluator, answer model, prompts, context budget, and inference settings. The published MemDerive BM25 and full-context/supporting-event references are controls, not scores for Broca.

Primary benchmark: [MemDerive](https://arxiv.org/abs/2610.25373).
