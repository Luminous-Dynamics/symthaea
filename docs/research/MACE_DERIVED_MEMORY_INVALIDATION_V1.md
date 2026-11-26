# MACE Derived-Memory Invalidation V1

**Snapshot:** 2026-11-26  
**Status:** In-memory dependency-index reference module and deterministic tests are draft code on `mace/derived-memory-invalidation-v1`. No durable projection database, external index deletion, production rebuild workers or hosted CI pass is claimed.

## 1. The stale-derived-memory problem

Correcting a conversation message is not enough. That message may already have contributed to:
- a compressed conversation summary;
- extracted facts or a user-preference model;
- a dense embedding or lexical index;
- a task plan, recommendation, answer cache or workflow skill;
- additional summaries and plans derived from those artifacts.

If only the raw message is corrected, a downstream projection can continue to influence a decision even though its source is no longer admissible. This is a central failure mode for long-term memory: the system may retrieve an answer that appears plausible while relying on obsolete information somewhere in its derivation chain.

The new design treats every derived memory artifact as a **versioned materialized projection with an explicit dependency graph**. A projection may be current only while every source event and parent projection it depends on remains current.

## 2. New Rust module

Path: `research/broca-mace-transfer-bench-v0-1/aml-adapter/src/temporal_projection.rs`

The module provides:
- typed projection kinds for summaries, embeddings, preference models, plans, fact indexes and answer caches;
- content and derivation-receipt commitments;
- exact source-event and parent-projection dependencies;
- a per-user/session sequence binding;
- transitive invalidation from source events through downstream projections;
- self-committing invalidation receipts;
- immutable rebuilds under new projection IDs;
- explicit replacement links from stale/quarantined versions to newly built current projections;
- erasure acknowledgements that clear the in-memory projection digest and dependency references after the caller provides a deletion-receipt commitment.

New projections can only be registered against the exact current ledger sequence. Direct source dependencies must be in the current active session snapshot. Parent projection dependencies must already be current. This prevents a new derived artifact from being constructed over an obviously stale source or stale parent projection.

## 3. Transition receipts now identify scope, targets and causes

The streaming ledger's transition receipt now binds:
- user and session identity;
- mutation identity and sequence;
- previous and resulting state digests;
- active event IDs after the transition;
- event IDs removed from the active view;
- source event IDs targeted by the lifecycle transition, including events that were already hidden;
- the invalidation cause: correction, quarantine, revocation or rights erasure.

The distinction between **invalidated event IDs** and **affected event IDs** handles an important edge case: an event may first be quarantined, then later revoked or erased. It is already absent from the active snapshot when the second transition occurs, but derived projections still need to escalate from quarantined to revoked/erasure-requested.

A projection invalidation receipt binds the exact source transition receipt, the index state before/after propagation, and the deterministic sorted list of affected projection IDs and status transitions.

## 4. Invalidation and rebuild semantics

| Source transition | Direct dependent projection status | Downstream effect |
|---|---|---|
| Correction / supersession | Stale | Descendants become stale |
| Quarantine | Quarantined | Descendants become quarantined |
| Revocation | Revoked | Descendants become revoked |
| Rights erasure request | Erasure requested | Descendants also require erasure |
| Revalidation of the original source | No automatic restoration | Old projections remain non-current until rebuilt |

A rebuild creates a **new projection ID** with a new content digest and derivation receipt. It must be built against the current session sequence and current source/parent dependencies. The old projection can then be linked to its replacement, but it remains a historical, non-current artifact.

Revalidation of an original message does not reactivate a summary, embedding or plan that was previously quarantined or stale. This is deliberate: the derived artifact itself must be rebuilt and re-attested.

Rights erasure is separate from ordinary staleness. After an erasure request, the projection index accepts a deletion-receipt digest and clears the in-memory content and dependency commitments for that projection. This is not proof that the external bytes, vector index, cache, replica, backup, or downstream artifact have been physically erased. The responsible storage operator and independent rights authority must verify those operations and provide a trustworthy receipt.

## 5. Deterministic tests

The module currently includes eleven authored test functions for:
1. source correction invalidating a dependent summary and downstream plan;
2. unrelated source events not invalidating an unrelated projection;
3. quarantine followed by revalidation not silently restoring an old derived artifact;
4. rebuilding a projection under a new ID and linking the replacement;
5. revocation and rights erasure cascading to descendants;
6. rights erasure after quarantine still escalating dependent projections;
7. cross-user/session dependency rejection;
8. exact invalidation receipt replay and conflict handling;
9. contiguous projection index sequence enforcement;
10. invalidation receipt self-integrity;
11. diamond-shaped projection dependencies invalidated exactly once.

The streaming ledger also carries its own event/correction/quarantine/revocation/erasure tests. These are authored tests; Rust compilation, test execution, rustfmt, Clippy and hosted CI have not been completed here.

## 6. Benchmark additions

The assurance benchmark should test the entire *source → evidence → projection → decision* dependency chain, not just whether the raw memory was updated.

| Metric | Proposed initial target |
|---|---|
| Direct dependency invalidation | 100% on exhaustive correction/quarantine/revocation/erasure fixtures |
| Transitive propagation | 100% of all descendants marked non-current |
| Stale projection acceptance | Zero on exhaustive lifecycle fixtures |
| Unrelated invalidation | Zero false invalidations on independent projection graphs |
| Rebuild correctness | 100% of rebuilt projections bind current source IDs, current parent projections and new derivation receipts |
| Revalidation resurrection | Zero old derived projections reactivated solely because a source event was revalidated |
| Revocation escalation | 100% of quarantined descendants escalated when the source is later revoked |
| Erasure propagation | 100% of dependent projections receive erasure-required status; external deletion receipts required before claiming erasure completion |
| Replay determinism | 100% same frozen transitions reproduce identical projection status and invalidation receipts |
| Cost | Report invalidation fan-out, rebuild count, index writes, bytes, tokens, p50/p95 propagation delay and all-in cost |

Required adversarial cases include shared descendants, diamond-shaped graphs, repeated corrections, simultaneous source dependencies, source revalidation after quarantine, revocation after quarantine, erasure after revocation, and a downstream plan that consumes two summaries where only one becomes stale.

Do not score only propagation recall. Measure both missed invalidations and false invalidations, because a system that marks every memory stale can achieve zero stale-use while destroying utility.

## 7. Relationship to external benchmarks

This layer is an assurance extension, not an AML leaderboard score. It should be evaluated alongside:
- AML Cycle 2 Streaming and Temporal & Event Sequencing;
- Memora/FAMA forgetting-aware accuracy;
- AMA-Bench causal inference and state updates;
- MemoryArena's interdependent agent-environment tasks;
- MEMOBench's process-level Storage, Update and Compression checkpoints;
- LongMemEval-V2 dynamic state tracking, workflow knowledge and premise awareness.

The central new evaluation question is: after a source is corrected or revoked, can the system prove that every downstream memory artifact and every later decision stopped depending on it, while preserving unrelated valid knowledge?

## 8. Production boundary

The current index is in-memory. Production requires:
1. a durable transactional projection registry;
2. a reverse dependency index with concurrency-safe updates;
3. recovery and reconciliation for unknown mutation outcomes;
4. rebuild workers that re-extract and re-qualify evidence;
5. separate authority/signature verification for derivation and rebuild receipts;
6. distributed invalidation across BM25, embeddings, caches, summaries, user preference stores and plans;
7. secure erasure protocols and verifiable receipts from every storage domain;
8. end-to-end concurrency, crash-recovery and propagation-latency benchmarks.

The reference module provides deterministic contracts for those services. It does not claim they are deployed.
