# MACE Streaming Temporal Session Ledger V1

**Snapshot:** 2026-11-26  
**Status:** In-memory Rust reference state machine and authored deterministic tests published on `mace/streaming-temporal-memory-ledger-v1`. It is not yet a hosted AML Add/Search service, durable database, distributed event log, or proof of physical erasure. Rust tests and CI have not been executed in this environment.

## 1. Why this is a competitive priority

The Agent Memory Leaderboard Cycle 2 textual results show a 13.46-point commercial advantage in streaming and a 9.29-point advantage in temporal sequencing between the leading commercial and open-source systems. Personalization/care and epistemic safety/privacy gaps are 15.88 and 13.39 points respectively. These are not source-assurance scores; they point to a different memory-engineering problem: a memory system must update its representation as the world and the user's preferences change, without returning stale or superseded state.

MACE already has a raw-session BM25 reference adapter and a LongMemEval-V2 trajectory parser. Those preserve raw source material and support retrieval experiments, but they are not a transactional, event-sourced session state machine. The new `streaming_session` module adds that missing state-transition core.

## 2. Paradigm shift: memory as a bitemporal event ledger

Ordinary conversation memory often stores a mutable summary or a set of extracted facts. Updating a fact can silently erase when the old information was believed, when the correction arrived, or what earlier decisions used it.

MACE now models two times:

- **Valid/effective time:** when the event or corrected information applies in the represented world.
- **Observed/transaction time:** when the system received and committed the update under its trusted consumer clock.

The distinction matters for late-arriving information. A report received on November 26 may correct a fact that was effective on November 20. The system must not pretend the correction was known on November 20.

The event sequence is a consumer-controlled contiguous ordering, distinct from event time. This permits effective-time reordering without allowing event replay, gaps or clock rollback to rewrite the ingestion history.

## 3. New Rust reference module

Path: `research/broca-mace-transfer-bench-v0-1/aml-adapter/src/streaming_session.rs`

The typed mutation operations are:

| Mutation | Effect |
|---|---|
| Append | Adds a new source-preserving session message |
| Correct | Supersedes a prior active message and appends a corrected version |
| Quarantine | Temporarily removes a message from the active retrieval projection |
| Revalidate | Restores quarantined evidence after a validation receipt |
| Revoke | Makes a message inadmissible for future active retrieval |
| Erasure request | Removes the payload from this in-memory ledger projection under a rights receipt |

The consumer supplies the observed-time value. The mutation sequence starts at one and is contiguous. An exact replay of a previous event ID and mutation digest returns the original receipt without advancing the ledger; reusing an event ID with different content is rejected.

Each successful transition emits a hash-linked receipt containing:
- event ID and sequence;
- valid/effective and observed times;
- exact mutation digest;
- prior and resulting state digests;
- the active event IDs visible after the transition;
- a digest of the receipt itself.

The implementation stages a transition on a clone and commits only after the state transition succeeds, so a rejected mutation does not partially alter the live in-memory ledger.

Snapshots order active messages by valid timestamp and then original ingestion sequence. Superseded, quarantined, revoked and erased entries do not appear in the active snapshot.

## 4. Critical lifecycle and privacy boundary

Quarantine is reversible and epistemic: evidence can be restored after a validation receipt. Revocation is terminal in this reference state machine. Rights erasure is distinct: the module removes the payload object and payload digest from the active in-memory ledger projection, but this is **not secure or complete erasure**.

The append history retains event and mutation commitments, including one-way hashes that were computed from the original message. Those commitments may still be linkable or sensitive in some threat models. The module does not erase:
- process memory snapshots or crash dumps;
- backups, replicas, filesystem blocks or database journals;
- external retrieval indexes, embeddings, caches or derived summaries;
- third-party source material or previously issued receipts.

A production rights-erasure service needs a separate authority, data-lineage closure, deletion receipts from every storage/index/cache domain, backup expiry or cryptographic erasure, and independent verification that future retrieval cannot surface the erased payload. No such production erasure system is claimed here.

## 5. Correctness and failure semantics

The reference state machine fails closed on:
- non-contiguous sequence values;
- event-ID reuse with changed mutation content;
- duplicate turn IDs on append;
- consumer observed-time rollback;
- invalid message/digest shape;
- corrections targeting non-active or unknown events;
- corrections that change the target turn identity;
- revalidation of anything other than quarantined evidence;
- invalid transition receipts and repeated erasure requests.

Late corrections may use an effective timestamp earlier than the last event, while observed time remains monotonic. This is a required distinction for bitemporal state, not a clock rollback.

The reference uses public typed structs and assumes the caller has already parsed and authenticated the raw-session message. In particular, it validates the shape of the original raw-line digest but does not independently reconstruct original JSONL bytes from the parsed message. Production ingestion must validate raw bytes at its own boundary.

## 6. Deterministic regression fixtures

Ten test functions are authored:
1. append visibility and before/after receipt binding;
2. exact idempotent replay;
3. event-ID content conflict rejection;
4. sequence-gap and observed-clock rollback rejection;
5. correction supersedes prior content without overwriting history;
6. quarantine and revalidation;
7. terminal revocation;
8. rights erasure payload removal from the active in-memory projection;
9. late correction preserving valid time and observed time;
10. atomic failure with unchanged state digest and sequence.

These tests are authored but not compiled or executed. The current environment has no Rust toolchain; no Cargo, rustfmt, Clippy or hosted CI pass is claimed.

## 7. Benchmark programme for streaming and temporal memory

The new module should be tested on both the official Agent Memory Leaderboard Streaming and Temporal & Event Sequencing tasks and a separate lifecycle benchmark. Do not infer that an event ledger improves AML scores until the official harness is executed.

### Required baselines

- Mutable-summary memory with rewrite-on-update.
- Append-only event log without a current-state projection.
- Timestamp-sorted event log without transaction/observation time.
- Bitemporal event ledger without source qualification.
- MACE ledger with qualification, lineage, sufficiency and revocation checks enabled.
- Retrieval-only BM25 and hybrid BM25+dense with no lifecycle transitions.
- Full-history context baseline under the same reader/token/tool budget.

### Task families

1. **Streaming visibility:** append events rapidly, then query immediately. Measure Add acknowledgement, durable-commit and Search-visible times separately.
2. **Out-of-order arrival:** deliver observations with valid times earlier than the latest committed event. Preserve transaction order and valid-time ordering separately.
3. **Correction chains:** change a preference, address, project status or world-state fact multiple times; ask questions about both current and historical values.
4. **Supersession propagation:** derive multiple summaries or plans from a fact, then correct it. Verify that downstream projections are revalidated or invalidated rather than continuing to use the old value.
5. **Quarantine and revalidation:** inject uncertain or disputed evidence, then resolve it. Measure false drops, false retains and restoration latency.
6. **Revocation and rights withdrawal:** invalidate a source, identity scope or permission. No affected evidence may remain active for a consequential decision.
7. **Erasure:** request deletion of a source payload and all its dependent retrieval representations; verify the independent deletion receipts across every storage/index/cache domain.
8. **Concurrency and retry:** duplicate requests, repeated event IDs, sequence gaps, retries after unknown outcomes, process restarts and competing writers.
9. **Tenant/session isolation:** replay event IDs or memory IDs across user and session boundaries; require exact scope binding.
10. **Preference drift:** distinguish stable preferences from temporary or superseded preferences; measure personalization quality without stale preference use.

### Proposed metrics and targets

| Metric | Initial target |
|---|---|
| Exact mutation replay | 100% deterministic protocol fixtures |
| Event identity conflict | 100% of changed-payload reuse rejected |
| Sequence gaps / clock rollback | 100% rejected in exhaustive fixtures |
| Correction visibility | 100% of corrected fixtures expose the latest valid state |
| Historical reconstruction | 100% of frozen valid-time / observed-time queries preserve both timelines |
| Stale/revoked/superseded use | Zero on deterministic lifecycle fixtures |
| Quarantine correctness | Zero inadmissible messages in active snapshots; reversible restoration on valid revalidation |
| Rights erasure | 100% payload/index/cache deletion receipts in a future end-to-end deployment; the current module does not meet this service-level target |
| Streaming latency | Publish p50/p95/p99 for acceptance, durable commit and search visibility; freeze a deployment-specific service-level target only after measuring the baseline |
| Update cost | Report bytes written, tokens reprocessed, dependent projections revalidated, tool calls and all-in cost per mutation |

The benchmark must separately report false deletion, stale retention, abstention and valid-evidence loss. A high correction success rate achieved by deleting all prior context is not acceptable. Historical questions must remain answerable when policy allows, while current-decision retrieval must not use superseded or revoked state.

## 8. Industry comparison and strategy

The official Agent Memory Leaderboard evaluates streaming as a distinct capability from explicit fact recall and memory governance. That distinction is important: low-latency writes are not enough if the active state is inconsistent, and correct historical recall is not enough if the current answer uses a superseded fact.

MACE's competitive strategy should be:
- **Parity:** reproduce the official AML Streaming and Temporal & Event Sequencing leaders under the exact official harness.
- **State correctness:** add bitemporal correction and event-ordering cases the leaderboard may not fully isolate.
- **Lifecycle assurance:** connect streaming events to qualification, source lineage, decision sufficiency, revocation and rights erasure.
- **Decision impact:** measure whether the corrected memory improves subsequent agent actions on MemoryArena and causal/state-update tasks on AMA-Bench.
- **Cost-aware operation:** optimize update and revalidation work, not merely retrieval accuracy. For a corrected fact, compute dependency closure and update only the affected projections while retaining a verifiable receipt trail.

The proposed external targets remain in the MACE benchmark parity programme: beat the current open-source Streaming capability by at least five points, then the commercial leader by at least two points, with the official scorer. These are targets, not achieved MACE results.

## 9. Production work still required

The next engineering increments are:
1. a durable append-only event store with transactional compare-and-swap sequence allocation and create-once idempotency;
2. a streaming Add/Search adapter that acknowledges only after durable write and immediate Search visibility;
3. recovery/replay and reconciliation for unknown write outcomes;
4. a dependency index from source events to facts, summaries, preferences, plans and decisions;
5. correction/revocation propagation through all derived projections;
6. independent rights-erasure execution and deletion receipts;
7. concurrency, crash-recovery, retention and load tests;
8. matched AML Streaming/Temporal benchmark runs.

The current branch delivers the typed in-memory state machine and its tests. It does not yet deliver those production services.
