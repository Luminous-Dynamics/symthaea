# AML Cycle 2 Coding Memory Adapter Contract

**Status:** research/implementation contract; no hosted service or benchmark submission is claimed.

**Contract source:** [Agent Memory Leaderboard API Guide](https://agentmemoryleaderboard.ai/api-guide), checked 2026-10-30.

## 1. Purpose and boundary

The Coding track evaluates retrieval, filtering, and reuse of engineering experience from the same repository under relevant and noisy conditions. Participant-controlled behavior is Add and Search; the platform owns Answer and scoring.

This work adds Rust wire types, validation, idempotency digest helpers, and an abstract storage/search backend boundary. It does **not** yet implement:
- HTTP routing, TLS termination, endpoint authentication or rate limiting;
- a concrete persistent backend / durable idempotency table;
- tenant isolation inside a concrete index;
- deployment, capacity engineering, platform access approval, Smoke, or Full evaluation.

The absence of these components is intentional and explicit; a Rust type layer is not a deployable competition entry.

## 2. Official Text/Coding wire contract

### Add request

Required JSON shape:

```json
{
  "request_id": "eval:<run_id>:repo:<session>:chunk-0",
  "messages": [
    {"role": "user", "timestamp": 1793350800000, "content": "debugging notes"},
    {"role": "assistant", "content": "root cause and fix"}
  ],
  "user_id": "eval:<run_id>:repo-42",
  "session_id": "eval:<run_id>:repo-42:session-0"
}
```

`messages` is an ordered array; roles are `user` or `assistant`; content is non-empty text; timestamp is optional Unix milliseconds. Preserve message order and source timestamps. The Coding track preserves repository/session engineering trajectories; do not apply the Textual track's deterministic 20-message / 2,000-word chunking rule to Coding requests by assumption.

`user_id` is the sole memory-isolation key. `session_id` organizes provenance and may group source history, but is not a Search filter.

### Add response

Return HTTP 200 only after the request has been durably committed and its memories are immediately searchable:

```json
{
  "success": true,
  "request_id": "exact request_id",
  "user_id": "exact user_id",
  "session_id": "exact session_id"
}
```

All three identifiers must be echoed byte-for-byte. HTTP 202 is only permitted if the submitted version has an approved Add task-status endpoint and includes its `task_id`; this adapter targets the simpler synchronous HTTP 200 contract.

Retries of a logical Add retain the same `request_id` and identical payload. The Rust module computes a stable payload SHA-256 and classifies same-payload retries as replay and same-ID/different-payload reuse as a conflict. A real backend must persist this receipt transactionally; the helper itself is not a durable ledger.

### Search request

```json
{
  "query": "Which previous change fixed this repository test?",
  "user_id": "eval:<run_id>:repo-42",
  "top_k": 100
}
```

`options` is optional and top-level for multiple-choice tasks; omit it for open questions. Do not send `session_id`, filters, `rerank`, `keyword_search`, or other unapproved fields. Search must use only the same `user_id` isolation boundary used at Add.

### Search response

```json
{
  "data": [
    {"id": "mem_123", "content": "Relevant engineering evidence", "score": 0.92, "created_at": "2026-10-29T08:00:00Z"}
  ]
}
```

The `data` array is mandatory; return `{"data":[]}` for no results. Each item requires a stable non-empty `id` and non-empty `content`; `score` and `created_at` are optional. Higher scores indicate greater relevance. The platform preserves participant result order. Do not return more than `top_k`; excess/invalid items are contract errors, not silently truncated. Unknown Search-response fields are ignored by the platform, so the Rust Search response type also tolerates them; this participant should emit only declared fields.

**Search returns evidence, never a generated final answer.** It must not receive or return gold answers, hidden labels, evaluator data, or prompt-injection instructions designed to manipulate the scoring pipeline.

## 2A. Official scored Coding benchmark

The current AML Coding scored suite is **CAMBench Coding**: 150 software-engineering tasks are run under relevant and noisy memory conditions, for 300 scored attempts. The documentation lists SWEContextBench as a reference catalog, not the current scored Coding benchmark. Do not substitute a SWEContextBench score or a Textual leaderboard score for CAMBench Coding.

The relevant/noisy pair is especially important for Broca: measure the score on relevant memories, the score under noisy memories, and the paired degradation. The public evaluation remains the official result; internal diagnostics should additionally record gold-evidence Recall@K / nDCG@K where licenses permit, stale-memory use, distractor intrusion, correct-repository attribution, relevant-to-noisy task-success delta, Search p50/p95 latency, returned evidence bytes/tokens, and storage/update cost. Keep each metric distinct; do not invent a blended score.

Before attempting a SOTA claim, establish the released CAMBench version, official evaluator, fixed answer model/prompt, model restrictions for the submission division, and exact task population. The official Cycle 2 results are scheduled for mid-November; as of 2026-10-30 there is no verified Cycle 2 public Coding result to beat.

## 3. Runtime, authentication, and privacy requirements

- Add/Search endpoints must be publicly reachable over HTTPS; do not embed credentials in endpoint URLs.
- Authentication supports Token, Bearer, and X-Api-Key; unauthenticated access is limited to public Smoke. An unauthenticated GET health endpoint may return any 2xx.
- HTTP 400/422: invalid payload; 401: invalid credentials; 403: forbidden user scope. Malformed bodies on 2xx still fail validation.
- Add retries can include 408, 409, 425, 429, 500, 502, 503, 504, and 524 (bounded by the platform). Search retries include 408, 425, 429, 500, 502, 503, and 504; Search 409 is not retried.
- Do not silently truncate Add/Search payloads or archived responses.
- Evaluation content is solely for evaluation: do not train on it, analyze/share it outside the evaluation, or reuse it for another purpose. Keep logs minimal and delete evaluation data within 30 days after the run.

## 4. Open-source Methods model restrictions

The official guide states that Open-source Methods entries must use:
- `text-embedding-v4` for embeddings;
- `gpt-4o-mini` for LLM-related components;
- rerankers are unrestricted.

These are competition-specific constraints, even though the core Broca architecture is Rust/HDC-first. The participant must disclose the actual models and methods; this module does not wrap, emulate, or bypass those requirements. If these model restrictions cannot be met for the intended submission division, do not represent the entry as compliant.

## 5. Code boundary

`crates/domains/symthaea-broca/src/memory_challenge_api.rs` defines:
- typed Add/Search request and response envelopes;
- validation for IDs, message order/content, options, top_k, stable IDs, duplicate IDs, score finiteness and ranked score order when all scores are supplied;
- byte-exact Add response echo validation;
- canonical Add payload SHA-256 and a retry classification helper;
- `AmlMemoryBackend`, the implementer boundary for persistent Add and user-scoped Search;
- `execute_aml_add`, which refuses success unless the backend receipt matches request ID and payload digest and confirms both durability and immediate searchability;
- `execute_aml_search`, which validates but does not re-rank or rewrite backend evidence.

The backend must still implement durable idempotency, correct user partitioning, source-order retention, ranking, and lifecycle deletion. The contract trait is not evidence that any of those have been implemented.

## 5A. Local persistent lexical baseline

The stacked research PR adds `memory_challenge_sled.rs`, a local Sled-backed implementation of the `AmlMemoryBackend` trait:
- Each Add request is stored as one immutable batch containing the exact ordered request, payload digest, stable evidence IDs, source pointers, session provenance, and insertion timestamp.
- A single-key Sled compare-and-swap binds each request ID to one payload; reusing an ID with different content is rejected.
- Add builds a candidate EvidenceIndex, verifies every new record is retrievable with its content query, commits the serialized batch, calls Sled `flush()`, then swaps in the candidate index before the executor can return success.
- Startup verifies each persisted batch against its canonical request digest, generated evidence records, and canonical request-derived storage key before rebuilding the lexical index.
- Search uses deterministic BM25 with `user_id` as the only scope. Different sessions for the same user are searchable together; the same session label for a different user does not share evidence.
- Explicit cutoff and full-purge methods remove persisted batches and rebuild the index.

This is only a **local research baseline**. It is not a deployment-ready competition entry:
- it uses BM25 rather than the required `text-embedding-v4` embedding component for Open-source Methods;
- it does not provide the required competition model configuration, HTTP/TLS/auth, a public endpoint, horizontal scaling, or hosted operations;
- it is a single-process in-memory lexical index over a local Sled store; use one active backend owner per store;
- lifecycle retractions, policy/use-validity checks at Search time, and proof packet compilation are not yet integrated into this backend;
- the retention purge is an explicit operator call, not a scheduled job, and logical deletion does not prove forensic media erasure;
- Sled 0.34 is documented as beta; its own docs recommend SQLite when reliability is the primary constraint. Do not claim production durability without a production database and crash/recovery qualification.

The local test suite exercises persistence across reopen, immediate BM25 retrieval, idempotent replay, conflicting ID reuse, user isolation, cross-session visibility for one user, top_k, purge, corrupt batch rejection, and non-canonical key rejection. The tests are authored but must be compiled and executed at the exact stacked PR head.

## 5B. SQLite/WAL persistence backend (preferred local baseline)

The repository workspace already defines `rusqlite 0.37` with bundled SQLite. The stacked research implementation adds `memory_challenge_sqlite.rs` and exposes `SqliteAmlMemoryBackend` behind the same `AmlMemoryBackend` trait.

Its contract:
- Open one local SQLite database file; configure `journal_mode=WAL`, `synchronous=FULL`, foreign keys, and a bounded busy timeout on each connection.
- Store one immutable Add batch per exact `request_id` primary key, including the complete ordered Add request, payload digest, generated evidence records, and storage timestamp.
- Use `BEGIN IMMEDIATE` to serialize writers before idempotency lookup and insert; same request ID/same digest replays, while same ID/different digest is a conflict.
- Validate a candidate BM25 index and each message's immediate retrievability before inserting.
- Commit the request and records in one SQLite transaction. Under WAL with `synchronous=FULL`, SQLite syncs the WAL at each transaction commit; success only follows a successful commit. See the [SQLite synchronous pragma documentation](https://www.sqlite.org/pragma.html#pragma_synchronous) and [WAL documentation](https://www.sqlite.org/wal.html).
- On restart, validate row metadata, payload digest, canonical source records, and schema version before rebuilding the in-memory evidence index.
- Explicit cutoff and full-purge operations remove logical rows in SQLite transactions and rebuild/clear the index.

The SQLite implementation has 10 authored tests covering persistence/reopen, immediate retrieval, cross-connection idempotency, conflicting request-ID reuse, user isolation, cross-session retrieval, top_k, WAL/FULL PRAGMA verification, retention purge, malformed rows, and failed-batch atomicity.

### SQLite backend deployment boundaries

SQLite WAL is a local/same-host storage mode, not a multi-host database: the SQLite documentation states that WAL does not work over a network filesystem and all processes using the database must be on the same host. The implementation therefore remains a **single-node local baseline** with a process-local EvidenceIndex cache and one active cache owner. It is not an HTTP deployment, distributed store, or substitute for service-level authentication, access control, backups, secret management, monitoring, and operator-run retention/erasure controls.

WAL mode also creates `-wal` and `-shm` sidecar files. Logical row deletion is not forensic erasure; when an evaluation's retention window expires, the operator must dispose of the main database, sidecars, backups, and snapshots under the applicable data-deletion policy.

### Sled comparison status

`memory_challenge_sled.rs` remains available as an experimental backend comparison. Sled 0.34 documents itself as beta; keep it for research comparison only. Prefer SQLite/WAL for the local single-node persistence baseline. Neither backend is yet the final public competition service, and neither removes the requirement to use `text-embedding-v4` for embeddings in Open-source Methods.

## 6. Conformance artifacts

- [JSON Schema](./coding-api-schema.json)
- [Positive/negative conformance vectors](./conformance-vectors.json)
- Rust unit tests in `memory_challenge_api.rs`

The fixture suite covers malformed requests, retry idempotency, exact echo, empty results, rank order, top_k overflow, stable result IDs, cross-user isolation obligations, immediate read-after-write, no answer generation, and privacy deletion. The deterministic vectors are compatibility tests, not a replacement for the platform Smoke test.

## 7. Submission gates

A valid entry still requires a fixed commit and version, publicly reachable endpoints, authentication/capacity/operational details, original-work attribution, platform access review, an issued Eval Key, a passing Smoke test, and a completed Full run. Open-source repository materials do not substitute for a deployed service. No endpoint, Eval Key, Smoke result, Full score, or leaderboard result is claimed by this repository work.
