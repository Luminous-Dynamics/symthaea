# Broca lexical evidence baseline and bounded worker

**State:** Rust worker and focused tests authored; not compiled or executed in the current environment. No benchmark score is claimed.

This branch is stacked on [PR #8682](https://github.com/Luminous-Dynamics/symthaea/pull/8682) and extends it with a transparent lexical retrieval baseline plus a persistent JSONL worker.

## What this increment adds

- `EvidenceRecord.source_pointer` binds an item to the exact string in the canonical trajectory JSON, not merely to a digest of the whole source object.
- `EvidenceIndex::insert_batch_atomic` validates every record and pointer before any mutation, so a malformed final record cannot leave half a trajectory indexed.
- `EvidenceIndex::search_bm25` provides deterministic, model-free BM25 retrieval over active text evidence, after scope/time/lifecycle and payload-digest checks.
- The `broca-lme-worker` binary accepts bounded JSONL requests, supports one pinned `web` or `enterprise` scope per process, ingests official-style trajectory objects, and returns reader context plus a separate provenance/score audit.
- The worker excludes root-level `outcome`, question IDs, question text, answer fields, screenshots and evaluator metadata from ingestion. It deliberately does not claim visual understanding or semantic embeddings.

## Build and focused tests

From the repository root when a Rust toolchain is available:

```sh
cargo test -p symthaea-broca --lib memory_evidence
cargo test -p symthaea-broca --bin broca-lme-worker
cargo run --locked -p symthaea-broca --bin broca-lme-worker
```

The environment used to author this change has no `cargo` or `rustc`. These commands are the verification gate; test functions present in source are **authored, not passed**.

## JSONL protocol

Every request must contain a non-empty `request_id` and an `op`. The first valid request must be `init`. The process handles one request at a time and writes exactly one JSON response line per request.

### Initialize the worker

```json
{"request_id":"1","op":"init","scope_id":"web","now_unix_ms":0,"bm25":{"k1":1.2,"b":0.75,"max_items":5}}
```

Only `web` or `enterprise` is allowed for LongMemEval-V2. Defaults are BM25 `k1=1.2`, `b=0.75`, `max_items=5`. These are un-tuned baseline parameters. Fix them before held-out evaluation and put the configuration in the run receipt.

### Insert a trajectory

```json
{"request_id":"2","op":"insert","trajectory":{"id":"traj-1","goal":"Open billing settings","start_url":"https://example.invalid/","states":[{"url":"https://example.invalid/billing","action":"click Billing Settings","text":"Billing Settings page"}]}}
```

The worker canonicalizes the trajectory with `serde_json`, derives deterministic IDs from the trajectory digest and JSON Pointer, then validates the complete batch atomically. Each returned text must exactly match the source string selected by its pointer. Root `outcome` and evaluator/question metadata are never candidate fields.

### Query

```json
{"request_id":"3","op":"query","query":"Where is Billing Settings?"}
```

A successful response includes:
- `result.context`: an array of official-shaped `{"type":"text","value":"..."}` items for the reader;
- `result.audit`: evidence and trajectory IDs, step/pointer, source/content digests, and BM25 scores;
- capability flags marking this arm as lexical-only and non-visual.

The adapter must pass only `result.context` to the reader. The separate audit is for receipts and analysis, not hidden reader prompt content.

### Bounds and failure behavior

- Maximum input JSONL line: 32 MiB.
- Maximum individual evidence text field: 4 MiB.
- Maximum evidence items per trajectory: 50,000.
- Malformed JSON, wrong request IDs, duplicate trajectory IDs, unknown operations, invalid source pointers, digest mismatches, and ID collisions fail closed.
- A non-empty image query is surfaced as `image_query_received_but_not_used`; no visual capability is claimed.
- Worker state is process-local. Persistent storage, save/load support, deletion across restarts, and a Python `Memory` adapter are **not implemented yet**.

## Experiment design

Compare this baseline with the exact pinned official arms, at matched reader-model/prompt/context/token settings:
1. `no_retrieval`;
2. official `rag_query_to_slice`;
3. this `broca_bm25` arm;
4. dense retrieval and the Broca HDC/hybrid arms only after their text query encoder is defined and separately evaluated;
5. AgentRunbook-C V2 as an additional current baseline while preserving the official historical LAFS reference frontier.

Report official LongMemEval-V2 metrics unchanged, per-question paired outcomes, p50/p95 query latency, retrieved context tokens/bytes, indexing time, memory storage, and governance failures. The BM25 score is a ranking diagnostic, not a task-success result. Keep synthetic and real benchmark results separate.
