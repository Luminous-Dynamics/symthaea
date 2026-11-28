# MACE TimlMemBench raw-native protocol adapter v0.1

**Status:** standalone JSONL protocol kernel and host-runtime interface implemented with synthetic fixtures. No official TimlMemBench data has been run, and this crate does not yet embed a production MACE runtime.

## Official protocol compatibility

The official TimlMemBench adapter is a long-running UTF-8 JSONL process: one request per input line, one response per output line, logs to stderr. Its raw-native lifecycle is `reset`, chronological `ingest_session`, `query`, `close`. Raw-native queries contain only sample/task identity, task type, case type, metric names and the current query. The official runner must not send expected behavior, answers, forbidden spans, expected state identifiers or scorer labels.

This adapter supports only `raw_native_cases`. Canonical-pool LLM-QA is deliberately a separate control track: it provides prepared memory items and cannot be blended with the raw-native result.

## Trust and state invariants

- `reset` establishes a fresh per-sample namespace and calls the host runtime's reset before acknowledging.
- Session ingestion is only allowed inside that namespace. Duplicate session IDs and duplicate turn IDs are rejected. Every session is hashed and assigned an ingest ordinal; the ordered session digests form a sample ingest manifest.
- Session payloads are scanned recursively for scorer-only keys such as expected answers/behavior, forbidden spans/state IDs, gold state IDs, answer keys and evaluator rationales. Unknown top-level request fields fail strict deserialization.
- Cross-sample and cross-language ingestion/query attempts fail before the host runtime is called.
- An ingest acknowledgement must match the session ID, canonical session digest and turn count and must state that persistence completed.
- The host may report raw-session claims as unverified until it has source/authority receipts. Raw dialogue is not automatically promoted to trusted facts.
- The host returns a typed evidence trace with the answer. `accept_and_act` is rejected unless a capsule digest exists, every exposed span has one canonical #13973 `DecisionEvidenceUseLinkV1` digest (ordered by context byte span), use-time revalidation covers every exposed item, unresolved evidence is zero, dependency closure is complete, provenance has been verified and hard gates pass.
- Any non-accepting decision path must return the benchmark's explicit `<ABSTAIN>` answer. The adapter will not turn retrieval failure, uncertain lineage or a failed hard gate into a confident answer.
- Query responses retain the official `answer`, `retrieved_memory_ids`, `retrieved_texts`, `metadata` and `usage` fields. MACE evidence lineage and ingest-manifest digests are attached under metadata. A separate strict response schema also binds the reset and ingest receipts and requires full evidence-trace conditions for accepted answers.

## Host runtime seam

`MaceTimlRuntimeV1` is deliberately injected. A real host must implement sample reset and namespace isolation; chronological raw-session ingestion preserving speaker, turn, timestamp, surface, language and tags; evidence-aware retrieval and decision policy; use-time revalidation and context-exposure links; resource accounting; and deterministic close/flush behavior.

The adapter does not implement a fake lexical engine and does not claim a synthetic test double is MACE. This keeps protocol and memory engine separate and lets the existing MACE/Broca runtime be integrated through a reviewed host implementation.

## Synthetic tests

Twenty deterministic tests cover official lifecycle sequence, scorer-label leakage in queries and sessions, cross-sample isolation, duplicate session/turn IDs, query-before-ingest, canonical/raw track separation, acceptance gate enforcement, unresolved/incomplete provenance rejection, hard-gate abstention, ingest-receipt binding, reset isolation JSONL close behavior, per-span link count, malformed link digest and duplicate-link rejection.

## Versioned JSON contracts

- `data/schemas/mace_timlmembench_raw_native_protocol_v1.schema.json` defines the strict request surface and recursively excludes scorer-only fields from raw sessions.
- `data/schemas/mace_timlmembench_raw_native_response_v1.schema.json` defines reset/ingest/query/close/error responses and the evidence-trace conditions required for accepted answers.

## Run

```bash
cargo test --manifest-path research/mace-timlmembench-adapter-v0-1/Cargo.toml
cargo clippy --manifest-path research/mace-timlmembench-adapter-v0-1/Cargo.toml --all-targets -- -D warnings
```

## External benchmark requirements

Before reporting TimlMemBench results, pin the official data commit, license, split, language, adapter host commit, reader and judge model/revisions, prompt, API endpoint and run date. Run canonical-pool and raw-native tracks separately. Keep the nine official axes separate. A heuristic smoke adapter is not paper-comparable; the official reproduction path uses semantic LLM judging.

Related work: #8427, #13825, #13346, #14033, #13967, #13973 and #13985.