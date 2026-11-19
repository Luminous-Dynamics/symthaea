# MACE Agent Memory Leaderboard Add/Search Adapter v0.1

**Snapshot:** 2026-11-19  
**Status:** Protocol core and authored contract tests committed. No hosted endpoint, official smoke pass, Cycle 2 submission, or leaderboard result is claimed.

## Purpose

Prepare Symthaea/Mycelix for a future eligible Agent Memory Challenge (AML) cycle. AML owns evaluation orchestration, Answer and scoring; participants provide Add and Search. Luminous should expose a narrow adapter over its canonical evidence/memory architecture, not replace Mycelix with a benchmark-only store.

Mycelix owns provenance, bitemporal identity, authority, revocation and tenant isolation. Symthaea/Broca owns evidence-grounded consolidation and retrieval policy. Symtropy supplies isolated execution and independently reconciled outcomes. The AML adapter projects a benchmark-specific memory view.

## Official status, November 19, 2026

The [official competition page](https://agentmemoryleaderboard.ai/competition/) still labels Cycle 2 results expected November 19 in the content checked for this snapshot; its public content still displays Cycle 1. No Cycle 2 rank is asserted.

Cycle 2 materials closed October 31, 2026 at 23:59 UTC+8; Full evaluation closed November 4, 2026 at 23:59 UTC+8. New submissions are not available for this cycle.

Cycle 1 is historical context, not a Cycle 2 result. The published Textual commercial leader MemoraX v0.5 scored 58.02 overall, including 89.89 explicit fact recall, 63.44 compositional inference, 60 temporal reasoning, 51.19 governance, 58.51 personalization, 29.99 Context Learning & Execution and 58.39 safety/privacy. The Cycle 1 open-method textual reference recorded in the public results was 45.06 overall. Do not compare these values with MACE synthetic utility or across tracks/divisions.

## Add contract

Required fields:
- request_id: unique logical write identifier; retries reuse the same ID and exact request body.
- messages: ordered, non-empty array with role user/assistant, non-empty content, optional Unix-millisecond timestamp.
- user_id: required memory-isolation key; Add and Search must use exactly the same value.
- session_id: required source-session grouping identifier; it is not a Search filter.

Textual/Coding content is a string. Multimodal content can be a string or ordered text/image_url parts.

Successful Add returns HTTP 200 only after the write is durable and immediately searchable. success must be true and request_id, user_id and session_id must echo byte-for-byte. HTTP 202 is only supported with an approved Add-status route; this prototype does not implement it.

Textual chunks are split at the first of 20 messages or 2,000 words, counted by the frozen AML adapter. Streaming uses incremental session/task units; Coding preserves repository/session trajectories; Multimodal follows benchmark source units and media limits.

The Rust adapter injects a FrozenAdapterWordCounter and records its version in TextualChunkPlan. The whitespace counter in tests is test-only and must not be used for an official run. A locally invented tokenizer would not establish compatibility with the frozen counter.

Chunk IDs are deterministic: logical request ID plus chunk index. An in-memory idempotency helper binds each request ID to a length-prefixed SHA-256 request digest. Same ID/same digest replays the in-flight/committed state; same ID/different digest is rejected. A hosted service must persist dedupe state atomically with the memory write.

## Search contract

- query is the original benchmark question, never a candidate answer or gold label.
- options is optional, top-level, and for multiple-choice/Streaming queries; omit for open questions.
- user_id is the sole retrieval-isolation key. Cross-user retrieval is prohibited.
- formal top_k is 100; return no more than requested.
- response is an object with data, an array in retrieval-rank order. No matches means data is an empty array.
- each item requires stable non-empty id and content. Optional score is numeric; higher means more relevant. Optional created_at is source/persistence time.
- preserve order; do not silently truncate excess results or malformed records.
- Textual/Coding use strings. Multimodal uses ordered text/image parts and inline Base64 data URIs, not remote image URLs.

## Multimodal limits

The public API guide specifies JPEG/PNG/WebP, maximum 10 MiB decoded per image, 30 MiB image data per Add request, and 30 MiB across one Search response. The adapter checks MIME, Base64 alphabet/padding and decoded-size estimates. It does not perform full image decoding, malware scanning, normalization or OCR.

## Security and privacy

Use public HTTPS endpoints; never embed credentials in URLs. Keep API keys out of source, logs, traces and PRs. Treat user_id as tenant partition; do not rely on session_id for access control. Use evaluation data only for the current job, never for model training, fine-tuning, unrelated analytics, dataset reconstruction or redistribution. Delete data and derived copies within 30 days after run completion unless AML approves another period in writing. Return Add success only after persistence and Search visibility. Reuse identical request_id and payload on retry; a 409 alone is not success.

## Luminous projection model

A successful Add does not make a provider trace trusted memory. Preserve message order, source IDs, provider event identity and source-span links. Keep provider/assistant claims, tool intents, tool outputs, external observations and reconciled outcomes distinct. Link each derived memory to the exact source and extractor/consolidation revision. Preserve valid-time and knowledge-time. Keep retraction, supersession, privacy erasure and active-index forgetting distinct. Search returns relevant evidence packets, not generated answers or unsupported causal claims.

## Targets for the next eligible cycle

| Dimension | Programme target |
|---|---|
| API contract | 100% frozen smoke-contract cases |
| Add integrity | Zero early acknowledgements; exact response echoes |
| Idempotency | 100% same-ID/same-payload replay; zero payload rebinding |
| Isolation | Zero cross-user leakage in adversarial tests |
| Search | Valid shape, stable IDs, preserved order, no over-top_k output |
| Textual chunking | 100% parity with the frozen AML counter |
| Overall | At least +5 percentage points over strongest official result in same track/division |
| Governance/Safety | At least +10 points over same-track baseline without reducing legitimate-use coverage |
| Efficiency | Report ingestion/search cost, memory bytes, p50/p95/p99 latency and cost per correct item |

These are programme goals, not achieved results. Recalculate absolute thresholds after official Cycle 2 publication.

## Implementation boundary

The workspace contains broca-aml-adapter with 14 authored unit tests. Implemented: typed contract core, injected-counter Textual chunking, SHA-256 Add identity/idempotency helper, Add echo/persistence-gate validation, track-aware Search validation, Top-K/result-order checks and multimodal data-URI/media validation.

Not implemented: HTTP service, authentication middleware, persistent store, atomic database idempotency, embeddings/retrieval/ranking, verified official word counter, live smoke/full evaluation, Cycle 2 submission, or end-to-end wiring from #12731 consolidation receipts into AML Add/Search.

Rust compile/tests/rustfmt have not run in this environment. Source-level checks are not test passes. No official compatibility pass or SOTA advantage is claimed.

## References

- [Official competition page](https://agentmemoryleaderboard.ai/competition/)
- [Official Add/Search API guide](https://agentmemories.ai/api-guide)
- [Cycle 2 rules](https://agentmemories.ai/rules)
- [Cycle 1 public leaderboard](https://agentmemories.ai/)
