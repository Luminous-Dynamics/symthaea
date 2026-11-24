# MACE AML Add/Search adapter contract v0.1

**Status:** contract and fixtures only; not an HTTP server, storage engine, deployed API, or AML submission.

**Tracks:** #13359. Composes with #13386 (candidate/source lineage), #13390 (on-policy total-effect evaluator), #13391 (label-blind retrieval-policy memory), #13410 (decision-scoped qualification), #13436 (pinned qualification trust), and #13446 (bitemporal rights-scoped evidence ledger).

## Why this is a separate integration layer

MACE already has internal candidate evidence, policy memory, qualification, temporal rights and causal evaluation work. The missing boundary is the participant-hosted Add/Search wire interface. AML Cycle 2 evaluates three separate tracks (Textual, Coding, Multimodal), and its official flow requires a participant-hosted Add and Search API; a GitHub repository or Docker image alone is not an endpoint. Cycle 2 closed on 2026-11-04 and its official results were released on 2026-11-19. This contract prepares a future cycle; it does not claim entry into Cycle 2.

Official references:
- API contract: https://agentmemoryleaderboard.ai/api-guide
- Cycle 2 results: https://agentmemoryleaderboard.ai/results/cycle-2/
- Public protocol source: https://github.com/AML-memory/agent-memory-leaderboard

This snapshot was authored against the public API guide as inspected on 2026-11-24. AML may change its protocol: a future submission must re-pin the current guide, API version, track-specific rules and organizer smoke test before deployment.

## Wire boundary

The intended adapter paths are deployment-configurable; the examples use:
- GET /health — unauthenticated readiness check; return any 2xx only when the service can accept work.
- POST /add — synchronous ingest; return HTTP 200 only after data is durably committed and immediately searchable.
- POST /search — retrieve ranked memory evidence; return {"data":[]} when no admissible candidates exist.

Supported authentication schemes are Authorization: Token ..., Authorization: Bearer ..., and X-Api-Key: .... Production must use authentication and HTTPS. The unauthenticated option is for public smoke only. Credentials must never appear in URLs, fixtures, logs or output.

The four JSON Schemas in data/schemas/ freeze the public request/response shapes used by this adapter:
- mace_aml_add_request_v1.schema.json
- mace_aml_add_response_v1.schema.json
- mace_aml_search_request_v1.schema.json
- mace_aml_search_response_v1.schema.json

The schemas intentionally reject undocumented request fields (including metadata, app_id, agent_id, async_mode, filters, rerank, and keyword_search). They also describe only fields MACE is supposed to emit. Internal evidence IDs, rights proofs, authority receipts, candidate-DAG links, evaluator labels and policy state stay inside the service and are never smuggled into the AML wire payload.

## Exact request semantics

### Add

Required fields:
- request_id: immutable identifier for one logical chunk write; all retries reuse the same ID and exact payload.
- messages: non-empty ordered list; each message has role (user or assistant) and non-empty content; optional timestamp is Unix milliseconds.
- user_id: sole AML retrieval-isolation key; Add and Search must use the exact same value.
- session_id: source conversation/session identity; used to organize memory, never as the Search filter.

Textual/Coding content is a non-empty string. Multimodal content is an ordered array with text and/or inline image_url parts. Images use Base64 data URIs for JPEG, PNG or WebP; remote URLs are not accepted. Preserve order and original media bytes. Never replace the only copy of an image with a caption.

Ordinary Textual input is deterministically chunked at either 20 messages or 2,000 words, using the frozen organizer-compatible word-count adapter. Streaming input is added as incremental frozen session/task units and Add/Search may interleave. Coding history retains repository/session trajectory context. No chunker may reorder messages or silently drop the tail.

### Add response

Return HTTP 200 and exactly the required echo fields:
success=true, request_id, user_id, session_id.

**Durability condition:** success means the full write is durable and query-visible before the response is sent. HTTP 202 is not part of this synchronous contract unless a future organizer-approved Add-status URL is explicitly bound to the deployed version.

### Idempotency and corrections

The adapter must provide exactly-once effects over at-least-once HTTP delivery:
1. First request_id + payload: persist once, update the searchable projection, commit the idempotency record and response atomically.
2. Same request_id + byte-identical/canonically identical payload: return the original successful echo without duplicating memory.
3. Same request_id + different payload: reject as a non-retryable client conflict (HTTP 422); never overwrite the first write.
4. A timeout or unknown commit result is reconciled by reading the idempotency record before another mutation. Unknown outcome means no blind second append.

The public Add schema has no correction/retraction fields. Corrections, revocations, supersession and erasure are handled by the internal lifecycle/evidence plane (including #13446), not invented as undocumented AML request fields. A correction is a new governed event that supersedes the prior claim; it is not an in-place rewrite.

### Search

Required fields:
- query: the original question, string for Textual/Coding or ordered content parts for Multimodal. Do not answer it inside Search.
- user_id: exact same isolation value used by Add.
- top_k: maximum returned item count. The formal AML configuration uses 100; never exceed the requested number.

Optional options is a top-level list of multiple-choice options. It must not contain gold answers. Do not send undocumented request filters or ask the evaluator's gold-label store for help.

Search response must be an object with a relevance-ordered data array. Each item contains a stable non-empty id and non-empty content; score and RFC 3339 created_at are optional. Return an empty array when nothing is admissible. Do not return a generated answer, do not wrap the array in items, do not silently truncate an oversized response, and do not emit an answer as a purported memory.

Multimodal search results may return ordered text/image content parts. Inline image payloads must be valid Base64 data URIs and must remain resolvable for the required evaluation retention window.

## Internal MACE processing order

The external API is intentionally simple. Internally, an Add should be processed as:

1. Authenticate the API principal and bind its tenant/service identity.
2. Validate request size, media types, content ordering, user/session identifiers and idempotency identity.
3. Durably append the raw source record and immutable source commitment.
4. Derive searchable text/HDC/graph/capsule projections while retaining a reversible pointer to original evidence.
5. Record the completion receipt only after the write and searchable projection are committed atomically.
6. Apply retention policy and ensure source content can be resolved for the evaluation's permitted use.

Search should be processed as:

1. Authenticate principal and authorize the exact user_id scope.
2. Resolve candidate identifiers, then filter by principal/tenant/user scope, purpose, authority, valid-time, knowledge-time, consent, retention, correction/supersession, retraction, erasure and required parent lineage **before exposing candidate content**.
3. Re-evaluate admissibility at the actual use boundary; a cached EvidenceBound<T> is a query-specific snapshot, not a reusable authorization token.
4. Rank only admissible candidates; preserve stable tie-breaking and original content/part order.
5. Return up to top_k admitted memory records and no answer.

The user_id field is the sole benchmark-level isolation key, but it is not sufficient as an internal security principal. The API-key principal and tenant boundary must also be checked server-side. session_id is organization metadata, not an alternate way to widen or narrow a Search request.

The rights/qualification projection may reduce the number of candidates below top_k. That is a correct empty/short answer, not a reason to bypass rights checks to fill the quota.

## Bounds, errors and operations

- Image formats: JPEG, PNG, WebP only. Maximum 10 MiB decoded per image; maximum 30 MiB total decoded media per Add request and per Search response. A JSON Schema cannot prove decoded byte size: the runtime must check the actual decoded bytes and verify file signatures rather than trusting MIME labels.
- One API request may be allowed up to the organizer's 30-minute request limit. Production server timeouts must be configured consistently with the submitted service capacity.
- Search response count must be <= top_k. An oversized result is an error, not silent truncation.
- Health is unauthenticated and must not disclose secrets or memory data.
- Use bounded concurrency, backpressure, request-body limits, timeouts, deterministic pagination internally, and actionable errors without payloads or credentials.
- Respect organizer retry semantics. Never turn a transport retry into a duplicate write.
- The current guide lists 400/422 for contract errors, 401 for authentication, 403 for denied scopes, 408/425/429 and transient 5xx for bounded retry, and 409 for temporary conflicts. A request-ID payload rebind is a permanent 422 in this adapter.
- The published evaluation data is evaluation-only: do not train on it, analyze or share it outside the evaluation, keep it private, and delete it within 30 days after the run. Production retention must be tied to a declared evaluation/run lifecycle, not an unbounded default.

## Competitive benchmark plan

Cycle 2's leaderboards are **separate by track and division**. Do not compare their scores as if they were one scalar:
- Textual: MemoraX 69.26 (commercial/closed) and Resurface 61.22 (open-source).
- Coding: MemoraX 68.00% resolved (commercial/closed) and InsightMem 63.33% resolved (open-source).
- Multimodal: MemoraX 53.68 (commercial/closed) and Khipu 43.26 (open-source).

These are Cycle 2 reported results, not a current Symthaea score or proof that the systems are directly comparable outside that cycle's fixed contract.

Suggested next-cycle qualification targets, not claims:
- Textual: >= 66 overall (at least 4.78 points above the Cycle 2 open leader 61.22), with no regression in safety/privacy or governance.
- Coding: >= 68% resolved (at least 4.67 percentage points above the Cycle 2 open leader 63.33%); report relevant and noisy conditions separately.
- Multimodal: >= 48 overall (at least 4.74 points above the Cycle 2 open leader 43.26), with evidence-preserving image retrieval and exact media-size compliance.
- All tracks: 100% Add/Search contract conformance; zero cross-user retrieval; zero unauthorized/retracted/erased evidence exposure; zero malformed or missing results; complete sample denominator; report p50/p95/p99 latency, error/retry counts, storage, bandwidth, and per-operation cost.

These are deliberately ambitious targets; they must be revised only after a matched local reproduction and a current organizer protocol review. A score target is not a promise or evidence of capability.

## Conformance and negative controls required before deployment

Positive fixtures:
- Textual Add + Search, including exact echo IDs and immediate read-after-write.
- Multimodal ordered parts and valid original image data URI.
- Empty Search result {"data":[]}.
- Same idempotency key and exact payload replays without duplicate records.
- Same idempotency key with changed payload rejects.
- Streaming interleaving preserves order and does not leak future events.

Negative controls:
- Search with another user_id or tenant returns no foreign evidence and is denied where the principal is unauthorized.
- Session-only match does not authorize retrieval.
- Wrong purpose, expired retention, future-effective data, revoked consent, superseded claim, retraction, erasure tombstone, missing parent or failed external qualification never leaks content.
- Invalid/remote image URLs, unsupported media type, forged MIME header, oversized decoded bytes and oversized aggregate media fail.
- Duplicate request ID with changed payload never overwrites the original.
- Simulated crash between durable write and index update recovers through reconciliation; no early success.
- Response with top_k + 1 candidates fails rather than being silently truncated.
- Search must not produce answer-shaped content where no source evidence exists.

## Status and implementation boundary

This PR tranche freezes the public Add/Search request/response schemas, examples and implementation/qualification contract. It does **not** implement the Axum HTTP server, durable backend, idempotency transaction, authentication/key custody, multimodal image decoder, deployment manifest, runtime smoke test or AML endpoint. Those remain subsequent implementation gates under #13359. No API is deployed and no AML score or benchmark run is claimed.
