# Agent Memory Leaderboard Cycle 2: Liquid Broca readiness plan

**Date checked:** 2026-10-22  
**Status:** feasibility and integration plan; no entry has been deployed or submitted  
**Tracking issue:** [#8475](https://github.com/Luminous-Dynamics/symthaea/issues/8475)

## 1. Why this is worth pursuing

The [Agent Memory Leaderboard (AML)](https://agentmemoryleaderboard.ai/) is an external shared evaluation platform for textual, coding-agent and multimodal memory. It provides an opportunity to measure the memory subsystem under a locked answer/scoring workflow instead of relying only on our own MIRE generator.

The official page currently lists Cycle 2 materials due **2026-10-31 23:59 UTC+8**, equivalent to **2026-10-31 17:59 Africa/Johannesburg**, with evaluation closing 2026-11-04 23:59 UTC+8 and results planned for mid-November. Dates must be rechecked before action.

This is a time-boxed opportunity, not evidence that the current Broca prototype is competitive.

## 2. The external API contract changes the problem

AML's current guide says participants must operate public Add/Search APIs. Public source code and a Docker image alone do not qualify as deployment; AML does not host participants' implementations.

### Add

The synchronous endpoint receives a unique `request_id`, ordered `messages`, `user_id`, and `session_id`. A successful response is only valid after the memory is durably stored and immediately searchable. It must return HTTP 200 and `success: true`, echoing the request, user and session identifiers exactly.

### Search

The endpoint receives `query`, `user_id` and `top_k`; the documented formal top K is 100. It must search only the matching `user_id` scope and return a JSON object with a relevance-ordered `data` array. Every result needs a stable non-empty `id` and non-empty `content`. Returning too many results, malformed results, or cross-user evidence can fail the API contract.

### Authentication, governance and runtime

The platform documents Token, Bearer and X-Api-Key authentication modes (none is limited to public smoke), an unauthenticated health endpoint, public endpoint review, capacity and operational declarations, and a formal smoke test before the full run. Treat endpoint availability, persistence semantics, top-K ordering and isolation as mandatory conformance tests.

The guide currently states that **open-source Methods** entries must use `text-embedding-v4` for embeddings and `gpt-4o-mini` for LLM-related components; rerankers are unrestricted. Confirm this written rule and any allowed exceptions before choosing a pure-HDC or hybrid design. Do not infer that an HDC representation automatically qualifies as, or exempts the system from, the platform's embedding rule.

Official contract references:
- [AML platform and evaluation flow](https://agentmemoryleaderboard.ai/)
- [AML Add/Search API documentation](https://agentmemoryleaderboard.ai/) (API Guide linked from the platform)

## 3. Candidate system design

Treat the AML adapter as an outer interface around an explicitly versioned memory backend. Keep API contract correctness separate from retrieval quality.

- **Source evidence:** preserve original message text, order, timestamps and session identifiers needed by the dataset; do not let one `user_id` retrieve another user's content.
- **Indexing:** start with a simple, auditable lexical retrieval baseline and benchmark a hybrid backend separately. Only add HDC-based indexing/aggregation when it improves the same held-out workload. Any embedding-dependent design must satisfy AML's published open-source component requirements.
- **Updates:** repeated Add requests with the same `request_id` should be idempotent or have a clearly documented duplicate policy. Record version, provenance and correction behavior; never allow replay to resurrect a superseded fact invisibly.
- **Search:** return stable IDs, non-empty content, valid finite scores when included, deterministic ordering for ties, and at most the requested K records.
- **Audit:** pin backend commit/config, tokenizer or embedding version, index parameters, storage budget, endpoint/API version and all method changes. Keep data split isolation intact.

Do not deploy directly from this research draft. A production endpoint needs a named operator, intentional hosting decision, HTTPS, firewall/rate limits, secure secret configuration, health monitoring, persistent storage and a verified rollback/recovery procedure.

## 4. Required conformance suite before a public smoke

| Area | Cases | Required behavior |
|---|---|---|
| Authentication | Missing/invalid key, supported header modes | Reject unauthorized access; never log tokens |
| Add schema | Empty content, missing identifiers, oversized input, invalid roles | Clear 4xx response; no partial success |
| Add durability | Restart immediately after HTTP 200 | Committed memory remains retrievable |
| Add idempotency | Repeated `request_id` | No silent duplicate inflation or conflicting rewrite |
| Isolation | Same text under two different `user_id` values | Cross-user Search returns no evidence from the other scope |
| Search contract | Empty store, `top_k=0`, `top_k=1`, `top_k=100` | Valid JSON object, no more than K items |
| Ranking | Exact query, paraphrase, tie and irrelevant query | Stable order; explicit relevance score semantics |
| Poison/update | Stale update, correction, delete/replay where supported | No unauthorized stale-value return |
| Load/recovery | Concurrent requests, restart, exhausted capacity | Bounded latency or explicit safe failure; no corrupted acknowledgement |
| Data contract | Invalid result ID/content/score | Rejected internally; never emit invalid records to AML |

These are test requirements, not reported passes.

## 5. Submission schedule and stop/go gates

### Gate A — feasibility

Before implementing a large adapter, confirm all three:
1. An approved, publicly reachable hosting target with HTTPS and persistent storage.
2. Secure access to any required embedding/LLM services and enough compute/network budget for ingestion and formal evaluation.
3. An operator available to submit the access request, respond to platform review, inspect results, and maintain the endpoint through the evaluation window.

If one is missing, use the rest of this cycle for contract testing only and plan for the next one.

### Gate B — contract implementation

Implement API handlers and conformance tests using a frozen backend. Keep API types independent of HDC internals. A correct API contract is necessary but does not imply good benchmark quality.

### Gate C — quality preflight

Run offline held-out checks for recall@K, false-positive retrieval, multi-hop evidence coverage, temporal/update correctness, cross-user isolation and memory lift against no-memory. Measure p50/p95 latency, Add/Search throughput, storage bytes and recovery. Do not tune on AML evaluation answers.

### Gate D — smoke and formal run

Only after a real public deployment, secure auth, the official access request/key, smoke success and fixed version declaration should a full evaluation be requested. Submit source attribution, method changes, public commit, operational limits, fixed version and required declarations. Keep all keys out of source control.

### Stop rule

Do not say “submitted,” “smoke-passed,” or “on the leaderboard” until the platform confirms those states. If a full run cannot be completed and reviewed by the deadline, publish the adapter and conformance results as engineering work and use a later cycle.

## 6. How this connects to the research already in flight

- [PR #8428](https://github.com/Luminous-Dynamics/symthaea/pull/8428) requires fair benchmark comparisons, full lifecycle cost accounting, and memory lift distinct from retrieval recall.
- [PR #8433](https://github.com/Luminous-Dynamics/symthaea/pull/8433) provides opt-in MemoryBridge intervention policies for controlled local evaluation.
- [PR #8460](https://github.com/Luminous-Dynamics/symthaea/pull/8460) explores governed consolidation decisions with a synthetic oracle and anti-controls.

The AML submission should be a separate, public, external experiment. Neither local PR scores nor the synthetic consolidation oracle can substitute for an official AML run.

## 7. Current limitation

No external endpoint has been deployed, no AML access request or key has been created, and no official smoke or full evaluation has run. This document is a preparation plan. Public endpoint deployment requires an intentional infrastructure and credential decision outside this repository change.
