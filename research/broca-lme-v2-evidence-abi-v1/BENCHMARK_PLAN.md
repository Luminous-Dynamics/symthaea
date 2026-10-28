# Broca Memory Benchmark and Research Plan

**Status:** proposed evaluation protocol and targets; no LongMemEval-V2, MemoryAgentBench, or Mem2ActBench score has been produced by this branch. The code provides exact-HDC retrieval from an encoded `BinaryHV` and a deterministic BM25 query path over committed text evidence. A strict Rust trajectory-to-text-evidence ingestion primitive is now authored, but it is not yet wired into the official worker. Important blocker: typed commits currently require a live HDC vector for every active evidence record, so the system cannot yet publish a scientifically clean lexical-only corpus without an encoder. Do not add synthetic vectors for that purpose; see [issue #8786](https://github.com/Luminous-Dynamics/symthaea/issues/8786). A pinned, validated text-to-HDC encoder is still required for semantic HDC retrieval.

## Research thesis

Treat memory as an **evidence-carrying state transition**, not a bag of similar vectors. A query should be able to account for (1) which committed generation it observed, (2) which candidates were ranked, (3) which records were eligible under scope, time and lifecycle rules, and (4) which exact payload bytes reached the reader. Learning from experience is only useful if the system can distinguish current state from obsolete state and can refuse to answer when evidence is missing or inconsistent.

This makes memory quality a multi-objective systems problem:

- **Epistemic correctness:** the supplied context actually supports the answer or action.
- **Temporal correctness:** changed preferences and state do not revert to obsolete values.
- **Authority and scope:** private, revoked, expired, or cross-environment records never leak.
- **Action transfer:** retrieved experience changes tool selection and parameter grounding correctly.
- **Operational efficiency:** useful evidence per unit of latency, memory, and compute.
- **Recoverability:** crashes expose either a valid committed state or an explicit integrity error—never a hybrid index/payload state.

## Public comparison points

The official [LongMemEval-V2 repository](https://github.com/xiaowu0162/LongMemEval-V2/) reports 451 curated questions, five memory abilities, web and enterprise domains, histories up to 500 trajectories / 115M tokens, and small/medium leaderboard tiers. Its published baseline page ([results](https://xiaowu0162.github.io/longmemeval-v2/)) currently lists:

| Method | Small accuracy | Small latency | Medium accuracy | Medium latency |
|---|---:|---:|---:|---:|
| No retrieval | 1.3% | 0 s | 1.3% | 0 s |
| RAG: query to slice | 42.8% | 0.1 s | 38.1% | 0.1 s |
| RAG: query to slice + notes | 51.0% | 0.2 s | 45.9% | 0.3 s |
| AgentRunbook-R | 58.6% | 26.9 s | 57.0% | 25.8 s |
| Codex | 69.9% | 177.2 s | 68.7% | 185.8 s |
| AgentRunbook-C | 74.9% | 108.3 s | 70.1% | 139.9 s |

These are **released baseline reference points**, not measurements made on this branch. The benchmark defines the fixed reference frontier from RAG query-to-slice + notes, Codex, AgentRunbook-R, and AgentRunbook-C; its official LAFS Gain measures improvement to that accuracy–latency frontier over budgets from 1 to 200 seconds. See the [official scoring definition](https://xiaowu0162.github.io/longmemeval-v2/) and [leaderboard packaging rules](https://github.com/xiaowu0162/LongMemEval-V2/blob/main/leaderboard/README.md).

A separate [AgentRunbook-C V2 research update](https://xiaowu0162.github.io/longmemeval-v2/agentrunbook-c-v2/) reports additional small-tier operating points:

| Model / effort | Controller | Accuracy | Average memory-query latency |
|---|---|---:|---:|
| GPT-5.4-mini, medium | AgentRunbook-C V1 | 70.30% | 70.00 s |
| GPT-5.4-mini, medium | AgentRunbook-C V2 | 72.73% | 56.68 s |
| GPT-5.4-mini, xhigh | AgentRunbook-C V1 | 74.90% | 108.30 s |
| GPT-5.4-mini, xhigh | AgentRunbook-C V2 | 75.61% | 130.54 s |
| GPT-5.6 Luna, medium | AgentRunbook-C V2 | 66.52% | 24.72 s |
| GPT-5.6 Luna, xhigh | AgentRunbook-C V2 | 75.17% | 48.94 s |

These points use different model/reasoning settings and the controller's measured online memory-query latency; do not compare them directly to an HDC kernel microbenchmark. The public leaderboard section on the benchmark overview still marks its tier tables as coming soon, so cite these as published method-page results, not as an official leaderboard rank.

Two complementary benchmarks matter because answer recall is not the whole job:

- [MemoryAgentBench (ICLR 2026)](https://mlanthology.org/iclr/2026/hu2026iclr-evaluating/) measures accurate retrieval, test-time learning, long-range understanding and selective forgetting through incremental multi-turn interaction.
- [Mem2ActBench (ACL 2026)](https://aclanthology.org/2026.acl-long.370/) evaluates active application of memory to tool selection and parameter grounding. Its released evaluation comprises 400 tool-use tasks built from 2,029 sessions, with human assessment indicating 91.3% are strongly memory-dependent.

## Target scorecard

All figures below are **goals**, not present results.

### 1. Correctness and non-leakage: hard gates

| Measure | Acceptance target |
|---|---|
| Deleted, revoked, superseded, expired or out-of-scope evidence emitted | **0** across deterministic tests, mutation tests and seeded fault schedules |
| Corrupt/missing/unknown-version committed state | **100% fail closed**; no silent fallback and no successful empty-answer masking of integrity errors |
| Recovery after injected interruption | Each restart yields the prior valid generation, the complete new generation, or an explicit error; never mixed generations |
| HDC/evidence active-ID parity | Exact set equality on every production commit and query |
| Query-generation consistency | Every result reports one generation and uses both index and evidence snapshot from that generation |
| Repeatability | Same frozen data, configuration and query gives identical exact-search candidates and context ordering |

Start with at least 10,000 seeded mutation/fault schedules across commit phases and lifecycle transitions, retaining seeds, artifact hashes and outcomes. That is a test target, not proof of power-loss safety: additionally kill a separate process at every commit boundary, restart it, and test each filesystem/device combination for which durability is claimed. Include actual delete/revoke/supersede transitions before, during and after interrupted commits.

### 2. Official long-term-memory performance

Use two levels of frontier targets rather than one stale scalar threshold:

1. **Official acceptance:** submit a positive official ΔLAFS / LAFS Gain against the benchmark's fixed reference frontier using the actual official packaging helper. Report every operating point and its model, reasoning effort, reader, context budget and latency definition.
2. **Matched V2 challenge points:** under the same model/reasoning setting and official harness, aim to beat at least one published AgentRunbook-C V2 point on both accuracy and memory-query latency—for example, exceed 72.73% while staying below 56.68 seconds at GPT-5.4-mini medium, and separately exceed 75.61% while staying below 130.54 seconds at GPT-5.4-mini xhigh. The GPT-5.6 Luna V2 points (66.52% / 24.72 s at medium; 75.17% / 48.94 s at xhigh) are separate settings and should be matched only using that same setup.
3. **Baseline frontier:** retain the original fixed reference-table goals (>74.9% under 108.3 seconds on small; >70.1% under 139.9 seconds on medium), but do not confuse these with a claim to beat every newer V2 operating point.
4. Report official LAFS Gain; do not replace it with an internal weighted score. Also report local p50/p95/p99 latency, tokens/context returned, peak resident memory, index size, cold-load vs warm-query times, query compute and cost per question.
5. Use paired bootstrap intervals (and paired correctness tests where appropriate); report per-ability results and failure cases. Only claim an improvement when the official run is reproducible and uncertainty supports the claim.

A result that is more accurate but slower may still improve LAFS at some latency budgets, but it does not Pareto-dominate the compared point. A result that is faster but loses state-tracking or multi-hop correctness is not a better system. Publish the whole accuracy–latency frontier and matched configuration details.

### 3. Dynamic memory and task execution

For MemoryAgentBench, report each of the four competencies separately; do not hide selective-forgetting regressions behind a high retrieval average. Add stress slices for changing preferences, contradictory facts, stale cached values, delayed corrections, and queries requiring information across multiple sessions.

For Mem2ActBench, report exact task success alongside:
- correct tool selection;
- argument/parameter grounding;
- use of the latest applicable preference/state;
- unnecessary tool calls and unsafe or unauthorized actions;
- failures when the needed fact is absent or has been revoked.

Use task-level controls where the same task is run with memory enabled, memory disabled, and a deliberately stale or wrong memory record. A system should improve task success with valid memory and should never become more confidently wrong when memory is stale.

## Required baseline arms

Freeze and compare these as separate arms. Do not conflate them into one advertised “memory” result.

1. **No-memory control:** same reader/model without retrieved history.
2. **Official simple RAG controls:** official query-to-slice and query-to-slice-plus-notes implementations.
3. **Official strong reference:** the AgentRunbook-C baseline as packaged by the benchmark, wherever the official evaluation setup permits a reproduction.
4. **Broca BM25 lexical baseline:** the deterministic `query_committed_text_context` path over the committed evidence snapshot. Tokenizer and BM25 parameters are pinned and dependency-free; this arm must be labeled lexical, not semantic HDC. A clean lexical-only persistence path is still blocked by the HDC-ID parity coupling in [issue #8786](https://github.com/Luminous-Dynamics/symthaea/issues/8786); do not make this arm appear runnable with synthetic HDC vectors.
5. **Broca exact HDC:** a pinned text/query encoder feeding exact search; exact search is the oracle for later ANN comparisons.
6. **Broca hybrid candidate generation:** lexical + HDC, with deduplication and bounded evidence context, only after each component has its own baseline.
7. **Ablations:** remove temporal state tracking, lifecycle filtering, provenance checks, notes, or HDC retrieval one at a time to show which component actually produces each gain.

All arms must receive the same allowed history and query. No benchmark question ID, gold answer, question type, evaluator metadata or future trajectory information may enter the memory write path, ranking features, or prompt. Pin dataset revisions, model/tokenizer hashes, code commits, prompt templates, context budget, seeds and evaluator versions.

## Paradigm-shift research agenda

### A. From vector similarity to evidence contracts

Current HDC search is candidate generation, not truth. Keep approximate search subordinate to an exact oracle, and treat provenance, lifecycle, scope and content digest as mandatory constraints. A candidate can be semantically close but still be wrong, obsolete or unauthorized.

**Research question:** can the system maximize task utility subject to zero forbidden-evidence emissions, rather than maximize retrieval similarity and hope a later reader catches errors?

### B. From summaries to temporal, event-sourced state

Represent a memory update as a source-linked event with an explicit time, scope, authority, validity interval, and relation to prior claims (supports, corrects, supersedes, or revokes). Derive current state from the event chain; keep prior evidence available for audit but ineligible as current truth after correction.

**Research question:** can this reduce contradiction and stale-state errors on dynamic-state tracking and selective-forgetting tasks without increasing context tokens?

### C. From retrieval-only to action-conditioned memory

Measure whether a memory changes an action and whether that change is correct. Reward useful tool choices and correct parameters, not the number of retrieved records. Include a no-memory control and a stale-memory negative control.

**Research question:** can Symthaea use outcome feedback to learn which prior experiences transfer to a new context, while refusing to generalize across incompatible scope, time, or system state?

### D. From “forgetting” flags to deletion semantics

Separate three claims: logical ineligibility, visible-file pruning, and physical erasure. The present pruning routine provides only visible-file removal. For stronger deletion requirements, investigate per-record envelope encryption, destruction of per-record keys, backup expiry, and independently auditable erasure policy. Crypto-erasure also needs key lifecycle, replica, backup and threat-model analysis; do not assert it simply because a key was deleted.

### E. From one score to a quality–cost frontier

Track answer/task quality against end-to-end latency, memory footprint, context tokens and inference cost. Use adaptive retrieval depth only behind deterministic safety filters, and compare its decisions to exact HDC results. Cache only with a versioned key that includes generation, scope, time policy and encoder identity; invalidate it on every relevant state transition.

**Research question:** can the system return better-supported answers and actions at lower latency/cost by spending retrieval effort only where additional evidence changes the decision?

## Implementation prerequisites and next sequence

This branch currently provides a binary-query API, not a complete benchmark backend. Proceed in this order:

1. Compile and run focused tests on the exact head; keep the PR draft until results are known.
2. Verify upgrade compatibility from a real prior V2 committed generation: recovery must retain the V2 snapshot/HDC invariants, rebuild the lexical index in memory without mutating old bytes, and publish V3 with a digest-bound index on the next typed commit.
3. A first Rust-native trajectory ingestion primitive is now authored at `crates/domains/symthaea-broca/src/longmemeval_v2_ingest.rs`. Its record/state fields were cross-checked against the public [LongMemEval-V2 data schema](https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/blob/main/SCHEMA.md) and the official [trajectory preparation code](https://github.com/xiaowu0162/LongMemEval-V2/blob/main/memory_modules/trajectory_store.py): required trajectory identity/domain/environment/goal/outcome/start URL, ordered state index, URL, optional original step/action/thought, accessibility-tree text and screenshot path. It rejects question/answer/evaluator metadata and unknown fields, resolves scope through caller-supplied trusted configuration, emits one text evidence record per state, and binds deterministic IDs plus canonical source digests. Screenshot paths are validated as relative beneath `screenshots/`, but image bytes are not yet materialized or emitted as image context. The Rust tests are authored but unexecuted; the module is not the official `insert(trajectory)`/`query(query, query_image)` worker. Compile/test it and run golden fixtures from a pinned upstream dataset revision before qualification.
4. Wire the trajectory ingester through typed HDC/evidence commits and the official worker, then run BM25 as the first raw-text retrieval baseline. **The immutable generation-bound inverted index and process-local verified-generation cache are now authored.** The cache is keyed by the exact CURRENT generation + manifest digest; warm queries reuse verified evidence, postings and the read-only HDC mapping, while pointer changes force a complete reload. Next benchmark cold-load vs warm-query p50/p95/p99, index/snapshot resident memory, and cache invalidation under update/delete/revoke. Scope/time/lifecycle stats and exact HDC ranking may still be linear; use the measurements to decide whether per-scope statistic summaries, a seekable/mmap posting layout or ANN candidate generation is justified.
5. Separately select and pin a real text-to-HDC encoder; compare it against BM25 without calling structured thought channels a text encoder.
6. Build the official LongMemEval-V2 `insert(trajectory)` / `query(query, query_image)` adapter. The query path must use the committed-generation resolver and return only the official `type/value` context items.
7. Add source-to-context and context-to-answer/action attribution, then run the official baselines and full suite.
8. Run MemoryAgentBench and Mem2ActBench as complementary evaluations; preserve their task-level outputs, not just aggregates.
9. Only after correctness and replay/restart gates pass, compare exact retrieval to approximate/hybrid retrieval and optimize the accuracy–latency frontier.

**Current claim boundary:** the exact-HDC and BM25 query APIs, generation-bound lexical artifact, verified-generation cache, and first trajectory-to-text-evidence ingester are authored on this branch. They are not compiled or executed in this environment. The ingester is not yet an official worker and does not encode text into semantic HDC vectors or process benchmark query images. No warm/cold latency or SOTA advantage is measured. There is no measured LongMemEval-V2 result, no measured SOTA advantage, no real process-kill qualification, and no physical-erasure claim.
