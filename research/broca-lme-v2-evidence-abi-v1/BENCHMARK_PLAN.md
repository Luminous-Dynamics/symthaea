# Broca Memory Benchmark and Research Plan

**Status:** proposed evaluation protocol and targets; no LongMemEval-V2, MemoryAgentBench, or Mem2ActBench score has been produced by this branch. The code now provides both exact-HDC retrieval from an encoded `BinaryHV` and a deterministic BM25 raw-text query baseline over committed text evidence. Raw trajectory ingestion and the official LongMemEval-V2 worker remain prerequisites; semantic text-to-HDC retrieval still requires a pinned, validated encoder.

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

These are **external reference points**, not measurements made on this branch. Hardware, model, prompt, cost and evaluator differences can invalidate a direct comparison. Use the official harness and record the benchmark revision and full run configuration before reporting a comparison. The official results section currently lists the released baseline frontier above; the separately labeled official leaderboard tables report that entries are coming soon. Treat the baseline table as the reproducible reference frontier and do not import third-party claims as official leaderboard results.

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

The first credible frontier target is deliberately demanding:

- **Small tier:** accuracy strictly above 74.9% and benchmark-reported latency strictly below 108.3 seconds.
- **Medium tier:** accuracy strictly above 70.1% and benchmark-reported latency strictly below 139.9 seconds.
- Submit the official **LAFS Gain** metric; do not replace the benchmark's score with an internal weighted score.
- Report the official score and latency, plus local p50/p95/p99 latency, tokens/context returned, peak memory, storage bytes, query compute and cost per question.
- Use paired bootstrap intervals (and paired correctness tests where appropriate); report per-ability results and failure cases. Only claim improvement when the official run is reproducible and uncertainty supports the claim.

A result that is more accurate but slower is not enough to claim that the current frontier has been beaten. A result that is faster but loses state-tracking or multi-hop correctness is not enough either. Publish the Pareto trade-off rather than selecting only a favorable metric.

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
4. **Broca BM25 lexical baseline:** the new deterministic `query_committed_text_context` path over the committed evidence snapshot. Tokenizer and BM25 parameters are pinned and dependency-free; this arm must be labeled lexical, not semantic HDC.
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
2. Implement a trajectory ingester that binds the exact canonical trajectory/step bytes to stable evidence identities and HDC vectors.
3. Run BM25 as the first raw-text retrieval baseline once trajectory ingestion exists. **The immutable generation-bound inverted index is now implemented** and its exact digest is bound by the V3 commit manifest. Indexed ranking eliminates per-query tokenization and text-digest hashing, but recovery still parses and structurally validates the entire lexical artifact, reconstructs evidence records, and scans the HDC checksum. The next optimization is a manifest-digest-keyed verified-generation cache or seekable/mmap posting layout, measured against this exact correctness reference.
4. Separately select and pin a real text-to-HDC encoder; compare it against BM25 without calling structured thought channels a text encoder.
5. Build the official LongMemEval-V2 `insert(trajectory)` / `query(query, query_image)` adapter. The query path must use the committed-generation resolver and return only the official `type/value` context items.
6. Add source-to-context and context-to-answer/action attribution, then run the official baselines and full suite.
7. Run MemoryAgentBench and Mem2ActBench as complementary evaluations; preserve their task-level outputs, not just aggregates.
8. Only after correctness and replay/restart gates pass, compare exact retrieval to approximate/hybrid retrieval and optimize the accuracy–latency frontier.

**Current claim boundary:** the exact-HDC and BM25 query APIs and tests are authored on this branch. They are not compiled or executed in this environment. There is no measured LongMemEval-V2 result, no measured SOTA advantage, no real process-kill qualification, and no physical-erasure claim.
