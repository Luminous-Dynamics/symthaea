# Broca Memory Benchmark and Research Plan

**Status:** proposed evaluation protocol and targets; no LongMemEval-V2, MemoryAgentBench, or Mem2ActBench score has been produced by this branch. The code provides exact-HDC retrieval from an encoded `BinaryHV` and a deterministic BM25 query path over committed text evidence. A strict Rust trajectory-to-text-evidence ingestion primitive, an opt-in trajectory-diversity selector, token-budget packer, adjacent-state expander, audited evidence-floor resolver, and query-specific evidence-set selector are authored but have not been compiled or evaluated; they are not yet wired into the official worker. Important blocker: typed commits currently require a live HDC vector for every active evidence record, so the system cannot yet publish a scientifically clean lexical-only corpus without an encoder. Do not add synthetic vectors for that purpose; see [issue #8786](https://github.com/Luminous-Dynamics/symthaea/issues/8786). A pinned, validated text-to-HDC encoder is still required for semantic HDC retrieval.

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

### Context efficiency and trajectory diversity

The trajectory ingester currently produces state-level text records and repeats trajectory-wide metadata in each state's rendered text. This makes a concrete retrieval-quality hypothesis testable: a query matching a goal can allow sibling states from one trajectory to monopolize top-k, even when the question needs multiple independent examples or a transition-specific state. The official AgentRunbook-R baseline explicitly separates raw-state slices, state-transition events, and procedure/hint notes; these are useful comparison arms, not mechanisms to assume are superior.

Evaluate with identical evidence, query set and length-bounded context budget:

- Current BM25 state records with repeated trajectory metadata.
- Normalized state text plus a separately indexed trajectory-level note.
- **New opt-in candidate policy (draft PR #8810):** oversample lexical candidates, then cap at a configured number of text states per source trajectory. The implementation preserves ranking scores and order semantics, deduplicates IDs, and rechecks scope/lifecycle/time validity. It is deliberately post-ranking and does not change the BM25 reference score.
- Query-conditioned adjacent-state expansion for transition/workflow questions only, compared against the no-expansion baseline.
- Oracle span selection, clearly labeled as a diagnostic upper bound and never included as a competitive method score.

Report top-k trajectory concentration (max share and entropy), distinct trajectories/states, relevant-state recall, contiguous-span precision, duplicate-token ratio, context tokens, query latency/cost, and answer accuracy by each of the five LME-V2 abilities. Static-state queries should penalize unnecessary neighbors; dynamic-state/workflow queries should test whether isolating single states loses transition evidence. Tune on development questions, freeze policy, then evaluate held-out test questions and official LAFS Gain.

This ablation separates a retrieval-policy win from a representation/index win. Keep it separate from HDC kernel throughput and exact-search parity.

### Context-budget packing (new opt-in policy)

The token-budget packer is a deterministic *context assembly* policy after ranking, not a new retrieval score. It processes candidates by descending relevance score and ascending HDC ID, skips an item that cannot fit in the remaining declared budget, and continues trying smaller candidates. It can optionally compose with a per-trajectory quota. This is a hypothesis about context utilization, not an asserted accuracy or latency improvement.

The cost map must use a pinned model tokenizer and the same canonical framing/serialization contract used by the reader. Byte length is not an acceptable substitute. Tokenization is not generally additive across concatenated items, so the caller must also count the final serialized context and fail closed if the actual rendered input exceeds the budget. The policy's additive cost check is not a substitute for that final guard.

Compare these arms using an identical oversampled candidate pool and held-out queries:

1. **Rank-prefix baseline:** preserve rank order and stop when the next item would exceed the budget.
2. **Skip-and-pack:** the new policy skips non-fitting items and continues through the ranking.
3. **Skip-and-pack + trajectory cap:** additionally limit sibling states, with the cap tuned only on development questions.
4. **Token-cost-aware reranking:** future, separate ablation (for example score per token); do not merge this with packing or infer that raw BM25 score is calibrated utility.
5. **Oracle-fit diagnostic:** an upper-bound packing analysis only; oracle labels and future answer outcomes must never be policy features.

Report answer accuracy overall and by all five LongMemEval-V2 abilities, relevant-state recall, context tokens from the final serializer, unused budget, trajectory concentration/entropy, duplicate-token ratio, retrieval latency, reader latency, and cost. Report paired differences with 95% paired bootstrap confidence intervals. Preserve per-question receipts that bind candidate IDs/scores, eligibility decisions, costs, selected IDs, final token count, generation, prompt/tokenizer identity, and answer/evaluator configuration.

**Hard gates:** zero final-prompt budget overruns; zero out-of-scope or inactive evidence; deterministic outputs for the same ranking/costs/generation; no silent fallback when a selected record or cost entry is missing; no answer-key or future-state features. Compare policies at the same *actual serialized-token* budget, not merely the same candidate count.

**Decision rule:** keep the new policy opt-in unless held-out paired evaluation shows a quality–cost improvement against the rank-prefix baseline. A possible project threshold is non-inferior answer quality within a preregistered 1 percentage point while reducing measured context or total lifecycle cost by at least 25%, or a statistically supported quality increase of at least 3 percentage points at equal declared budgets. These are proposed local decision rules, not accepted industry standards; freeze the criterion before viewing held-out results.

### Query-conditioned adjacent-state expansion (new opt-in policy)

LongMemEval-V2 includes both static-state recall and questions about state changes and recurring workflows. A single relevant state may be insufficient for a transition question, but blindly retrieving neighboring states wastes context on static questions. Test an explicit, provenance-preserving expansion policy after retrieval.

The Rust helper keeps eligible ranked seeds first and appends only the immediate predecessor/successor from the same trajectory and trusted scope. Added states carry the seed HDC ID that justified expansion; they do not receive invented relevance scores. Missing indices are not bridged, inactive/out-of-scope/non-text siblings are excluded, and an ambiguous same-trajectory/state coordinate fails closed. The helper is opt-in, not integrated into the official worker, and requires the same final serialized-token budget guard as every other arm.

Compare under identical seed rankings, reader/model, prompt, and final serialized-token budget:

1. **Seed-only:** retrieval seeds without temporal expansion.
2. **Adjacent ±1:** append at most the predecessor and successor for each seed.
3. **One-sided ablations:** predecessor-only and successor-only.
4. **Ability-conditioned expansion:** enable neighbors only for a policy fixed on the development split; do not condition on gold labels or evaluator metadata.
5. **Unrelated-neighbor negative control:** replace adjacent states with same-budget states from another trajectory to test whether any gain is merely more context.
6. **Oracle-neighbor diagnostic:** may quantify an upper bound, but must never use future answer labels at inference.

Report accuracy separately for static state recall, dynamic state tracking, workflow knowledge, environment gotchas and premise awareness; evidence-state recall; transition/ordered-span recall; irrelevant-context rate; actual final prompt tokens; reader and retrieval latency; and cost. Use paired question-level comparisons with 95% bootstrap confidence intervals and preserve the exact seed-to-neighbor attribution in receipts. The static-state category is a key regression guard: an overall average must not hide a static recall loss.

**Hard gates:** no state-index jumping across missing steps; no cross-trajectory or cross-scope expansion; no inactive or invalid neighbor; no invented relevance score; deterministic expansion from the same snapshot and ranking; final serialized context always remains within budget. Test lifecycle mutations (revoke/delete/supersede/expire) and restart/snapshot restoration.

**Decision rule:** keep expansion opt-in unless a held-out, paired comparison improves dynamic-state/workflow outcomes or offers a quality–cost improvement without an unacceptable static-state regression. Freeze the acceptable static-regression margin and cost criterion before inspecting held-out results. These are experiment acceptance rules, not current measured outcomes or an industry standard.

### Audited evidence coverage and abstention boundary (new opt-in contract)

The resolver now has an opt-in audited path that emits the selected, provenance-bearing context plus one disposition for each input HDC hit. Decisions distinguish emitted evidence, duplicate IDs, unknown IDs, invalid records, inactive lifecycle, cross-scope evidence, not-yet-valid evidence, expired evidence, and evidence that passed metadata eligibility checks but was not payload-verified because the output limit was already reached. Corrupt eligible payloads remain hard errors; they are not converted into an empty result.

An EvidenceFloorPolicy can require a minimum emitted-item count and a minimum number of distinct source trajectories. The resulting audit reports observed counts and whether the configured **operational coverage floor** was met. It explicitly does *not* claim semantic sufficiency, entailment, contradiction detection, or answer correctness. Two correlated records can satisfy the floor yet fail to support the question; a single authoritative record may be enough for some questions and still fail a two-source policy. Treat the floor only as a calibrated abstention/escalation feature, not a truth detector.

Evaluate this alongside [Memora / FAMA](https://github.com/geniesinc/Memora), whose Forgetting-Aware Memory Accuracy penalizes reuse of obsolete or invalidated memories, and [AgentAbstain](https://agentabstain.github.io/), which evaluates paired near-identical tasks where the agent should act in one case and abstain in the other. AgentAbstain's public summary reports 263 paired tasks across 42 environments and a best paired accuracy of 59.5% across 17 evaluated models; these are broad agent-abstention reference points, not memory-resolver scores or a directly comparable benchmark.

Build memory-specific paired cases where a tiny change flips the correct decision:
- sufficient, current, in-scope evidence versus one missing critical state;
- unique evidence versus two conflicting active records;
- current claim versus a superseded/expired predecessor;
- exact environment/scope versus a near-match from a different namespace;
- authorized candidate set versus only unknown, corrupt, or tombstoned evidence.

The no-evidence and insufficient-evidence cases must be paired with solvable near-twins so "always abstain" cannot score well. Report **coverage**, task accuracy conditional on proceeding, paired act/abstain accuracy, false-action rate on should-abstain cases, false-abstention rate on solvable cases, obsolete-memory reuse (FAMA-style), and quality–cost curves. If a calibrated probability model is later added, report Brier score/ECE and risk–coverage curves with confidence intervals; do not pretend the count-based floor is a probability estimator.

Compare at least: (1) resolver without audit/floor, (2) audit only, (3) configured count/diversity floor, (4) an independently evaluated semantic support/contradiction gate once one exists, and (5) always-answer / always-abstain controls. Keep integrity failures separate from ordinary low evidence coverage. Record the exact candidate dispositions and floor outcome per example.

**Hard gates:** a rejected, invalid, stale, or cross-scope candidate must never be emitted; an eligible but corrupt payload must produce an explicit integrity error; failed floor must be visible to the caller; no empty-result fallback may disguise integrity failure. The downstream policy must not proceed on a failed floor unless the application explicitly chooses and audits a separate escalation path.

**Decision rule:** treat floor compliance and decision quality as separate axes. First achieve zero policy bypasses in adversarial fixtures; then require held-out paired improvement in act/abstain accuracy without unacceptable loss of solveable-task coverage. The operational floor's threshold is tuned only on development data. No semantic sufficiency or SOTA claim is implied by this resolver.

### Query-specific evidence sets and refinement (new opt-in selector)

Recent work makes the relevance–sufficiency gap explicit: a record can be individually relevant while the answer still lacks a required fact, relation, time constraint or supporting source. QEMem formulates memory utilization as constructing a query-specific evidence set and combines query-guided discovery, source-grounded ViewCards and sufficiency-guided refinement ([arXiv:2610.22668](https://arxiv.org/abs/2610.22668)). Its authors report a 91.43% headline LLM-judge accuracy on LoCoMo with Claude Opus 4.5; its detailed ablation also reports 89.6% for full QEMem on a 1,540-question setup. Treat these as the paper's configuration-specific results, not as LongMemEval-V2 results or direct evidence that a Rust selector outperforms QEMem.

The new Rust experiment implements only a small selection primitive, not the QEMem framework. The caller supplies (a) query-required information-role labels and (b) an explicit role set for each candidate. The selector greedily chooses eligible text candidates that cover the most still-uncovered roles per declared token cost; retrieval score is a tie-break, not an estimate of evidence strength. It reports covered and missing roles. It does not extract roles, prove a role annotation, detect entailment/contradiction, or decide that the answer is semantically supported. Role extraction and query planning must remain separately source-grounded, versioned, and measured.

Compare under identical histories, reader, prompt, and actual final serialized-token budget:

1. **Score-prefix baseline:** standard top-ranked records, budget bounded.
2. **Role-coverage selector:** maximize uncovered required roles per declared token cost.
3. **Coverage + trajectory diversity:** add a per-trajectory constraint as a separate ablation; do not assume independence from distinct trajectories.
4. **Coverage + adjacent-state expansion:** include ordered neighbors only as an explicit additional arm.
5. **Iterative gap refinement:** after the first set, obtain a structured list of uncovered requirements and issue a second bounded retrieval pass. The missing-requirement generator must not see gold answers or evaluator metadata.
6. **Oracle role coverage:** diagnostic upper bound only; oracle annotations are prohibited from inference-time selector inputs.

Construct held-out tasks where answers require complementary pieces from different records, intermediate relations, time constraints, or a source excerpt. Include near-twins with one critical role removed, contradictory candidate roles, wrong-time evidence, duplicate records from one source family, and source-mapping errors. A high coverage percentage cannot override source-integrity or lifecycle failures.

Report requirement-role recall/precision, source-span visibility/traceability, complete-evidence-set rate, answer/task accuracy conditional on coverage, unsupported-answer rate, false-coverage rate from incorrect role annotations, number of refinement rounds, additional retrieval cost, actual context tokens, and paired 95% confidence intervals. Separate the role selector's conditional algorithmic behavior from the role extractor's semantic accuracy. Use mutation tests to falsify coverage claims by deleting one required role, swapping source maps, duplicating one source across multiple projections, or reordering candidate inputs.

**Hard gates:** every candidate must have an explicit role annotation (an empty set means explicitly annotated as covering no required roles, not missing metadata); unknown/missing annotations fail closed; inactive and cross-scope evidence never contribute; eligible corrupt text is an error; uncovered required roles remain explicit; final serialized context must satisfy the actual token budget. The complete result means only that the supplied role labels cover the supplied required-role labels. It is not a truth or entailment certificate.

**Decision rule:** keep the selector experimental until a matched, held-out comparison improves evidence coverage and downstream accuracy without increasing unsupported-answer rate or violating cost budgets. If the role extraction/planning stage is not calibrated, role coverage results are only a software contract test and must not be presented as cognitive sufficiency.

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
