# Liquid Broca Memory: Cross-Benchmark Quality–Cost Frontier

**Status:** research protocol / proposed evaluation plan; no model benchmark results are claimed  
**Date:** 2026-10-22  
**Tracking:** [issue #8427](https://github.com/Luminous-Dynamics/symthaea/issues/8427)  
**Related:** [MIRE protocol PR #8377](https://github.com/Luminous-Dynamics/symthaea/pull/8377), [MIRE reference harness PR #8395](https://github.com/Luminous-Dynamics/symthaea/pull/8395), [selective intervention issue #8392](https://github.com/Luminous-Dynamics/symthaea/issues/8392), [bounded-payload lifecycle issue #8411](https://github.com/Luminous-Dynamics/symthaea/issues/8411)

## 1. Thesis: memory must earn its place in the loop

The goal is not to maximize the number of stored facts, retrieved chunks, or HDC similarities. A memory system is useful when it helps an agent take a more correct, timely, grounded action or answer while staying within a declared cost and safety envelope.

This creates three independent questions:

1. **Integrity:** Is the record current, scoped to the right person/episode/entity, valid under the writer/version policy, and within the deletion contract?
2. **Utility:** Does intervening with this memory improve the actual answer or action on this query, compared with not intervening?
3. **Economics:** Does that improvement justify the *total* memory and compute cost, including ingestion, consolidation, index and control metadata, retrieval, prompt tokens, latency, and recovery?

A system must pass integrity gates before a task-average score can be considered. Then its task quality and full lifecycle costs belong on a Pareto frontier. A single weighted score must not allow excellent recall to compensate for stale-memory resurrection or a deletion-policy violation.

This protocol extends, rather than replaces, the controlled state-machine checks in MIRE. The dependency-free harness in PR #8395 is not a model benchmark and currently does not load Liquid Broca, an HDC store, a language model, or a CUDA kernel.

## 2. External benchmark portfolio

Run public suites through their official data and scoring paths where licensing and environment allow. Do not merge different suites into one unlabelled percentage.

| Benchmark | Primary question | Why it matters to Broca | Required report |
|---|---|---|---|
| [LongMemEval (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html) | Information extraction, multi-session reasoning, temporal reasoning, updates, abstention | A baseline for persistent conversational memory across hundreds of questions and scalable histories | Official task accuracy and category breakdown; identify exact LongMemEval version/split |
| [MemoryAgentBench (ICLR 2026)](https://mlanthology.org/iclr/2026/hu2026iclr-evaluating/) | Incremental retrieval, test-time learning, long-range understanding, selective forgetting | Tests memory as an evolving process, not just a static context dump | Report each competency separately; no hiding selective-forgetting failure in an average |
| [Memora / FAMA (Findings of ACL 2026)](https://aclanthology.org/2026.findings-acl.1337/) | Remembering, reasoning, recommending while user facts evolve | Directly surfaces stale facts and failure to reconcile changed state | FAMA and original component metrics; include the obsolete-memory subset |
| [Mem2ActBench (ACL 2026)](https://aclanthology.org/2026.acl-long.370/) | Use retained context to select tools and ground tool arguments | Retrieval may be correct while execution still applies the wrong account, target, time, or parameter | Tool selection accuracy, exact argument match, task success, invalid/unsafe action rate |
| [BEAM (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d7f0cfa0fe759b033d5262e1bb7d4065-Abstract-Conference.html) | Long, coherent conversations up to 10M tokens with 2,000 validated questions | Tests whether behavior survives extreme history length and diverse probing | Official metrics by conversation length and question family |
| [AMA-Bench (ICML 2026)](https://proceedings.mlr.press/v306/zhao26bs.html) | Memory over long-horizon agent/environment trajectories | Adds observations, actions, and tool outputs beyond dialogue-only memory | Task success by trajectory horizon and state/action dependency |
| [MIRE](BROCA_MEMORY_INTEGRITY_BENCHMARK_2026-10-22.md) | Adversarial updates, stale replay, deletion, namespace binding, abstention and capacity pressure | Controlled failure detection and regression protection | Per-query trace; mutation controls must fail as intended; exact reference must match evaluator labels |

These are complementary datasets, not exchangeable measures. LongMemEval's authors have published a V2 update for agentic context; a report must name the exact revision, not simply say “LongMemEval.” If a public benchmark's licence, download or scoring setup prevents a faithful reproduction, state that limitation and do not imply an official score.

### Minimal first portfolio

Start with **MIRE + LongMemEval + MemoryAgentBench**. Add **Mem2ActBench** before claiming the system improves tool-using agents. Add BEAM/AMA-Bench when the first adapter is stable. This order gives us a cheap integrity gate, a known conversational baseline, incremental-memory competencies, and then execution-level memory utility.

## 3. Fair baseline matrix

Separate the memory substrate from the answer model. In the primary comparison, freeze the same answer model, system prompt, generation parameters, official evaluator, and answer-token budget across memory configurations wherever the suite allows it.

| Baseline | Purpose | Fair-comparison rule |
|---|---|---|
| No persistent memory | Measures what the backbone already knows from the current turn | Same answer model and current-turn input |
| Full-history / maximum-context | Strong context-only reference, where feasible | Explicitly report input tokens, truncation policy and context limit |
| BM25 / lexical retrieval | Cheap, interpretable retrieval reference | Same source corpus and candidate pool |
| Dense retrieval | Standard embedding-based comparator | Freeze embedding model and index settings |
| Dense + reranker / reciprocal-rank fusion | Strong retrieval-stack reference | Tune on development data only; expose reranker latency and cost |
| Versioned exact key-value memory | Separates lifecycle correctness from semantic retrieval | Evaluator keys must be available only in the reference/task oracle, not leaked into semantic query generation |
| Summary/observation memory | Tests compact persistent state without HDC | Count summarization and background consolidation |
| Recurrent-only Liquid Broca | Isolates persistent recurrent state | Same training/development split and reset rules |
| HDC retrieval + current MemoryBridge bundling | Measures the current memory path | Freeze store/index/config and record vector count and bytes |
| HDC + calibrated intervention gate | Tests whether retrieval should affect the current thought at all | Tune only on development data; use paired memory-relevant/irrelevant cases |
| HDC + alternate aggregator | Separates retrieval selection from bundling/aggregation | Keep the gate fixed when comparing aggregators |
| Label-informed oracle | Diagnostic upper bound only | Never present as deployable or eligible to win |

Run two experiments rather than mixing hypotheses: **(A)** gate policy with aggregation held fixed; **(B)** aggregation policy with gate held fixed. Include no-blend, always-blend, top-score threshold, top-score-plus-margin, a calibrated utility gate, and a label-informed oracle. Similarity is not proof of relevance: test high-similarity irrelevant distractors and moderate-similarity relevant memories.

## 4. Score task utility, not retrieval theatre

Report at least these outcomes separately:

- **Answer utility:** official exact-match / benchmark score, plus official category metrics.
- **Action utility:** correct tool, correctly bound arguments, and successful execution on Mem2ActBench-like tasks.
- **Temporal integrity:** stale-answer rate after a valid update; stale-replay error rate; obsolete-memory penalty such as FAMA.
- **Binding integrity:** wrong entity, attribute, episode/person, relation direction, or source namespace.
- **Abstention quality:** correct abstention when current payload is absent, evidence conflicts, or the evaluation context cannot establish the answer; false abstention when evidence is available.
- **Forgetting:** removal/invalidation of the intended record, collateral forgetting of adjacent valid facts, and the clearly declared scope of deletion. Logical invalidation is not proof of physical secure erasure.
- **Selective intervention:** paired delta between gating policies and no-memory on relevant and irrelevant query pairs. Fewer interventions alone is not a success.
- **Capacity/retention:** utility versus event count, number of unique keys, distractor rate, update/delete frequency, payload capacity, and total stored bytes.
- **Resource cost:** read/write/consolidation compute, tokens, memory traffic, total RAM/VRAM/disk, index/metadata/journal size, energy where measurable, and p50/p95/p99 end-to-end latency.

For each metric show numerator and denominator, exclusions, per-task family, and confidence interval. Retain per-example outcomes so a reviewer can identify whether an aggregate gain hides a catastrophic slice.

## 5. Cost accounting: count the whole memory lifecycle

A system does not get to call itself “small memory” by counting only payload vectors. Include:

- payload values and embeddings;
- index structures, codebooks, routing tables, and cached query state;
- entity bindings, namespaces, version vectors, tombstones, provenance and confidence metadata;
- journal/checkpoint/snapshot/index bytes and compaction overhead;
- ingestion, deduplication, summarization and consolidation work;
- query encoding, retrieval, reranking, bundling and answer-generation tokens;
- recovery time and resources after restart or damaged state.

Report **steady-state cost** and **amortized lifecycle cost** separately. Where exact byte or energy measurement is unavailable, label the estimate and its method. The unbounded lifecycle ledger in PR #8395 is a semantic diagnostic, not an equal-total-memory win; its size must not disappear from a cost calculation.

Make comparisons at matched declared budgets. Use several budget points to draw the Pareto frontier rather than selecting one convenient capacity. At minimum vary payload capacity, history length, distractor rate, update/delete frequency and query rate. Include both a fixed-latency and a fixed-memory view if either constrains intended deployment.

## 6. Proposed decision gates — targets, not results

These are internal research thresholds for deciding what to investigate next, not claimed industry standards.

### Gate 0 — scorer trustworthiness

- The exact versioned reference matches every generated label.
- Every declared mutation class is detected by its negative control.
- A malformed, incomplete, or wrong-revision receipt fails closed.
- Report the actual number of adversarial cases run; a passing 100-case test is not evidence of zero failures at scale.

### Gate 1 — integrity and scope

For the deterministic mutation suite, target **zero observed stale-value resurrection, cross-namespace leakage, or prohibited post-delete return** under the tested policy. Expand to at least 10,000 generated/adversarial transitions across multiple seeds before making a robust empirical claim. Any failure blocks overall task-quality promotion until explained and fixed. Physical secure-erasure claims require a different threat model and verification method.

### Gate 2 — no regression in useful memory

On a held-out split, compare the proposed system with the strongest fair baseline using paired per-example analysis. A proposed gate is eligible only if it stays within a preregistered **1 percentage-point non-inferiority margin** on memory-relevant tasks and does not worsen the critical integrity slices.

### Gate 3 — meaningful quality/cost improvement

A proposed architecture change earns a positive claim if it meets one of these preregistered outcomes:

- **Efficiency win:** task quality within the 1-point non-inferiority margin, while reducing **measured total lifecycle cost by at least 25%** at the same workload and quality target; or
- **Quality win:** at least a **3 percentage-point paired gain** at the same declared cost budget, with a 95% confidence interval excluding zero on the primary outcome and no critical integrity regression.

Use paired bootstrap confidence intervals or an appropriate paired test. Predeclare the primary metric and control multiple comparisons when testing many gates, aggregators, or task families. These thresholds are a decision policy, not proof of SOTA; one benchmark alone should not support a broad general-capability claim.

### Gate 4 — reproducibility

A report must pin dataset version and split digest, repository commit, model/checkpoint and tokenizer hashes, prompt/config, evaluator version, commands, runtime/hardware, resource-accounting method, random seeds, all per-example outcomes and an immutable receipt/log digest. Distinguish `PASS`, `FAIL`, `NOT RUN`, `INVALID`, and `SKIPPED`. A queued or skipped workflow is not a passing result.

## 7. How to compare honestly with current industry and research

Publicly reported scores indicate that the target is competitive, but they are not one compatible league table:

- The original LongMemEval is a 500-question benchmark with categories for extraction, multi-session reasoning, temporal reasoning, updates, and abstention; LongMemEval-V2 is a separate revision for more agentic settings. See the [official paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html) and [official repository/V2 notice](https://github.com/xiaowu0162/LongMemEval).
- Redis reported 86.5% task-averaged accuracy for a combined Remis + Instruct strategy on LongMemEval with a GPT-4o answer backbone in its [September 2026 report](https://redis.io/resources/agent-memory-benchmark-report/). This is a vendor-reported result on its stated setup, not a universal ranking.
- Mastra reported 84.23% with GPT-4o and 94.87% with GPT-5-mini for its Observational Memory setup in a [February 2026 report](https://mastra.ai/research/observational-memory). Different answer models alone make the percentages non-comparable as a memory-substrate ranking.
- Mamba-3's ICLR 2026 paper reports a 1.8-point total average downstream-accuracy advantage for its MIMO variant over the stated next-best model at 1.5B scale. This is a recurrent-model architecture result, not a memory-agent leaderboard score; use it to motivate a matched backbone/latency comparison, not to claim Broca has matched Mamba-3.

The research strategy must therefore produce **two reports**:
1. **Memory-substrate bake-off:** same frozen answer model, prompt/evaluator, data and declared total budget; isolate memory architecture.
2. **End-to-end system bake-off:** strongest allowed backbone and full agent stack for every system; transparently report model versions, hosting, token prices, hardware, latency and cost.

Never use a number from one report as if it were a result from the other. Pin leaderboard snapshots by date. Treat a leaderboard as a lead for reproducible experiments, not as evidence until the evaluation can be independently reproduced on its declared configuration.

## 8. Experiment sequence

1. **Integrity first:** run MIRE reference tests and mutation controls on an exact source revision; do not interpret skipped CI as a pass.
2. **Substrate-only pilot:** use small, fixed data and one frozen answer model to compare no-memory, BM25, dense retrieval, versioned KV, current HDC/MemoryBridge and selective intervention. Keep the experiment cheap enough to run locally.
3. **External suite adapter:** first LongMemEval and MemoryAgentBench, preserving official scorers and dataset versions. Do not copy benchmark data into the training path.
4. **Task-grounded follow-up:** add Mem2ActBench or an equivalent frozen tool sandbox. Verify actual selected tool and exact arguments, not merely whether memory was retrieved.
5. **Frontier sweep:** cross several memory budgets with fixed workloads, multiple seeds, paired confidence intervals and full metadata accounting.
6. **Scale and robustness:** extend to long context / agent trajectories, crash/restart/replay tests and concurrent or out-of-order writers only when the baseline adapter is stable.
7. **Publish negative results too:** if HDC, recurrent state, a learned gate, or a larger mechanism loses to versioned KV or conventional retrieval under equal budgets, report that. Remove complexity that fails to produce repeatable value.

## 9. Success criterion

The ambitious claim is not “HDC memory beats RAG” or “liquid state replaces transformers.” It is:

> Given a fixed task, data, answer backbone and cost budget, the system selects the right prior information when it improves an action, does not inject stale or out-of-scope information, abstains when the evidence is unavailable, and demonstrates a reproducible quality–cost advantage on public tasks.

Until the external suites and the actual MemoryBridge are executed on a pinned revision, this document defines an experiment and decision policy. It reports **no Liquid Broca score, no measured efficiency gain, and no SOTA claim**.
