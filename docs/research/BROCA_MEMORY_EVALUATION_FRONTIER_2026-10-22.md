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
| [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2) | Agentic long-term memory with explicit accuracy–latency trade-offs | Tests whether an agentic retrieval/workflow improves answers enough to justify its runtime | Use the V2 harness and report its own frontier metric (LAFS Gain), size tier, accuracy and end-to-end latency separately from V1 |
| [MemoryAgentBench (ICLR 2026)](https://mlanthology.org/iclr/2026/hu2026iclr-evaluating/) | Incremental retrieval, test-time learning, long-range understanding, selective forgetting | Tests memory as an evolving process, not just a static context dump | Report each competency separately; no hiding selective-forgetting failure in an average |
| [Memora / FAMA (Findings of ACL 2026)](https://aclanthology.org/2026.findings-acl.1337/) | Remembering, reasoning, recommending while user facts evolve | Directly surfaces stale facts and failure to reconcile changed state | FAMA and original component metrics; include the obsolete-memory subset |
| [Mem2ActBench (ACL 2026)](https://aclanthology.org/2026.acl-long.370/) | Use retained context to select tools and ground tool arguments | Retrieval may be correct while execution still applies the wrong account, target, time, or parameter | Tool selection accuracy, exact argument match, task success, invalid/unsafe action rate |
| [BEAM (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d7f0cfa0fe759b033d5262e1bb7d4065-Abstract-Conference.html) | Long, coherent conversations up to 10M tokens with 2,000 validated questions | Tests whether behavior survives extreme history length and diverse probing | Official metrics by conversation length and question family |
| [AMA-Bench (ICML 2026)](https://proceedings.mlr.press/v306/zhao26bs.html) | Memory over long-horizon agent/environment trajectories | Adds observations, actions, and tool outputs beyond dialogue-only memory | Task success by trajectory horizon and state/action dependency |
| [MIRE](BROCA_MEMORY_INTEGRITY_BENCHMARK_2026-10-22.md) | Adversarial updates, stale replay, deletion, namespace binding, abstention and capacity pressure | Controlled failure detection and regression protection | Per-query trace; mutation controls must fail as intended; exact reference must match evaluator labels |
| Consolidation decision audit / ConsolidationBench paper | Decide whether evidence stays raw, remains provisional, is summarized, promoted to a skill/rule, or retired | Tests the write-path and abstraction decision, not just retrieval; includes destructive forgetting and cross-level transfer | Keep the published ConsolidationBench evaluation distinct from our proposed synthetic prototype; report downstream transfer and governance separately |

These are complementary datasets, not exchangeable measures. **LongMemEval V1 and V2 are separate evaluation tracks**: V1's 500-question task-accuracy breakdown is not the V2 accuracy–latency frontier. Always name the dataset, release, split and metric. Also disambiguate the two unrelated uses of “Memora”: ACL 2026's *From Recall to Forgetting* is a benchmark using FAMA, while Microsoft Research's *Memora: A Harmonic Memory Representation Balancing Abstraction and Specificity* is a memory architecture/report. Do not treat them as the same contribution. If licensing, download, or evaluator setup prevents faithful reproduction, state the limitation rather than implying an official score.

### Minimal first portfolio

Start with **MIRE + LongMemEval V1 + MemoryAgentBench**; run **LongMemEval-V2** as a separately reported accuracy–latency track as soon as its adapter is stable. Add **Mem2ActBench** before claiming the system improves tool-using agents. Add BEAM/AMA-Bench when the first adapter is stable. This order gives us a cheap integrity gate, a known conversational baseline, incremental-memory competencies, and then execution-level memory utility.

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
| OAS-style retain/merge/abstract/rewrite policy | Tests budget-conditioned operator selection with pre-generation features and held-out harm calibration | Use the published LongMemEval and LoCoMo split/evaluator where available; sweep tight and loose budgets; report operator-specific outcomes and raw-evidence replacement harms |
| MemPO (policy-optimized self-memory) | End-to-end agent policy learns to manage its own memory for long-horizon tasks | Reproduce its official task/evaluation setup; report base model, policy-training data/compute, exact actor-token budget and memory overhead. Treat paper-reported gains as external results until matched locally |
| ReMe (dynamic procedural memory) | End-to-end experience distillation, context-adaptive reuse and utility-based refinement; paper reports results on BFCL-V3 and AppWorld | Use those native task suites and official metrics when the implementation/evaluator is available; keep this as a whole-agent comparator rather than attributing its gains to a memory index alone |
| Cue-anchor / dual-resolution memory | Uses compact abstraction to route into specific supporting evidence instead of storing only summaries or only raw chunks | Include abstraction, raw-evidence store, anchors and navigation/index costs; measure multi-hop and source-grounding failures |
| Microsoft Research Memora-style harmonic representation (where a faithful implementation is available) | Tests the published abstraction/specificity strategy as a competing representation | Treat published scores as motivation, not results; reproduce the stated model, data and scoring setup before making claims |
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


## 4A. Consolidation is a decision, not a compression ratio

A fresh line of work, [The Epistemics of Agent Memory (September 2026)](https://arxiv.org/abs/2609.33013), shifts attention from “what did the system retain?” to “was the decision to retain, summarize, abstract, or forget justified, and can that decision be audited and reversed?” It introduces ConsolidationBench. Its authors report both positive results and important failures: their boundary learner missed a preregistered criterion, and a promotion/compression policy reported a task-success gain alongside destructive forgetting and cross-level transfer regressions. Treat these as that paper's reported findings, not results for Broca.

This creates a complementary research target for our system: **a lifecycle controller** that separates evidence ingestion from durable belief formation. An observed fact can remain provisional while evidence accumulates. A summary should point back to source episodes. A skill or rule should be promoted only under declared criteria; promotion must not delete the raw evidence it depends on. Retiring a derived abstraction should not accidentally erase still-valid episodes.

### Luminous Dynamics experiment: provisional-to-durable promotion

Tracked in [issue #8445](https://github.com/Luminous-Dynamics/symthaea/issues/8445). This proposed prototype is *not* a reproduction of the published ConsolidationBench.

Compare the policies below on a deterministic oracle-by-construction stream before attempting real traces:

1. Always keep raw records, no consolidation.
2. Fixed-interval consolidation.
3. Deduplication-only.
4. Recency/frequency/utility-threshold retention.
5. Summary-only promotion.
6. Typed lifecycle policy: raw → provisional → summary → skill/rule, with explicit retire/retract transitions.
7. Oracle lifecycle choice, reported strictly as a diagnostic upper bound.

The synthetic ground truth should specify which future tasks depend on which evidence clusters and which abstraction level is appropriate. This lets the evaluator compute decision regret, but the oracle choice and future task labels must never appear in the policy's observable features.

### Mandatory anti-cheating and anti-degeneracy tests

- **Provenance gate:** mechanically inspect observable fields so the policy cannot read future outcomes, answer keys, or generator-internal state.
- **Probe gate:** test whether individual observable features trivially reveal oracle labels above a preregistered margin.
- **Count/budget matching:** give all policies the same event stream, query exposure, storage budget and distractor distribution.
- **Degenerate-forgetting gate:** a policy must not earn high “forgetting precision” by simply forgetting or refusing to write almost everything. Report coverage, load-bearing retention and transfer.
- **Poison/reversal gate:** inject plausible but false candidate rules, then corrections; require complete provenance and safe rollback without overwriting supporting episodes.
- **Distribution-shift gate:** vary prevalence, repetition frequency, noise, distractor density and query mix in held-out configurations.

### Metrics

Report end-task success; cross-level transfer accuracy; oracle-normalized decision regret by action; false promotion rate; stale abstraction rate; correction/rollback latency; load-bearing destructive-forgetting rate; coverage–utility curves; compression with complete byte accounting; provenance completeness; and write/consolidation/query latency. Report governance and task utility as separate dimensions.

This distinction is essential: a score computed from a constructed synthetic oracle can establish decision quality **within that generator**, but it cannot establish real-world transfer. Promotion policies must also run on held-out task outcomes, and a governance guarantee does not substitute for task utility.


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

The current field has several useful reference points, but their metrics and setups differ materially:

An October 12, 2026 landscape review notes there is **no single official agent-memory leaderboard**: most rankings are operated by vendors and depend on each owner's answer model, judge, memory reset rules, prompt budget and dataset version. In one common-harness comparison using GPT-4.1-mini for answers and GPT-4o-mini as judge, published vendor numbers differed significantly from the reproduced values, and the leading system varied by benchmark length. See the [October 2026 landscape review](https://aiagentmemory.org/articles/agent-memory-leaderboard/) and the [open agent-memory benchmark methodology](https://github.com/RonsenbergVI/agent-memory-benchmarks).

Therefore every comparison here should report **three separate outcomes**:
- **Retrieval recall:** did the memory system return evidence that can answer the question?
- **Task-answer/action accuracy:** did the downstream agent answer or act correctly?
- **Memory lift:** how much better did it perform than the same answer model on the same query without persistent-memory evidence?

Also capture false-positive retrievals, source grounding, ingest and query token costs, answer and judge model IDs, memory reset/isolation policy, and paired confidence intervals. A retrieval-only score is not a memory-agent score; a high answer accuracy with near-zero memory lift may largely measure the backbone rather than the memory system.
- **LongMemEval V1 vs V2:** V1 is the original 500-question benchmark with extraction, multi-session reasoning, temporal reasoning, updates and abstention categories. [LongMemEval-V2's official repository](https://github.com/xiaowu0162/LongMemEval-V2) and [leaderboard](https://xiaowu0162.github.io/longmemeval-v2/) explicitly emphasize accuracy–latency trade-offs. Its published baselines include AgentRunbook-C at 74.9% on the “small” track and 70.1% on “medium”, with reported latencies of 108.3s and 139.9s respectively; these are V2-specific results and must not be compared directly with V1 task accuracy. Use V2's own LAFS Gain and exact harness.
- **Redis (vendor report, September 2026):** Redis reports 86.5% task-averaged accuracy for Remis + Instruct on LongMemEval using a common GPT-4o answer backbone, with a modeled cost of about $0.0703 per session and 1.78s query latency in its reproduced setup. Its report also notes that published results varied substantially from its reproductions. See the [report and methodology](https://redis.io/resources/agent-memory-benchmark-report/).
- **Microsoft Research's Memora architecture (not the ACL benchmark of the same name):** its [research report](https://www.microsoft.com/en-us/research/blog/memora-a-harmonic-memory-representation-balancing-abstraction-and-specificity/) reports 87.4% on LongMemEval and 86.3% LLM-judge accuracy on LoCoMo, plus up to 98% fewer tokens versus full-context inference. These are authors' reported scores on their setup, not yet a matched result for Broca. The interesting architectural hypothesis is a harmonic/dual-resolution memory: abstractions and cue anchors navigate to precise evidence so compactness does not erase specificity.
- **Supermemory's retrieval report:** [its research page](https://supermemory.ai/research/) reports 97% Recall@20 on LongMemEval-S. That is a retrieval Recall@20 metric, **not** the same as end-to-end answer accuracy; it belongs in a retrieval table, not a direct task-accuracy ranking.
- **Mastra (vendor report):** reports 84.23% with GPT-4o and 94.87% with GPT-5-mini for its Observational Memory setup in its [February 2026 report](https://mastra.ai/research/observational-memory/). The changed answer backbone alone prevents treating those percentages as a pure memory-substrate comparison.
- **Mamba-3 (ICLR 2026):** reports a 1.8 percentage-point average downstream-accuracy improvement for its MIMO variant over the stated next-best model at 1.5B scale. This is recurrent sequence-model evidence, not a memory-agent leaderboard score; use it to motivate a matched backbone/state-size/latency comparison, not to claim Broca has matched it.
- **ACL 2026 Memora / FAMA:** [From Recall to Forgetting](https://aclanthology.org/2026.findings-acl.1337/) is a long-term memory benchmark contribution focused on invalidated facts and FAMA. It is a separate work from Microsoft Research's harmonic memory representation despite the shared name.
- **OAS (Retain or Consolidate?):** authors report a budget-dependent crossover on LongMemEval and LoCoMo, with up to 48% absolute accuracy improvement under tight LongMemEval budgets and retention preferred when evidence fits. This motivates the fixed multi-budget sweep, but the numbers are not a Broca result and require a matched reproduction: [paper](https://arxiv.org/abs/2607.17545).
- **MemPO (Findings of ACL 2026):** authors report a 7.1 absolute F1 gain over their previous SOTA baseline while reducing token use by 73.12% relative to that baseline. Treat this as a high-priority end-to-end comparison target; reproduce the same task, base model, token accounting and official scorer before comparing: [paper](https://aclanthology.org/2026.findings-acl.1166/).
- **ReMe (Findings of ACL 2026):** authors report SOTA on BFCL-V3 and AppWorld and report Qwen3-8B + ReMe outperforming memoryless Qwen3-14B. This makes ReMe a relevant whole-agent comparator for procedural memory, but not a matched HDC-backend result: [paper](https://aclanthology.org/2026.findings-acl.829/).

This evidence suggests two promising design hypotheses to test, not assume:
1. **Intervention control:** let a memory alter the current thought only when a calibrated policy earns it on actual task outcomes. Raw similarity and resonance are predictors to validate, not utility itself.
2. **Dual-resolution evidence:** keep compact cue/abstraction anchors for navigation while retaining access to source-specific facts and temporal provenance. Evaluate whether this improves multi-hop recall and cuts tokens without amplifying obsolete/conflicting abstractions.

The research strategy must therefore produce **two system-level reports** and keep retrieval metrics separate:
1. **Memory-substrate bake-off:** same frozen answer model, prompt/evaluator, data and declared total budget; isolate memory architecture. Show retrieval-only metrics such as Recall@k separately from answer/action accuracy.
2. **End-to-end system bake-off:** strongest allowed backbone and full agent stack for every system; report model/version, dataset revision, host/runtime, token prices, hardware, p50/p95 latency, ingest and query costs.

Never combine V1 task accuracy, V2 frontier metrics, Recall@20, LLM-judge scores, or scores produced with different answer models into one “SOTA” ranking. Pin leaderboard snapshots by date. A reported number is a lead for reproduction; it is not evidence of a Broca win until the evaluation can be independently reproduced on its declared configuration.
## 8. Experiment sequence

1. **Integrity first:** run MIRE reference tests and mutation controls on an exact source revision; do not interpret skipped CI as a pass.
2. **Substrate-only pilot:** use small, fixed data and one frozen answer model to compare no-memory, BM25, dense retrieval, versioned KV, current HDC/MemoryBridge and selective intervention. Keep the experiment cheap enough to run locally.
3. **External suite adapter:** first LongMemEval V1 and MemoryAgentBench, preserving official scorers and dataset versions; run LongMemEval-V2 as a separate frontier experiment, never merging its score with V1. Do not copy benchmark data into the training path.
4. **Task-grounded follow-up:** add Mem2ActBench or an equivalent frozen tool sandbox. Verify actual selected tool and exact arguments, not merely whether memory was retrieved.
5. **Frontier sweep:** cross several memory budgets with fixed workloads, multiple seeds, paired confidence intervals and full metadata accounting.
6. **Scale and robustness:** extend to long context / agent trajectories, crash/restart/replay tests and concurrent or out-of-order writers only when the baseline adapter is stable.
7. **Publish negative results too:** if HDC, recurrent state, a learned gate, or a larger mechanism loses to versioned KV or conventional retrieval under equal budgets, report that. Remove complexity that fails to produce repeatable value.

## 9. Success criterion

The ambitious claim is not “HDC memory beats RAG” or “liquid state replaces transformers.” It is:

> Given a fixed task, data, answer backbone and cost budget, the system selects the right prior information when it improves an action, does not inject stale or out-of-scope information, abstains when the evidence is unavailable, and demonstrates a reproducible quality–cost advantage on public tasks.

Until the external suites and the actual MemoryBridge are executed on a pinned revision, this document defines an experiment and decision policy. It reports **no Liquid Broca score, no measured efficiency gain, and no SOTA claim**.
