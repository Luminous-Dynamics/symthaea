# MACE Evaluation Paradigm and Benchmark Frontier v0.1

**Status:** research plan and competitive analysis. The leaderboard figures below are external reference points, not MACE scores. The benchmark adapters, matched baseline runs and independent evaluator are not yet complete.

## 1. Paradigm shift: evaluate the memory lifecycle, not a retrieval snapshot

The most important recent benchmark shift is toward interdependent, long-horizon agent-environment loops.

- [MemoryArena (ICML 2026)](https://proceedings.mlr.press/v306/he26am.html) evaluates interdependent subtasks where agents learn from earlier actions/feedback and use distilled memory to guide later actions. It reports that near-saturated performance on long-context dialogue benchmarks can coexist with poor agentic task performance.
- [AMA-Bench (ICML 2026)](https://proceedings.mlr.press/v306/zhao26bs.html) evaluates long-horizon trajectories of states, actions, observations and tool outputs, across real and synthetic applications. Its paper reports 57.22% accuracy for AMA-Agent and says the strongest baseline was exceeded by 11.16% in its own protocol. Reproduce the official setup before using that figure as a MACE comparison.
- [WorldMemArena (2026)](https://worldmemarena-mem.github.io/) evaluates 400 multi-session multimodal tasks and makes four lifecycle stages observable: write, maintain/update, retrieve, and use/act. Its design is especially relevant to stale state and evidence-chain failures.
- [Agent Memory Leaderboard Cycle 2](https://agentmemoryleaderboard.ai/results/cycle-2/) publishes separate Textual, Coding and Multimodal tracks and separate open-source / commercial / closed-source divisions.

MACE should therefore report stage-level diagnoses and end-to-end decision utility, not only recall accuracy or a single final score.

### Four lifecycle stages

1. **Write:** Did the system preserve future-useful evidence, provenance, uncertainty, causal/objective context and enough raw evidence to audit the abstraction?
2. **Maintain:** Did it supersede, retract, expire, correct, selectively repair and erase stale or poisoned memory without reviving deleted facts from backup?
3. **Retrieve/qualify:** Did it surface the evidence relevant to the exact task while passing authority, scope, provenance, temporal, contradiction, privacy and calibration gates?
4. **Use/act:** Did the downstream agent use the evidence correctly, avoid overclaiming, abstain or acquire missing evidence when necessary, and improve actual task utility?

The same event and source lineage should be traceable through all four stages. A high retrieval score must not hide a harmful write or failed retraction; a high answer score must not hide unsupported evidence or privacy violations.

## 2. Competitive anchors (Cycle 2, published 2026-11-19)

The official page reports nine leaderboards and 289 track entries. Do not pool tracks or divisions.

| Track | Open-method leader | Score | Commercial leader | Score |
|---|---|---:|---|---:|
| Textual memory | Resurface | 61.22 | MemoraX | 69.26 |
| Coding memory | InsightMem | 63.33% resolved | MemoraX | 68.00% resolved |
| Multimodal memory | Khipu mm-2 | 43.26 | MemoraX | 53.68 |

The visible open-to-commercial gaps are therefore 8.04 points textual, 4.67 percentage points coding, and 10.42 points multimodal. These are descriptive leaderboard differences, not controlled causal comparisons. Cycle 2 reports different task schemas and metrics per track; coding resolution percentages must not be directly compared with the textual or multimodal scores.

**Stretch targets (not achieved):** exceed the open-method leader by at least five points on the exact same track/protocol: textual 66.22, coding 68.33% resolved, multimodal 48.26. A credible submission must also show matched budgets, repeated runs, uncertainty and governance/cost submetrics; a higher aggregate score achieved through weaker safety or much higher resource use is not a Pareto win.

### Two-tier competitive targets

The open-leader +5 targets above are the first competitive milestone, not the final definition of superiority. A separate cross-division challenge should aim to exceed the current commercial reference by at least one point on the exact same track and scoring scale:

- Textual: 70.26 versus the 69.26 MemoraX reference.
- Coding: 69.00% resolved versus the 68.00% MemoraX reference.
- Multimodal: 54.68 versus the 53.68 MemoraX reference.

These are aspirational thresholds calculated from the published Cycle 2 results, not predictions or achieved MACE scores. The open-source submission would still be ranked in its own division; a cross-division result is only a measured comparison against the commercial reference.

To claim a genuine improvement, require a matched protocol and no regression in the relevant governance, privacy, evidence-quality and noisy-input controls. For coding, report new-feature and bug-fix separately. For Textual/Multimodal, report every official subscore. In parallel, aim for at least 25% lower all-in lifecycle cost at parity with the commercial reference, or a strictly better quality-cost Pareto frontier at the same measured budgets. Cost must include memory writing, updates, indexing, retrieval, reasoning, external tool calls, storage, energy and recovery—not just query latency.

A one-point aggregate win with a material safety regression, hidden additional model calls or an unreported token budget is not superiority. The system must publish the full scorecard and matched budget receipts.

The published textual subscores show specific opportunities: Resurface is strongest among open methods in governance, while the commercial leader has larger leads in personalization/care, epistemic safety/privacy, streaming and temporal/event sequencing. On multimodal, the visible gaps include retrieval, continual learning and relational reasoning. Coding requires a separate bug-fix improvement programme while preserving new-feature performance. Use the official detailed table for the exact values and versions.

## 2A. Industry architecture baselines beyond the leaderboard

Two additional public LongMemEval references shape the reproduction plan:

| System | Published protocol/result | Architectural lesson | Comparison caution |
|---|---|---|---|
| Redis Remis + Instruct | Redis' September 21, 2026 report: 86.5% on the 500-question LongMemEval Small split, GPT-4o answer model and judge; report shows 1.78 s query latency and about $0.0703 per session under its cost assumptions | Preserve raw chunks with hybrid BM25+dense retrieval and neighboring-context expansion, while separately retrieving extracted facts; the two stores cover different failure modes | Redis' earlier June report shows 86.1% under a distinct configuration. Keep those snapshots separate. Its cost model uses a stated average turn count and list pricing; reproduce it rather than copying it |
| Microsoft Research Memora | The ICML 2026 work reports 86.3% LLM-judge accuracy on LoCoMo and 87.4% on LongMemEval | Balance abstract memory entries with concrete values through primary abstractions and cue anchors; consolidate updates while allowing graph-like cue traversal beyond direct similarity | Paper-reported results use their own setup; do not compare directly with Redis, AML Cycle 2 or MACE without matching model, judge, split and pipeline |

Sources: [Redis June 2026 LongMemEval study](https://redis.github.io/redis-ai-research-public/longmemeval-agent-memory/), [Redis September 2026 reproduction report](https://redis.io/resources/agent-memory-benchmark-report/), and [Microsoft Research Memora](https://www.microsoft.com/en-us/research/publication/memora-a-harmonic-memory-representation-balancing-abstraction-and-specificity/).

### Required retrieval/representation baselines

MACE should implement and compare these arms against the same candidate reader and exact benchmark split:

1. Raw full-history / full-context ceiling.
2. Raw chunk retrieval: BM25, dense cosine, and reciprocal-rank fusion (RRF), with adjacent/neighbor context expansion.
3. Extracted fact memory with create/update/delete and contradiction handling.
4. Hybrid raw chunks + extracted facts (Redis-style), preserving provenance to the original turn.
5. Hierarchical abstraction + cue-anchor expansion (Memora-style).
6. Structured state / causal-objective graph.
7. Typed-content HDC/HDC-LTC candidate generation, followed by unchanged MACE evidence gates.
8. MACE decision capsule: smallest evidence representation meeting a preregistered regret/risk bound, with a replayable chain to source records.
9. No-memory and full-context baselines.

This is a deliberately stronger test than "HDC vs vector retrieval." MACE should show where HDC adds value (temporal/relational cue binding, memory footprint, update cost, source-preserving retrieval or decision regret) and where conventional hybrid retrieval is better. If a conventional arm wins on task quality and cost without compromising assurance, use it.

**LongMemEval stretch target:** after reproducing the exact public benchmark configuration, target at least 88.4% task-averaged accuracy (one point above the published 87.4% Memora reference), while reporting per-question-type accuracy, ingest/write costs, query latency, token use and lifecycle governance. This is an aspirational threshold, not a same-protocol direct comparison or a MACE result today. Do not count a score against a different split/model/judge as a win.

## 2B. Deployment is part of the benchmark, not an afterthought

The AML Cycle 2 participation contract requires participant-hosted, stable Add and Search endpoints, API authentication/capacity details and an integration smoke test. A public repository or Docker image alone does not constitute an eligible deployment. Cycle 2's published submission windows are already closed as of November 23, 2026; the strategic use of this requirement is to prepare a repeatable deployment package for the next cycle, not to suggest that this cycle can still be entered.

MACE therefore needs a provider-neutral production Add/Search service, not just a research runner:
- idempotent, bounded Add with immutable request IDs and explicit completion/abandon receipts;
- Search that returns source-linked evidence, not generated answer text;
- text, coding and ordered multimodal input preservation, including the original image parts and timestamp/speaker/session metadata;
- strict tenant/principal/authority scoping and data retention/erasure semantics;
- deterministic request accounting, rate limits, retries, health/readiness, capacity and p50/p95/p99 telemetry;
- evaluator smoke fixtures and a deployment manifest pinning image/binary, schema, config and dependencies;
- a stable endpoint that can be hosted and audited independently of the research evaluator.

The AML Add/Search score should be reported separately from MACE's decision-assurance tests. A good Add/Search adapter is a prerequisite for leaderboard participation, not evidence of causal memory transfer or adversarial lifecycle resilience.

## 2C. Agentic benchmark qualification targets

The recent agentic benchmarks expose four kinds of evidence MACE needs to preserve:

| Benchmark | Task structure / primary axes | MACE target and measures |
|---|---|---|
| AMA-Bench | 207 trajectory samples / 2,482 QA pairs across six domains (Spider/Text2SQL, GAIA, WebArena, BALROG, ALFWorld, SWE-bench); Recall, Causal Inference, State Updating, State Abstraction | Reproduce each domain with fixed agent/backbone; aim for ≥5 absolute points over the strongest matched baseline on an agreed primary metric, while separately reporting four capability axes, tool/environment cost and memory tokens |
| MemoryArena | Interdependent multi-session subtasks; earlier actions/feedback must be distilled and used in later web navigation, preference-constrained planning, progressive search and formal reasoning | Measure Session-A→B/C success lift vs no-memory and raw/full-context; require source-linked lessons, correct revision after feedback, and lower regret at matched memory/context budgets |
| WorldMemArena | 400 multi-session multimodal tasks with gold memory points, updates, distractors and evidence chains across write / maintain / retrieve / use | Publish stage-level correctness and end-to-end task success; report stale-state and visual-evidence retention, update/retraction propagation, retrieval support fidelity, and action success separately |

The AMA-Bench paper reports 57.22% for AMA-Agent and an 11.16-point improvement over its strongest baseline in that paper's own protocol. That is a reproduction anchor, not a MACE result or a direct cross-benchmark target. MemoryArena explicitly finds that near-saturated long-context recall can still fail on interdependent agent-environment tasks. WorldMemArena makes the four lifecycle stages independently visible. The lesson is to measure both what a memory system stores and how it changes later agent behaviour.

## 2D. Model-conditioned memory dosage is now a research priority

IBM Research's November 19, 2026 paper, *How Much Should an Agent Remember?*, reports that useful memory quantity depends on target-agent model capability. In its AppWorld experiments, the weaker gpt-oss-120b baseline was 39.9% task-completion; all 199 guidelines raised it to 51.2%, but a selected subset of about 29 reached 55.4% (mean of two runs). DeepSeek-V3.2 improved with the full 191-guideline set. The authors also report failure modes including dropped provenance during consolidation and swallowed tool/API errors that produced plausible logs.

MACE already has a model-conditioned memory budget prototype (v0.16) and a receipt-gated qualification path. The next step is not to hard-code 29 or 191 as universal budgets. Instead:
1. estimate a target-model-specific memory dose-response curve on calibration clusters;
2. qualify a candidate budget on disjoint held-out clusters and source/task families;
3. fit hierarchical model-family fallback when a model has insufficient calibration data;
4. evaluate leave-one-model-family-out generalization and model updates as distribution shift;
5. report utility, regret, abstention, tokens, retrieval/write latency, energy, storage and monetary cost at each dose;
6. preserve source lineage through lesson extraction and consolidation, and count failed/unreconciled tool calls as failures rather than successful experiences.

**Paradigm shift:** memory capacity should be treated like a control variable selected for a particular agent, task distribution and risk envelope—not a universal context limit or a fixed number of retained facts. The output is a qualified memory-dose policy, not just a summary size.

## 3. MACE's differentiated scorecard

MACE should run external utility tracks plus an internal decision-assurance track. Never collapse all of these into one headline percentage.

### A. Utility and decision quality

- Official track primary score (Textual / Coding / Multimodal / AMA-Bench / MemoryArena / WorldMemArena).
- Paired utility lift versus no-memory and strongest matched baseline.
- Mean and worst-group regret; compression-induced regret gap.
- Cost-regret frontier: regret or utility against tokens, CPU/GPU time, wall-clock, storage, write cost, retrieval cost, energy and dollars.
- Calibration: risk-coverage, AURC, worst-group selective risk, false sufficiency and abstention coverage.
- Evidence precision/recall, temporal-state accuracy, supersession/retraction correctness and source-lineage recovery.

### B. Governance and adversarial resilience

- Unsupported-action rate and causal overclaim rate.
- Poison acceptance and persistence at L1 single-record, L2 compositional and L3 dormant-trigger levels.
- Detection, containment and selective-repair latency.
- Retraction propagation and derived-memory invalidation completeness.
- Erasure completeness including old snapshots/backups/replicas and prevention of post-restore resurrection.
- Privacy leakage, authority/scope violations and stale-state use.
- Evidence-source diversity after common-cause/root collapse; repeated copies from one source are not independent support.

### C. Operational performance

- Add/write/update/search/query p50, p95 and p99.
- Recall/decision quality at fixed token and storage budgets.
- All-in cost per successful, evidence-supported decision.
- CPU/GPU peak and average use, energy, storage bytes, index build cost, recovery/replay overhead.
- Availability and degraded-mode correctness; unknown execution outcome must not trigger a new mutation until reconciliation.

## 4. Benchmark portfolio and correct use

| Benchmark | Primary capability | MACE use | What it does not prove by itself |
|---|---|---|---|
| AML Cycle 2 Textual | Recall, temporal, governance, personalization, streaming | Competitive text-memory baseline | Causal transfer or environment success |
| AML Cycle 2 Coding | Repository context and coding task resolution | Bug-fix/new-feature track with pinned agent and sandbox | General memory quality outside coding tasks |
| AML Cycle 2 Multimodal | Cross-modal retrieval and reasoning | Original image/audio/video evidence preservation and temporal binding | Correct actions in a live environment |
| AMA-Bench | Long-horizon trajectory memory | Causal/objective retention, state update, abstraction, retrieval | Independent security or source authenticity |
| MemoryArena | Interdependent agent-environment tasks | Session-A memory → Session-B/C actions and feedback | Broad multimedia lifecycle coverage |
| WorldMemArena | Write/maintain/retrieve/use in multimodal environments | Stage-level diagnosis and stale-world correction | Statistical causal inference as a whole |
| LongMemEval / LongMemEval-V2 | Long-history QA and evidence retrieval | Matched retrieval-only and temporal retrieval baselines | Agentic action utility |
| MemSecBench / MemoryDojo / MemPoison | Memory-specific attack and lifecycle stress | Poisoning, delayed triggers, persistence and repair | Full product security without an isolated deployment |
| MACE counterfactual suite | Decision-sufficiency and evidence ablation | Regret under event/source/root knockouts, matched random controls | External validity while public/co-located labels remain |

Run each benchmark with its own official data, scoring, license, model, tool and split protocol. Never compare LongMemEval, AML, AMA-Bench, MemoryArena and WorldMemArena percentages as if they measure the same thing.

## 5. Paradigm-shifting research agenda

### 5.1 Memory as a decision-sufficient representation

Store the smallest representation that preserves the set of downstream decisions within a preregistered regret tolerance. The decision capsule should carry evidence references, task/estimand scope, uncertainty, provenance, temporal validity, rights and a replayable derivation—not merely a summary embedding.

Compare:
- raw full-context memory;
- chunked BM25 + dense retrieval + reranking;
- structured facts/state graph;
- causal/objective graph;
- typed-content HDC/HDC-LTC;
- hybrid raw-excerpt + structured capsule;
- adaptive persist / locator / snapshot / discard;
- no-memory baseline.

Use the same downstream agent and task protocol, not a different LLM per system.

### 5.2 Evidence-aware active memory

Make memory acquisition a decision: accept current evidence, acquire more, retrieve a different source, abstain, or request an authorized probe. Use expected value of sample information only after deterministic authority and lifecycle gates; calibration must be group-conditional and evaluated under shift. Report realized probe cost and whether additional evidence changed the decision beneficially.

### 5.3 Lifecycle-aware provenance graph

Represent memory as a graph of claims and source events with multi-parent lineage, bitemporal validity, authority, rights, supersession, retraction and derivation. Retraction must invalidate descendants and all derived views; backup restore cannot resurrect erased content. This is not just a vector-index deletion operation.

### 5.4 Adversarial evaluation as a first-class benchmark

Run attacks after memory write, during consolidation, after retrieval, and during restore. Include single-record, compositional, common-cause, source impersonation, dormant triggers, contradictory update, tool-success-without-reconciliation, retraction races, and pre-erasure backup restore. Score persistence and downstream impact, not just whether a detector raised an alert.

### 5.5 Independent evaluator by construction

A separate candidate package is now an initial step (see [MACE Isolated Candidate Protocol v0.1](MACE_ISOLATED_CANDIDATE_PROTOCOL_V0_1.md)). The benchmark's evaluator must remain outside candidate reach, with one-way task delivery, sealed labels, separate credentials and signed result receipts. A separate Cargo crate or two JSONL files does not qualify as isolation.

## 6. Matched experimental design

For every external or internal claim, freeze:

- exact code commit and dependency lock;
- model, embedding, reranker, tokenizer and prompt digests;
- data version, split, source roots and task seeds;
- evaluator/judge implementation and rubric;
- agent/tool/sandbox versions;
- token, CPU/GPU, memory, storage, network and wall-clock budgets;
- warm/cold cache policy and concurrency;
- repeated run count, cluster unit and uncertainty procedure.

Use paired episodes and cluster-aware bootstrap by world/task/root rather than treating correlated tasks as independent. Keep calibration, development, public test and sealed evaluation splits distinct. Do not repeatedly tune on leaderboard/public test labels. Include negative controls and component ablations.

Every capability report should separate: **implemented**, **source-inspected**, **schema-validated**, **tests-authored**, **tests-executed**, **CI-qualified**, **sandbox-qualified**, **externally evaluated**.

## 7. Execution order

1. Land candidate artifact contract and test it locally; fix compile/runtime gaps.
2. Rewire evaluator to launch candidate subprocess; remove candidate execution from evaluator's in-process path.
3. Add resource-bound child runner and negative controls.
4. Move hidden truth/interventions into custodian-controlled service/repository.
5. Add signed artifact/request/result receipts and independent replay.
6. Build matched retrieval baselines: BM25, dense, RRF, rerank, neighbors, raw excerpts, structured facts and hybrid.
7. Run AML textual/coding/multimodal adapters separately; run AMA-Bench, MemoryArena and WorldMemArena lifecycle adapters.
8. Run adversarial-memory and selective-risk qualification on sealed splits.
9. Publish scorecards and cost-regret Pareto frontiers only after complete exact-head evidence is available.

## References

- Agent Memory Leaderboard, [Cycle 2 official results](https://agentmemoryleaderboard.ai/results/cycle-2/), published 2026-11-19.
- He et al., [MemoryArena](https://proceedings.mlr.press/v306/he26am.html), ICML 2026.
- Zhao et al., [AMA-Bench](https://proceedings.mlr.press/v306/zhao26bs.html), ICML 2026.
- Liu et al., [WorldMemArena](https://worldmemarena-mem.github.io/), 2026 benchmark site and paper.
- Luminous Dynamics, [MACE evaluator-custody qualification issue #13287](https://github.com/Luminous-Dynamics/symthaea/issues/13287) and [candidate extraction issue #13350](https://github.com/Luminous-Dynamics/symthaea/issues/13350).
