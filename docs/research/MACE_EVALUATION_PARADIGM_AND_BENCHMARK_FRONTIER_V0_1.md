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

The published textual subscores show specific opportunities: Resurface is strongest among open methods in governance, while the commercial leader has larger leads in personalization/care, epistemic safety/privacy, streaming and temporal/event sequencing. On multimodal, the visible gaps include retrieval, continual learning and relational reasoning. Coding requires a separate bug-fix improvement programme while preserving new-feature performance. Use the official detailed table for the exact values and versions.

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
