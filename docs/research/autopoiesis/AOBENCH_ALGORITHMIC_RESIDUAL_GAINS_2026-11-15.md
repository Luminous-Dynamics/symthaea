# AOBench — Evidence-Backed Residual Gains Beyond Causal Algorithm Portfolios

**Status:** Initial deterministic graph-edit evaluator committed; exact-head Rust qualification pending  
**Date:** 2026-11-15  
**Tracking:** [AOBENCH-016 #12238](https://github.com/Luminous-Dynamics/symthaea/issues/12238)  
**Research basis:** [Ma, Ding & Wang, Heuresis (arXiv:2610.22846)](https://arxiv.org/abs/2610.22846)

## 1. Why the benchmark objective changes

A causal agent can improve a result for two very different reasons:

1. It selects a stronger existing algorithm.
2. It inspects the data, performs additional tests or interventions, and recovers information absent from every candidate algorithm output.

These should not be conflated. Heuresis explicitly separates the best-chosen-algorithm result from the residual improvement requiring closer analysis of the data. It uses anonymous simulated datasets and multiple mechanism families, and traces every edge an agent changes. Its authors report GPT-6.1 Sol exceeding the best chosen algorithm across six evaluated families, with a 14 percentage-point adjacency-F1 gain on the most difficult family; Claude Opus 5.5 gained 7.4 points there. The paper reports that roughly one third of the true edges restored by these two agents on the hardest families were absent from all algorithm outputs. These are the paper's findings, not AOBench measurements.

The benchmark artifacts were not assumed to be publicly released or reproduced here. AOBENCH-016 implements an independent audit contract inspired by this evaluation design.

## 2. AOBench paradigm: measure the residual contribution

The benchmark must freeze the algorithm baseline on data disjoint from the test worlds. The agent then receives the selected baseline graph, the candidate portfolio outputs, and a declared evidence ledger. It proposes explicit edge additions, removals or reorientations. Every edit must reference one or more evidence receipts.

The evaluator holds the true graph and receipt adjudications outside the agent-facing task. It separately checks whether the evidence receipt is authentic/relevant to the edit and whether the resulting edge is actually correct.

This distinction matters:
- An authentic diagnostic may support a false hypothesis. The edit can be evidence-backed but still be a false positive.
- An unsupported edit must not be counted as an auditable scientific correction, even if it happens to match hidden truth.
- A valid final graph must be exactly reproducible by replaying the recorded edits over the frozen baseline.
- The ex-post best algorithm in the portfolio is an evaluator-only ceiling, never the deployable baseline.

## 3. Executable evaluator

The nested Rust package now contains src/algorithmic_residual.rs and exports its API from the crate root.

Agent-facing scenario:
- declared graph nodes and DAG requirement;
- calibration-selected baseline algorithm ID;
- algorithm portfolio outputs for the test instance;
- available evidence receipt IDs and descriptions;
- maximum edge-edit budget;
- mechanism-family label.

Evaluator-only oracle:
- the hidden ground-truth graph;
- evidence receipt disposition (Verified, Rejected, Unavailable);
- exact edge operations supported by each receipt.

Agent trace:
- selected baseline ID;
- final graph;
- ordered Add / Remove / Reorient operations;
- evidence receipt IDs for every operation.

The evaluator checks:
- frozen baseline selection;
- node set, endpoint, self-edge and DAG constraints;
- receipt lookup, independent status and operation support;
- edge-edit budget;
- exact replay consistency between baseline, edits and final graph.

It computes adjacency and directed precision/recall/F1, structural Hamming distance, residual gain over the frozen baseline, the ex-post suite ceiling, headroom closed, edit precision/recall, evidence-backed edit rate and recovery of true directed edges absent from every portfolio output.

A verified receipt is not a proof of edge truth. Graph metrics remain independently scored against the hidden oracle.

## 4. Deterministic fixtures

Eleven deterministic tests cover:
1. a verified, novel true edge improves the frozen baseline;
2. missing or unknown evidence;
3. rejected evidence;
4. unavailable evidence produces Inconclusive;
5. a receipt that does not support the exact edit;
6. an agent attempting to switch away from the calibration-selected baseline;
7. final graph mismatch against edit replay;
8. a DAG cycle despite a valid receipt;
9. an evidence-backed but false edge being scored as a graph error rather than hidden;
10. the ex-post algorithm-suite ceiling remaining distinct from the frozen baseline;
11. a reorientation being counted as one edge operation and one SHD change.

These are source-authored deterministic controls, not a mutation campaign, statistical benchmark result or compiler pass.

## 5. Baselines and fair comparison

- Frozen calibration-selected PC, GES, FCI, LiNGAM, NOTEARS or family-appropriate method.
- Frozen router trained/selected only on calibration worlds.
- Ex-post best-of-portfolio ceiling, diagnostic only.
- Majority, union and intersection ensembles.
- Always retain baseline / no-edit control.
- Random edit control under identical edit budget.
- Agent with bounded statistical tests, diagnostics and intervention tools.
- Hidden oracle graph as an upper bound only.

Match graph family, mechanism family, sample size, missingness, intervention count, significance thresholds, multiple-testing correction, algorithm versions, compute budget and agent tool budget. Report algorithm-selection gain separately from agent residual gain. Do not let the ex-post ceiling leak into the agent's baseline selection.

## 6. Proposed qualification targets (not achieved)

- 100% consistency between the final graph and replayed edit sequence on deterministic fixtures.
- Zero accepted edits with missing, unknown or rejected evidence receipts.
- Zero accepted cycles on DAG tasks.
- At least 90% evidence coverage for all accepted graph edits.
- Positive median residual adjacency-F1 gain over the frozen baseline on at least four of six preregistered mechanism families.
- At least 25% of baseline-to-oracle adjacency-F1 headroom closed on sealed hard-family tasks while maintaining edit precision no lower than the frozen baseline.
- On designated algorithm-blind-spot tasks, recover at least 5% of true directed edges absent from every candidate algorithm without inflating false-positive rate.
- Report every family and worst-family result. Non-positive residual gain is a legitimate result; do not weaken the baseline or edit budget to force improvement.

Thresholds must be calibrated on pilot worlds and frozen before confirmatory evaluation.

## 7. How this fits the broader benchmark landscape

CausalArena adds controlled breadth over synthetic, semantic-operational and formula-grounded SCMs, as well as public real-data checks. Its full benchmark describes 1,200 SCM specifications; the current public Hugging Face release includes 500 synthetic, 50 semantic and 50 formula-grounded configurations, alongside selected results and real-data source indices. AOBench should use these data as external comparison material where licensing and protocol permit, while keeping task manifests and model exposure explicit.

Causal-Discovery-Bench evaluates agent-led sequential experiment design and reports that most recorded errors occur during experiment design rather than final inference. AOBENCH-016 should connect its edit ledger to AOBENCH-011 intervention receipts, not merely score post-hoc graph edits.

TCD-Arena contributes a large assumption-violation ladder. The residual-gain benchmark should include compound violations and report per-family degradation rather than only pooled F1.

Relevant references:
- [Heuresis paper](https://arxiv.org/abs/2610.22846)
- [CausalArena paper](https://arxiv.org/abs/2609.11897)
- [CausalArena public dataset](https://huggingface.co/datasets/LAMDA-Tabular/CausalArena)
- [Causal-Discovery-Bench](https://arxiv.org/abs/2611.09934)
- [TCD-Arena](https://tcd-arena.github.io/)

## 8. Claim boundary

A positive residual gain demonstrates additional data analysis under a frozen protocol, not universal causal discovery. The baseline selection split, algorithm portfolio, task generator, evidence adjudication, edge-edit budget, multiple-testing policy, model/tool versions and compute budget must be fixed and published. A verified receipt establishes provenance/relevance under the evaluator contract; it does not establish that the causal edge is true.