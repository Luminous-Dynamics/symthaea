# HME query-cost estimation and proof-guided scheduling

**Date:** 2026-11-07  
**Status:** Research protocol / benchmark design; no runtime implementation or performance result is claimed  
**Parent:** #10915 — HDC-native Holographic Memory Engine  
**Related owners:** #5471, #10918, #10945, #10973, #10974, #10978, #11000, #11022

## Executive thesis

The next retrieval optimization is not simply a better similarity kernel. It is deciding which unit of work should be performed next, which evidence it is expected to reveal, and what that evidence can legitimately authorize.

HME should investigate **proof-guided value of computation (PG-VoC)**:

- A cost model predicts which candidate, block, metadata record or page is likely to produce the greatest useful progress per unit of cost.
- The query scheduler uses that prediction to order work and allocate a declared budget.
- An independent canonical exact scorer and conservative bound verifier alone determine whether a candidate may be removed from an ExactOnly query.
- The completion contract (#10973) decides whether the request finished exactly, finished under a declared approximate profile, or stopped incomplete.

This separates *where to spend effort* from *what the evidence proves*.

This issue owns retrieval-cost estimation, drift qualification and benchmark integration. It does not create a second query-budget API, metareasoning controller, ANN index, exact-score oracle or general benchmark runner. Compose the existing owners.

## 1. Competitive baseline: query-aware retrieval is now a distinct research area

### 1.1 ANNiE — learned graph-query cost estimation

Wang et al., PVLDB 19(11):3820–3833 (2026), DOI 10.14778/3836663.3836728.

ANNiE estimates the distance-computation effort required for a target recall on a particular graph index. It performs a preliminary "homing" search, extracts local graph-structural features, and uses a lightweight GBDT model. Its paper reports 6x better estimation accuracy than its baselines and 2.3x speedup for ANNiE-S over those baselines.

**Guarantee qualification:** its RTG theorem assumes a globally optimal model minimizing expected quantile loss over random model inputs X and query-cost labels Y. The resulting property is Pr[F(X) >= Y] >= p. This is a population/conditional-quantile statement, not a guarantee over index randomness for every fixed query in the sense of PQR-Search. The paper explicitly notes that the theorem only proves RTG on the training distribution and that finite test sets, train/test distribution shifts and practical optimization can violate it. In its cross-modality experiment, IMG→TXT achieved 58.55% empirical coverage against a 95% target. HME must preserve this distinction.

Reference implementation: https://github.com/CaucherWang/ANNiE  
Paper: https://doi.org/10.14778/3836663.3836728

### 1.2 QBAT — model-based per-query budget autotuning

Bae et al., PVLDB 19(11):3091–3104 (2026), DOI 10.14778/3836663.3836675.

QBAT targets clustering-based ScaNN. It explores a gradient-boosted model and an interpretable AlphaEvolve-derived heuristic to allocate query budgets for performance or recall-consistency objectives. The published abstract reports up to 68.8% lower total searched budget in its consistency mode while enforcing its per-query recall target.

HME should reproduce its *budget-allocation discipline* in a comparable workload, not claim that its own guarantees are identical.

Paper: https://doi.org/10.14778/3836663.3836675

### 1.3 DARTH and DARTH+

Chatzakis, Papakonstantinou and Palpanas, DARTH (PACMMOD 2025) and DARTH+ (March 2026 preprint).

DARTH predicts per-query recall at runtime and terminates when the desired level is reached. DARTH+ improves prediction/training efficiency and adds optional probabilistic quality guarantees for its recall predictions. This is an essential comparator for any HME probabilistic mode.

Author-maintained research overview: https://mchatzakis.github.io/declarative-recall/

### 1.4 Ada-ef — distribution-aware HNSW exploration

Zhang and Miller, "Distribution-Aware Exploration for Adaptive HNSW Search," SIGMOD 2026, arXiv:2512.06636.

Ada-ef uses a statistical query-to-database similarity model to adapt HNSW's exploration factor per query. Its paper reports up to 4x lower online latency and 50x lower offline computation / 100x lower offline memory than its evaluated learning-based adaptive baselines. It is a necessary comparison for query difficulty modelling and update-friendly calibration.

Paper: https://arxiv.org/abs/2512.06636

### 1.5 PQR-Search

Loh and Lee, arXiv:2611.07579 (2026-11-05).

PQR-Search's per-fixed-query probability is over an internal random orthogonal transform, under its stated condition that queries and updates do not depend on that transform. It is a stronger theorem class than an empirical query-population cost model, but it addresses a different problem: probabilistic per-query recall, rather than predicting HME's exact work cost.

Paper: https://arxiv.org/abs/2611.07579

### 1.6 I/O-first ANN systems

Li et al., "I/O Optimizations for Graph-Based Disk-Resident Approximate Nearest Neighbor Search: A Design Space Exploration," PVLDB 19(7), 2026, DOI 10.14778/3801059.3801064, develops a page-level I/O model and combines memory-resident navigation, dynamic width, page shuffle and page search in OctopusANN. It reports 4.1–37.9% higher throughput than Starling and 87.5–149.5% higher throughput than DiskANN at matched Recall@10=90%.

Peng et al., "SVFusion: A CPU-GPU Co-Processing Architecture for Large-Scale Real-Time Vector Search," PVLDB 19(5):1074–1087 (2026), DOI 10.14778/3796195.3796216, evaluates CPU/GPU/disk collaboration and reports 20.9x average throughput over its baselines under its workloads.

These are system-level comparators. HME must account for data movement, update costs and concurrency, not compare only a CPU similarity function against an end-to-end database.

## 2. HME research direction: proof-guided value of computation

### 2.1 Current exact-search state

For each unresolved candidate i, HME can maintain:

- pinned candidate / segment identity;
- a canonical exact incumbent score when fully scored;
- a conservative upper bound U_i for an incomplete candidate;
- the current exact top-k threshold theta_k;
- block/page metadata generation and validation state;
- bytes already read and remaining payload bytes.

For a query-and-candidate block b, define:

- r_b = ||q_b||_2 ||x_b||_2, a real-arithmetic upper contribution;
- s_b = q_b dot x_b, available only after evaluating the payload block;
- Delta_b = r_b - s_b >= 0, the bound tightening provided by reading it.

A compact sketch can predict s_b or expected bound reduction. It cannot use a predicted s_b as an admissible bound.

At candidate level, an approximate index (HNSW, IVF, LSH or PQR) may suggest evaluation order. It cannot define the candidate set or omit eligible IDs in an exact query. An exact incumbent can raise theta_k, making later admissible pruning easier.

### 2.2 Action space

The scheduler may choose among legal work actions:

1. read eligibility metadata for a pinned candidate/page;
2. fetch a candidate block;
3. fully exact-score a promising candidate;
4. verify a residual/bound metadata record;
5. fetch a page likely to contain a high-value candidate;
6. perform a canonical full scan when ordering hints are missing or untrusted.

Only #11003's eligibility policy decides whether payload access is permitted. Only the canonical score and conservative bound verifier decide exact pruning. Missing or stale generation-bound metadata cannot be treated as a low score or a prune.

### 2.3 PG-VoC objective

Use modelled utility only for scheduling. A candidate action a may be prioritized by an estimate such as:

E[future eligible payload bytes safely avoided | pre-action features, a]
---------------------------------------------------------------------
metadata bytes + payload bytes + CPU cost + I/O-latency cost

A second objective can estimate the rise in the exact kth threshold from a full candidate score. For multi-query execution, add fairness/age terms so hard queries cannot monopolize resources.

These are estimated utilities, not proof terms. No predicted utility or predicted score can authorize pruning, complete a query, or turn an incomplete result into ExactComplete.

Compare simple deterministic heuristics before GBDT:
- fixed storage order;
- descending query block energy;
- query-energy × candidate block norm;
- predicted bound-slack per byte;
- candidate-order hints from an approximate index;
- query-cost quantile model;
- hybrid scheduler with fallback to fixed order when out of distribution.

Exact-slack hindsight order is a diagnostic upper ceiling only, never a deployable policy.

## 3. Cost model and prediction targets

Train/evaluate separate models for separate cost outcomes. Do not collapse all work into "latency."

### Query-level targets

- exact completion work: candidate evaluations, coordinates, payload bytes and bound checks;
- exact completion latency / I/O wait;
- approximate candidate policy cost for a declared target (tau, delta);
- exact-fallback cost after an approximate query is not qualified;
- cold-open / warm-query / update-maintenance cost.

### Action-level targets

- expected bound reduction after reading the next block;
- probability a block causes a candidate to cross below theta_k;
- expected future payload bytes avoided;
- expected exact-threshold improvement from an exact candidate score;
- cache/page locality and concurrent resource contention.

### Features available before action

- query dimension, representation and profile ID;
- query block norms and energy distribution;
- current theta_k and exact top-k margin;
- segment/corpus generation and declared statistics;
- certified candidate/block norms or residual bounds;
- current candidate upper-bound gap;
- block size, byte cost, metadata size and page mapping;
- work already performed and recent observed bound shrink;
- query profile / distribution identity and model calibration generation.

Do not use post-action scores, final recall, or future outcome data to construct a pre-action feature.

## 4. OOD and model-trust boundary

ANNiE's empirical OOD result makes this a hard requirement.

Every model receipt must bind:
- model artifact digest and feature-schema version;
- training corpus / query panel / index-generation digests;
- target cost, target recall or target quantile;
- calibration panel and finite-sample method;
- active segment / manifest / projection / transform generation;
- OOD detector profile and state;
- observed input-feature envelope;
- prediction and actual cost when the query completes.

OOD triggers to test:
- embedding modality or encoder change;
- HDC dimension, metric, score-contract or quantization change;
- corpus/index rebuild or major update generation;
- feature range / covariance shift;
- query locality, sparsity, periodicity or block-energy shift;
- adaptive feedback queries;
- calibration residual or quantile coverage breach.

Fallback policy:
- ExactOnly: disable the predictor's ordering recommendation and use deterministic fixed order or another already-qualified exact scheduler; continue exact scanning. If the caller's deterministic budget expires, return Incomplete.
- ApproximateAllowed: a model may only use the guarantee class actually qualified for its active population and index generation. A detected calibration breach revokes that empirical profile until recalibrated; use an explicitly permitted exact fallback or return Uncertified/Incomplete.
- BestEffort: provisional output remains Incomplete until the completion contract is met.

A drift detector is not proof of distribution equivalence. A model confidence score is not a recall guarantee. Recalibration does not retrospectively certify prior queries.

## 5. Benchmark protocol

### Baselines

1. fixed budget per profile;
2. simple interpretable query-cost heuristic;
3. QBAT-inspired offline GBDT / interpretable rule on a clustering baseline, where reproducible;
4. ANNiE cost estimator and ANNiE-S on a graph baseline, where reproducible;
5. DARTH / DARTH+ recall prediction, where accessible and license-compatible;
6. Ada-ef on HNSW, where accessible;
7. HME fixed exact scan;
8. HME progressive exact fixed order;
9. HME predicted bound-slack per byte;
10. HME PG-VoC with OOD fallback.

Use each paper's official implementation and protocol when available. Record exact version, compiler, model, datasets and index parameters. Do not compare different embeddings/metrics as if they share ground truth.

### Workload strata

- in-distribution training-like queries;
- held-out query identities from same population;
- held-out corpus and encoder;
- modality / semantic profile shift;
- index generation and update shift;
- sparse and block-localized support;
- periodic / phase-aliasing structure;
- near-duplicate and hard kth-margin queries;
- high-correlation and clustered corpora;
- adversarial fixed-before-index queries;
- adaptive-feedback queries.

Use the benchmark's independent exact oracle for every query. In adaptive exact mode, query and candidate order hints may change work, never the accounted eligible set.

### Metrics

Cost estimator:
- MAE, MAPE and Q-error;
- quantile pinball loss;
- upper-tail underestimation frequency;
- empirical RTG/quantile coverage by query stratum and index generation;
- OOD detection delay and false-positive/false-negative rates;
- inference time and model memory.

HME scheduler:
- exact top-k oracle ID/order/score agreement;
- false pruning count;
- candidates fully scored / safely pruned / unprocessed;
- bound reduction per byte read;
- exact candidates eliminated per payload page;
- bytes and pages read, metadata/sketch overhead, allocations and peak RSS;
- query latency p50/p95/p99 and batch throughput;
- model inference overhead;
- total index build, model training, update, refresh, compaction and recovery cost;
- energy per correct retrieval only where physically measured.

### Promotion gates

These are project targets, not industry standards or measured results:

1. Exact agreement: 100% under canonical score and tie contract; zero false prunes.
2. Model inference overhead: no more than 5% of the corresponding retrieval budget on the declared workload, and report absolute overhead for tiny queries.
3. Empirical cost-quantile model: observed held-out coverage must meet its declared target with a predeclared one-sided confidence bound for every required stratum; never promote this to a theorem.
4. Drift: a detected profile breach disables model-guided approximate claims; ExactOnly remains correct by fallback.
5. PG-VoC efficiency: target at least 1.5x end-to-end throughput or equivalent verified payload-byte reduction at equal exact output, after model/metadata/build/update cost is included.
6. Tail latency: p95 regression <=10% and p99 separately reported; no mean-only promotion.
7. Total cost: include bytes on disk, index metadata, model artifact, build, refresh, updates, recovery and memory residency.
8. If the heuristic/learned policy cannot improve the common workload, retain fixed order. Negative results are acceptable.

## 6. New research hypotheses

### H1 — Proof progress is a better scheduling target than recall proxy

For exact HME, predict the expected reduction in *unresolved admissible bound gap per byte*, rather than trying to predict recall. Test if this better generalizes across corpus/query shifts because its target is grounded in HME's own observable proof state.

### H2 — Candidate and block scheduling compose

An approximate index can schedule promising candidates; an HDC-specific block predictor can schedule their internal payload blocks. Test the composition against fixed/fixed and each single-level policy. All eligible candidates remain accounted for.

### H3 — Calibration can be revoked online

Use small audit panels, residual/coverage monitors and generation-bound profiles. Detect when the current query population no longer matches the model's training envelope, and disable only the model-dependent claim—not the exact algorithm.

### H4 — Cost estimates can be safely useful without being correctness evidence

A query cost prediction may allocate a budget or choose an order while exact upper bounds remain the sole pruning authority. This may permit useful learning without importing ANN recall risk into ExactOnly retrieval.

## 7. Relationship to existing owners

- #5471 owns the general shadow cognitive value-of-computation scheduler. HME contributes retrieval-specific action/cost receipts; it does not create another controller.
- #10973 owns the HME request budget, execution and completion-state contract.
- #10945 owns progressive exact bounds and false-prune qualification.
- #11000 owns exact block-ordering heuristics.
- #10974 owns approximate sampling profiles.
- #11022 owns cross-algorithm per-query recall reliability comparison.
- #11003 owns metadata-first eligibility.
- #10979/#11001 own learner checkpoints and consistent-cut recovery.

## 8. Nonclaims

No query-cost model, PG-VoC scheduler, OOD detector or HME adaptive runtime is implemented by this research protocol. The published QBAT, ANNiE, DARTH+, Ada-ef, OctopusANN and SVFusion results are not HME results. No SOTA win, theorem or runtime speedup is claimed.

All learned and heuristic outputs remain advisory. The exact score contract, conservative interval bounds, eligibility generation and typed completion state remain the authorities for exact-mode claims.


## 2026-11-08 score-contract binding

Cost models and scheduling receipts must bind the exact score semantics used to produce their observed margins, bounds and completion cost. For ContinuousHV/cosine, use `continuous_hv_cosine_full_f32_group8_v1`.

The profile admits only finite components in [-1, 1] and pins full-coordinate f32 eight-coordinate grouped accumulation plus ordered tail and the near-zero denominator rule. A model trained against a different metric, reduction order, range or dimension profile is not interchangeable; invalidate/recalibrate its model receipt when these semantics change.

The source now contains the versioned contract ID and admission checks, but they are not yet integrated into segment storage or the cost-model runtime.