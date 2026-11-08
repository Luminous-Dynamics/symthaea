# HME benchmark and paradigm-shift execution plan

**Date:** 2026-11-08  
**Status:** benchmark specification and research agenda; no HME performance result is claimed  
**Branch:** research/hlb-memory-engine-20261107  
**Parent:** [#10915 — HDC-native Holographic Memory Engine](https://github.com/Luminous-Dynamics/symthaea/issues/10915)

## Executive thesis

HME should not try to win by being another generic vector database. Its defensible target is **proof-carrying, representation-aware memory retrieval**:

1. preserve the canonical score and memory semantics;
2. establish complete, policy-filtered corpus coverage;
3. eliminate candidates only when their certified upper bounds cannot cross the exact top-k boundary;
4. fetch original payloads only for the unresolved refinement set;
5. return an auditable completion state and enough evidence to reproduce the decision.

This combines exact retrieval, data-local execution, per-query work allocation, temporal/governance semantics and evidence provenance. It is a stronger claim than a speed-only ANN leaderboard result.

## 1. Paradigm shifts with falsifiable payoffs

### 1.1 Proof-carrying retrieval, not bare nearest-neighbour results

Treat each retrieval as a chain of independently checkable claims:

- **Snapshot claim:** exact manifest, eligible membership, rights scope, valid-time/known-time and representation generation.
- **Synopsis claim:** a segment/group/page bound was built from every member of that pinned scope.
- **Numeric claim:** each interval encloses the score produced by the versioned canonical f32 scorer.
- **Refinement claim:** every unresolved payload was read and exactly reranked.
- **Completion claim:** the result is ExactComplete only when eligible coverage and all refinement work are complete.

A similarity score is not evidence of truth, freshness, authority or permission. Hard eligibility must precede ranking and payload access. A corrupted or stale proof component means no prune, a full exact fallback, or an explicit incomplete/error state—never a quiet best-effort exact label.

**Research payoff:** make correctness compositional and independently auditable while reducing the number of original payload bytes that must be touched.

### 1.2 Group geometry should exploit the joint residual, not discard it

The current group path intersects two enclosures: coordinate minima/maxima (a box) and a deterministic centroid plus verified L2 radius (a ball). The box is robust but forgets the joint residual budget. The ball keeps it, but the current score bound uses separate Cauchy–Schwarz dot and reverse-triangle norm bounds, which can be loose.

A promising next step is a **spherical-cone bound**. When a ball B(c,r) does not contain the origin (r < ||c||₂), every nonzero vector in the ball lies within a cone around c whose half-angle is α = asin(r / ||c||₂). For unit query direction u = q / ||q||₂, set d = u · c / ||c||₂ = cos(θ). The real-arithmetic cosine extrema over the cone are:

- upper: 1 when θ ≤ α, otherwise cos(θ − α);
- lower: −1 when π − θ ≤ α, otherwise cos(θ + α).

Away from the saturation cases, the endpoints are d cos(α) ± sqrt(1 − d²) sin(α).

This is a **research candidate**, not a prune-authorizing implementation yet. It must be turned into outward-rounded intervals for the exact f32 score contract, with conservative treatment of centre norm, query-centre dot, asin/sqrt or algebraic alternatives, denominator threshold and all rounding error. If the ball contains the origin, return [-1,1]. Compare this cone interval against the existing joint-ball and coordinate-box intervals and intersect only after proving each enclosure independently valid.

**Acceptance:** zero containment escapes across adversarial dimensions and targets; strictly tighter median/p95 interval widths on a named clustered workload; no regression on random, low-margin or near-zero-norm groups; and measured payload bytes avoided after synopsis overhead.

Owner: extend #11078 / #11059. Do not replace the current bound until a proof and differential test corpus are reviewed.

### 1.3 Optimize the decision boundary, not compression MSE

A code is useful when it preserves the top-k decision, not merely when its reconstruction has a low average error. Record exact score margin between ranks k and k+1, whether quantization changes either side of that boundary, residual norm and direction relative to the query, interval width and refinement-set cardinality.

A lower-MSE code may still be worse at ranking near-ties. Co-design the codebook, residual certificate and query synopsis against rank-boundary preservation. Keep this separate from any probabilistic quality claim.

### 1.4 Schedule work by certified value-of-computation

The query planner should choose the next action that most reduces expected unresolved work per unit cost, while keeping the proof condition independent of the predictor. Candidate actions include opening a segment synopsis, evaluating another block, decoding a compact reconstruction, reading an original payload, or stopping with a truthful partial result.

Use QBAT as a comparator for query-specific budget allocation, but learn only the ordering/budget hint. The hard correctness gate remains the validated interval and full eligible coverage. A budget exhausted before proof means Approximate/Incomplete—not ExactComplete.

### 1.5 Memory should represent events and revisions, not just vectors

Semantic similarity should retrieve candidate events/claims; the authoritative memory object must also retain source, revision lineage, temporal validity, knowledge time, scope, rights and revocation dependencies. A correction is not another equally valid vector. It changes which claims are current and which descendants can still be used.

Measure the full lifecycle: ingest, revise, supersede, retract, forget, consolidate, recover and answer. This is where HME can exceed vector-database semantics rather than simply imitate them.

## 2. Industry landscape as of 2026-11-08

| System / benchmark | What it contributes | How HME should compare | Important limitation |
|---|---|---|---|
| FAISS Flat / canonical scalar scan | Exact brute-force ground truth | Same vectors, metric, k and tie policy | O(ND), but the correctness anchor |
| HNSW / DiskANN / Vamana | Strong graph-based ANN baselines | Recall–latency–memory frontier and dynamic-update tests | Average recall does not describe per-query tail failures |
| IVF-PQ / ScaNN / RaBitQ | Compact-code and coarse-quantized search | Same bytes/vector, exact rerank, candidate count and recall | Index-specific candidate coverage differs |
| PQR-Search (arXiv 2611.07579, Nov 5 2026) | Per-fixed-query probabilistic recall target Pr[Recall(q)<τ]≤δ, with an internal random transform | Compare per-query failure audits, OOD, update/delete stability, latency and memory | Probabilistic guarantee over the index transform, not deterministic 100% exact top-k |
| Recast (arXiv 2611.02765, Nov 2 2026) | Refinement-centred storage; compact in-memory codes plus raw vectors on disk and uncertainty-driven candidate reads | Compare cold payload I/O, update/delete/compaction, recall and matched-memory QPS | High-probability intervals and selected IVF lists; not global deterministic coverage |
| QBAT (PVLDB 2026) | Query-specific budget allocation; reported up to 68.8% lower searched budget in ScaNN consistency mode | Compare work allocation against a static budget and learned scheduler | Budget prediction is not itself a correctness proof |
| VIBE | Current algorithm-level ANN benchmark with modern in-/out-of-distribution embedding datasets | Use benchmark datasets and query methodology where representation/distance is compatible | VIBE explicitly is not a vector database or cloud-service benchmark |
| VectorDBBench | End-to-end database/platform behavior, filtering and operational cases | Use for service-level comparisons where supported vector profiles match | Service-level results are not HDC-native score semantics |
| MLPerf Storage v3.0 Vector Database | Storage-path benchmark for an on-disk DiskANN-backed Milvus database | Adopt its I/O, cache, concurrency, exact-ground-truth and latency discipline | Published CLOSED workload is 10M × 1,536D float32, top-10; not directly 16K–256K ContinuousHV |
| LongMemEval / MemoryAgentBench / Mem2ActBench | Downstream memory utility, temporal updates, selective forgetting and action grounding | Hold reader and context budget fixed; compare the memory substrate, not different LLMs | Answer quality is not vector retrieval quality |

### Research findings worth treating seriously

- **PQR-Search** reports a per-query probability guarantee and audits across 20 datasets up to 100M vectors. Its paper reports 26.4× geometric-mean batch throughput over exhaustive FAISS in static settings and 69–154× at 100M, while single-query latency is higher than HNSW in the reported comparison. It is a current direct comparator; reproduce its guarantee conditions rather than translating it into an HME result.
- **Recast** reports 1.56–2.01× QPS over its strongest matched-memory baseline at 0.99 recall, 1.22–6.75× less external storage, and 3.10–37.27× faster batch updates than SPFresh/OdinANN in its tested settings. Its method is a close architectural neighbour to HME’s refinement path, so matched payload-I/O and update workloads are essential.
- **QBAT** reports up to 68.8% less searched budget in ScaNN consistency mode by adapting work to query difficulty. HME should target a better *certified* work scheduler, not just learn the same budget predictor.
- **VIBE** includes in-distribution and OOD embedding datasets and compares ANN algorithms; its maintainers distinguish that scope from database/cloud benchmarking.
- **MLPerf Storage v3.0** added a vector database workload. The published CLOSED case uses 10 million 1,536D float32 vectors in Milvus/DiskANN, top-10 search and recall checked against brute force. It provides useful storage discipline, not a directly comparable dimensional profile.
- **LongMemEval** tests information extraction, multi-session and temporal reasoning, knowledge updates and abstention over 500 questions. **MemoryAgentBench** adds accurate retrieval, test-time learning, long-range understanding and selective forgetting. **Mem2ActBench** evaluates proactive memory use for tool-argument grounding, using 2,029 sessions and 400 tool-use tasks; it addresses a weakness in benchmarks that ask only explicit fact questions.

## 3. Frozen benchmark tracks

### Track A — exact HDC retrieval

**Reference:** the canonical full-resolution ContinuousHV score and exhaustive full-payload top-k using the pinned stable-ID tie policy.

Compare:
1. exact full scan;
2. exact scan with current page/segment layout;
3. progressive exact scan (#10945/#11000);
4. per-record certified compressed refinement (#10978);
5. group/page envelopes (#11059/#11078);
6. hybrid selection that uses a synopsis first and exact-scores all survivors.

Hard gates:
- 100% top-k ID and order agreement with the canonical oracle;
- zero false-prune across every frozen and adversarial query;
- exact output scores are canonical f32 scores, not reconstructed approximations;
- complete, generation-bound eligibility coverage;
- no excluded, revoked or unauthorized payload is read or emitted;
- all missing/corrupt/stale proof metadata falls back or returns explicit incomplete/error.

Measure:
- eligible count, interval count, survivors, excluded count;
- cold original-payload bytes avoided and actual device bytes read;
- synopsis/code/certificate/index bytes and construction cost;
- p50/p95/p99 latency, QPS and batch throughput;
- cache residency, page faults, CPU cycles/instructions and allocations where reliably available;
- build, insert, delete, compaction, crash/reopen and rebuild costs.

Primary HME claim to earn: **100% exact top-k agreement with at least 50% fewer cold original-payload bytes on a named workload, after all synopsis/code/index overhead is counted.** If the index saves bytes but increases end-to-end latency or total storage beyond the declared Pareto bound, report that honestly.

### Track B — approximate ANN

Use VIBE-compatible public dense-embedding datasets and, separately, HME-native high-dimensional synthetic/learned HDC datasets. Keep representation and distance type explicit; do not compare BinaryHV Hamming directly to float cosine or represent 256K HDC as a backend-supported profile when it is not.

Baselines: FAISS Flat ground truth, HNSW, DiskANN/Vamana, IVF-PQ, ScaNN, RaBitQ and PQR-Search. Include Recast for cold refinement/I/O and QBAT for query-budget scheduling.

Report Recall@1/10/100 and exact top-k agreement where claimed; per-query recall distribution, minimum, p1/p5 and fraction below target; confidence interval and independent query/transform counts; in-distribution vs OOD and pre-/post-update cohorts; bytes/vector, RAM, build/update/delete/compaction; QPS, p50/p95/p99, batch throughput and cold/warm behavior; recall–latency–memory–I/O Pareto frontiers.

Proposed separate result profiles:
- **Approx-Standard:** target τ≥0.99, δ≤0.01.
- **Approx-High-Reliability:** stretch target τ≥0.999, δ≤0.001.

These are project targets, not industry standards. An empirical audit is not a mathematical guarantee; distinguish observed tail frequency, a calibrated statistical guarantee and a formal per-query guarantee.

### Track C — agent-memory utility

Use official versions/splits/evaluators for:
- LongMemEval / LongMemEval-S: temporal reasoning, multi-session updates and abstention;
- LoCoMo: long conversational histories and multi-hop/temporal questions;
- MemoryAgentBench: test-time learning, long-range understanding, selective forgetting and accurate retrieval;
- Mem2ActBench: implicit memory use and correct grounding of tool arguments;
- Agent Memory Leaderboard where its current official submission/API protocol is available.

Freeze the same downstream reader/backbone, judge, prompt, context-token budget, tool-call budget, history, task split and decoding settings. Compare no memory, full-history oracle, lexical, dense, hybrid, current Broca/HDC and HME. Also run a canonical-memory-pool reader control to separate retrieval substrate from ingestion/extraction quality.

Metrics:
- answer correctness and strict evidence recall;
- temporal/state-update accuracy and obsolete-fact reuse;
- abstention precision/recall on unanswerable questions;
- selective forgetting and deletion/resurrection controls;
- source attribution and evidence localization;
- correct tool-action/parameter grounding;
- context tokens, calls, p50/p95/p99, storage and background lifecycle cost.

Promotion target: hold answer quality within a preregistered 1 percentage-point non-inferiority margin under the same reader and token budget while reducing the full steady-state + retrieval + consolidation cost by at least 25%; alternatively, improve paired quality by at least 3 percentage points at equal total cost. Do not pool scores across benchmark families.

## 4. Test corpus and adversarial controls

Build one frozen corpus manifest with exact vector bytes, IDs, metric, dimension, transformation/codebook identity, temporal scope, rights scope and expected top-k. Include:
- random isotropic and orthogonal vectors;
- clustered, anisotropic and highly correlated vectors;
- query/corpus distribution shift;
- sparse one-coordinate signals and phase/periodic structures;
- identical and near-duplicate items;
- exact ties and margins around the kth boundary;
- near-zero vector norms and denominator threshold overlap;
- dimensions 2K, 4K, 8K, 16K, 32K, 64K, 128K and 256K where supported;
- high-fan-in/correlation fixtures already owned by #7772;
- insert, delete, update, supersede, revoke and restart sequences;
- corrupt, stale, cross-generation and incomplete-scope metadata.

Negative controls are required: put the true nearest item only in the last page/block; make a weak centroid hide a strong outlier; make one member violate the claimed radius; invalidate one payload digest; change one eligibility generation; make the kth comparison equal; and test when all scores are so close that the safe answer is to refine everything.

For statistical profiles, use held-out seeds, untouched test query sets, OOD cohorts and confidence intervals. Do not tune on test queries and then report them as a blind audit.

## 5. Systems protocol

Freeze and publish:
- source commit, compiler/toolchain, feature flags, CPU model/ISA, RAM, storage device/filesystem and OS;
- dataset and query manifest digests;
- index/segment configuration, compression settings and thread counts;
- cache condition (warm, process-cold, OS-page-cache-cold or device-cold—do not conflate);
- batch size, concurrency, arrival pattern, query mix and duration;
- durability mode, fsync policy and whether build/compaction is inside or outside the timed interval;
- exact oracle version and score contract.

Run enough repetitions for stable tail estimates. Publish raw per-query output, not only averages. Use paired query order and identical hardware; separate CPU-only and GPU-accelerated tracks. Report total build and maintenance costs alongside query costs.

## 6. Proposed scorecard

| Dimension | Promotion gate / stretch target |
|---|---|
| Exact top-k | 100% ID/order/score-contract agreement |
| False pruning | 0 on the complete frozen/adversarial corpus |
| Eligibility and governance | 0 unauthorized reads/returns; complete pinned scope |
| Exact-mode byte savings | ≥50% cold original-payload bytes avoided on one named workload after fast-tier overhead |
| Exact-mode speed | ≥1.5× useful throughput or a measured Pareto improvement; no hidden quality trade |
| Approx-Standard | τ≥0.99, δ≤0.01 |
| Approx-High-Reliability | stretch target τ≥0.999, δ≤0.001 |
| Scheduler overhead | ≤5% of retrieval budget on target workloads |
| Agent-memory efficiency | ≤1 pp quality loss and ≥25% lower total cost, or ≥3 pp paired quality gain at equal cost |
| Update resilience | report recall/QPS and proof coverage after 10%, 25%, 50% corpus churn |
| Recovery | 100% correct last-committed generation restoration or explicit fail-closed error |

Every number in this table is a proposed project gate, not an achieved result and not an industry standard.

## 7. Immediate execution sequence

1. Finish numeric qualification of the residual verifier, group-ball kernel and box/ball intersection; run authored tests when a Rust toolchain is available, but continue source review now.
2. Add the angular-cone group bound as an independent optional interval and intersect only after proof and differential containment tests.
3. Persist the synopsis with the #10917 immutable segment manifest, exact membership, eligibility generation and checksummed content. Reopen must revalidate binding before any prune.
4. Feed group intervals into #11078's global kth-lower-threshold selector. Ensure all eligible groups are represented; open any group whose upper bound reaches the threshold.
5. Add a query-cost scheduler that orders synopsis evaluation and exact payload I/O. Predictor outputs affect ordering only; interval proof decides exclusion.
6. Run Track A first against exact scan. Only then add ANN comparators and downstream memory integrations.
7. Publish per-query outputs, workload manifests and negative controls, including failed or uninformative cases.

## 8. Nonclaims

No HME speedup, exact-complete production receipt, industry-leading score, deterministic cross-platform interval proof or memory-agent utility gain is claimed by this document or the current prototypes. The current group envelope remains in-memory; persistent segment verification and end-to-end read accounting are not yet integrated.

## References

- VIBE benchmark and scope: https://vector-index-bench.github.io/ and https://github.com/vector-index-bench/vibe
- PQR-Search, arXiv:2611.07579 (submitted 5 Nov 2026): https://arxiv.org/abs/2611.07579
- Recast, arXiv:2611.02765 (submitted 2 Nov 2026): https://arxiv.org/abs/2611.02765
- QBAT, PVLDB 2026: https://research.google/pubs/qbat-model-based-query-budget-autotuner-for-clustering-based-approximate-nearest-neighbor-search/
- MLPerf Storage v3.0 and vector database workload: https://mlcommons.org/2026/09/mlperf-storage-v3-0-results/ and https://github.com/mlcommons/storage/blob/main/vdb_benchmark/README.md
- LongMemEval, ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html
- MemoryAgentBench, ICLR 2026: https://mlanthology.org/iclr/2026/hu2026iclr-evaluating/
- Mem2ActBench, ACL 2026: https://aclanthology.org/2026.acl-long.370/


## 9. Current geometric-bound prototype and evidence (2026-11-08)

A separate real-arithmetic spherical-cap interval prototype now exists in crates/core/symthaea-core/src/hdc/continuous_hv_interval.rs:

- Public result type: ContinuousHVRealCosineInterval.
- Method: ContinuousHV::real_cosine_interval_for_l2_ball_center.
- The radius-to-centroid norm, normalized dot, square roots, algebraic sqrt(1 - z^2) terms, and endpoint products are propagated with outward f64 intervals.
- The formula avoids asin/acos; it computes sin(alpha) = r / ||c|| and cos(alpha) = sqrt(1 - sin(alpha)^2).
- It returns [-1,1] when the ball may contain the origin, the query or center is zero/uncertain, or arithmetic cannot be safely bounded.

**Critical type boundary:** this is a real-cosine geometry result, not an enclosure for the canonical f32 score. It is intentionally not accepted by ScoreIntervalRecord or select_refinement_set, and it must not authorize payload pruning. The eventual f32 score enclosure needs a separate derivation that includes the canonical reduction, denominator, near-zero branch and rounding behavior. The current exact interval path remains the only candidate for deterministic pruning.

Three Rust test groups have been authored for this prototype: generated high-dimensional member containment; aligned/antipodal saturation; and origin-containing-ball fallback. A further property test generates random residual directions across dimensions 1 through 16,384 and independently verifies each residual before testing real-cosine containment. None has been executed because this environment lacks cargo and rustc.

A separate Python numerical simulation exercised 2,400 randomly generated query/center/ball/member cases at dimensions 1, 7, 8, 9, 17, 257, 2,048 and 16,384. It observed zero real-cosine interval escapes. This is an exploratory simulation, not execution of the Rust implementation, a formal proof, a canonical-f32 result, or a performance benchmark.

A second prototype now exists: ContinuousHVSphericalCapF32ScoreInterval from similarity_interval_full_resolution_for_l2_ball_center_spherical_cap. It maps the real spherical-cap cosine interval through separate gamma/eta bounds for the canonical dot/norm reductions, the f32 norm-product multiply, square root and quotient. This type is not accepted by the HME selector; it remains an unqualified research candidate.

A separate Python f32 arithmetic simulation compared this candidate interval against the existing joint-ball canonical interval on 900 generated members at dimensions 8, 9, 17, 257, 2,048 and 16,384. It observed zero canonical-score escapes for both and candidate mean interval width 0.0585 vs 0.0732 for the joint-ball interval; the candidate was narrower in all 900 samples. This is not Rust execution, proof, a benchmark run or permission to prune.

A separate aligned-cluster geometry comparison with 1,000 query/centroid cases per dimension (8, 128 and 16,384), at radius 0.08 times centroid norm, observed:
- spherical-cap mean interval widths: 0.0402, 0.0426 and 0.0427;
- independent Cauchy/reverse-triangle surrogate mean widths: 0.1814, 0.1817 and 0.1818;
- cone interval narrower in all 3,000 sampled cases.

The comparison is synthetic and uses real-arithmetic enclosures. It does not establish safe canonical pruning or end-to-end speedup. See #11142 for the qualification contract.

## 10. Agent-memory frontier: lifecycle and relationship are now first-class benchmark axes

Do not stop at LongMemEval, LoCoMo or retrieval Recall@K. Current evaluations expose two gaps particularly relevant to Luminous:

### MEME — multi-entity and evolving memory

MEME (arXiv:2605.12477) evaluates six tasks: Exact Recall, Aggregation, Tracking, Deletion, Cascade and Absence. Its 100 controlled episodes test whether a system updates dependent facts when upstream facts change, honors deletion, and knows when an old conclusion is no longer justified. The paper reports mean baseline accuracy of 3% on Cascade and 1% on Absence despite adequate static retrieval. This is an author-reported benchmark result, not an HME result. A separate MEME benchmark report published by N71 describes its own June 12–16, 2026 full-suite run and reports iteration-3 overall 0.574, Cascade 0.628 and Absence 0.42. This is a vendor-published system result, not an independent replication; it is useful as a stronger moving comparison point than the paper's default field average, but only a matched reproduction can support ranking claims.

Use MEME to test whether HME's event/provenance links and #9075 bitemporal ledger can invalidate or revise dependent memories, rather than merely retrieving the most similar historical vector. The authoritative dependency graph and temporal semantics belong to the existing ledger; HME should be its fast, rebuildable retrieval materialization, not a second truth store.

### DyadMem — user-conditioned relational agent memory

DyadMem (arXiv:2610.03020) explicitly evaluates both user-side memory and relational agent memory—how the agent should work with a particular user as the shared history evolves. Its release describes 3,065 episodes, 50,961 sessions and 61,210 QA instances, with session-level Capture/Update and query-level Recall annotations. It separates Gold-Memory QA from Full-Pipeline QA.

This distinction is important: a high answer score with gold memory does not prove the system can capture, update, retrieve and use the right memory in a real pipeline. Compare:
- Gold-Memory: tests the reader and memory representation given correct memory;
- Full-Pipeline: tests capture, consolidation, retrieval, revision and answer integration;
- user facts vs agent-user working agreements/preferences;
- correct retrieval vs correct use of a memory, including when it should not be surfaced.

Add DyadMem as a separate agent-memory track; do not pool its score with MEME or LongMemEval.

### Luminous HME-Revise diagnostic suite

The first synthetic contract fixture is [hme_revise_lifecycle_v0.json](fixtures/hme_revise_lifecycle_v0.json). It covers late-arriving corrections, bitemporal queries, dependency invalidation, deletion/resurrection after reopen, alternative proof sets and typed absence. It is not a runtime test or benchmark result.

Use these public benchmarks plus a small project-owned deterministic diagnostic suite. Each fixture should be a graph of source events, claims, derived claims, temporal scopes and typed dependencies. It is not a replacement for the public benchmark.

Required sequences:
1. admit a source event and derive one or more conclusions;
2. update or supersede the source, including a late-arriving correction with an earlier valid time;
3. retract or delete the source, then test dependent conclusion availability;
4. query both "what was true at valid time V, as known at knowledge time T?" and "what do we now believe was true at V?";
5. compact, restart and rebuild all retrieval indexes;
6. check that obsolete conclusions are not emitted or resurrected and that unsupported absence is not represented as certainty.

First-class metrics:
- dependency-cascade precision and recall;
- stale-conclusion emission rate;
- false-absence and false-unknown rate;
- bitemporal leakage rate;
- deleted/revoked descendant read and emission count;
- invalidation-to-unavailability latency;
- resurrection rate after compaction/reopen/replay;
- provenance completeness and independent-source diversity;
- correct abstention with a typed reason (unknown, stale, superseded, retracted, not authorized, or not observed).

Use MEME for dependency closure/deletion/absence, DyadMem for capture/update/relationship-specific memory, Memora's FAMA for stale-memory penalties, and LongMemEval/MemoryAgentBench/Mem2ActBench for temporal reasoning, selective forgetting and action grounding. Reuse #8427 for the common reader/cost harness, #9075 for valid-time/known-time semantics, #9048 for alternative AND/OR justifications and explicit Unsupported state, and #9139 for governed consolidation, forgetting and erasure policy.

### Paradigm shift: absence and invalidation are data

A missing nearest neighbor is not evidence that a proposition is false; a tombstoned memory is not the same as a never-observed fact; a superseded claim is not the same as an unknown claim. Model these states distinctly and preserve the source of the state transition. The reader should receive a small typed answerability/evidence status rather than an unlabelled empty result.

A second paradigm shift is **write-time revision with read-time proof checking**:
- source updates propagate supersession/invalidation through the authoritative dependency graph;
- HME indexes are rebuilt or generation-invalidated as derived views;
- query-time proof checks catch stale materializations and policy changes;
- every response is tied to the bitemporal snapshot and current authorization scope.

This directly addresses cascading stale memory without asking a larger LLM to reconstruct every dependency from a large transcript at answer time.

## 11. Competitive interpretation and promotion rules

The frontier is not one aggregate score:
- **PQR-Search** is the strongest comparator for per-query probabilistic ANN guarantees under updates. Its guarantee is not the same as deterministic complete exact top-k.
- **Recast** is the closest storage/refinement architecture comparator. Match its memory budget, raw vector bytes, query distribution, update/delete volume and recall target before comparing QPS or device I/O.
- **QBAT** is the query-specific budget-allocation comparator. HME should let a scheduler predict work order, while only certified bounds may reject a candidate.
- **MEME / DyadMem / LongMemEval / MemoryAgentBench** test semantic and lifecycle quality. A retrieval speedup is irrelevant if HME returns stale or unauthorized memory.
- **MLPerf Storage** supplies system/I/O discipline; use its native profile for that workload, and separately publish HME-native 16K/64K/256K traces.

For every competitor, identify whether the reported result is author/vendor-reported, independently reproduced, or run under the same HME harness. Do not rank results across different dimensions, metrics, readers, context budgets, or guarantee classes.

## References added

- MEME (arXiv:2605.12477): https://arxiv.org/abs/2605.12477 ; project page: https://seokwonjung-jay.github.io/meme-eval/
- DyadMem (arXiv:2610.03020): https://arxiv.org/abs/2610.03020 ; project page: https://renaissancet.github.io/DyadMem/
- Rust f64 primitive precision contract (sqrt and other operations): https://doc.rust-lang.org/std/primitive.f64.html


### Proposed lifecycle benchmark promotion targets

These are HME project targets, not published standards or achieved scores:

| Evaluation | Target |
|---|---|
| MEME Exact Recall / Tracking / Aggregation | ≥0.95 each under the frozen reader |
| MEME Deletion | ≥0.99, with zero resurrection in deterministic reopen/replay controls |
| MEME Cascade / Absence | ≥0.90 each as a stretch target; no false claim of certainty when replacement evidence is missing |
| DyadMem Full-Pipeline vs Gold-Memory | ≤5 percentage-point gap on matched QA scoring, with Capture/Update support recall ≥0.95 |
| HME-Revise deterministic suite | 100% bitemporal correctness; zero late-knowledge leakage, stale dependent emission, deleted-item resurrection or unauthorized pre-read |
| LongMemEval / LoCoMo downstream | non-inferior strict answer accuracy at equal reader/context budget, with stale-use and abstention measured separately |

For public benchmarks, use official splits, official scorers, licensed data and reproducible reader/judge settings. The project-owned HME-Revise fixture is a deterministic diagnostic gate, not an alternative public benchmark and not a substitute for human evaluation.

A high MEME or DyadMem score cannot compensate for data-governance violations. The safety invariants are independent hard gates, not weighted terms in a single aggregate.


## 12. External agent-memory leaderboard and embodied-memory frontier

### Agent Memory Leaderboard (AML)

AML is now a useful industry-facing benchmark target in addition to paper-specific datasets. Its Cycle 1 results (published 12 August 2026) include separate textual, multimodal and coding tracks, with metrics covering explicit fact recall, compositional inference, temporal/event reasoning, memory governance, personalization/care, context learning/execution, and safety/privacy.

Published Cycle 1 textual results show:
- commercial products: MemoraX v0.5 at 58.02 overall;
- open-source methods: InvMem at 45.06 overall.

The AML landing page states that Cycle 2 results are expected in mid-November 2026. As of 8 November 2026, Cycle 2 has not yet been published, and its materials submission window is closed. Do not infer that HME has a current submission or score.

The evaluation path requires participant-hosted Add/Search APIs and a formal evaluation after a smoke test; it is not a repository-only offline submission. This is a valuable future productization gate: build a stable, versioned public Add/Search interface only after local correctness and governance qualification.

AML scores should be reported separately by track and metric; don't compare its overall score directly with MEME or LongMemEval.

### MEMORA-Bench — embodied and multimodal memory

MEMORA-Bench (Findings of EMNLP 2026) evaluates memory formed from 45 hours of egocentric video, with retrospective memory-faithfulness and prospective memory-grounded planning tasks. Its framework maintains typed spatial, object-identity/state and temporal-action evidence, then consolidates repeated experience into routines, habits and preferences.

This is especially relevant as an optional HME multimodal/embodied track:
- index HDC or other supported representations of objects, states, places and action episodes;
- preserve links to source video/time evidence and observation provenance;
- measure retrospective faithfulness separately from prospective planning utility;
- hold perception and answer/planning model constant between memory baselines;
- report OOD goal composition and physical/action grounding separately from textual memory scores.

Do not conflate MEMORA-Bench (embodied video memory) with Memora (ACL 2026 personalized-agent benchmark using FAMA).

### Updated comparative targets

The proposed MEME 90% Cascade/Absence target remains an aspirational stretch, but should not be the only milestone. N71 publishes a vendor-reported full 100-episode MEME run with iteration-3 Cascade 0.628 and Absence 0.42. A credible near-term bar is to reproduce those two scores under the official protocol, then exceed them with source-linked dependencies and typed absence; the longer-term stretch remains ≥0.90 on both.

For AML, the first milestone is protocol compliance and a clean local evaluation, followed by an Add/Search service implementation. Only submit to a future open cycle after:
- Add/Search API version and security review;
- equal benchmark settings and all required tasks;
- full per-task score and resource disclosure;
- governance/privacy controls that pass separately from the quality score.

For MEMORA-Bench, first target faithful retrospective retrieval and evidence grounding. Prospective planning is a later integration test involving perception, memory and policy; it must not be credited to HME alone.

## 13. Canonical f32 scorer regression and qualification status

A cancellation-sensitive regression test is now authored in the canonical scorer module and a paired candidate-interval test is authored in the HME interval module. They distinguish a separately rounded product followed by addition from a fused multiply-add result. The exact primitive arithmetic contract is documented in Rust's f32/f64 documentation; release target/IR/assembly and FTZ/DAZ qualification remain open under #11176.

Repository build metadata inspected on the research branch:
- Rust toolchain pinned to 1.96.0;
- test profile: opt-level 1, codegen-units 16;
- release profile: opt-level 3, thin LTO, codegen-units 1;
- Nix flake native shells export target-cpu=native in selected shells.

The actual effective compiler flags and runtime floating-point mode still need to be captured from real invocations. A target-cpu setting does not by itself prove fast-math or FMA contraction, and a flake comment is not a substitute for an executed compiler command line. No Rust build/test run is claimed.

### Updated references

- Agent Memory Leaderboard evaluation and protocol: https://agentmemories.ai/evaluation
- AML Cycle 1 official results: https://agentmemories.ai/results/cycle-1/
- AML documentation and Add/Search evaluation flow: https://agentmemories.ai/docs
- MEMORA-Bench, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.1211/
- MEME official project and benchmark results: https://seokwonjung-jay.github.io/meme-eval/
- Rust f32 precision contract: https://doc.rust-lang.org/core/primitive.f32.html
- Rust f32 square root: https://doc.rust-lang.org/std/primitive.f32.html
- Rust f64 square root: https://doc.rust-lang.org/std/primitive.f64.html
- LLVM floating-point contraction semantics: https://llvm.org/docs/LangRef.html#fast-math-flags
