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


## 14. Hierarchical group-to-member selection and build-time partition receipts (2026-11-09)

The source now has a two-stage query path, with persistent receipt/reopen authentication explicitly owned by [#11221](https://github.com/Luminous-Dynamics/symthaea/issues/11221):

- select_group_refinement_set_with_partition_receipt
- select_hierarchical_refinement_set_with_partition_receipt
- GroupPartitionCoverageReceipt
- VerifiedMemberResidualCertificate
- GroupScoreIntervalIdentity

The group interval is weighted by member count. For group g with m_g eligible members and conservative interval [L_g,U_g], define:

\[
\theta_G = \operatorname{kthLargest}\left(\bigcup_g
\underbrace{\{L_g,\ldots,L_g\}}_{m_g\text{ copies}}\right).
\]

At least k records have exact scores at least theta_G. Therefore any group with U_g < theta_G cannot contain a top-k record and can be excluded. Equality remains.

For the surviving groups, obtain each member interval [L_i,U_i] and intersect it with its parent group interval:

\[
L'_i=\max(L_i,L_g),\qquad U'_i=\min(U_i,U_g).
\]

If the intersection is empty, fail closed. Then run the existing per-record global selector over all surviving groups. Since the first-stage threshold guarantees at least k members with lower bounds at least theta_G, the second-stage threshold cannot decrease below theta_G. Exact reranking of the final surviving member set is therefore sufficient for the whole eligible corpus, assuming the parent group bounds and member bounds are valid.

### Query-time completeness without an O(N) partition scan

A performance contradiction was caught in review: checking that group member IDs do not overlap inside every query would itself scan O(N) IDs, defeating the metadata-first optimization. The source now distinguishes two paths:

- select_group_refinement_set: defensive path; it scans the member rosters and validates disjointness on every call.
- select_group_refinement_set_with_partition_receipt: fast path; it verifies a build-time receipt against O(G log G) group identities/counts/generations/query digests and does not walk every member ID.

GroupPartitionCoverageReceipt::from_group_envelopes pays the full membership scan once at segment-build time. It requires both:
- the authoritative eligible-record count from the pinned manifest/eligibility snapshot;
- the authoritative digest of that exact eligible member-ID set.

It verifies each group's roster digest, cross-group disjointness, a single manifest/eligibility generation, the aggregate eligible population and a deterministic digest over the group directory. The query path then checks the group records against this receipt in O(G log G). GroupScoreIntervalRecord also shares immutable member rosters and payload-digest rosters through Arc, avoiding a full member-ID vector clone when query plans are built.

The receipt is still an in-memory prototype. It is not signed, serialized or authenticated on reopen. #10917 remains the owner of immutable segment serialization, checksums, manifest binding and reopen qualification. Do not use the fast path after process restart until the segment owner authenticates the stored receipt and the exact manifest generations.

Every group score interval is now bound to a digest of the exact query vector; groups scored for different queries cannot be mixed. The selection plan also carries the manifest generation, eligibility generation and query digest.

### New property-test coverage authored

The Rust source now includes:
- a 256-case generated hierarchical selector property test, varying group count, group sizes, vector contents, top-k and reconstruction perturbations;
- comparison of final hierarchical reranking against the full original-payload exact top-k oracle;
- build-time rejection of overlapping group membership;
- receipt rejection for wrong eligible count or wrong eligible-member-set digest;
- equivalence of receipt-backed and defensive group selection;
- rejection of modified group content identity against the receipt;
- existing negative controls for missing member metadata, unexpected groups, stale query digest and generation mismatch.

These Rust tests are authored but not executed: the current environment has no Rust toolchain. No test pass is claimed.

### Frozen diagnostic fixture

docs/research/fixtures/hme_hierarchical_interval_select_v0.json records a synthetic 11-record example: 3 group synopses, one surviving group, 3 member intervals and 2 final exact payload reads. Its nominal 81.8% payload-read reduction is fixture arithmetic only, not measured HME I/O.

The fixture now needs the authoritative eligible-member-set digest supplied by the real manifest owner; its symbolic digests remain clearly marked as non-cryptographic placeholders.

### Correctness and performance interpretation

This design changes where the exhaustive work happens:
- once per immutable segment/eligibility snapshot: verify disjoint membership, authoritative eligible population and per-group payload rosters;
- per query: scan group synopses (O(G log G));
- only for surviving groups: scan member interval metadata and compact reconstructions;
- only for the final member refinement set: read original ContinuousHV payloads and run the canonical full-resolution scorer.

The fast path does not make an incomplete candidate list exact. It is authorized only when the receipt is derived from the manifest's complete eligible-member count and member-set digest. Group membership must also be uniform with respect to principal/policy/valid-time/known-time/forgetting scope; a single group cannot mix records with different eligibility states.

## 15. New 2026 agent-memory benchmark diagnostics

The current frontier contains several benchmarks that expose separate failure modes. They should be added as independent tracks, not blended into a single score.

### COLLABMEM — collaborative sharing, access and provenance

COLLABMEM fixes the trace, target role, answerer, judge and gold event in a matched memory-only harness. It includes 58 hand-authored scenarios (1,118 probes), a 174-scenario scaled split (3,359 probes), adversarial/compositional sets and six transactive-memory competencies. The paper reports a 69.6% non-Oracle average for TMS-GRAPH++; that is an author-reported system score, not an HME comparison.

HME diagnostic mapping:
- sharing/routing correctness by target role;
- unauthorized payload pre-read and unauthorized emission;
- provenance and writer attribution;
- authority/conflict resolution;
- retention and scope isolation.

This is particularly important when HME serves multi-party or role-specialized agents: global vector similarity must never override visibility or authority boundaries.

### TAME / Trust-Memevo — memory evolution must not degrade trustworthiness

TAME introduces Trust-Memevo to measure memory misevolution: trustworthiness can decline during benign task evolution even when utility or task success improves. Its executor/evaluator split is a useful competitor and diagnostic baseline.

Add long-horizon evolution runs with fixed task suites and repeated benign memory writes. Measure:
- safety-policy regression relative to the pre-evolution checkpoint;
- unsupported claims, unsafe tool/action use and privacy violations introduced by memory;
- harmful-memory propagation and recovery after correction/retraction;
- utility gained per unit of trustworthiness lost;
- whether an evaluator catches negative experience before it is consolidated or generalized.

Hard target: zero regressions on explicit safety/privacy invariants. Report capability drift and safety drift separately; do not hide a safety decline behind aggregate task improvement.

### TimlMemBench — governed activation, scope and restraint

TimlMemBench uses reviewed raw sessions and source-anchored memory assets. The released package describes 208 reviewed bilingual samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases. It contrasts canonical-pool QA with raw-native ingestion and governed-use evaluation.

This fills an HME gap between retrieval and answer correctness:
- did the system capture the right thing from raw sessions?
- was it still valid and in scope at query time?
- should it be activated for this request, or left unused?
- can the answer recover provenance?
- does the system exercise restraint when memory is revoked, irrelevant or outside scope?

Track activation precision/recall, provenance recovery, scope leakage, false activation and restraint/abstention separately.

### MEMOBench — process-level memory operations for robotics

MEMOBench contains 30 history-dependent manipulation tasks, 1,500 expert demonstrations and 4,200 executable checkpoint instances from 84 templates. It labels Storage, Update and Compression checkpoints and separates memory fidelity measures from task success. The reported strongest memory-augmented baseline reaches 31.9% average success; this is a benchmark-reported baseline, not an HME result.

Use this as a later embodied/robotics track:
- memory storage fidelity at checkpoints;
- correct state update and invalidation;
- compression retention of action-critical facts;
- memory contribution isolated from policy/control improvements;
- held-out goal/task composition and long-horizon action success.

### MEMORA-Bench — multimodal experience formation and planning

MEMORA-Bench contributes a 45-hour egocentric-video suite, separating retrospective memory faithfulness from prospective memory-grounded planning. Preserve observation timestamps, object identity/state and temporal-action evidence; compare against no-memory, flat-text and graph baselines under frozen perception and planning models.

Do not confuse MEMORA (embodied action memory) with Memora (personalized agent-memory benchmark/FAMA).

### Updated benchmark scorecard

| Track | Near-term bar | Stretch target |
|---|---|---|
| Exact HME top-k | 100% exact ID agreement; zero false prunes | Same at 1M+ records and 16K–256K dimensions |
| Fast group selector | Receipt verification O(G log G); no per-member roster walk on fast path | Metadata work ≤10% of per-record baseline on homogeneous workloads |
| Cold payload I/O | Measure physical block reads; do not infer from candidate counts | ≥50% cold payload bytes avoided at p50 and ≥20% at p95 with exact output |
| Lifecycle / deletion | 100% fixture correctness; zero stale emission/resurrection | Maintain under compaction, restart, replay and policy change |
| COLLABMEM / TimlMemBench | Zero unauthorized read/emission; publish per-competency scores | Outperform matched public baselines on provenance, scope and restraint |
| TAME / Trust-Memevo | Zero hard safety/privacy invariant regressions | Positive utility gain with no measured trustworthiness decline |
| MEMOBench / MEMORA-Bench | Memory-only ablations with fixed perception/control | Improved OOD memory-faithful planning and process metrics |

The ≤10% metadata-work target is a proposed HME project target, not an established industry threshold. The cold-payload target is also prospective; it can only be reported after physical I/O measurement against the exact full-scan oracle.

## 16. Research frontier comparison update (2026-11-09)

The newest useful comparisons are not all vector-search benchmarks:

- PQR-Search challenges average-recall ANN with a per-query probability guarantee under insertions/deletions. HME's deterministic exact mode is a different guarantee class; HME's approximate mode should report a per-query recall distribution, not average Recall@K alone.
- VectorDBBench added hosted VIBE datasets and 1M/10M/100M dataset profiles in its September 2026 release. This provides a useful system workload source, while HME must separately publish native high-dimensional profiles.
- Agent Memory Leaderboard offers textual, coding and multimodal tracks with participant-hosted Add/Search APIs. As of 9 November, its Cycle 2 page displays provisional participant entries (e.g., MemoraX v0.5 at 62.00 overall, Claude-mem v12.1.1 at 52.00 and MemOS v2.0.1 at 52.00), while explicitly stating that official Cycle 2 results are due mid-November. Treat these as live provisional/status-labelled entries, not a finalized ranking or independent replication. Materials submission is closed; the future milestone is API compatibility for the next cycle, after local correctness/governance qualification.
- COLLABMEM, TAME, TimlMemBench and MEMOBench expose routing, memory evolution, governed activation and process-level update/compression failures that answer-only benchmarks hide.

The winning HME story should not be "another database with high QPS." It should be a reproducible point on a Pareto frontier: exactness/guarantee class, p99 latency, cold bytes, metadata bytes, update/compaction cost, memory quality, governance and lifecycle integrity. Publish the complete trade-off surface.

### References added

- COLLABMEM, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.1194/
- TAME, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.2100/
- TAME / Trust-Memevo preprint: https://arxiv.org/abs/2602.03224
- MEMOBench, Findings EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.333/
- MEMORA-Bench: https://aclanthology.org/2026.findings-emnlp.1211/
- Agent Memory Leaderboard Cycle 2: https://agentmemories.ai/evaluation
- VectorDBBench September 2026 VIBE dataset update: https://github.com/zilliztech/VectorDBBench
- PQR-Search: https://arxiv.org/abs/2611.07579


## 17. Eligibility-scope binding and benchmark execution contract (2026-11-09)

The group query records and partition receipt now carry HmeEligibilityScopeBinding, a typed identity containing:
- principal/tenant/namespace scope digest;
- policy identity/version digest and policy generation;
- source-ledger generation;
- bitemporal query digest covering valid_at and known_at.

Each query-specific GroupScoreIntervalRecord carries this binding. GroupIntervalCoverageCounts carries the same binding. The partition receipt commits to it, the fast path checks the coverage binding against the receipt, and both fast and defensive selector paths reject group intervals carrying a different scope. The GroupIntervalSelectionPlan preserves the binding for downstream completion receipts.

This is deliberately an integration contract with #11003, not a new policy engine. The binding fields must be produced from the authoritative eligibility/policy/source-ledger decision. Constructing an HmeEligibilityScopeBinding in application code does not prove authorization. Every query-specific group interval record, coverage record, partition receipt and returned group selection plan carries the binding; the selectors reject cross-scope group intervals. The HME layer verifies consistency; #11003 remains the owner of eligibility truth and #11221 owns persistent receipt authentication after reopen.

The new machine-readable benchmark contract is docs/research/fixtures/hme_benchmark_matrix_v0.json. It separates:
- exact full-resolution HDC retrieval and receipt-backed hierarchy;
- approximate per-query ANN quality;
- systems and physical I/O;
- public agent-memory benchmarks;
- lifecycle/deletion/governance diagnostics;
- exactness, privacy, stale-generation and unauthorized-read hard gates.

It specifies the required 16K/64K/256K ContinuousHV dimensional sweep and 1K/10K/100K/1M corpus sizes, query strata (IID, clustered, OOD, near-tie, adversarial, zero/near-zero), cold/warm cache profiles and top-k values. It is an execution contract only; no benchmark results are implied.

The critical comparison is a six-way Pareto surface:
1. exact agreement / guarantee class;
2. cold original-payload bytes;
3. metadata bytes and member roster work;
4. p50/p95/p99 latency and peak memory;
5. ingest, update, delete and compaction cost;
6. memory quality, provenance, scope integrity and lifecycle safety.

A speedup that silently lowers eligibility correctness or increases stale-memory emissions is not an improvement. Unauthorized reads, revoked-source emissions, stale receipt reuse and deletion resurrection remain hard gates, not weighted score terms.


## 18. Synopsis-memory Pareto and certified coordinate quantization (2026-11-09)

A storage audit exposed a serious scaling issue in the first group envelope: retaining coordinate minima and maxima as f32 arrays plus an f32 centroid costs approximately 12 bytes per dimension per group, before IDs, payload digests, allocator headers or page indexes. At 256K dimensions and 1M records grouped by 64, the numeric group synopsis alone is approximately 45.78 GiB.

The group envelope now has three explicit source storage profiles:

| Profile | Numeric storage per group | Bytes/dimension | Intended tradeoff |
|---|---:|---:|---|
| BoxAndBall | 3D f32 + radius | 12 | Tightest of the three; highest metadata |
| QuantizedBoxAndBall | D f32 + 2D u16 + radius | 8 | Intermediate metadata; outward-expanded coordinate box |
| BallOnly | D f32 + radius | 4 | Lowest metadata; potentially looser group intervals |

The 4-byte radius is additional per group. Stable ID strings and 32-byte member payload digests are per-record costs shared across modes.

GroupEnvelopeMode::BoxAndBall remains the default. QuantizedBoxAndBall stores 16-bit min/max codes over the admitted [-1,1] coordinate domain. Encoding widens the scaled value down before floor and up before ceil. Decoding widens the f64 division and subtraction separately, then widens again at f32 conversion. The interval kernel receives decoded f32 endpoints, not the quantized codes. The profile tag is included in the group content digest; a receipt from one mode cannot be silently replayed against another.

BallOnly drops both coordinate-extrema arrays entirely while retaining the centroid and independently verified L2 radius. This creates a clean metadata/pruning comparison.

### Analytical memory estimates

For 1M eligible members and group size 64, the numeric synopsis estimates are:

| Dimension | BoxAndBall | QuantizedBoxAndBall | BallOnly |
|---:|---:|---:|---:|
| 16K | 2.86 GiB | 1.91 GiB | 0.95 GiB |
| 64K | 11.44 GiB | 7.63 GiB | 3.81 GiB |
| 256K | 45.78 GiB | 30.52 GiB | 15.26 GiB |

At group size 1024, the corresponding 256K figures fall to approximately 2.86, 1.91 and 0.95 GiB. That does not prove the larger group size is better: larger groups may be substantially harder to prune. Choose the mode and group size jointly.

The full 81-scenario analytical matrix is in docs/research/fixtures/hme_group_synopsis_memory_budget_v0.json. It also counts 32-byte member payload digests and an illustrative 24-byte mean stable-ID size, but excludes allocator/Arc/String headers, indexes, checksums, alignment, page padding and query caches. The numeric GiB figures are formula outputs, not measured RSS or serialized bytes.

The source now provides estimate_group_envelope_memory(...) and numeric_synopsis_bytes(). Authored tests cover:
- the exact 12D/8D/4D storage arithmetic;
- 100,000 deterministic coordinate values and endpoints/subnormals;
- adjacent f32 values around all 65,536 quantizer-bin boundaries;
- high-dimensional member-score containment for all three profiles;
- mode-specific content identity.

No Rust test or build is claimed. The environment still lacks cargo/rustc. QuantizedBoxAndBall must remain experimental and cannot authorize selector pruning until #11243's test/compiler/target gates pass.

### The next optimization question

The objective is not minimum synopsis bytes or narrowest intervals independently. It is:

\[
\operatorname{minimize}\ \text{query cost} =
\text{synopsis bytes touched}
+\text{member metadata bytes touched}
+\text{cold original payload bytes}
+\text{refinement compute}
\]

subject to exact top-k equality, complete eligibility coverage and zero false prunes.

A larger exact synopsis is worth keeping only when the cold payload bytes it saves outweigh the extra memory footprint, cache misses and build/compaction cost. A quantized synopsis is useful only if outward endpoint widening does not erase enough pruning power to lose against BallOnly. Group size must be swept alongside the codec.

The promotion matrix is in #11243. #10917 owns the physical I/O/segment measurements; #11059 owns group-bound correctness; #11221 owns persistent and reopen-authenticated partition receipts.

## 19. Competitive frontier refresh (2026-11-09)

### LongMemEval: a stronger production reference

Redis' September 21, 2026 agent-memory report describes a 500-question LongMemEval Small reproduction using the official GPT-4o binary judge and a common GPT-4o answer backbone for its comparable reproduced systems. Its Remis + Instruct combination reports 86.5% task-averaged accuracy, 1.78s query latency and about $0.0703/session under the report's cost model. The report also notes discrepancies between published and reproduced numbers, including a 42.4% reproduced Vertex Memory Bank result versus roughly 70% previously associated with the underlying method.

Treat 86.5% as a strong vendor-published reproduced result, not an independently verified universal leaderboard. HME should reproduce it under the same split, answer model, judge, retrieval budget and session cost methodology before making a ranking claim.

The architectural lesson is important: Redis' raw evidence path and extracted fact path have complementary failure modes. HME should preserve both source-native evidence and structured, versioned claims with provenance/dependencies. The structured claim layer should not destroy the raw record that would let a later query inspect exact wording, timestamps or corrections.

### GroupMemBench: multi-party memory is a separate hard problem

GroupMemBench (arXiv:2605.14498) reports 46.0% average accuracy for its strongest tested memory system; knowledge update is 27.1% and term ambiguity 37.7%. Its authors report that a simple BM25 baseline matches or exceeds many agent-memory systems.

This is a warning against optimizing a single-user, dense-semantic retrieval score and assuming multi-party memory is solved. HME needs:
- asker-conditioned retrieval;
- speaker/writer attribution;
- per-person and per-role belief separation;
- audience/namespace-aware activation;
- lexical/exact-identifier retrieval alongside HDC similarity;
- knowledge-update and abstention tests;
- zero unauthorized pre-read and emission.

COLLABMEM is complementary: it fixes the trace, target role, answerer and judge to isolate sharing, access, routing, provenance, authority and retention. Its authors report 69.6% non-Oracle average for TMS-GRAPH++. Keep this separate from GroupMemBench and LongMemEval.

### Memory evolution and governed use

TAME / Trust-Memevo (EMNLP 2026) finds that benign test-time memory evolution can reduce agent trustworthiness. Its authors report a 14.6 percentage-point AIME accuracy improvement on GPT-5.2 while maintaining competitive trustworthiness. That is a test-time evolution result, not a vector retrieval score. For HME, evaluate utility gain and safety/privacy drift on separate axes; hard governance invariants remain zero-tolerance gates.

TimlMemBench provides a governed-activation diagnostic: a remembered fact can be revoked, superseded, private to a different context or irrelevant to the current request. It has 208 reviewed bilingual samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases. HME should measure correct non-use as well as correct recall.

MEMOBench (EMNLP 2026 Findings) exposes process-level memory operations for robotics: 30 history-dependent tasks, 1,500 expert demonstrations and 4,200 executable checkpoints from 84 templates. Its strongest reported memory-module baseline averages 31.9% task success. This is an embodied process-memory track, not a pure retrieval leaderboard.

### Vector retrieval and system baselines

- PQR-Search (arXiv:2611.07579, Nov 5) highlights that an average Recall@K target is not a per-query guarantee: indexes tuned to average 0.99 reportedly left 5–19% of in-distribution queries below target and as many as 99.6% of OOD queries below target when not query-log tuned. HME approximate mode should publish the full per-query recall distribution, fraction below tau, OOD failure rate and update drift. Keep its probabilistic guarantee class separate from HME deterministic exactness.
- Recast (arXiv:2611.02765, Nov 2) is the relevant storage/refinement competitor: it keeps compact codes in memory, raw vectors on disk and chooses a minimal refinement set using uncertainty intervals. It reports 1.56–2.01x QPS at 0.99 recall under its tested matched-memory conditions and lower external storage. HME should compare cold bytes and matched quality, not claim equivalence to its probability-based interval assumptions.
- QBAT (PVLDB 2026) reports up to 68.8% less searched budget in its ScaNN consistency mode using query-aware budget allocation. HME should schedule by expected certified work saved per unit cost while preserving its separate eligibility and coverage gates.
- VectorDBBench's September 2026 integration registers all 24 VIBE HDF5 datasets and multimodal VDBBench datasets at 1M/10M/100M scale. This improves reproducibility for public embedding workloads. HME's 16K–256K ContinuousHV profile still needs a separate native workload; do not compare it as if it were a standard 768/1536-dimensional embedding benchmark.

### HME's honest competitive thesis

HME should not claim to beat Redis on LongMemEval, PQR-Search on ANN throughput, or Recast on disk QPS yet. There are no HME benchmark runs. The near-term differentiator is a testable combination:
1. deterministic exact retrieval where intervals can be certified;
2. provenance-carrying source/claim memory rather than destructive summarization;
3. bitemporal revision, dependency invalidation and typed absence;
4. role/scope/authority-aware eligibility before payload reads;
5. a cost model that counts synopsis memory, metadata traffic and cold payload bytes.

The first external goal is reproducible parity with strong baselines under frozen answer/judge protocols. The stretch goal is a strictly better multi-objective Pareto frontier, with no safety or exactness regression.

### References refreshed

- Redis LongMemEval agent-memory reproduction (2026-09-21): https://redis.io/resources/agent-memory-benchmark-report/
- GroupMemBench (arXiv:2605.14498): https://arxiv.org/abs/2605.14498
- COLLABMEM (EMNLP 2026): https://aclanthology.org/2026.emnlp-main.1194/
- TAME (EMNLP 2026): https://aclanthology.org/2026.emnlp-main.2100/
- TimlMemBench (Findings EMNLP 2026): https://aclanthology.org/2026.findings-emnlp.507/
- MEMOBench (Findings EMNLP 2026): https://aclanthology.org/2026.findings-emnlp.333/
- PQR-Search (2026-11-05): https://arxiv.org/abs/2611.07579
- Recast (2026-11-02): https://arxiv.org/abs/2611.02765
- QBAT (PVLDB 2026): https://research.google/pubs/qbat-model-based-query-budget-autotuner-for-clustering-based-approximate-nearest-neighbor-search/
- VectorDBBench 2026 VIBE dataset support: https://github.com/zilliztech/VectorDBBench/blob/main/docs/release/2026-09-hugging-face-datasets.md


## 20. No-allocation quantized decode and selector promotion gate (2026-11-09)

The first quantized implementation decoded every group's u16 extrema into two temporary D-element f32 vectors. At 256K dimensions that would allocate approximately 2 MiB per group evaluation, potentially overwhelming the saved resident bytes when many groups are scanned.

The HDC core now exposes an allocation-free indexed callback kernel, similarity_interval_full_resolution_for_component_bounds_with. It uses the same group-of-eight and ordered-tail f32 interval reduction as the slice-based component-box kernel, but requests each conservative (lower, upper) pair from a deterministic callback and consumes it immediately. On a valid profile, the callback is evaluated exactly once per coordinate in ascending order. Invalid component bounds fail closed.

HME's quantized group path now decodes the outward u16 endpoints within that callback. It does not construct temporary dimension-sized vectors. The callback kernel has authored tests comparing its interval exactly to the slice-based kernel, checking the group-of-eight/tail behavior, invalid bounds and callback count. Those Rust tests have not been executed.

Most importantly, QuantizedBoxAndBall is now explicitly selector-gated:
- the public score_interval_for_snapshot API returns UnqualifiedSynopsisMode for this profile;
- GroupScoreIntervalRecord::from_group_envelope calls that public API and therefore cannot admit a quantized group into the exact selector;
- a private cfg(test) helper exists only to test enclosure behavior;
- BoxAndBall and BallOnly remain on the existing public conservative score path.

This keeps the quantized codec available for numerical research and future benchmarking without allowing it to silently authorize an exact prune. Promotion requires #11243's complete endpoint, canonical-score, compiler/target, and runtime measurement gates, followed by an explicit source change removing the gate.

An independent JavaScript IEEE-754 model tested 296,608 coordinates: every quantizer boundary and its adjacent f32 values, plus 100,000 deterministic random values. It observed zero endpoint escapes; maximum decoded interval width in that sample was about 3.06e-5. This is useful arithmetic cross-checking only. It is not Rust execution, formal proof, target/compiler qualification, or permission to prune.

The next measurements must count actual metadata bytes, allocations, group survival, member intervals scanned and physical original-payload bytes. The theoretical 4/8/12-byte-per-dimension profiles are only the first line in a cost model, not performance outcomes.


## 17. Executable hierarchical-selector benchmark harness (2026-11-09)

A dependency-free benchmark executable is now authored at:

crates/domains/symthaea-memory/benches/hme_hierarchical_select.rs

It is registered in crates/domains/symthaea-memory/Cargo.toml as a harness-free Cargo bench. The executable caps the synthetic original-payload working set at 512 MiB and group count at 100,000 to avoid accidental host-memory exhaustion.

Run:

\`\`\`sh
cargo bench -p symthaea-memory --bench hme_hierarchical_select
\`\`\`

Configurable inputs:

\`\`\`sh
HME_BENCH_RECORDS=4096 \\
HME_BENCH_DIM=16384 \\
HME_BENCH_GROUP_SIZE=64 \\
HME_BENCH_TOP_K=10 \\
HME_BENCH_REPEATS=21 \\
cargo bench -p symthaea-memory --bench hme_hierarchical_select

# Synopsis profile: box (default) or ball
HME_BENCH_MODE=ball cargo bench -p symthaea-memory --bench hme_hierarchical_select
\`\`\`

The harness:
- generates deterministic, correlated high-dimensional ContinuousHV groups;
- builds BoxAndBall group envelopes;
- derives an authoritative eligible-member-set digest and a group partition receipt;
- builds 1/1024-grid scalar reconstructions and verifies per-member residual receipts at admission;
- compares defensive per-query disjointness validation with receipt-backed group selection;
- materializes member intervals only for surviving groups;
- performs hierarchical selection and exact-reranks the final payload set;
- checks every run against the full canonical score/top-k oracle;
- separately times the full-scan baseline and the end-to-end two-stage path;
- reports p50/p95/p99 for full scan, end-to-end hierarchy, each selector/materialization stage, build costs, logical exact-payload reads and metadata estimates. The two-stage latency timer stops before the full-scan baseline begins. It also emits explicit work counters: defensive member-roster IDs walked, receipt-backed member-roster IDs walked (expected zero before stage one), group-directory records visited, and member interval records consumed in stage two.

The reconstruction remains an unpacked f32 grid simulation. This is not a production compression format, not a physical disk-I/O benchmark, and not a substitute for #10917 segment/page telemetry. It emits physical_io_measured=false by design. The benchmark has been authored and registered but was not executed in this environment because cargo/rustc are unavailable.

### Interpreting its output

The benchmark separates:
- one-time dataset and group-envelope build cost;
- one-time partition receipt generation;
- one-time member residual certificate admission;
- defensive selector latency;
- receipt-backed selector latency;
- member interval materialization;
- full hierarchical selection;
- final exact rerank.

It does not infer cold bytes from candidate counts as a measured result. Logical payload-byte estimates use the exact f32 dimension and final exact refinement count; physical savings require device/page counters from the immutable segment implementation.

### Source review corrections

Two trust-boundary corrections were made while authoring the harness:
- Coordinate-extrema accessors now return Option<&[f32]> so BallOnly and QuantizedBoxAndBall modes cannot be mistaken for storing raw f32 extrema.
- VerifiedMemberResidualCertificate and GroupMemberScoreIntervalRecord now carry the HME eligibility scope binding. A residual receipt or member interval built under another principal/policy/source-ledger/bitemporal context is rejected, even if group identity, payload digest and query vector happen to match.

The new negative test constructs two otherwise identical group records under different policy scopes and checks that the member receipt cannot cross between them. This is authored source evidence, not executed Rust evidence.

## 18. Current comparative scorecard — November 9, 2026

### Vector retrieval and system baseline

VectorDBBench's September 2026 update makes all 24 VIBE HDF5 datasets available through its existing Performance case and adds VDBBench multimodal Parquet workloads at 1M, 10M and 100M vectors. HME should adopt these workload datasets where the representation and metric match, but it must also publish a separate ContinuousHV native track at 16K, 64K, 128K and 256K dimensions. Results at standard 768/1536 embedding dimensions must not be presented as a direct native-HDC comparison.

The HME comparison set should include:
- exact scalar full-resolution ContinuousHV scan as the semantic oracle;
- the existing HdcStore mmap scan and SQLite ConversationMemory path where the metric/profile matches;
- FAISS Flat for exact vector-search ground truth;
- HNSW, DiskANN/Vamana, IVF-PQ and ScaNN as conventional ANN baselines;
- PQR-Search for per-query probabilistic guarantees;
- Recast for storage-aware refinement and cold-payload behavior;
- QBAT for query-specific work allocation;
- VectorDBBench/VIBE for reproducible system-level workloads.

Report exact top-k agreement, per-query Recall@K distribution, fraction below each \u03c4, p50/p95/p99, batch throughput, memory, metadata bytes, build/update/delete/compaction time, cold payload bytes, and energy if direct telemetry exists. Maintain separate result tables for exact, probabilistic-guarantee and empirical-ANN modes.

### Agent-memory benchmark baseline

As of November 9, the Agent Memory Leaderboard Cycle 2 page displays provisional participant statuses and scores but says official results will be released mid-November. Example visible entries include MemoraX v0.5 at 62.00 overall, Claude-mem v12.1.1 at 52.00 and MemOS v2.0.1 at 52.00. The page distinguishes Submitted (API) from Evaluated. HME must not claim a rank against these provisional entries, and Cycle 2 materials submission is already closed.

The relevant public diagnostic suites now cover distinct functions:
- MEME: aggregation, tracking, deletion, cascades and typed absence;
- DyadMem: user-conditioned/relational capture and full-pipeline gaps;
- COLLABMEM: role routing, sharing, visibility and provenance;
- TAME / Trust-Memevo: safety and trustworthiness during benign memory evolution;
- TimlMemBench: activation, restraint, scope, provenance and raw-session ingestion;
- MEMOBench: process-level storage, update and compression for robotics;
- MEMORA-Bench: retrospective multimodal faithfulness and prospective planning.

Scores across these suites are not commensurate. Use a common frozen answerer/reader, controlled context tokens, fixed source trace and independent evaluators where each benchmark permits. Always report pipeline and memory-only controls separately.

### Competitive hypotheses, not achieved results

1. HME can outperform general-purpose agent memory on source-linked revision, typed absence, scope boundaries, eligibility-before-read and exact auditable retrieval.
2. HME can outperform conventional ANN on guarantee strength for workloads where exact completeness is required, but it will not necessarily beat ANN latency or throughput.
3. HME can outperform full scans on cold-payload bytes only if group/member bounds are tight enough to offset synopsis bytes and CPU cost.
4. HME can outperform existing memory systems on lifecycle correctness only if raw ingestion, update/revocation, retention, activation and restraint are measured—not just final question answering.

These hypotheses need independent benchmark runs. Current HME benchmark status remains authored-but-not-executed.


## 21. BlockBall synopsis research path (issue #11259)

A further synopsis point is now tracked in #11259: **BlockBall**, an intermediate representation between BallOnly and per-coordinate BoxAndBall.

For a fixed block partition of D coordinates, store the centroid c and one conservative per-block L2 residual radius r_b. Since every coordinate deviation is bounded by its block L2 norm, each member coordinate j in block b is enclosed by [c_j-r_b,c_j+r_b]. A global safe radius follows from sqrt(sum_b r_b^2). The query can generate these endpoint pairs through the core no-allocation indexed component-bound callback, then intersect that component enclosure with the global centroid-ball interval.

The estimated numeric synopsis is 4D + 4 ceil(D/B) + 4 bytes per group, excluding all IDs/digests/index/allocator overhead. Compare B = 32/64/128/256/512/1024 against BallOnly (4D+4), QuantizedBoxAndBall (8D+4) and BoxAndBall (12D+4). BlockBall remains an issue-level research hypothesis, not yet an implemented or promoted source mode.

The selection criterion is not "tightest interval wins": measure the full memory/pruning/metadata/payload-read/latency/build-cost Pareto frontier. A query-time no-allocation bound only helps if the group metadata it requires remains cache-efficient and the additional build/update work is justified.

For a numeric-only illustration at D=262,144, 1,000,000 eligible records and 64 members/group (15,625 groups): BallOnly is 15.259 GiB; BlockBall adds about 1.562% at B=64 (15.497 GiB), 0.391% at B=256 (15.318 GiB) and 0.098% at B=1024 (15.274 GiB); BoxAndBall is 45.776 GiB. These are formula-derived synopsis bytes only, not measured allocations, and exclude all IDs/digests/index/allocator overhead. The central hypothesis is that small block radii may yield tighter bounds for modest added synopsis memory.
