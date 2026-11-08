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
