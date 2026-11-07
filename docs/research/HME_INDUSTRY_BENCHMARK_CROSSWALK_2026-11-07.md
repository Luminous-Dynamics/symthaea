# HME industry benchmark crosswalk and research priorities

**Date:** 2026-11-07  
**Status:** Research / protocol design; no HME benchmark result is claimed  
**Parent:** [#10915 — HDC-native Holographic Memory Engine](https://github.com/Luminous-Dynamics/symthaea/issues/10915)  
**RFC:** [PR #10916](https://github.com/Luminous-Dynamics/symthaea/pull/10916)

## Executive decision

HME should compete on a narrower and more defensible proposition than "a faster vector database":

> A representation-aware memory substrate that preserves exact HDC semantics, separates semantic storage from live dynamical state, moves less data per useful retrieval, and makes memory history/recovery independently verifiable.

There is no single industry benchmark that measures all of this. Use three independent qualification layers and join them only through explicit, source-bound receipts:

1. **HDC representation and dynamics:** #10917, #10918, #7772, #6560/#6561 and #7749.
2. **Agent-memory utility and governance:** #8427, #9145, #9075, #9177 and #9139.
3. **Storage / retrieval systems:** #7922, #7934, #7955, #7864/#7866 and #10896.

A score in one layer must not be presented as a result in another. For example, a lower vector-search latency does not prove better episodic recall, better temporal reasoning, safe forgetting, or preserved liquid trajectories.

## 1. Industry comparison: use the right comparator for each claim

| Claim being tested | Appropriate comparator | What must be matched | HME's possible differentiator |
|---|---|---|---|
| Exact HDC retrieval | Canonical scalar oracle; current in-memory scan; existing HDC store | Same vectors, dimension, algebra, metric, IDs, tie policy and query set | HDC-aware exact layouts and progressive exact scan |
| On-disk approximate vector retrieval | MLPerf Storage vector-database workload; ANN-Benchmarks; compatible Qdrant / pgvector / Vespa profiles | Dataset, dimension, metric, index, recall target, query concurrency, hardware and I/O boundary | Exact HDC oracle plus explicit recall / bytes-read / recovery accounting |
| Embedded persistent storage | Existing SQLite and symthaea-hdc-store; Fjall, redb and RocksDB candidates | Same write durability, batch size, read mix, cache state, crash model and host | HDC-specific immutable segments without recreating a general DBMS |
| Long-term conversational memory | LongMemEval-S / V2, LoCoMo, MemoryAgentBench, TimlMemBench where the official release/license is available | Same sessions, reader/backbone, prompt/context budget, retrieval policy and evaluator | Temporal, source-aware, resolution-aware memory under bounded cost |
| Governed memory lifecycle | Memora/FAMA, ConsolidationBench, StateMemBench, TEM, MEME and project-owned symbolic fixtures | Exact benchmark protocol; separate lifecycle and downstream utility measures | Auditable lineage, correction/revocation propagation, reversible consolidation and safe forgetting |
| HDC capacity | #7772's controlled fan-in / interference / query-multiplicity matrix; VSA capacity literature | Same distribution, task, capacity definition and number of comparisons | Demonstrate workload-specific associative capacity rather than infer it from D |

### Storage benchmark guidance

MLPerf Storage v3.0 added a vector-database workload in 2026. Its published CLOSED configuration uses 10 million 1,536-dimensional float32 vectors (about 61 GB raw) in Milvus with DiskANN, measures sustained top-10 queries, reports QPS and p99 latency, and checks recall against exact brute-force ground truth. It is a useful on-disk ANN / random-read storage comparator, but its embedding dimension and ANN semantics are not the same as Symthaea's 16K–256K HDC profiles. HME should use its measurement discipline and only claim MLPerf compliance if it runs the exact suite and configuration. Treat results as workload-specific, not as a universal storage ranking.

For vector search, report the recall–throughput–latency–index-size frontier. Treat the original ANN-Benchmarks as a legacy reference rather than the current frontier: its upstream project now says it is no longer actively maintained and points users toward VIBE. VIBE (2026) is an algorithm-level ANN benchmark over modern embedding datasets, including in-distribution and out-of-distribution sets; it is not a vector-database/cloud-service benchmark. VectorDBBench adds database/platform cases and, in its September 2026 release, registered the VIBE datasets plus 1M/10M/100M multimodal corpora. MLPerf Storage adds an on-disk storage-path case. These answer different questions. Use the same dataset, metric, filter, recall target and hardware configuration wherever a suite permits. None replaces HME's canonical exact HDC oracle.

MLPerf Storage v3.0's published CLOSED vector-database case uses 10 million 1,536-dimensional float32 vectors in Milvus/DiskANN (~61 GB raw), measures sustained top-10 queries and reports QPS, p99 latency and recall against exact brute-force ground truth. This is a useful on-disk ANN/storage profile, not a direct semantic or dimensional match for 16K–256K HDC. Read QPS and recall together; a faster result at lower recall is not a fair speedup.

The existing symthaea-hdc-store is a critical incumbent: its format-v2 is a fixed 16,384-bit BinaryHV mmap store with redundant checksummed headers, strict HdcStoreReader, deterministic exact full-store scan, optional LSH scan with examined/total/exact/fallback diagnostics, journaled batch updates, recovery/failpoints, snapshots, compaction and portable restore. HME should delegate this representation family to that store rather than clone it. Its schema is not a generic 2K–256K ContinuousHV store; a distinct ContinuousHV segment profile remains a possible HME extension after the exact semantics are established. #7922 and #7934 own the existing store's missing layout and workload evidence.

### Agent-memory benchmark guidance

Use the official evaluators and keep each benchmark's score separate:

- **LongMemEval-S / LongMemEval-V2:** multi-session recall, temporal reasoning, updates, abstention and evolving context.
- **EverMemBench:** multi-party, multi-group conversations beyond one million tokens, with temporally changing facts and role-specific profiles; a useful stress test for cross-topic and multi-party retrieval.
- **LoCoMo:** long conversational histories and temporal, causal, commonsense and multi-hop questions.
- **MemoryAgentBench / BEAM / AMA-Bench:** broader memory-agent abilities and long-horizon behavior where their official protocols apply.
- **StateMemBench / TEM / Memora-FAMA / MEME:** current-state tracking, versioned historical facts, obsolete-memory penalties, cascade/deletion/absence behavior.
- **TimlMemBench:** retain its two modes—canonical-pool reader control and raw-native ingestion-to-answer—after confirming the official data release and license.
- **ConsolidationBench:** memory promotion/compression decisions and transfer; governance properties remain separate from utility.

Reuse #8427 for the cross-benchmark cost-quality frontier and #9145 for TimlMemBench. Reuse #9075 for valid-time / known-time behavior, #9177 for proof-aware minimal sufficient context, and #9139 for consolidation governance. Do not create duplicate adapters or benchmark runners.

A storage-only experiment cannot establish that HME improves an agent's answers. The downstream track must integrate the actual Broca reader and report both retrieval and reader results separately. Keep backbone comparisons distinct from memory-substrate comparisons.

### Published vendor numbers are reference points, not a leaderboard

Public vendor pages currently report, for example, Mem0 LongMemEval 94.4% / LoCoMo 92.5% at a top-200 retrieval budget and roughly 6.8–7.0K mean tokens, while Zep reports LongMemEval 90.2% / LoCoMo 94.7% with its own reader/judge, context sizes and latency accounting. These are vendor-published claims, not independent common-harness measurements and not directly rankable against each other or against Symthaea. Their value is to reveal the competitive range and the metrics vendors expose. HME/Broca must reproduce the same benchmark version, exact question set, reader/backbone, judge, prompt, context budget, token accounting and latency boundary before making a comparative claim. Preserve per-category scores; do not optimize only a single aggregate.

## 2. Paradigm shifts worth testing

### A. Semantic address space is not the live evolution state

Keep semantic width (D_s) and local dynamical width (D_e) independent. A high-dimensional memory address may be useful even when the continuously evolving state is much smaller. Compare full-width evolution, compact evolution, random/orthogonal/manifold controls and explicit residual/novelty pathways under #7749/#3348.

The crucial resource accounting is total work, not just state bytes: projection/lifting, residual maintenance, synchronization, cache movement and any hidden full-width recurrent update must be included. A compressed profile only counts as compressed if the full-width operation is genuinely removed from the hot path.

### B. Progressive exact search, not merely approximate nearest-neighbor search

New research child: [#10945 — HME-EXACT-PRUNE-001](https://github.com/Luminous-Dynamics/symthaea/issues/10945).

Test admissible partial-score bounds that allow a scan to stop reading a candidate only after proving it cannot enter top-k. Every candidate remains accounted for; no approximate index may omit a candidate. For packed Hamming distance, the best possible final match score is bounded by the mismatches already observed. For ContinuousHV cosine, Cauchy–Schwarz bounds the contribution of the unexamined suffix.

The difficult part is floating-point safety and metadata integrity: suffix norms/bounds must be conservative, versioned and checksummed; ambiguity must cause continued scanning or full exact fallback. Tie behavior must preserve the stable-ID ordering. If the bound is too loose or the metadata cost is too high, keep the ordinary exact scan.

### C. Memory placement should be based on reuse, deadlines and governance—not vector size alone

Classify data by hotness, expected reuse, update rate, deadline, durability, rights, temporal validity and recovery cost. Place it in live RAM, resident segment cache, mmap/NVMe or external storage accordingly. Make movement observable and reversible. This composes with #7919 rather than introducing a competing tier manager.

### D. Retrieval should return an admissible evidence packet, not just similar vectors

HDC similarity is a candidate-generation signal, not proof that an item is true, current, authorized or independently corroborated. The canonical memory object should preserve identity, source/revision lineage, valid-time and known-time, access scope, content digest and representation transform. Where the task asks for a supported claim or action, compose with #9177's proof-aware minimal sufficient context rather than treating top-k as the final product.

### E. Treat indexes and summaries as rebuildable materialized views

The canonical source/event and payload lineage remains authoritative. Indexes, compressed representations, caches and summaries are derived artifacts with exact generation and transform identities. On corruption, stale lineage, source retraction or rights change, fail closed and rebuild/invalidate according to the owning policy. Do not let an ANN hit or cached summary become the authority record.

### F. Optimize correct useful work per byte moved

A fast inner-product kernel can lose end-to-end when storage reads, decoding, index traversal, transfers, filtering, reranking or answer-context construction dominate. Measure:
- exact useful answers / query and correct answers / byte read;
- p50/p95/p99 end-to-end latency;
- bytes moved across memory / storage / device boundaries;
- total resident and peak memory including metadata and indexes;
- index build, update, compaction, recovery and rebuild costs;
- energy per correct result only when directly measured over a declared system boundary.

## 3. Frozen comparison design

### HDC and HME core

Reuse #10917 and #10918; preserve distinct representation families.

- Continuous f32: 2K, 4K, 8K, 16K, 32K, 64K, 128K and 256K when the active code and resource budget admit them.
- Packed BinaryHV: only dimensions and algebra supported by the canonical implementation.
- Keep continuous, binary/bipolar, quantized, sparse and manifold representations distinct; never compare scores across unlike metrics.
- Corpus sizes: 64, 256, 1K, 4K, 16K, 64K, 256K and 1M where the declared host can run them.
- Tasks: single exact top-1/top-k, batched queries, all-pairs/clustering, fan-in/role-filler retrieval, correlated distractors and adversarial near-duplicates.
- Dynamic state: fixed and irregular dt, long horizon, online learning, high→low→high transitions, snapshots and restore under #6560/#6561/#7749.

### Storage systems

Minimum internal baselines:
1. Contiguous exact in-memory scan.
2. Existing SQLite conversation HDC BLOB scan for its actual workload.
3. Existing symthaea-hdc-store exact reader and LSH candidate path for its supported BinaryHV workload.
4. Read-only HME segment candidate (after #10917).
5. SQLite, Fjall, redb and RocksDB only under identical persistence durability and recovery contracts.

External profiles under #10896 only where actual dimension, representation, metric and filtering semantics match. Do not compare a 256K ContinuousHV as if it were a supported single dense vector in a backend that cannot represent it.

### Agent-memory integration

At minimum, compare:
- no-memory and full-history/context controls;
- BM25/lexical;
- dense retrieval;
- hybrid retrieval with reranking;
- current Broca/HDC memory;
- recurrent-only / selective-intervention controls where the task supports them;
- HME-backed retrieval with the same reader;
- oracle-minimal packet only as a diagnostic upper bound.

Use same frozen reader/backbone, prompt policy, context tokens, calls, sessions, task split and compute budgets. Report raw-native ingestion separately from canonical-pool reader evaluation. Include hard negatives for stale facts, temporal leakage, retractions, scope violations, source poisoning, false corroboration, missing premises and unanswerable questions.

## 4. Metrics and gates

### Exactness and correctness

- 100% deterministic top-k identity/order agreement with exact full-scan oracle.
- Scores satisfy the exact oracle's declared numerical contract; do not hide ranking changes behind score tolerance.
- Zero acceptance of wrong dimension, representation, metric, stale transform or invalid segment header.
- Zero false-prune from progressive exact bounds; invalid bound metadata forces exact fallback/error.
- Zero silent state resets across any admitted resolution transition.
- 100% replay/rebuild parity on deterministic fixtures.
- Zero unauthorized or revoked payload emissions in the deterministic lifecycle controls.

### Approximate retrieval

Report Recall@k, precision, candidate count, false-match risk at actual query multiplicity, calibration, score margins, and exact-fallback rate. Separate one-query O(N) and all-pairs O(N²) risk. Do not call sampled stride exact.

### System performance

Report p50/p95/p99; throughput; bytes read and moved; page residency/fault deltas where measurable; CPU/SIMD utilization; peak RSS; metadata/index overhead; allocation and rebuild cost; durable update and recovery latency. Keep cold process start distinct from truly cold media. Do not infer physical device I/O or energy from logical byte counts.

### Proposed promotion gates (not achieved results)

- Exactness and recovery invariants are hard gates; no performance gain compensates for a violation.
- Candidate storage/layout promotion requires a reproducible Pareto gain on a named workload and equivalent durability semantics.
- Existing project stretch target: at equal task quality, either ≥1.5× end-to-end useful retrieval throughput or ≥30% lower directly measured joules per correct retrieval, while p95 is no more than 10% worse for that declared workload.
- Agent-memory efficiency candidate: maintain task quality within a preregistered 1 percentage-point non-inferiority margin while reducing total steady-state plus background lifecycle cost by ≥25%; alternate quality gate: ≥3 percentage-point paired improvement at equal budget. Reuse #8427; these are project decision rules, not industry standards.
- No universal "256K is best" rule. Select the minimum sufficient semantic/evolution width on the task-quality, false-match, trajectory, latency and memory Pareto frontier.

## 5. External research anchors and how to interpret them

- **VIBE (2026):** modern embedding ANN algorithm benchmark with in-distribution and OOD datasets, quantized 8-bit/binary representations, CPU/GPU and HPC profiles. Its maintainers explicitly distinguish algorithm benchmarking from vector database/cloud service benchmarking. https://vector-index-bench.github.io/ https://github.com/vector-index-bench/vibe
- **VectorDBBench (2026):** vector database benchmark with current registered VIBE corpora and VDBBench multimodal corpora at 1M/10M/100M vectors, plus cloud-oriented insert, payload, multi-tenant and cold-latency cases. Treat its sponsor/leaderboard context and case-specific scoring transparently; use per-case raw metrics rather than a single composite. https://github.com/zilliztech/VectorDBBench
- **MLPerf Storage v3.0:** official MLCommons on-disk vector-database workload. Use as an external storage-path benchmark, but do not claim conformance without running its prescribed workload and harness. https://mlcommons.org/2026/09/mlperf-storage-v3-0-results/
- **ANN-Benchmarks:** useful historical reference for recall-versus-QPS/index-size curves, but the upstream project now states it is no longer actively maintained; use VIBE for the current algorithm-level comparison where appropriate. https://ann-benchmarks.com/
- **NVMe Computational Programs / Subsystem Local Memory:** ratified command sets describe device-side program execution and subsystem-local read/write/copy primitives. These are future hardware interfaces to investigate only when an identified device, firmware, driver and host stack actually support them. Standards publication is not hardware availability or proof of acceleration. https://nvmexpress.org/specification/computational-programs-command-set/ https://nvmexpress.org/specification/subsystem-local-memory-command-set/
- **LongMemEval:** the original 500-question benchmark tests multi-session memory abilities including information extraction, temporal reasoning, updates and abstention. It is useful for downstream agent memory, not raw HDC-store throughput. https://arxiv.org/abs/2410.10813
- **LongMemEval-V2:** a May 2026 extension contains 451 questions on static state recall, dynamic state tracking, workflow knowledge, environment gotchas and premise awareness; its histories can reach 500 trajectories and 115M tokens. Its paper reports 72.5% for AgentRunbook-C versus 48.5% for its strongest RAG baseline in that paper's setup. These are paper-specific results, not an HME-comparable score until the same data, reader, budget and evaluator are run. https://arxiv.org/abs/2605.12493
- **EverMemBench:** a 2026 benchmark with 1,000+ QA pairs over million-token, multi-party / multi-group conversation histories; its paper highlights unresolved multi-hop and temporal-versioning problems. Use as a stress test after confirming the official release and evaluator. https://arxiv.org/abs/2602.01313
- **LoCoMo:** a long-conversation benchmark useful for multi-session, temporal and multi-hop conversational recall. Keep its data/evaluation distinct from LongMemEval. https://aclanthology.org/2024.acl-long.747/
- **Capacity Analysis of Vector Symbolic Architectures (IBM Research / JAIR, 2026):** useful theoretical comparator for capacity under specific VSA operators and tasks. Its capacity definition is not automatically equivalent to Symthaea's fan-in, query-scale false-match or dynamic-state capacity. https://research.ibm.com/publications/capacity-analysis-of-vector-symbolic-architectures
- **Memristive HDC language-identification systems:** hardware-algorithm co-design demonstrates that locality and representation matter. It does not establish 256K liquid-state equivalence or an HME software speedup. https://www.nature.com/articles/s41467-026-76067-5

## 6. Execution order (no CI dependency)

1. **Close semantic ambiguity first:** #10918 owns the exact scalar reference and explicit stride-sampled semantics. Audit callers; preserve legacy behavior until safe migrations are justified.
2. **Establish a read-only exact HME baseline:** implement #10917 within the existing symthaea-memory boundary unless source ownership review proves another crate is canonical. No writes, WAL, compaction or remote dependencies in this slice.
3. **Run layout and query experiments:** reuse #7955/#7934/#7922 and #7864/#7866. Compare the full exact scan against the segment reader using identical data, IDs and result semantics.
4. **Prototype progressive exact retrieval:** #10945 may be developed as an isolated algorithm/fixture after the canonical score contract is stable. It cannot replace the full scan until all false-prune controls pass.
5. **Only then qualify durability candidates:** compare SQLite/Fjall/redb/RocksDB against explicit acknowledged-write, crash, restore and compaction contracts. Prefer the simplest backend that clears the gate.
6. **Integrate dynamics:** consume #7749/#6560/#6561 for semantic/evolution width and state continuity. No HME-specific duplicate resize algorithm.
7. **Prove downstream utility:** integrate the HME retrieval adapter into the existing Broca evaluation crosswalk (#8427/#9145/#9075/#9177/#9139) using official datasets/evaluators and matched baselines.
8. **Hardware last:** explore NVMe local compute, CXL or FPGA only against an identified physical device and measured end-to-end workload; keep simulated, synthesized and measured evidence distinct.

## 7. Claim boundary

This document is a research plan and benchmark crosswalk. It reports no new HME implementation, no Rust compilation/test result, no storage-engine bake-off, no physical I/O/energy measurement, no MLPerf compliance and no SOTA result. The purpose is to make future comparisons fair, reproducible and difficult to overclaim.
