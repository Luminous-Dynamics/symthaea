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


### Agent Memory Leaderboard: a practical external integration target

The [Agent Memory Leaderboard (AML)](https://agentmemories.ai/) runs separate Textual, Coding Agent and Multimodal Memory tracks. It requires participant-hosted, fixed-version Add/Search APIs; a repository or Docker image alone is not accepted as a substitute for a working endpoint. Search returns memory evidence rather than the final answer, while the platform runs Answer/Eval. It also requires sample/user isolation, source attribution and evaluation-version traceability.

**Status on 2026-11-07:** Cycle 2 material submissions closed October 31 (UTC+8); formal evaluation closed November 4; the official results page states that results are expected mid-November and full evaluations are still being processed. Do not claim a Cycle 2 submission or score for Symthaea. Track the outcome and use [#8475 — submission readiness](https://github.com/Luminous-Dynamics/symthaea/issues/8475) as the existing owner for next-cycle preparation.

For a future Symthaea submission, the Coding Agent Memory track is a natural first applied test: it evaluates retrieving and reusing experience within the same repository under relevant and noisy memory conditions. The Textual and Multimodal tracks should remain separate follow-on work; their results are not interchangeable. Reuse [#8427](https://github.com/Luminous-Dynamics/symthaea/issues/8427) for the underlying frozen-reader / memory-substrate cost-quality comparison.

Operationally, AML changes the success criterion: HME must support an honest, stable Add/Search service with durable Add acknowledgment, tenant/sample isolation, bounded Search, exact version pinning and enough capacity to survive evaluation concurrency. The API adapter belongs in the existing Broca/benchmark integration, not in the HME hot cognitive update loop.

Official references:
- https://agentmemories.ai/evaluation
- https://agentmemories.ai/
- Existing project readiness issue: https://github.com/Luminous-Dynamics/symthaea/issues/8475

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


## 8. New deterministic fixtures — sampling alias and query completion

The branch now includes [hme_query_sampling_and_completion_v0.json](fixtures/hme_query_sampling_and_completion_v0.json). It contains small exact cosine controls where fixed-origin stride 4:
- misses two identical sparse vectors entirely (full cosine 1.0, sampled-path zero-norm fallback 0.0);
- reports sampled cosine 1.0 when the full cosine is 0.0 because the only disagreement lies in an omitted coordinate;
- misses final-coordinate and shifted-phase sparse patterns.

The fixture also contains typed completion cases for ExactComplete, ApproximateComplete, Incomplete after budget exhaustion and UnavailableOrInvalid. These are deterministic contract fixtures, not a benchmark run. The sampling study is owned by #10974; query completion semantics by #10973.


## 9. Additional HME research: bounded queries, sampling and certified compression

Three new work packages extend the cross-benchmark programme:
- [#10973 — HME query budgets](https://github.com/Luminous-Dynamics/symthaea/issues/10973): measure exact-completion probability, approximate quality and incomplete-result behavior under deterministic byte/candidate/coordinate budgets and optional deadlines. A timeout cannot be reported as exact success.
- [#10974 — phase-balanced approximate similarity](https://github.com/Luminous-Dynamics/symthaea/issues/10974): compare fixed-origin stride with stratified, phase-rotated and adaptive sampling. The fixture includes cases where stride 4 changes full cosine 1.0 to 0.0 and full cosine 0.0 to 1.0.
- [#10978 — certified compressed exact retrieval](https://github.com/Luminous-Dynamics/symthaea/issues/10978): use compact codes plus verified residual bounds to avoid reading original ContinuousHV payloads for candidates that cannot enter top-k, while exact-scoring every survivor from original payload.

Machine-readable fixtures now on the HME branch:
- [hme_query_sampling_and_completion_v0.json](fixtures/hme_query_sampling_and_completion_v0.json)
- [hme_progressive_exact_bounds_v0.json](fixtures/hme_progressive_exact_bounds_v0.json)

The progressive-bound fixture exhaustively enumerated 19,200 small continuous real-arithmetic prefix cases and 768 four-bit binary cases. It found no mathematical under-bound within the declared enumeration tolerance. It also contains a binary64 numerical counterexample: the ordinary floating evaluation of the ContinuousHV Cauchy–Schwarz expression returned 0.4999999999999999 although the full cosine was 0.5, enough to falsely prune at a threshold of 0.49999999999999994. This is why HME requires outward-rounded intervals, not ordinary floating values treated as proofs. These are small synthetic sanity checks, not production proofs or benchmarks.

## 10. Practical competitive positioning

Do not make HME compete as a universal replacement for a modern vector database:
- On standard 384–3,072D embedding workloads, VIBE and VectorDBBench are the correct algorithm/database reference points; use matched datasets, metrics, filters, recall and hardware.
- On on-disk vector I/O, MLPerf Storage is the appropriate storage-path reference.
- On long-term agent memory, LongMemEval, LongMemEval-V2, LoCoMo, TimlMemBench and the Agent Memory Leaderboard are separate downstream tests.
- On 16K–256K HDC capacity and dynamics, #7772/#7749/#6560/#6561 are the native qualification layer.

A fair head-to-head requires more than equal vector counts. A 16,384D continuous-f32 payload is 64 KiB; a 256K payload is 1 MiB; a 1,536D float32 embedding is 6 KiB. Therefore report both:
1. **Equal-representation task comparisons** (same underlying corpus and task, each representation scored under its own exact oracle).
2. **Equal-total-memory / equal-cost comparisons** (include all payloads, codes, indexes, metadata, provenance, journaling, snapshots, cold storage and query CPU).
3. **Equal-quality Pareto comparisons** (same downstream success / recall floor, then compare bytes, latency and energy with declared measurement boundaries).

If HME uses substantially more bytes per memory than a conventional embedding store, it must demonstrate a task-utility, interference, provenance or state-continuity improvement that justifies the extra footprint. The benchmark must be allowed to show that a conventional embedding or database path wins for a workload; HME should be selected only where its representation-aware semantics are valuable.


The bounded-query contract also has a structural artifact:
- [hme_query_receipt_v0.schema.json](schemas/hme_query_receipt_v0.schema.json)
- [hme_query_receipt_states_v0.json](fixtures/hme_query_receipt_states_v0.json)

The synthetic receipts distinguish exact full scan, exact safe pruning, approximate candidate completion, budget-exhausted incomplete search and invalid generation. The schema is structural; exact candidate-accounting arithmetic and mathematical bounds require an independent verifier.


Certified-compression research also includes a blockwise residual-bound fixture:
[hme_certified_compression_block_bounds_v0.json](fixtures/hme_certified_compression_block_bounds_v0.json).
It shows a synthetic query-weighted block residual dot bound of 2.0 versus 10.1 for a global residual-radius bound. The corresponding cosine upper bounds are 0.01048 versus 0.05291 in real arithmetic. This motivates measuring blockwise certificate metadata overhead against saved full-payload reads; it is not a measured query-speed improvement.


## 11. Concrete integration finding: SQLite conversation retrieval consumes ContinuousHV similarity

A source audit found that `crates/domains/symthaea-memory/src/conversation.rs` implements `ConversationMemory::find_similar()` by loading SQLite hypervector BLOBs, deserializing `Vec<f32>`, constructing `ContinuousHV` and invoking `query_embedding.similarity(&hv)`.

This makes #10918's hidden stride/build discrepancy relevant to persisted conversation retrieval. The method also lacks decoded-vector dimension and finite-value admission checks before calling an API that asserts equal dimensions; it sorts by score without a stable conversation-ID tie-break and maps NaN partial comparisons to Equal.

HME integration must not silently alter this legacy behavior. The first exact baseline should compare:
- existing method under its actual build/stride configuration;
- a declared exact full-coordinate score contract;
- invalid-dimension and non-finite BLOB handling;
- corrupted-BLOB diagnostics;
- deterministic score-descending / conversation-ID-ascending tie ordering.

The semantic API owner is #10918; the read-only HME comparison is #10917. The intended result is an explicit migration decision, not a shadow scoring implementation.


## 12. HDC-LTC checkpoint correctness

The HME architecture now has a dedicated checkpoint-completeness work package, [#10979](https://github.com/Luminous-Dynamics/symthaea/issues/10979). It distinguishes evolution-only snapshots from complete learner checkpoints and replay checkpoints. The distinction is essential because a state snapshot may restore current dynamics while omitting weights, optimizer momentum or mutable normalization statistics needed for continued learning. Checkpoint size comparisons must cover identical field sets; a smaller artifact that omits training state is not a compression win.


The HDC-LTC checkpoint contract has a source-derived field inventory:
[hme_dyn_checkpoint_field_matrix_v0.json](fixtures/hme_dyn_checkpoint_field_matrix_v0.json).
It captures the distinction between an EvolutionSnapshot (dynamic state + outputs + clocks) and a LearnerCheckpoint (parameters, optimizer state, mutable statistics, config, bindings and dynamic state), plus ReplayCheckpoint and WeightsOnlyArtifact.


## 13. Bridge projection footprint and low-rank update comparison

New research child [#10981](https://github.com/Luminous-Dynamics/symthaea/issues/10981) investigates structured/regenerated projection bases and rank-one output-projection update factors. This matters directly to HME's 2K–256K resource profile: at D=256K with 256-wide input and output, current dense bridge projections account for 512 MiB of logical f32 payload.

Machine-readable arithmetic fixture:
[hme_bridge_projection_footprint_v0.json](fixtures/hme_bridge_projection_footprint_v0.json).
It includes the dense pair, theoretical multiply-accumulate counts, and exact factor-log size across rank values. These are analytical byte/op counts, not RSS or runtime results.


The source-derived checkpoint inventory now includes the HdcLtcBridge adapter, not just its inner HdcLtcUnifiedNetwork: bridge config, input/output projection state, cached output, step/diversity/resize counters and GenesisSeed lineage. The Bridge type itself is not serde-derived; scratch buffers are transient. See [hme_dyn_checkpoint_field_matrix_v0.json](fixtures/hme_dyn_checkpoint_field_matrix_v0.json).


## 14. Direct HDC projection prior art: Hadamard submatrix encoding (2026)

The projection crosswalk now includes Salahudeen, Maalouf and Elfadel, “Deterministic orthogonal projections for hyperdimensional regression: A Hadamard submatrix approach,” *Intelligent Systems with Applications* 31 (September 2026), DOI [10.1016/j.iswa.2026.200707](https://doi.org/10.1016/j.iswa.2026.200707).

The publisher abstract reports a deterministic Hadamard-submatrix encoder, tested on six regression datasets across associative, iterative-gradient and FLASH-style closed-form ridge learners; it reports up to 9× reduction in selected hyperspace dimension and up to 19× wall-clock speedup versus its default HDR baselines. The paper explicitly reports mixed but competitive predictive performance, dataset/learner dependence, strongest results with closed-form learning, and power-of-two / non-universal compression constraints. Its data is available on request. These are the paper's reported results, not Symthaea results or a reproduction.

This is a direct prior-art baseline for #10981. HME/Symthaea must compare its current dense bridge projection and proposed structured/low-rank profiles against a Hadamard-submatrix arm, then test actual HDC-LTC temporal behavior separately from static regression.


## 15. Direct projection prior art and HDC-LTC structured transform candidate

The projection comparison must include the September 2026 paper by Salahudeen, Maalouf and Elfadel, “Deterministic orthogonal projections for hyperdimensional regression: A Hadamard submatrix approach,” *Intelligent Systems with Applications* 31, DOI 10.1016/j.iswa.2026.200707. Its publisher abstract reports six regression datasets, associative / iterative-gradient / FLASH-style closed-form learners, up to 9× smaller selected hyperspace and up to 19× speedup versus its default HDR baselines. It also reports mixed task-dependent predictive performance, strongest benefits with a closed-form solver, and structural limitations. Dataset availability is on request. These are paper-reported results, not Symthaea measurements or a reproduction.

[#10981](https://github.com/Luminous-Dynamics/symthaea/issues/10981) now includes two specific bridge fast-transform candidates:
- Input 256→D: zero-pad, seeded sign/permutation, normalized fast Walsh–Hadamard transform (FWHT).
- Output D→256: independently seeded sign/permutation, FWHT and selected-coordinate readout, or a Hadamard-submatrix alternative; compare with an exact rank-one learned adapter.

The model is O(D log D) transform work versus the current dense O(256D) projection arithmetic, but operation counts are not latency claims. These transforms alter the projection profile; compare both static regression controls and actual HDC-LTC temporal/online-learning behavior. The paper's regression gains cannot be generalized to liquid dynamics without those experiments.


## 16. Fast Quantized JL (October 2026): approximate compression comparator

A current theoretical comparator is Korakitis and Nakos, “Fast Quantized Johnson–Lindenstrauss Transforms,” arXiv:2610.16289 (posted October 12, 2026). It proposes D-FQJL (Hadamard sketch plus Gaussian dithering) and P-FQJL (Hadamard sketch plus permutation) for one-bit asymmetric quantized inner-product estimation, with O(d log d) application time and different embedding-dimension guarantees. Its motivating application is transformer KV-cache compression.

This work is relevant for HME's fast-tier approximate code, but its concentration/failure guarantees are statistical; the returned inner-product estimate is not a deterministic worst-case upper bound on an exact ContinuousHV score. Therefore:
- benchmark D-FQJL and P-FQJL as explicitly approximate profiles;
- measure estimator bias/variance, concentration failure, memory, compute and task quality;
- allow them to order candidates or trigger refinement;
- do not let their confidence estimate authorize a zero-false-prune ExactComplete result unless a separate, conservative, verified upper certificate is constructed.

HME's distinctive test is to compare probabilistic quantized sketches against its certified global/blockwise residual bounds under the same task, memory budget and exact full-payload oracle. Keep exact assurance, statistical confidence and empirical recall as three different labels.

Official/preprint references:
- https://arxiv.org/abs/2610.16289
- https://alma.di.uoa.gr/node/771



## 12. New SOTA comparator: PQR-Search (2026-11-05)

[Per-Query Recall Guarantees for Approximate Nearest Neighbor Search in Dynamic Vector Databases](https://arxiv.org/abs/2611.07579), by Woong-Kee Loh and Suan Lee (arXiv, November 5, 2026), is a highly relevant new comparator.

The paper introduces PQR-Search, which targets a per-query probabilistic guarantee: P(Recall(q) < τ) ≤ δ for each query under its randomized orthogonal-transformation assumption. It stores one-bit rotated residual codes relative to IVF centroids and uses exact quantiles of residual errors for pruning. Its reported evaluation includes 20 datasets up to 100M vectors and dynamic insert/delete workloads. The abstract reports failure rates at or below δ in the reported audits, and compares against average-recall-tuned HNSW and exhaustive FAISS.

This is a different guarantee from HME's intended ExactComplete:
- PQR-Search is approximate but aims to bound the probability that an individual query falls below a recall threshold.
- HME full scan is deterministic exact against a declared oracle.
- HME progressive/certified pruning seeks deterministic exact top-k if every prune is mathematically admissible.
- HME ApproximateComplete may select PQR-like profiles only under an explicit, versioned quality guarantee; it must never be promoted to ExactComplete.

Use PQR-Search as a head-to-head external algorithm baseline where representation, distance metric, dataset, k, hardware and dynamic update semantics are compatible. Report its τ, δ, transformation seed assumptions and per-query failure audit separately from HME exactness. Do not call a probabilistic guarantee deterministic, and do not compare abstract paper speedups with local HME timings without a common setup.

Paper: https://arxiv.org/abs/2611.07579


## 13. Adaptive exact block search and coherent checkpointing

New HME research:
- [#11000 — query/candidate-informed block order](https://github.com/Luminous-Dynamics/symthaea/issues/11000). This applies representation-aware block bounds to exact HDC progressive scans while explicitly recognizing BlockMaxWAND and variable-block-max WAND as established prior art. Its goal is HDC-specific exactness + data locality, not claiming the general block-pruning concept.
- [#11001 — consistent-cut cognitive checkpoint](https://github.com/Luminous-Dynamics/symthaea/issues/11001). It binds bridge learner checkpoint, HME manifest/segments and consumed/materialized event watermarks into one coherent, recoverable generation. It composes the existing memory substrate and checkpoint owners.

The branch includes deterministic synthetic fixtures:
- [hme_adaptive_block_order_v0.json](fixtures/hme_adaptive_block_order_v0.json)
- [hme_consistent_cut_states_v0.json](fixtures/hme_consistent_cut_states_v0.json)

Both fixtures demonstrate contract shapes only; they are not Rust runtime tests or storage crash-injection results.


## 14. Eligibility and lifecycle governance at the HME query boundary

[HME-ELIGIBILITY-001 / #11003](https://github.com/Luminous-Dynamics/symthaea/issues/11003) adds metadata-first bitemporal, principal/scope, retraction, revocation, retention and forgetting eligibility before any HDC payload is read.

The query receipt schema and examples now bind the pinned eligibility snapshot and reconcile physical, eligible and excluded populations. The machine-readable [eligibility filter fixture](fixtures/hme_eligibility_filter_v0.json) expects zero payload reads for excluded candidates, including a stale LSH hint that points at a revoked item.

This composes the existing #9075/#9139/#9177 owners. It explicitly separates query exclusion, active-index forgetting, physical purge and cryptographic erasure.


## 15. Quantized-vector decision stability: evaluate margins, not just distortion

A second 2026 paper is directly relevant to #10978 and #11000: Xiao & Cao, [When Does Low-Bit Quantization Preserve the Decisions of Vector Search?](https://arxiv.org/abs/2609.09854) (September 9, 2026).

The paper studies whether quantization preserves the local rank/prune decisions consumed by a vector-search algorithm. It shows why MSE and global rank correlation can fail to predict selected near-neighbor decision flips. Its analysis decomposes flip risk into exact-margin boundary mass and calibrated residual tails, accounts for covariance between residuals sharing a query/candidate, and presents a held-out block certificate for a frozen quantized decision rule.

Implications for HME:
- Record exact top-k margin distributions and risk by hard-margin quantiles.
- Measure ranking and pruning decision flips on selected candidate populations, not random pairs only.
- Measure covariance of paired residuals rather than assuming independence.
- For held-out finite-sample selective certificates, report coverage and failure bound; these produce an explicitly probabilistic ApproximateComplete profile.
- Do not mistake an empirical or statistical certificate for the deterministic worst-case residual upper bound required by ExactComplete.

The paper reports 10.1% ranking flips for scalar int4 on its hardest fifth of margins in one inspected configuration, despite strong aggregate fidelity. This is an external paper result, not an HME measurement. <Cite refs={["turn684205view0","turn443733view0"]}/>


Machine-readable decision-margin example:
[hme_quantized_decision_margins_v0.json](fixtures/hme_quantized_decision_margins_v0.json).
It demonstrates large- versus small-margin ranking stability and shared-query residual cancellation versus anti-correlation. This is a synthetic arithmetic fixture, not an empirical quantizer result.


## 2026-11-07 addendum: per-query reliability and selective exactness

The PQR-Search paper is now treated as a direct competitive baseline, not merely an ANN comparison. Loh & Lee, *Per-Query Recall Guarantees for Approximate Nearest Neighbor Search in Dynamic Vector Databases*, arXiv:2611.07579 (2026-11-05), states:

\[
\Pr[\operatorname{Recall}(q)<\tau]\leq\delta
\]

for every fixed query, with probability over the index's random orthogonal transformation and the stated condition that queries and updates do not depend on that transformation. The authors report 20 datasets up to 100M vectors and dynamic-update audits. Their abstract reports that, in compared settings, PQR-Search incurred 2.5x HNSW single-query latency and 1.3x lower batch throughput where HNSW met the 0.99 per-query target, while average-recall-tuned HNSW left 3–29% of queries below 0.99. These are paper-reported figures, not reproduced HME comparisons.

**Guarantee boundary:** per-fixed-query probabilistic recall is not deterministic exact top-k. A client choosing later queries after observing index-dependent responses requires a separate adaptive-query threat model; do not claim the original independence premise covers that case without proof.

Xiao & Cao, arXiv:2609.09854 (2026-09-09), provide a complementary decision-margin and covariance-aware quantization framework. Its frozen-candidate / frozen-trace results do not independently establish candidate-generation coverage or end-to-end recall. HME must report quantization/ranking/pruning decision flips separately from candidate coverage and final Recall@k.

The new [#11022 — HME-PQR-001: per-query recall guarantees and selective exact fallback](https://github.com/Luminous-Dynamics/symthaea/issues/11022) owns a common reliability audit across PQR-Search, conventional ANN, and HME profiles. It reuses #10945, #10973, #10974, #10978 and #11000 rather than creating another ANN index or query engine.

New protocol and fixture:
- [Per-query reliability and selective exactness protocol](https://github.com/Luminous-Dynamics/symthaea/blob/research/hlb-memory-engine-20261107/docs/research/HME_PER_QUERY_RELIABILITY_2026-11-07.md)
- [Machine-readable query-strata / guarantee matrix](https://github.com/Luminous-Dynamics/symthaea/blob/research/hlb-memory-engine-20261107/docs/research/fixtures/hme_per_query_reliability_v0.json)

The benchmark separates five classes: ExactFull, ExactProgressive, ProbabilisticPerQuery, EmpiricalSelective, and IncompleteOrUncertified. It freezes a non-adaptive query panel before index-transform generation, then adds a separate adaptive-feedback panel. Dynamic insert/delete traces are audited at declared update watermarks. Report per-query recall quantiles, failure fractions by stratum, candidate coverage, decision flips, latency tails, index memory, bytes read, build/update/compaction cost and fallback frequency.

For a zero-failure binomial audit, the one-sided upper limit is \(p_{upper}=1-\alpha^{1/n}\). Approximate audit planning counts are 299 independent transforms per fixed query for \(\delta=0.01\) at 95% confidence, 459 at 99%, 2,995 for \(\delta=0.001\) at 95%, and 4,603 at 99%. These are empirical-audit planning values, not proof of a theorem; use an exact upper interval when failures occur and predeclare multiple-claim correction. Do not pool heterogeneous queries to claim per-query reliability.

Project promotion targets: exact profiles require 100% canonical-oracle agreement and zero false pruning; a probabilistic standard tier targets \(\tau\geq0.99,\delta\leq0.01\) per fixed query under a reviewed proof and explicit assumptions; a stretch tier targets \(\tau\geq0.999,\delta\leq0.001\) only if practical and proven. Empirical selective bounds remain explicitly empirical.

No HME runtime result, PQR reproduction, per-query theorem, or SOTA win is claimed by these research artifacts.


## 2026-11-07 addendum: learned query-cost estimation, declarative recall and I/O-first systems

The current comparison is expanded beyond PQR-Search and quantization.

### Query-aware cost and budget baselines

- **ANNiE** — Wang et al., PVLDB 19(11):3820–3833 (2026), DOI 10.14778/3836663.3836728. A graph-local-feature GBDT estimates distance-computation cost for a target recall. Reported ANNiE-S speedup is 2.3x over the paper's baselines, and cost-estimation accuracy improves 6x over those baselines. Its formal RTG property is a conditional-quantile relation over random model-input / query-cost variables under an ideal globally optimal expected quantile-loss optimizer. The paper explicitly limits the theorem to the training distribution and warns about finite samples, optimization and train/test shift. In its Text2Image10M modality-transfer experiment, IMG→TXT empirical RTG coverage is 58.55% against a 95% target; same-modality cases are materially better. Do not equate this with PQR-Search's probability over index randomness for each fixed query.
- **QBAT** — Bae et al., PVLDB 19(11):3091–3104 (2026), DOI 10.14778/3836663.3836675. Compares GBDT and an interpretable AlphaEvolve-derived heuristic for query-adaptive ScaNN budgets; the paper's abstract reports up to 68.8% lower total searched budget in consistency mode while enforcing a strict per-query recall target. This is a direct cost/budget allocation comparator.
- **DARTH / DARTH+** — DARTH predicts runtime per-query recall for early termination; DARTH+ improves prediction and training efficiency and adds optional probabilistic quality guarantees in its March 2026 preprint. This belongs in the declarative-recall comparison, not the exact-search guarantee column.
- **Ada-ef** — Zhang & Miller, SIGMOD 2026, arXiv:2512.06636. Uses a statistical query–database similarity model to adapt HNSW exploration per query, reporting up to 4x lower online latency and substantial offline computation/memory reductions versus its evaluated learning-based adaptive baselines.

References:
- ANNiE paper/code: https://doi.org/10.14778/3836663.3836728 ; https://github.com/CaucherWang/ANNiE
- QBAT: https://doi.org/10.14778/3836663.3836675
- DARTH+: https://mchatzakis.github.io/declarative-recall/
- Ada-ef: https://arxiv.org/abs/2512.06636

### OOD graph robustness

Hua et al., "Dynamically Detect and Fix Hardness for Efficient Approximate Nearest Neighbor Search" (PACMMOD 2025 / SIGMOD 2026), defines Escape Hardness and uses RFix/NGFix to repair graph regions online, reporting up to 2.25x faster OOD search at 99% recall than RoarGraph and 6.88x than HNSW in evaluated settings. This is an index-structure repair approach, distinct from HME's proposed query-cost OOD monitor. Treat as a graph-index comparator, not an HME guarantee.

Reference: https://doi.org/10.1145/3769783 ; https://arxiv.org/abs/2510.22316

### Disk I/O and heterogeneous system baselines

- **OctopusANN / I/O-first graph ANN design-space study** — Li et al., PVLDB 19(7) (2026), DOI 10.14778/3801059.3801064. Reports 4.1–37.9% higher throughput than Starling and 87.5–149.5% higher throughput than DiskANN at matched Recall@10=90%, using a composition of memory-resident navigation, dynamic width, page shuffle and page search. Its abstract states SSD I/O can contribute 70–90% of query latency for disk-resident graph ANN workloads.
- **SVFusion** — Peng et al., PVLDB 19(5):1074–1087 (2026), DOI 10.14778/3796195.3796216. CPU/GPU/disk co-processing, workload-aware caching and dynamic-update concurrency control; reports 20.9x average throughput and 1.3x–50.7x lower latency than its baselines under its evaluated streaming workloads.

These results strengthen the HME requirement to measure pages, bytes, cache state, data movement, concurrency and update cost. They do not imply a directly comparable HDC result.

### New HME workstreams

- [#11022 — HME-PQR-001](https://github.com/Luminous-Dynamics/symthaea/issues/11022) owns the cross-algorithm per-query reliability audit and separates theorem-backed per-fixed-query results from population-level / empirical quantile models.
- [#11041 — HME-COST-001](https://github.com/Luminous-Dynamics/symthaea/issues/11041) owns retrieval-cost estimation, calibration / OOD qualification and proof-guided value-of-computation scheduling without duplicating #5471/#10973/#11000.

New protocol:
- [HME query-cost and proof-guided scheduling](https://github.com/Luminous-Dynamics/symthaea/blob/research/hlb-memory-engine-20261107/docs/research/HME_QUERY_COST_AND_PROOF_GUIDED_SCHEDULING_2026-11-07.md)
- [Machine-readable cost-model / model-trust matrix](https://github.com/Luminous-Dynamics/symthaea/blob/research/hlb-memory-engine-20261107/docs/research/fixtures/hme_query_cost_estimation_v0.json)

The differentiating HME hypothesis is **proof-guided value of computation**: predict which candidate/block/page is likely to reduce future exact work per byte, while the predictor only schedules work and conservative score bounds alone authorize pruning. On ExactOnly workloads, an OOD predictor is disabled and a qualified deterministic exact schedule is used; if its budget expires, return Incomplete. On ApproximateAllowed workloads, a calibration breach revokes the model-dependent empirical claim until recalibration or an explicitly permitted exact fallback.

No HME cost estimator, OOD detector, PG-VoC scheduler, Rust engine, SOTA win or performance result is claimed by these protocols.
