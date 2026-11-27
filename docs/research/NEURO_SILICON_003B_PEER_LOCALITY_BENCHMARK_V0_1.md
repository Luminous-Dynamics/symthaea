# NEURO-SILICON-003B — Peer-Resident HDC Retrieval and Query-to-Data Placement

**Status:** preregistered paired experiment; no peer run or physical measurement is claimed  
**Version:** 0.1  
**Parent:** [NEURO-SILICON-003 #7864](https://github.com/Luminous-Dynamics/symthaea/issues/7864)  
**Host baseline:** [PR #7866](https://github.com/Luminous-Dynamics/symthaea/pull/7866)  
**Platform policy contract:** [Luminous Platform DWF-001 #351](https://github.com/Luminous-Dynamics/luminous-platform/issues/351)

## 1. Research question

For one exact HDC retrieval task and one exact vector-store manifest, when should execution remain local, move to the data, move data to compute, or replicate once and query repeatedly?

The experiment must account for the complete path: task/input identity, index residency/build, serialization, dispatch, network, queueing, retrieval, response, independent verification, origin/peer energy and privacy policy.

This is a measurement and comparison child of #7864. It does not create a second HDC implementation, generic scheduler, transport, memory topology, EnergyMeter, device inventory or trust system.

## 2. Falsifiable hypotheses

H1 — Query-to-data wins for a large resident store and small query when query/response bytes plus session/dispatch overhead are materially smaller than transferring the required store or segments.

H2 — Data-to-compute or replication wins for small stores, large batches or repeated local queries when initial transfer is amortized over enough queries and local execution meets the same quality/latency contract.

H3 — There is no universal winning placement. The break-even surface depends on store size, query batch, network RTT/bandwidth/loss, peer load, store residency, index-update frequency, result size, energy scope and policy.

H4 — Privacy and authorization are eligibility constraints, not optimization weights. A local-only or unauthorized data class cannot be offloaded even if remote compute is faster or more energy efficient.

The experiment must be allowed to conclude that local CPU/SIMD wins all measured conditions, remote execution is never worthwhile, or evidence is insufficient.

## 3. Canonical owners and composition

- #7866 provides the canonical local host baseline, fixtures, timing observations, receipt format and exact BinaryHV operations. Reuse it; do not copy or fork its retrieval algorithm.
- #5805/#5808 own canonical HDC workload identity, semantics and correctness oracle.
- #9341 owns typed memory topology and state-placement evidence. This experiment records actual store residency/transfer paths and references the QARCH contract; it does not invent a topology schema.
- #7919/#7515 own persistent memory/storage architecture, workload capacity and degradation semantics.
- #10460 owns Rust-native hardware energy-meter capability discovery and energy receipts. Unsupported sensor scope must remain Unsupported/Inconclusive, not estimated as measured joules.
- Xenia owns authenticated peer transport/session scope. Session authentication does not prove device attestation, workload identity, execution, result correctness or local compute authority.
- Mycelix #3972/#3992 own hardware/workload admission and physical TPM/measured-boot profiles. Use only when an exact supported profile is qualified. LEX-NET #1149 is not a device-compute permission.
- Luminous Platform DWF-001 #351 owns product-level task placement and data policy. This Symthaea issue owns the HDC-specific benchmark experiment and measurement.

## 4. Experiment arms

All comparable arms use identical store bytes, query bytes, oracle, Hamming semantics, batch size, result limit, expected IDs and tie-breaking. One exact store manifest and query manifest bind every run.

### A — Local execution

The origin device holds the exact store in its declared memory/storage tier and runs the canonical CPU implementation. Use PR #7866 as the local anchor. Measure cold load and warm residency separately.

### B — Query-to-data

An owner-controlled Node holds the exact committed store/segment manifest. The origin sends only the query batch and minimum permitted metadata. The Node performs exact HDC retrieval and returns top-k IDs/distances plus a bounded execution receipt. The raw vector store remains on the Node.

Account for peer handshake/session setup, query serialization, network bytes, queueing, retrieval, response serialization and validation.

### C — Data-to-compute

Transfer the exact store or a declared required segment set to the origin, verify its digest, then run the same local search. Measure transfer and verification separately from query serving. Do not report only the search kernel as end-to-end performance.

### D — Replicate once, query many

Copy and verify the exact store once, then issue a preregistered sequence of local queries. Include replication energy/bytes, persistent storage use, refresh/update cost and the query count over which replication amortizes.

### E — Remote ingestion/build versus serving

Where the fixture supports it, separate index creation/build, persistence, restart/load/verification, incremental update, store replication and query serving. Never collapse one-time build cost into per-query serving or omit it from lifecycle comparisons.

## 5. Fixed workloads and conditions

Begin with the exact workload and dimensions already used by #7866:
- store sizes: 64, 256, 1,024 and 4,096 vectors;
- query batch sizes: 1, 4, 16 and 64;
- preserve BinaryHV dimension, byte order, deterministic query perturbation and tie-breaking.

Larger stores (16K, 131K, 1M) are a later scale tier. Promote only when memory/working-set feasibility, source fixture identity and resource boundaries are recorded. Use #9341 to describe residency and transfer edges.

Network conditions are separate strata:
1. Same-host loopback: serialization/dispatch control, not a peer result.
2. Same-LAN physical peer: record actual link, RTT, throughput, loss and endpoint identities.
3. Rate-limited LAN: controlled bandwidth, jitter and loss.
4. WAN-like emulation: synthetic-only; never pool with physical-LAN results.
5. Peer reconnect/rekey and idle-to-first-query cold start.
6. Warm persistent store versus load/verify after peer restart.
7. Concurrent peer load and competing read workloads.
8. Partition before dispatch; partition after dispatch and before response.

Synthetic vectors are first. No confidential codebook, personal data, credentials, production index or private user content may be transferred for this experiment.

## 6. Industry baselines and representation boundaries

- FAISS IndexBinaryFlat is a relevant optimized exhaustive Hamming baseline. Compare exact top-k IDs/distances under identical bytes, dimensions, tie policy and queries.
- FAISS IndexBinaryIVF, IndexBinaryHNSW and binary hash variants are separate partitioned/approximate profiles. Report recall@k, exact-neighbor agreement and memory/build costs; do not call them exact if they can miss a neighbor.
- Milvus BIN_FLAT and BIN_IVF_FLAT are additional binary-vector comparators where reproducible deployment and exact byte/metric mapping are confirmed.
- Qdrant binary quantization is an adjacent dense-embedding compression track. It is not automatically equivalent to native BinaryHV because its representation derives from float-embedding quantization and distribution-specific accuracy behavior.
- MLPerf Inference v6.1 End-to-End RAG is the closest current system-level methodology reference. It separates vector-database ingestion/build from multi-hop query serving and exposes placement, co-residency, scheduling and CPU/GPU/NIC/storage interaction. It is not a directly comparable HDC score; this project does not claim MLPerf compliance.

References:
- MLPerf End-to-End RAG (26 Aug 2026): https://mlcommons.org/2026/08/endtoend-inference/
- MLPerf Inference v6.1 analysis (17 Sep 2026): https://mlcommons.org/2026/09/chairs-mlperf-inference-v6-1/
- FAISS binary indexes: https://github.com/facebookresearch/faiss/wiki/Binary-Indexes
- Milvus binary-vector metrics/indexes: https://milvus.io/docs/index-vector-fields.md
- Qdrant quantization: https://qdrant.tech/documentation/manage-data/quantization/

## 7. Measurement contract

Every run binds:
- exact source commit/tree and clean/dirty state for runner and workload;
- store/query/expected-output digests, dimensions, vector count, batch size and result limit;
- origin and peer exact SKU/board revision, CPU/GPU/NPU model, memory, storage and residency state;
- OS/kernel, firmware, driver/runtime, backend and Xenia transport/session profile;
- network link/profile, direction, RTT/latency distribution, loss, throughput and congestion;
- cold/warm store state, index build/load/verification/update state;
- per-stage wall time and p50/p95/p99 end-to-end latency;
- bytes in both directions, protocol/control overhead, retransmissions and result size;
- exact output IDs/distances, tie behavior and independent-oracle outcome;
- origin and peer energy receipts separately, with sensor scope, raw counters, supported status, wrap/uncertainty, calibration and boundary;
- memory topology, working set, RSS/VRAM, residency and transfer path;
- trial order, warm-up, repetitions, failures, timeouts, retries and all excluded observations with reasons.

Do not sum overlapping GPU, CPU-package and whole-device energy scopes. Modelled energy, TDP-derived estimates, OS battery telemetry and calibrated meter results are distinct evidence classes.

## 8. Break-even analysis

Report the break-even query count Q* for replicate-once versus query-to-data only when inputs exist: one-time store transfer/build/verification cost; per-query upload/download/dispatch/network cost; per-query origin and peer energy; expected query volume and store refresh frequency; measured or explicitly modelled latency/energy distribution.

Q* must be an interval with uncertainty or NOT_COMPARABLE. Never derive it from one mean latency while ignoring updates or peer contention. Show the full Pareto vector and no-winner outcomes.

## 9. Adversarial and negative controls

The companion fixture is research/neuro-silicon-003b/peer_locality_scenarios_v0_1.json. It must cover at least:
1. tiny store where local CPU wins;
2. large resident store/small query where query-to-data reduces bytes;
3. large batch where query upload dominates;
4. transfer-once then repeated local queries;
5. remote index-build cost omitted or double-counted;
6. one-bit-mutated store manifest;
7. stale store after update;
8. valid session but missing execution scope;
9. local-only data with attempted peer egress;
10. partition before dispatch;
11. partition after dispatch/unknown completion;
12. signed but incorrect top-k result;
13. different byte order or HDC version;
14. approximate FAISS/Milvus candidate misses exact neighbor;
15. unsupported or overlapping energy counters;
16. advertised NPU but observed CPU fallback;
17. peer overload/thermal limit;
18. replication digest mismatch;
19. store restart and unverified reload;
20. no peer available: local/deferred behavior is correct.

Every fixture is synthetic. An independent evaluator must implement expected dispositions separately from the fixture generator.

## 10. Proposed gates (not results)

- Exact retrieval: 100% exact top-k agreement for exact remote/local paths against the independent oracle on the frozen fixture.
- Transfer accounting: 100% of task payload, store bytes, response bytes and declared protocol overhead included in each end-to-end boundary.
- Privacy: zero unauthorized query/store egress in local-only negative controls.
- Trust/integrity: zero acceptance of stale/mismatched store, workload, runtime or required session/admission evidence.
- Energy: measured joules bind to supported #10460 receipts and explicit non-overlapping scopes; otherwise Unsupported/Inconclusive.
- Reproducibility: exact artifacts/configuration and raw per-run observations retained; at least 30 paired repetitions per scored condition where feasible, with candidate order interleaved/randomized.
- Product advantage: only claim remote-placement superiority after matched quality and complete end-to-end evidence shows at least 10% lower measured joules per accepted query or 10% lower p95 latency for a declared non-sensitive profile, with no privacy, availability or correctness regression.

The 10% values are internal starting targets. They are not current results or industry thresholds.

## 11. Claim ceiling

This protocol does not claim distributed HDC retrieval exists, that Xenia sessions prove a trusted physical device, that remote execution is faster or more energy efficient, or that Symthaea outperforms FAISS/Milvus. It defines a falsifiable experiment that can find a local winner, remote winner, different winners under different conditions, or no comparable result.
