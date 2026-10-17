# MEMORY-SUBSTRATE-001: Workload-Aware RAM and Persistent Storage v0.1

**Status:** research architecture + analytical sizing matrix; no hardware/software benchmark or storage durability result is claimed.  
**Tracking:** [issue #7919](https://github.com/Luminous-Dynamics/symthaea/issues/7919), child of [COMPUTE-ARCH-001 #7905](https://github.com/Luminous-Dynamics/symthaea/issues/7905).  
**Machine-readable scenarios:** `crates/domains/symthaea-silicon/schemas/memory_storage_capacity_matrix_v0_1.json`.

## 1. Executive design

Do not start by attempting to invent commodity DRAM cells or NAND flash. Start by designing a memory system where the physical tiers, controller policies, persistent format and compute dataflow are jointly optimized for real Symthaea workloads. The initial candidate is a **persistent two-plane store with a workload-scheduled memory hierarchy**:

- **Query plane:** immutable, versioned, sequentially laid out vector/codebook segments; the hot working set and small exact-search buffers reside in DRAM or accelerator-local SRAM/BRAM; exact Hamming/reduction runs next to the store where practical; only compact results cross the system boundary.
- **Update plane:** append-only, checksummed versioned delta log; a durable commit record and atomic manifest publication; background snapshot/merge/compaction converts deltas into immutable segments. Query foreground traffic receives bounded latency/QoS protection from compaction.
- **Recovery plane:** content hashes/checksums, format version, generation/epoch and stable item IDs; on restart, select the last valid committed manifest, verify all referenced extents, replay only valid committed records, and surface corruption/missing data explicitly rather than returning a partial result as success.

The research hypothesis is that reducing avoidable data movement, random writes, repeated index loads and foreground stalls can improve **joules per correct query, latency tails, durability and repairability** more reliably than pursuing a new memory-cell technology as the first step.

## 2. Memory is a scheduleable hierarchy

| Tier | Candidate technology | Intended role | What must be measured |
|---|---|---|---|
| T0 | Registers and hardware cache | Active operation state and hot metadata | stalls, locality, occupancy |
| T1 | SRAM / FPGA BRAM | Query buffers, reducer state, small hot codebook subset | usable capacity after other RTL, ports/banking, latency, resources, timing |
| T2 | Host DRAM | Resident codebook, index metadata, bounded cache | RSS, bandwidth, random access, pressure, cache state, NUMA/placement where available |
| T3 | NVMe SSD | Immutable corpus segments, snapshots, logs, cold vectors | sequential and random IO, p95/p99, flush semantics, sustained load, recovery, wear telemetry where available |
| T4 optional | CXL-attached/poolable memory | Additional memory capacity or shared hot datasets | actual device support, latency, bandwidth, contention, RAS, software placement |
| T5 exploratory | NVMe computational programs / subsystem-local memory | Compute or move data inside a storage subsystem | actual implementation availability, supported operation set, end-to-end gain |
| T6 long-term | Custom SRAM macro / controller silicon; possibly separately researched NVM | Only if a measured bottleneck justifies custom hardware | qualified RTL/formal/PPA, physical macro/PDK support, process/packaging/test dependency |

These are separate technologies and claims. DRAM, SRAM and NAND are not interchangeable. An SRAM generator creates SRAM design views for supported profiles; it is not a DRAM/NAND fabrication capability. CXL and computational-storage standards are interfaces, not proof the user's machine or a candidate SSD supports them.

## 3. A Symthaea-specific size model

Canonical packed `BinaryHV` is 16,384 bits, so raw vector payload is **2,048 bytes (2 KiB) per vector**, before stable ID, index, alignment, integrity and allocator metadata.

| Codebook | Raw vector payload | Approximate scale | Design question |
|---:|---:|---|---|
| 64 vectors | 128 KiB | Small hardware experiment | Can all payloads reside in local SRAM/BRAM with room for buffers/logic? |
| 256 vectors | 512 KiB | Small on-device store | When does host DRAM / external-memory access become necessary? |
| 1,024 vectors | 2 MiB | Useful screening tier | Is the controller or popcount datapath the bottleneck? |
| 4,096 vectors | 8 MiB | FPGA external-memory regime on small parts | What sustained bandwidth can the memory interface actually feed? |
| 16,384 vectors | 32 MiB | Medium corpus | Does persistent local residency justify preload cost? |
| 131,072 vectors | 256 MiB | Larger store | How does hot-set caching compare with all-resident host DRAM? |
| 1,048,576 vectors | 2 GiB | Large store | Does the workload favor RAM scanning, NVMe sequential scans, or an index that trades recall/latency? |

This table is exact arithmetic for payload sizes only. It is not an estimate of a board's available RAM or the full index size.

A 4-KiB page/block is included as one **hypothetical layout profile** in the JSON: two 2-KiB vectors could occupy a 4-KiB unit if alignment, metadata and actual device/file-system behavior allow it. It is not an assumption about every NVMe device's physical block size or NAND read granularity. Real layout must record logical block size, file system/direct-I/O mode, request sizes, alignment and IO traces.

## 4. Proposed on-storage format and update protocol

### Immutable segment

Each segment should have a versioned header with the format/schema version, vector width, item count, stable-ID range/lookup metadata, body length, byte order, corpus/representation identity and integrity hash. Keep payload extents sequential where scans are expected. Store metadata/indexes separately so startup doesn't have to load every vector merely to find segment boundaries.

Checksumming options need a concrete threat/cost profile: a fast per-block checksum detects accidental corruption; a cryptographic digest binds segment identity and the manifest. Do not conflate them with ECC or cryptographic authentication. Authentication/signatures may be useful where tampering is in scope; neither replaces reliability testing.

### Delta and commit

A possible write path is:

1. Validate incoming record and assign stable ID, generation and idempotency identity.
2. Append record and integrity metadata to a write-ahead/delta log.
3. Invoke the storage API's required persistence operation; acknowledge only when the declared durability contract says the commit is durable.
4. Publish a versioned commit/manifest record that references the new generation.
5. Background compaction merges older immutable segments plus committed deltas into a new immutable generation.
6. Atomically select the new manifest only after every referenced extent has been completely written and verified.
7. Retire previous segments only after no active reader or recovery path needs them.

The concrete atomicity/durability mechanism depends on the real storage API, filesystem, device and flush/FUA behavior; do not assume a host write returned = power-fail durability. Include every boundary in the profile.

### Recovery and degradation

On startup, choose the newest valid committed manifest—not merely the newest-looking file. Validate references and checksums before making a generation queryable. If a segment is missing/corrupt, mark the affected corpus scope incomplete and fail/abstain or reconstruct from a known-good copy; never invent a match from partial data. Keep version and integrity checks in the query receipt so a stale cache cannot silently answer for a newer generation.

Power-cut testing requires actual controlled hardware power-loss experiments. Process termination and simulated crash points test different, weaker fault models and must be labelled accordingly.

## 5. Controller/dataflow co-design

For the FPGA retrieval tile in [#7908](https://github.com/Luminous-Dynamics/symthaea/issues/7908):

- Keep only the measured hot subset and necessary FIFOs in local BRAM/SRAM.
- Use multi-bank memory only when the selected device can supply the actual lane width/parallelism; each extra distance lane creates an extra data-feed demand.
- Compare row/bank-aware scheduling (where DRAM interface details are available), bounded read-ahead, query batching, and fair queueing between foreground queries and background compaction.
- Use local nearest/top-k reduction so each candidate does not return every score to the host.
- Report separately: cold codebook load, warm query service, external bytes per query, bytes read locally, latency tail, memory contention and actual resource use.
- On memory exhaustion or controller errors, backpressure, reject, or fail explicitly; never silently drop a vector or return a truncated ranking as complete.

For the persistent layer, prefer sequential segment construction/compaction for immutable data. If the actual NVMe supports Zoned Namespaces, measure a zone-aware sequential append profile as a distinct candidate; do not assume all SSDs expose ZNS. Only test NVMe Computational Programs / Subsystem Local Memory when the exact device and software stack support the relevant command set.

## 6. Benchmarks, baselines and qualification

### Workloads

Use existing canonical owners rather than duplicating the benchmark harness:

- **Exact HDC scan/retrieval:** #7864/#7866 and the canonical BinaryHV oracle fixtures. Exact outputs must match the oracle.
- **Memory pressure:** #7515; capture peak/steady/transient RSS, allocation/working set, budget overrun, degraded-mode result and task quality.
- **Persistent update/restart:** synthetic immutable segments + append-only delta; deterministic records, known expected manifest and corpus digest. Inject failures at each protocol stage.
- **Storage IO:** sequential full-codebook scan, random index probes, mixed read/write, cold/warm cache, background compaction and multiple concurrent queries. An index/search-quality result is not interchangeable with exact full scan.
- **Physical hardware:** specific FPGA board and memory device first; actual host DRAM/NVMe incumbent; CXL or computational storage only as available. Synthetic timing models are for workload generation and measurement design, not a replacement for these baselines.

MLPerf Storage v3.0's Vector Database workload is a useful methodological reference because it measures on-disk ANN queries and verifies recall against exact brute-force ground truth. It also separates other storage regimes such as checkpointing and KV-cache offload. We can adapt its emphasis on representative load and quality checks, but must not call our result MLPerf-compliant unless we run the actual rules.

### Required metrics

**RAM / controller:** bandwidth and latency p50/p95/p99; working-set size and locality; controller stalls/bank conflicts where observable; request/byte counts; cache hit/miss and transfer overhead; actual FPGA LUT/FF/BRAM/DSP and timing with exact tool/target reports.

**Persistent media:** throughput/IOPS under declared block size, concurrency, queue depth and read/write ratio; p50/p95/p99; cold-start vs warm-cache; sustained behavior and compaction/GC interference; bytes read/written; flush/durability contract; data-integrity and recovery result; wear/endurance only where actual device telemetry supports it.

**Whole task:** exact correctness and corpus manifest, task quality, cold build/load, query p50/p95/p99, throughput, bytes moved per useful result, update cost, peak memory, recovery time and measured joules per successful query/update only when a trustworthy meter covers the stated boundary.

Repeat at least 30 measured batches/configuration where practical, preserve raw samples and uncertainty, and keep repeated durability/power-loss tests on their own declared protocol. Report the negative results.

### Proposed promotion gates—not achieved

- **Data integrity:** zero silently corrupted, lost or duplicated acknowledged updates in the test corpus; every fault results in either validated recovery or explicit incomplete/corrupt status.
- **Exact semantics:** 100% oracle match for exact profiles and zero results from incomplete codebook generations. Approximate or compressed representations need their own version and predeclared quality floor.
- **System benefit:** equal task quality and either >=1.5× useful-query throughput or >=30% lower directly measured joules per correct query, with p95 latency no more than 10% worse for the declared scenario.
- **No tail hiding:** show p50/p95/p99 with cold/warm cache and compaction/GC pressure; don't promote a faster average that causes severe p99 or crash-recovery regressions.
- **Custom hardware:** measure on a named board and memory part, then justify ASIC effort by improvement over the qualified CPU/FPGA/NVMe alternatives and a transparent fabrication/packaging/test dependency budget.

## 7. The real paradigm shifts

1. **Memory as a schedule, not a static tier:** placement follows locality, reuse, persistence and deadlines; the same corpus may have a resident hot slice and cold immutable tail.
2. **Two-plane mutation:** stable knowledge is immutable and compactable; new observations go to an append-only delta. This may reduce random writes and lock contention, but only if compaction cost, read amplification and crash guarantees measure well.
3. **Bring queries to persistent data:** move reduction near a resident store when useful; avoid materializing/transferring every intermediate distance.
4. **Treat integrity/recovery as part of the memory architecture:** manifests, generations, checksums, commit rules and explicit degraded status are in the design contract, not merely filesystem implementation details.
5. **Make memory faults observable and recoverable:** bounded queues, telemetry, data checks, optional ECC where supported, and verified restart instead of silent corruption.
6. **Keep media replaceable:** portable versioned segments, verifiable export/import, immutable source manifests and rebuildable indexes reduce dependence on one SSD controller or memory vendor.
7. **Explore device-side execution only when available:** computational storage and CXL may open useful paths, but the specification and hardware-support claims are distinct.

## 8. Research anchors (checked 17 Oct 2026)

- [CXL Consortium](https://computeexpresslink.org/about-cxl/): CXL 4.0 advertises 128 GT/s, bundled ports and memory RAS enhancements; investigate only where compatible hardware exists.
- [NVM Express Computational Programs](https://nvmexpress.org/specification/computational-programs-command-set/) and [Subsystem Local Memory](https://nvmexpress.org/specification/subsystem-local-memory-command-set/): current specification set lists v1.3 ratified 31 July 2026; device-side execution/local memory is a possible standard interface, not a guarantee of product availability.
- [NVMe Zoned Namespaces](https://nvmexpress.org/specification/nvme-zoned-namespaces-zns-command-set-specification/): sequential-write zones can suit append-heavy data structures and may reduce device-side write amplification when supported by hardware, firmware and host stack.
- [MLPerf Storage v3.0](https://mlcommons.org/2026/09/mlperf-storage-v3-0-results/) / [official suite](https://mlcommons.org/benchmarks/storage/): added Vector Database and KV Cache workloads to characterize different storage patterns.
- [OpenRAM setup and technology notes](https://github.com/VLSIDA/OpenRAM/blob/stable/docs/source/basic_setup.md): illustrates why supported memory macro generators are technology-specific; current documented support does not include GF180 SRAM generation.

These anchors inform the hypotheses; none establishes a performance win for this design.

## 9. Work sequence without waiting on runners

1. Establish host memory and NVMe baselines using the maintained #7515/#7864 owners and a pinned synthetic corpus.
2. Validate the capacity/transfer scenario matrix against independent formula calculations.
3. Prototype immutable segment + delta + manifest + restart/replay in the existing Rust storage owner after inspecting that component; test injected failures locally before claiming crash consistency.
4. Compare layout/read-ahead/working-set and foreground/background scheduling under frozen traces.
5. Integrate FPGA local-buffer + external-memory profile after choosing a real board and controller.
6. Explore custom SRAM macro/controller only after physical FPGA evidence shows that the memory subsystem is a meaningful bottleneck.
7. Only explore new NVM devices/cells as a separate research programme with defined retention, endurance, variability, error-rate, energy and fabrication qualification.

## 10. Explicit nonclaims

This proposal does not create a RAM chip, SSD, NAND/DRAM process or qualified SRAM macro. It does not establish data durability, zero data loss, higher memory bandwidth, lower latency/energy, FPGA operation, CXL/ZNS/Computational Programs compatibility, MLPerf compliance or ASIC readiness. The accompanying JSON includes only exact payload-size arithmetic and explicitly hypothetical bandwidth/page-size scenarios.
