# COMPUTE-ARCH-001 — Workload-Driven Computer Architecture Frontier v0.1

**Status:** research/benchmark protocol; all numerical gates below are proposed and unmeasured.  
**Tracker:** [Symthaea issue #7905](https://github.com/Luminous-Dynamics/symthaea/issues/7905).  
**Parent:** [COMPUTE-MFG-001 #5796](https://github.com/Luminous-Dynamics/symthaea/issues/5796).

## Executive hypothesis

Symthaea should not start by claiming it can replace a modern desktop CPU, GPU, DRAM fab, or NAND fab. It should discover **which measurable improvements can be obtained by co-designing algorithms, instruction/dataflow, memory hierarchy, parallel execution, and storage**—then implement the narrowest candidate that survives independent evaluation.

The unit of success is a useful, correct end-to-end workload with an explicit resource and lifecycle boundary. CPU speed, GPU throughput, RAM bandwidth, and SSD bandwidth are important diagnostics, not universal proxies for system value.

## Ownership and compose rules

This protocol composes existing canonical owners rather than introducing parallel systems:

- #5796 / #5798 / #7658 / #7681 / #7698: repairable computer capabilities, BOM/import dependence, and successor capability.
- #7515: workload-specific peak-memory budgets, resource envelope and graceful degradation.
- #5804 / #5805 / #5808 / #5818 / #5800: HDC/CfC workload baseline, exact HDC semantics, HDL/EDA pipeline, FPGA/fabrication evidence.
- #6968 / #6991 / #7003: typed GPU Fabric, backend identity, acceleration-claim boundaries and concrete resource synchronization.
- #7864 / #7866: end-to-end HDC retrieval and locality benchmark/receipt.
- Existing test/receipt and Mycelix capability/provenance owners retain their domains; no duplicate evidence ontology or closure kernel is introduced.

## Paradigm shifts

### 1. From peak specs to useful work per constrained resource

Evaluate the same workload across candidate configurations. Keep correctness, quality, throughput, p50/p95/p99 latency, memory capacity and movement, power/energy, area, repairability, cost and dependencies separate. Report the Pareto frontier; do not hide trade-offs inside one invented total score.

### 2. Treat memory movement and residency as architectural choices

Represent a task as a dependency graph of compute, memory and storage stages. Explore cache locality, batching, streaming, data placement, transfer count/bytes, queueing and locality-sensitive scheduling before adding arithmetic units. Include cold start and store/index construction as well as warm steady state.

### 3. Specialize only after a conventional baseline points to a bottleneck

CPU/SIMD, GPU, FPGA/soft-core, near-memory and storage-side execution remain competing hypotheses. Negative controls are mandatory; a specialized accelerator is allowed to lose. Only then investigate task-specific ISA instructions, local scratchpads, fused datapaths or controller policies.

### 4. Co-design hardware and software semantics without weakening them

Keep the canonical Rust implementation as the reference for exact operations. Any approximate representation is versioned and evaluated separately against an application-specific quality floor. Compiler decisions, fallback paths, device dispatch and data-layout changes are visible in the run receipt.

### 5. Include repairability, lifecycle and supply dependencies

A candidate is not just a die: consider board, package, cooling, interconnect, firmware, calibration, test, substitute parts and repair. Local assembly with imported chips is a legitimate, bounded achievement; it is not local semiconductor fabrication.

## Four-track design-space map

| Track | Near-term achievable targets | Later research | Do not claim without evidence |
|---|---|---|---|
| CPU | Host/SIMD baselines; open RISC-V core comparison; instruction/cache/data-layout experiments; formal properties and ISA compliance | Custom HDC/vector extension, custom controller core, externally fabricated mature-node IP | Desktop/server-class competitive performance from a small soft core; production silicon from generated RTL |
| RAM / memory | Measure working sets, cache behavior, bandwidth, access latency, batching and tier transitions; optimize software layout, reuse and bounded buffering | SRAM macro/cache/buffer RTL; controller/ECC experiments; compatible CXL memory system when actual hardware is available | A memory controller or SRAM generator means a DRAM die/fab exists; simulated CXL is physical evidence |
| GPU / parallel | CPU/SIMD versus actual GPU; kernel and transfer-inclusive timing; one HDC/parallel workload; use open GPU architecture as reference | Task-specific HDC accelerator; scheduler, scratchpad and DMA co-design; custom GPU extensions | CPU fallback/software Vulkan is GPU acceleration; one kernel proves general GPU superiority |
| Storage | Host storage baseline, read/write and random/sequential mixes, cold/warm runs, queue depth and persistence/recovery tests | Data layout/tiering/cache/controller experiments; real computational-storage capability where supported | Cache-only performance represents durable storage; software design proves custom flash hardware |

### CPU track

Use an established open processor as an independent reference rather than a fresh home-grown CPU as the only candidate. **Ibex** is appropriate as an embedded/control CPU baseline, not as a proxy for a desktop processor. Evaluate another high-throughput open RISC-V design only if the frozen workload justifies it.

Record CPU model/configuration, toolchain, ISA/feature profile, clock/power mode, code size, correctness, cycles/task where instrumentable, wall-clock latency/throughput, branch/cache counters where supported, synthesis area/timing for RTL candidates, and measured energy only with a declared instrument/boundary.

CoreMark/Embench or other appropriate established benchmarks can describe CPU behavior; frozen Symthaea workloads decide the project-specific result. ISA/formal compliance and ordinary task outputs are hard gates, not optional score bonuses.

### RAM / memory track

Start with **memory system engineering**, not an assumption that Symthaea can soon replace commodity DRAM manufacturing. Distinguish:
- representation footprint versus allocator/runtime overhead;
- cold-start versus warm-resident memory;
- sequential bandwidth versus random-access latency;
- peak transient memory versus steady working set;
- host DRAM versus GPU memory, on-chip SRAM/BRAM and nonvolatile storage;
- memory-capacity cliff versus compute saturation.

First compare bounded batches, streaming, buffer reuse, cache policy and explicit data layout against an unchanged semantic baseline. Then explore bounded SRAM/cache RTL or a controller policy under a named FPGA/PDK/EDA profile. For ECC/RAS, inject controlled faults and require correct detection/reporting; do not imply an ECC result from ordinary timing tests. CXL is a system profile only when compatible physical hardware exists.

### GPU / parallel track

Compose the existing GPU Fabric rather than build a competing runtime. Treat CPU reference, simulator, software Vulkan/lavapipe, actual GPU execution and synthesized/FPGA estimates as different evidence categories. Start with:
1. HDC Hamming/retrieval or another canonical binary workload;
2. one dense and one sparse vector workload;
3. a CfC/state-update path only if the production path and reference semantics are clear.

Report both kernel-only and full task latency; include dispatch, synchronization, transfer, staging and fallback. Record device, driver and runtime. A new GPU instruction, memory tile, queue or fused operation advances only when workload evidence identifies the bottleneck and exact semantic comparison remains valid.

### Storage track

Use the existing commodity device as the first baseline. Freeze corpus and cache state; record working set, read/write ratio, random/sequential pattern, queue depth, concurrency, filesystem or object layer, durability barriers, flush behavior and crash-recovery expectations.

MLCommons Storage v3.0 provides reference workloads for vector-database access and KV-cache read/write; its 2026 suite also expanded the access layer to S3 alongside POSIX. These are methodology/comparator references, not a claim of formal MLPerf compliance.

NVM Express ratified Computational Programs Command Set 1.3 on 31 July 2026, defining a standardized mechanism for programs to operate on data in an NVM subsystem. Investigate it as a near-data-computation direction; only run such a profile if an actual device supports the capability. Do not emulate support and present the emulation as physical storage hardware.

## Cross-layer benchmark matrix

### Frozen starter workloads

1. **HDC retrieval:** canonical 16,384-bit BinaryHV store, deterministic target queries, Hamming scoring and deterministic rank/tie rules; compose with #7864.
2. **Streaming/memory-bound workload:** sequential scanning at increasing working-set sizes with output checksum and exact output contract.
3. **Random-access workload:** deterministic pseudo-random index traversal / lookup, distinguishing warm-resident and cold working set.
4. **Storage persistence workload:** fixed corpus write/read/verify with declared flush durability, plus random and sequential access mixes.
5. **Parallel compute workload:** one canonical dense and one sparse operation, compared with CPU and actual GPU only when available.

Each workload manifest binds source commit/tree, fixtures and SHA-256, build features, compiler/tool versions, platform/device identity, input/output contract, warm-up, repetition, concurrency and measurement boundary. Synthetic tasks remain labelled synthetic until the task-level outcome is representative.

### Required metrics (separate, not collapsed)

- **Correctness/quality:** exact-reference agreement or preregistered task metric/floor; data integrity and recovery outcomes.
- **Latency:** p50/p95/p99 plus cold-start, setup, transfer and steady-state slices.
- **Throughput:** useful completed tasks/s, items/s, queries/s, IOPS or bytes/s with workload and queue depth named.
- **Memory:** peak RSS/VRAM, working-set/capacity, bytes moved, measured bandwidth, access-latency distribution, spill/tier-transition behavior.
- **Energy:** joules per successful/useful task and power only when measured with declared boundary/instrument; modelled estimates are separate.
- **CPU/GPU/RTL:** device/config, cycles/instructions/counters when available, kernel versus end-to-end time, RTL resources, synthesis area/fmax bound to exact flow/target.
- **Storage correctness:** flush/durability semantics, checksum errors, crash recovery, write amplification/endurance only where reliable telemetry/hardware supports it.
- **Lifecycle:** bill of materials, import/service dependencies, replaceability, repair procedure, calibration, test equipment, thermal/power limits and known-good fallback.

### Proposed promotion gates

All numeric values are proposed project gates—not achieved results, industry standards or universal truths.

- **Exact digital correctness:** 100% agreement on the admitted frozen corpus; zero silent truncation, stale results or accepted malformed frames.
- **Material end-to-end improvement:** at equal task quality, either >=1.5x throughput or >=30% lower *measured* joules/useful task, with p95 latency no more than 10% worse for the same serving scenario.
- **Repeatability:** at least 30 measured batches/configuration after preregistered warm-up where the metric permits, preserving raw observations, dispersion and exclusions. Long-duration durability tests use a duration/cycle profile appropriate to the claim.
- **Capacity/resilience:** pass declared memory pressure, out-of-space, device-loss, malformed-input and recovery controls; missing evidence is not zero failures.
- **CPU:** required ISA/interface compliance and independent correctness; use established CPU scores as a reference, not a replacement for Symthaea workload results.
- **RAM:** report bandwidth, latency, capacity and task results together; do not compare a smaller but semantically different workload as an improvement.
- **GPU:** actual device identity plus kernel and end-to-end evidence; CPU fallback, simulator and lavapipe cannot satisfy a physical acceleration gate.
- **Storage:** same integrity and durability contract, measured p95/p99 under declared workload/cache/queue conditions; never compare warm-cache-only results to cold full-stack results.
- **PPA and energy:** estimated PPA/energy must be labelled and pinned to tools/target. Physical measurements require direct observation with method/instrument and raw capture.
- **Pareto:** publish candidate losses and wins. Any numerical gate revision must occur after baseline and before candidate comparison.

## Staged execution without waiting for queued runners

### Stage 0 — inventory and baseline contract
Map existing repo code, benchmark owners, available local tools, and actual maintained workloads. Output exact path/owner/limitation inventory. No architecture performance claim yet.

### Stage 1 — host-software evidence
Use the existing Rust workload owners to collect CPU, memory-footprint, data-movement and storage baselines. A run is valid only for the source/configuration actually captured. This stage needs no custom silicon or FPGA.

### Stage 2 — candidate modelling
Use simple analytical estimates and candidate matrices to reject obviously dominated options. Label estimates; do not call them measurements. Change one architectural variable at a time and preserve the negative control.

### Stage 3 — RTL / simulation / formal
Choose one small CPU/controller/memory/accelerator element only after workload evidence supports it. Reuse the HDL/EDA and evidence bridge. Require semantic/reference tests and applicable formal properties before synthesis.

### Stage 4 — synthesis and FPGA
Record exact target/tool/constraints, area/resource use, timing, and the exact board/bitstream/firmware when physically executed. Compare full-system metrics against frozen baselines.

### Stage 5 — external fabrication, only when justified
Use the existing PDK/MPW/package/test evidence ladder. A design, GDS, clean DRC/LVS, tapeout submission, returned die and qualified component are distinct states. Foundry/packaging/material dependencies remain explicit.

## Research anchors (checked 17 October 2026)

- Compute Express Link Consortium: [CXL](https://computeexpresslink.org/) currently advertises CXL 4.0 with 128 GT/s signaling, bundled ports and enhanced memory RAS. This motivates a system memory-pooling profile where actual compatible hardware exists; not a local RAM chip design claim.
- [Vortex open GPU architecture](https://vortexgpgpu.github.io/) provides open GPU hardware/software research and FPGA support.
- [lowRISC Ibex](https://github.com/lowRISC/ibex) provides a configurable, widely verified open 32-bit RISC-V CPU with published performance/synthesis configurations.
- [NVM Express Computational Programs Command Set](https://nvmexpress.org/specification/computational-programs-command-set/) defines an execution interface for programs operating on data in an NVM subsystem (revision 1.3 ratified 2026-07-31).
- [MLCommons Storage v3.0 results](https://mlcommons.org/2026/09/mlperf-storage-v3-0-results/) (1 September 2026) includes vector-database and KV-cache test profiles and an S3 access layer alongside POSIX.

## Explicit non-goals

- Claiming competitive desktop/server CPU, general GPU, commodity DRAM or NAND dies can be manufactured from the current repository or by a small initial workshop.
- Using synthetic or simulated data as proof of production performance.
- Replacing existing HDC algebra, GPU Fabric, memory management, benchmark receipt or computer-closure owners.
- Choosing a hardware candidate before the conventional workload and claim boundary are frozen.
