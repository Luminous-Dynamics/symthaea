# QARCH-022: Memory topology and state-placement contract

**Status:** typed Rust validator + schema proposal  
**Version:** `symthaea.quantum_comp.architecture_memory_topology.v1`  
**Parent evidence model:** QARCH-020 receipts (#9305)  
**Workload-side descriptor:** QARCH-021 regimes (#9317)  
**Canonical execution owners:** HDC retrieval (#7864), HDC-LTC (#7749), manifold dynamics (#3348), memory resilience (#7515)

## 1. Design thesis

Memory must be represented as a **multi-tier directed topology plus logical state placement**, not reduced to one working-set number or one bandwidth scalar.

A workload declares its state demand. A candidate architecture declares its memory regions, transfer paths, and placement of those state classes. QARCH compares the two only when both are bound to the same workload fingerprint, observation, precision contract, and measurement boundary.

This separation is essential:
- A workload does not declare that it uses HBM, SRAM, or a GPU cache.
- A candidate does not get to redefine the workload's logical state or quality criterion.
- A memory region is not a compute architecture family.
- A communication technology is not a compute substrate. Co-packaged optical links remain interconnects; photonic matrix compute remains a separate compute family.
- A topology description is not proof that the device actually achieved its advertised bandwidth or energy.

## 2. Three-layer model

### A. Workload state demand (QARCH-021 side)

State classes should represent semantically distinct live data:
- HDC codebook / vector store;
- active fast continuous state;
- compact slow context;
- retrieval index;
- event/input queue;
- model parameters;
- scratch or temporary state;
- snapshots/checkpoints.

A workload-side state-demand record may include live-set bytes, lifetime, reuse-distance distribution, reads/writes per unit of work, update cadence, access pattern, consistency requirement, and reconstructibility. Values are optional unless the workload contract requires them. Unknown is absent, never zero.

A future extension to QARCH-021 should add this as workload demand; do not insert architecture-specific region assignments into the regime profile.

### B. Candidate memory topology (this RFC)

A topology contains three typed collections.

**Memory regions (nodes)** describe where bytes can reside:
- tier/medium: registers, scratchpad, L1/L2/L3, SRAM, HBM, DDR/LPDDR, CXL-attached memory, persistent storage, remote memory, or a named other tier;
- scope: thread, core, cluster, package, device, host, rack, or remote;
- optional capacity and occupancy/high-water intervals; occupancy is the existing reserved/background allocation excluding logical state placements in this record, so it can be added to the placements when proving a guaranteed fit;
- directional read/write access latency and sustained bandwidth;
- optional directly measured energy per byte;
- optional access granularity, persistence/volatility, and coherence model.

**Transfer edges** describe movement between regions:
- stable edge ID and source/destination region IDs;
- direction and fabric;
- the QARCH-021 communication-boundary ID that accounts for the physical bytes crossing that boundary;
- transfer/conversion/serialization costs where separately observed;
- no duplicated accounting of one transfer at multiple abstraction layers.

**State placements** map logical state classes to one or more regions:
- state-class ID and target region ID;
- pinned, cache-managed, streamed, replicated, spillable, or reconstructed mode;
- resident bytes, access/update rates and required consistency/precision contract;
- migration, promotion/demotion and cold-load cost when measured.

Multiple placements for one state class are valid for replication or tiering, but their roles and byte accounting must be explicit. A state class may not be silently assumed to be resident in every tier.

### C. Evidence receipt (QARCH-020 side)

The topology is architecture evidence and must bind to:
- exact architecture ID and observation record;
- exact workload fingerprint and comparison-contract fingerprint;
- source artifact and extraction provenance;
- the QARCH-020 evidence receipt;
- device, software, toolchain and measurement boundary already required by that receipt.

A schema-valid description is not a measured result. Published specifications, analytical estimates, simulation, synthesis, and physical measurements remain distinct evidence origins.

## 3. Graph invariants and accounting

1. Every node ID is unique; every edge endpoint and placement region reference must resolve.
2. Edge direction must be explicit. A route is an ordered sequence of edges; it must connect the declared source and destination without inventing an implicit hop.
3. The overall topology may contain cycles (for example, coherent cache fabrics). A *declared route* must be finite, ordered and internally connected; cycle rejection applies only where the route or algorithm contract requires an acyclic path.
4. Region capacity and observed occupancy are distinct quantities. Unknown capacity is not unbounded capacity.
5. Capacity arithmetic is checked; report capacity headroom and high-water marks rather than assuming all declared regions are simultaneously available.
6. One physical transfer is counted once at its named boundary. A boundary may have one read edge and one reverse-direction write edge to describe distinct full-duplex traffic; duplicate edges in the same direction or overlapping accounting are rejected. A host read that fetches a line from DRAM through cache hierarchy must not be added again as though each level transferred the full line independently unless the actual bytes at each boundary were separately observed.
7. Capacity admission adds the region's existing/background occupancy (excluding the named placements) to the aggregate live bytes of the placements. Guaranteed residency requires known capacity, known background occupancy and known live-set bounds; an unknown occupancy cannot be treated as zero.
7. Cache hit rate, reuse distance and residency claims require trace-backed or instrumented evidence. A synthetic trace can validate code and schema, not certify hardware behavior.
8. Preserve p50/p95/p99 latency or a raw distribution where possible. A mean cannot hide queueing and tail stalls.
9. Cold load, warm steady state, concurrent pressure, capacity overflow, eviction, spill, migration, fallback, and recovery are separate scenarios.
10. Energy per correct answer must include the same end-to-end boundary for all candidates: compute, memory, host, fabric, conversion and required post-processing. If a component is omitted, the omission remains visible.
11. State placement must preserve the workload's representation, precision, consistency and output contract. Approximate/analogue memory is a separately versioned profile with a task-quality floor.
12. The topology contract does not enable runtime migration, policy learning, or architecture routing.

## 4. First controlled experiment matrix

Reuse existing owners, not another runner.

| Workload owner | Controlled memory regimes | Main falsifier |
|---|---|---|
| HDC retrieval #7864 | small resident store; fits local SRAM/BRAM; exceeds local capacity; host DDR/HBM spill; cold upload vs pinned residency; sequential vs random/gather | If CPU/SIMD wins for small stores or transfer cost erases kernel gains, retain CPU |
| HDC-LTC #7749 | full-width state; fast state plus compact slow context; slow state in local memory vs host memory; different update cadences | If context does not improve long-horizon retention/recovery at equal cost, reject it |
| Manifold dynamics #3348 | full-width vs compressed state; projection/lift and residual path measured separately | If residual traffic recreates full-width movement every step, compression has not solved locality |
| Memory resilience #7515 | cold/warm cache; concurrent pressure; OOM/eviction; checkpoint/recovery; constrained RAM/VRAM | If a budgeted mode drops state or emits incomplete output as success, reject it |

Use exact CPU scalar and optimized SIMD baselines. Include GPU, FPGA and neuromorphic candidates only where the same task, precision and end-to-end boundary can actually be run. PIM, near-memory, CXL and optical links remain separate topology variants, not assumed winners.

## 5. Metrics and promotion

Report independent columns:
- exact result agreement, dynamical trajectory error and task quality;
- end-to-end p50/p95/p99 latency and throughput for declared batch/streaming conditions;
- reads/writes by logical state class and bytes crossing every named boundary;
- cache/residency rate, reuse distance, queue/stall time, migration and spill/fallback count;
- resident, peak and high-water memory by tier;
- cold-start, warm-start, checkpoint/restore and recovery time;
- joules per correct retrieval/update/sequence, only from a directly metered comparable full path.

For HDC retrieval, reuse the existing proposed target from #7864 as a preregistered hypothesis, not a result: at equal correctness/task quality, aim for either at least 1.5× end-to-end throughput or at least 30% lower directly measured joules/query, while p95 latency is no more than 10% worse in the stated serving scenario. Always retain regimes where CPU is the correct choice.

## 6. External benchmark alignment

- **NeuroBench** provides the right separation between hardware-independent algorithm correctness/complexity and deployed system task-level timing/efficiency. Its system-track principle is particularly important: include the whole system path, not just a kernel.
- **MLPerf Tiny** gives constrained-edge comparisons fixed tasks, quality targets, latency and energy per inference, while allowing stack-level optimization. Reuse the discipline, not an unearned compliance claim.
- The 2026 dual-memory-pathway SNN work motivates the fast/slow state hypothesis, but its model and hardware results are not Symthaea results.
- 2026 co-packaged-optics research is relevant to communication boundaries, conversion, thermal and packaging costs; it does not imply photonic compute.

## 7. Acceptance sequence

**Stage 0 — schema and fixtures**
- JSON Schema v1 and synthetic positive/negative fixtures.
- Semantic validation tests for duplicate IDs, missing endpoints, disconnected routes, prohibited route cycles, double counting, capacity overflow, absent/unknown metrics, invalid units, placement mismatch and source provenance.
- Synthetic values are explicitly labelled and never enter measured architecture rankings.

**Stage 1 — typed Rust model (initial implementation in `architecture_memory.rs`)**
- Typed memory region, transfer edge, explicit route and state placement structs.
- Structural checks for unique IDs, valid endpoints, route connectivity, optional acyclicity, non-overlapping boundary accounting (with one explicit reverse-direction read/write pair allowed), interval domains and capacity feasibility including background occupancy.
- Cross-record validation against the QARCH-020 evidence record and QARCH-021 workload-regime profile, including exact observation/architecture/workload/contract/precision identity.
- Synthetic fixtures are rejected by the comparison validator. Unknowns remain absent; invalid or incomplete evidence fails closed.

**Stage 2 — one canonical consumer**
- Integrate with an existing #7864 or #7749 run without creating a new runner.
- Preserve raw output, exact source/fixture/device identity and QARCH-020 receipt.

**Stage 3 — measured comparison**
- Report per-regime Pareto frontiers and confidence/coverage; do not collapse to a universal “best architecture.”
- Promote only when correctness/quality and hard constraints pass, with a measured gain on identical full-path boundaries.

## Current implementation boundary

The typed API is an initial contract validator, not a hardware collector or benchmark runner. It validates declared topology records and joins their identities to existing QARCH records. The schema and fixture values remain synthetic; no physical memory bandwidth, occupancy, cache hit rate, latency, or energy is asserted. Integration with the existing HDC retrieval or HDC-LTC run owners is still required before any comparison result can be produced.

## References

- NeuroBench (Nature Communications, 2025): https://www.nature.com/articles/s41467-025-56739-4
- MLPerf Tiny v1.4 results (MLCommons, July 2026): https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/
- Dual memory pathways (Nature Machine Intelligence, June 2026): https://www.nature.com/articles/s42256-026-01255-3
- Co-packaged optics for HPC and AI (Nature Electronics, August 2026): https://www.nature.com/articles/s41928-026-01681-6

## Non-claims

This contract does not show that HBM, SRAM, CXL, PIM, near-memory, neuromorphic or optical systems are universally superior; that slow context improves HDC-LTC; that 128K/256K dimensions are better; or that any hardware has been measured. Those claims require source-bound runs under the existing benchmark owners.
