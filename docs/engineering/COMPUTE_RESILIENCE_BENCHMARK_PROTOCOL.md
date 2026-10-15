# Compute Resilience: Task-Bound Memory Benchmark Protocol

**Status:** Research protocol linked to [COMPUTE-RESILIENCE-001 #7515](https://github.com/Luminous-Dynamics/symthaea/issues/7515). This document specifies how to measure memory use without hiding capability loss. It does not claim that any optimization has yet improved Symthaea.

## Thesis

The actionable near-term response to expensive/scarce RAM is not to claim that source code can replace DRAM fabrication. It is to maximize **verified useful work per byte, watt and dollar**, extend the serviceable life of existing machines, and preserve a longer-horizon open-hardware/silicon track.

Symthaea uses HDC/CfC-style representations rather than a conventional transformer-only stack, so LLM inference measurements such as bytes/token or KV-cache size are not directly transferable as primary outcomes. A recent survey of LLM inference argues that data movement and memory/I/O can dominate inference latency and calls for decomposed memory/bandwidth/energy metrics; that is useful as a measurement lesson, not proof about Symthaea's bottlenecks ([Chowdhury, 2026](https://doi.org/10.1007/s10462-026-11651-1)). This protocol uses **task-bound** metrics and profiles the actual implementation first.

## Phase 0 — choose real workloads and quality oracles

Select maintained, executable Symthaea workloads after confirming their current canonical owner, inputs and outputs. At minimum, the initial campaign should cover:

1. **Representation/encoding workload:** a named HDC encoding/binding/bundling or similarity task using a frozen corpus.
2. **Retrieval/working-set workload:** a named memory-search or knowledge-retrieval task with a reference result set and known corpus cardinality.
3. **Temporal workload:** a bounded CfC/LTC or other recurrent state-evolution task with a reference trajectory/acceptance criterion.
4. **End-to-end workload:** a real CLI/API or integration scenario that exercises the maintained product path rather than a synthetic allocation loop.

Do not invent crate names, commands or benchmark targets before inspecting the current default branch and owning module. For every workload, define what makes the output correct, useful or acceptable. If no defensible quality oracle exists, label that workload exploratory and do not make a capability-preservation claim.

## Phase 1 — freeze environment, inputs and baseline

A benchmark result is bound to all of:
- exact source commit; enabled Cargo features; build profile; compiler/toolchain; dependency lock; relevant build flags;
- operating system/kernel, CPU model/core configuration, physical RAM, GPU and VRAM if used, driver/runtime, and storage context at a level sufficient to reproduce the run without publishing personal identifiers;
- input corpus/source version, input sizes/cardinality, content digest, data provenance and expected-result/reference version;
- seed and determinism settings where meaningful; thread count, affinity/concurrency policy, power/performance mode, warm-up and cache-state protocol;
- the selected quality oracle, tolerances, rejection rules and missing-output semantics;
- measurement tool/version, sampling resolution and known blind spots.

Keep cold-start, warm-cache, steady-state and concurrent-pressure runs separate. Do not compare results from different source commits or materially different machine profiles as if one controlled variable changed.

## Phase 2 — measure resource use independently from task quality

### Required per-run observations

| Metric | Requirement |
|---|---|
| Peak host RSS | Measure for the complete process and state the measurement source/resolution; distinguish peak RSS from heap allocation and total system memory |
| Peak GPU memory | Record separately where a GPU path is used, including driver/runtime and what the selected counter includes |
| Working set / input scale | Record loaded corpus size, number and dimensions of represented items, cache contents and any retained prior state |
| Latency | Record cold-start and steady-state latency; report distributions/quantiles, not only the fastest run |
| Throughput | State exact units (items/s, queries/s, frames/s, etc.) and workload cardinality |
| Outcome quality | Independently recompute correctness, retrieval quality, trajectory discrepancy or task-specific acceptance; do not infer this from resource counters |
| Reliability | Count OOM, timeout, panic, rejected input, incomplete output and abstention distinctly |
| Energy | Include only when measured by a stated reliable meter/counter; do not infer energy from an unqualified power estimate |

A smaller RSS number is not success if the task silently skips cases, returns stale/cache-only data, truncates output or degrades below the predeclared quality requirement. Report missing or unmeasurable metrics as unknown.

### Representation accounting

Use source inspection plus measurement to explain memory rather than assuming a logical dimension is a byte count. For reference, a single 16,384-dimensional vector would occupy 2 KiB when bit-packed, 32 KiB at 16 bits per dimension, or 64 KiB at 32 bits per dimension before metadata/alignment/allocator overhead. Those are arithmetic illustrations only; they do not assert that any current Symthaea type uses one of these layouts.

For each implementation, record actual element representation, packing/alignment, temporary buffers, copies, indexes/metadata and number of simultaneously live vectors. Separate persistent corpus/index memory from query-temporary state. If a small per-vector representation multiplied by a large retained corpus dominates memory, the best experiment may be bounded indexing, eviction/streaming or selective materialization—not simply quantizing individual vector elements.

## Phase 3 — fair comparison sequence

Compare the current implementation against transparent variants, one change at a time:

1. **Baseline:** current code and configuration at the frozen source head.
2. **Budget-only profile:** explicit concurrency/batch/cache limits without semantic changes.
3. **Streaming/chunking:** bound transient buffers and process data incrementally where the algorithm permits it.
4. **Lifetime/copy reduction:** shorten unnecessary live ranges, reuse buffers or avoid redundant copies where ownership/semantics remain correct.
5. **Cache policy:** test bounded, lazy, incremental or evictable cache/index configurations; measure both working-set reduction and extra I/O/latency.
6. **Compact representation experiments:** try bit-packing, lower precision, sparsity or reduced dimensions only as separately versioned candidates with task-specific quality tests. Do not assume semantic equivalence from dimensional or numeric similarity.
7. **Combined profile:** only after effects of individual changes are recorded.

Use the same inputs, quality oracle, device constraints and comparison budget for baseline and candidate. Select workloads whose memory bottleneck is evidenced by measurement; do not optimize the most visible counter merely because it is convenient to change.

## Phase 4 — repeatability and analysis

For exploratory screening, run repeated trials (at least five as a minimum smoke check where the workload is stable and affordable) and record every trial. That minimum does not by itself qualify a statistical performance claim: final repetition count and uncertainty method must be chosen before confirmatory evaluation based on observed variability and experiment cost.

Report median and spread/interval for peak resource use and latency. Keep per-workload results separate before any aggregate. For a small change, compare paired runs on the same machine/configuration when possible. Freeze the final source, inputs and evaluator; no retrospective fixture or metric changes after results are inspected. Publish raw receipts or machine-readable run summaries with artifact digests, while avoiding secrets and unnecessary host identifiers.

## Required negative/adverse controls

- Input that exceeds the declared memory budget: explicit controlled failure/abstention, not unbounded allocation or success-shaped truncation.
- GPU path when no compatible device is available or allocation fails: explicit status and no silent CPU/GPU path change without recording it.
- Cold cache vs warm cache and concurrent execution pressure.
- Missing, corrupted, duplicated or incomplete input/state.
- Early termination or output truncation that would look like a success unless the result is checked.
- Compact/quantized representation with deliberately sensitive cases that expose a quality regression.
- Stale cache/index that returns older results while appearing cheap.
- OOM, timeout, panic and abstention hidden by averaging only successful runs.
- Mismatch between reported and independently recomputed input/output cardinality.

## Go/no-go criteria

A memory optimization may be called an improvement only if:
1. The exact baseline and candidate builds/inputs are reproducible.
2. At least one predeclared memory metric improves by a stated amount/interval on a named workload.
3. Output quality/correctness meets the frozen oracle/tolerance; missing cases and errors remain visible.
4. The latency, throughput, reliability and energy trade-offs are reported, not omitted.
5. The result survives an independent recomputation of the metrics/receipt and relevant adverse controls.
6. No generalization is claimed beyond the evaluated workload, device profile, corpus and configuration.

If no candidate preserves required quality while reducing the target resource cost, the correct result is a negative finding, not a forced win. Return **insufficient evidence** or a specific next experiment when the quality oracle or workload representativeness is unresolved.

## Longer-horizon computer hardware resilience

This protocol is the software/runtime layer. It composes [COMPUTE-MFG-001 #5796](https://github.com/Luminous-Dynamics/symthaea/issues/5796), the existing productive-closure ladder:
- favour inspected and tested replacement/refurbished modules and repairable designs;
- keep exact device/part revision, compatibility and test evidence for recovered memory/hardware;
- separate local enclosure/PCB/assembly/repair from imported CPU/GPU/DRAM dies;
- keep the silicon path distinct: RTL → formal/FPGA validation → synthesis/PPA/DRC/LVS → external foundry/MPW → package/test/board qualification.

OpenRAM is an open-source **SRAM** compiler, not a drop-in DRAM fabrication path ([OpenRAM](https://openram.org/)). Open-source EDA, RTL and PDK work can reduce design and verification barriers, but does not supply the specialized process integration, yield, equipment, materials or scale needed to manufacture modern commodity DRAM. See the [Google/SkyWater SKY130 repository](https://github.com/google/skywater-pdk), which describes its PDK as an experimental preview not intended for production use.

## Research references

- Chowdhury (2026), [I/O for LLM inference: a survey of storage and memory bottlenecks](https://doi.org/10.1007/s10462-026-11651-1). Memory/I/O measurement ideas; not directly transferable to Symthaea's non-transformer workloads.
- [Peak-memory-aware partitioning and scheduling for multi-tenant DNN inference](https://doi.org/10.1016/j.sysarc.2026.103696) (2026). Evidence that peak-memory scheduling is an active systems area; its results require independent application-specific comparison.
- [OpenRAM](https://openram.org/): SRAM compiler scope.
- [Google/SkyWater SKY130 PDK](https://github.com/google/skywater-pdk): experimental-preview status.
- [COMPUTE-RESILIENCE-001 #7515](https://github.com/Luminous-Dynamics/symthaea/issues/7515): execution tracking.

## Nonclaims

This protocol reports no current Symthaea peak-memory result, no measured improvement, no universal low-memory compatibility, no proof that a compact representation preserves every HDC/CfC task, and no claim of local DRAM manufacturing. It defines the experiment needed to establish any of those claims.
