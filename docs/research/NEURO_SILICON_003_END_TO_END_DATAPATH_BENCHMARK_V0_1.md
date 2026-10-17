# NEURO-SILICON-003: End-to-End HDC Retrieval and Data-Locality Benchmark v0.1

**Status:** research protocol; proposed targets, not measured results  
**Tracking:** [Symthaea issue #7864](https://github.com/Luminous-Dynamics/symthaea/issues/7864)  
**Owners to compose:** #5804, #5805, #5808, #5818, #5800, and the existing streaming RTL PR train

## 1. Research question

When does an HDC accelerator improve a complete Symthaea workload once memory movement, invocation/setup, data residency, result reduction, and task correctness are included?

This protocol tests a shift from *kernel-first optimization* to *workload-and-dataflow co-design*. A fast XOR or popcount datapath is useful only if the application can feed it efficiently and the whole workload benefits.

The initial target is exact 16,384-bit `BinaryHV` retrieval. One hypervector occupies 16,384 bits = 2,048 bytes = 2 KiB. Higher-dimensional continuous HDC-LTC state is a different workload and must not be inferred from results here; compose with #7749 and the existing HDC-LTC trajectory/resolution evidence instead.

## 2. Paradigm shifts and falsifiable hypotheses

### H1 — move compute to data

Compare repeatedly transferring vectors to a device against a profile that keeps a frozen vector store in local memory and moves the query to the store. Include every transfer and invocation in the end-to-end boundary. The hypothesis is locality helps only once avoided movement exceeds residency/setup costs.

### H2 — optimize energy per correct answer

Keep correctness / retrieval quality, tail latency, throughput, power, energy/query, capacity and update cost as separate metrics. Do not collapse these into one opaque “neuromorphic score” or use TOPS/W to imply application-level advantage.

### H3 — co-design dataflow, not just datapath width

After the current stream interfaces are qualified, compare lane width together with query batching, local-memory residency, pipelining, backpressure, and fusion of adjacent already-qualified operations. Lane width alone is not an architecture conclusion.

### H4 — heterogeneous compute is allowed to win

Keep canonical CPU and optimized CPU/SIMD baselines. Add a GPU baseline only when the same workload and measurement boundary can be run. Explicitly expect tiny working sets or transfer-bound workloads to favor ordinary compute.

### H5 — exact and approximate hardware are separate experiments

The initial comparison is exact. Analogue / memristive / reduced-precision profiles must have an explicit representation version, independent reference and error accounting, declared task-quality floor, and their own results. Never use approximation to silently change the exact BinaryHV oracle.

### H6 — the evidence receipt is part of the artifact

Every run binds code, workload, fixtures, device/profile, tools, measurement method and raw outputs. Simulated, synthesized-estimated, physically measured and modeled quantities have distinct labels. Missing evidence is incomplete, not zero and not a pass.

## 3. Canonical workload and scope

The workload boundary is:

```text
frozen query/input
  -> canonical query encoding
  -> optional existing bind / permutation
  -> retrieve against a frozen vector store
  -> exact Hamming/similarity reduction
  -> deterministic ranked or thresholded result
```

Use the existing Rust `BinaryHV` definition and canonical source-bound fixture generator. Do not write a second HDC algebra to simplify hardware tests. The result contract must define deterministic tie-breaking by stable item ID. If a supported production retrieval path is not yet callable from a benchmark, declare that missing adapter and begin with a source-bound synthetic workload; label its findings as synthetic.

**First milestone:** a runnable host-only baseline and receipt validator, useful without RTL CI or a physical board. It must not claim an accelerator result. Existing semantic and differential gates remain prerequisites for adding corresponding hardware profiles.

### Comparison profiles

| ID | Profile | Include in end-to-end cost |
|---|---|---|
| P0 | Canonical Rust CPU oracle | Query preparation, scan/retrieval, reduction, result |
| P1 | Optimized ordinary CPU path, including SIMD where available | Same input/output semantics and all host work |
| P2 | Per-operation accelerator offload | All copies, synchronization, invocation and result transfer |
| P3 | Persistent/local-memory accelerator | Store load/residency cost, query transfer, reduction and returned result |
| P4 | Batched/streaming profile | Queueing, backpressure, batch formation, drain and per-query tail latency |
| PG | GPU | Only when a real matching hardware profile and complete comparable run exist |

Do not infer GPU or FPGA results from a simulator. Do not compare a warm-resident accelerator run with a cold CPU run without publishing both cases.

### Initial screening matrix

Hold the vector width at canonical 16,384 bits initially. Sweep store sizes of **64, 256, 1,024 and 4,096 vectors** and query batch sizes of **1, 4, 16 and 64** for host baselines. Each vector store represents 128 KiB, 512 KiB, 2 MiB or 8 MiB before metadata and alignment. Record actual memory tier; never presume that all of it fits in FPGA BRAM or on-chip SRAM.

Reuse existing lane profiles **64, 128, 256, 512 and 16,384 bits**. Initially screen representative, preregistered lane/memory/batch combinations; do not blindly run the entire Cartesian product if it adds no decision value. Expand the sweep only after the host baseline identifies the bottleneck.

Declare and preserve:
- cold start vs warm resident state;
- store-build/load time vs amortized steady-state query time;
- single-query latency vs batch throughput;
- whether queries are independent or re-use cached/store state;
- supported update/adaptation path and update cost, if updates are in scope.

## 4. Required metrics

Report these independently; omit a measurement only with an explicit reason.

### Task and semantics
- exact output match against the canonical oracle, fixture count and mismatch count;
- deterministic ranking/tie behaviour;
- retrieval/classification quality on a named, frozen task corpus if one is admitted;
- malformed frame, truncation, stale result and backpressure outcomes;
- update/learning cost if the profile supports online updates.

### Performance
- end-to-end p50, p95 and p99 latency with raw observations;
- steady-state throughput and query/batch count;
- startup/compile, store load, synchronization, device transfer and result-reduction costs;
- queue depth, stalls/backpressure and host/device bytes moved;
- memory capacity, working-set size, memory tier/bandwidth and residency assumptions.

### Hardware and energy
- LUT/FF/BRAM/DSP and fmax only with an identified synthesis/implementation flow and target;
- board power and joules/query only with a named meter, calibration/state, sampling procedure, and declared measurement boundary;
- estimated power/energy as a separate estimated metric with model/tool identity;
- thermal/frequency/power mode and clock settings when available.

### Reproducibility
- repository, commit SHA, tree SHA when available, workload and fixture-manifest SHA-256;
- compiler/toolchain, simulator, synthesis tool and versions as applicable;
- hardware model/board, memory configuration, runtime/driver, clock/power profile;
- exact command, exit status, warning output and raw result-artifact SHA-256;
- warm-up count, measured-batch count, run ordering and exclusions.

## 5. Proposed targets and promotion gates

These are **project targets to preregister, not achieved results or universal standards**.

1. **Exactness:** 100% oracle agreement on the admitted frozen fixture set; zero accepted malformed/truncated protocol frames. Include boundaries, fixed-seed random cases and negative controls.
2. **Boundary completeness:** every workload stage and every host/device transfer in the declared serving path is either timed or explicitly listed as outside the claim. No hidden precomputation.
3. **Repeated observations:** at least 30 measured batches per configuration after a declared warm-up, preserving raw observations and dispersion. Interleave candidate order where practical to expose thermal/drift effects. A batch is not necessarily one independent query; say which it is.
4. **Material advantage:** only promote a profile as materially better when equal-quality measurements on the same hardware/measurement boundary show **either** at least 1.5× end-to-end throughput **or** at least 30% lower *measured* joules/query, while p95 latency is no more than 10% worse for the same serving scenario. Publish the full Pareto table, including profiles that lose.
5. **Approximate profiles:** decide the task-specific quality floor before testing; report error distribution and all failure cases. Never blend approximate and exact results in a mean.
6. **Resource claims:** synthesized area/timing and estimated power must name their target and tool. Physical FPGA measurements require a board-observation receipt. None implies ASIC readiness or fabricated silicon.

If initial host baselines show that these numerical promotion gates are inappropriate, revise them **before** evaluating accelerator candidates and preserve the rationale/version history.

## 6. Mandatory negative controls

- Small store / small batch where accelerator invocation cost should erase any kernel benefit.
- Transfer-dominated operation where a faster primitive does not improve end-to-end latency.
- Working-set capacity cliff when the store spills from on-chip memory to off-chip DRAM.
- Batching that improves aggregate throughput but worsens p95/p99 single-query latency.
- Ordinary SIMD or GPU profile that outperforms a specialization on dense/vectorizable work.
- malformed/truncated stream, output backpressure, reset, and frame-boundary cases that must not emit partial or stale results.
- a receipt with modeled energy mislabeled as measured energy, which must be rejected by validation.
- missing/mismatched source or fixture identity, missing device, and skipped/queued tool execution, which must yield `incomplete` or `not_comparable`, never `pass`.

## 7. Receipt contract and status semantics

The schema lives at [`end_to_end_receipt_v0_1.schema.json`](../../crates/domains/symthaea-silicon/schemas/end_to_end_receipt_v0_1.schema.json). The example file is intentionally an **incomplete planning receipt**; it contains no performance or energy claim.

Schema v0.1 covers exact digital **end-to-end workload runs** on CPU/SIMD, GPU, or physically observed FPGA. Lower-level RTL differential and synthesis qualification remain owned by #5818 and the existing silicon PR train; future approximate/analogue hardware needs a separate contract/version rather than silently weakening exact-result rules. A `pass` receipt means the declared run/evidence contract was satisfied; it does **not** by itself prove a candidate beats the CPU baseline or crosses the material-advantage gate.

- `pass`: the declared checks ran on the exact profile/input identities, all required invariants met and evidence is complete for the claim ceiling.
- `fail`: a required check actually ran and failed, such as an oracle mismatch or a prohibited malformed frame accepted.
- `incomplete`: required evidence/execution is absent, queued, skipped or unfinished.
- `not_comparable`: identities, workload boundary, units, device profiles, or measurement conditions do not support the requested comparison.

Missing metrics are absent—not zero. A successful schema parse verifies only receipt structure, not hardware behaviour.

## 8. Staged execution without runner dependency

| Stage | Deliverable | Claim ceiling |
|---|---|---|
| A | Freeze workload/manifest; implement host-only oracle baseline + receipt validator | Software baseline only |
| B | Run adversarial fixture and determinism checks locally | Only the commands actually executed |
| C | Attach already-qualified exact RTL profile and differential receipt | Simulator correctness only |
| D | Named synthesis profile and resource/timing report | Synthesis estimates only |
| E | Board-level FPGA execution with direct power/energy capture | That particular board/workload/measurement profile |
| F | Separate approximate or analogue experiment | Only the declared representation/device/corpus profile |

Work on stages A/B while upstream workflows are queued. Do not alter `--locked`, relax semantic acceptance, or mark CI as passed to make progress.

## 9. Research anchors

- Huang et al., [“Hyperdimensional in-memory computing with analogue memristive crossbar arrays,” *Nature Communications* 17, 9162 (2026)](https://www.nature.com/articles/s41467-026-76067-5), published 28 July 2026. The study demonstrates HDC language identification on a memristive system-on-chip and discusses peripheral/ADC costs and hardware-aware encoding. It makes locality/co-design worth testing; it does **not** establish an advantage for Symthaea's workload or device.
- [MLPerf Tiny](https://mlcommons.org/benchmarks/inference-tiny/) and [MLPerf Edge](https://mlcommons.org/benchmarks/inference-edge/) provide useful principles: frozen task/quality targets, explicit latency scenarios, and careful energy methodology. This protocol reuses those principles without claiming MLPerf compliance.
- Existing Symthaea dependencies and evidence contracts are linked above. This document adds no new HDC semantics or silicon/fabrication claims.

## 10. Non-goals

- A replacement for #5805's conventional-baseline contract, #5808's HDC semantic contract, #5818's HDL/EDA pipeline, or #5800's physical evidence ladder.
- A new cognition architecture, duplicate BinaryHV implementation, speculative universal accelerator, or single aggregate score.
- A claim that generated RTL is synthesized, an FPGA is physically present, energy was measured, or custom silicon exists merely because source/configuration files were added.
- A dependency on completion of queued CI before the host-only protocol work can proceed.
