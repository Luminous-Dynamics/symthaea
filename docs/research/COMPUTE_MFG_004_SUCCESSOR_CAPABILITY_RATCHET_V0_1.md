# COMPUTE-MFG-004 — Successor capability ratchet benchmark (v0.1)

**Status:** draft synthetic benchmark contract; no real successor gain has been measured  
**Tracking:** [Symthaea COMPUTE-MFG-004 #7698](https://github.com/Luminous-Dynamics/symthaea/issues/7698)  
**Profile projection:** [COMPUTE-MFG-003 #7681](https://github.com/Luminous-Dynamics/symthaea/issues/7681) / draft [PR #7682](https://github.com/Luminous-Dynamics/symthaea/pull/7682)  
**Reference corpus:** `crates/domains/symthaea-capability-closure/fixtures/successor-capability-ratchet-v0.1.json`

## Why move beyond replication?

Replicating an existing computer is a useful bootstrap milestone, but identical copies do not establish that the next generation is more capable. A successor may improve one metric while quietly losing offline recovery, exact semantics, latency guarantees, repairability, or test coverage.

The next unit of progress is therefore a **capability ratchet**: each successor claims only the specific capability it actually demonstrates; protected prior capabilities remain intact; regressions and trade-offs stay visible; and the next experiment is chosen from the evidence-backed Pareto frontier rather than one universal score.

This composes canonical domain owners:
- #5782 owns multi-generation capability-envelope comparison.
- #5783 owns capability-frontier/Pareto planning.
- #6228 owns blocker-to-engineering/materials feedback.
- #5805 owns conventional workload baselines.
- #5808 owns exact HDC RTL semantics.
- #5800 owns the FPGA-to-external-MPW evidence bridge.

No second closure, HDC algebra, capability-envelope or Pareto engine is proposed.

## Two tracks up the chain

### Track A — improve the cell's ability to make and qualify things

1. Reproduce a modest computer/controller under C4/C5 only after witnessed host-removal evidence.
2. Add a qualified test/programming fixture and revision-bound acceptance receipts.
3. Improve repair, replacement and calibration renewal under a declared profile.
4. Add measured capability for board assembly or selected local mechanical parts.

These can increase useful production capability even when the CPU/SoM, memory and bare PCB remain imported.

### Track B — make a more capable compute successor

1. Freeze exact Symthaea HDC/CfC workloads and the CPU baseline (#5805).
2. Preserve exact `BinaryHV` semantics in the small RTL core (#5808).
3. Simulate/formally check; synthesize; evaluate resource/clock profiles.
4. Observe the design on a physical FPGA board; compare end-to-end workload performance, including host/device overhead.
5. Only if measured improvement survives the hard constraints, investigate an external mature-node ASIC route using the pinned design flow and #5800's evidence gates.
6. Separately qualify fabrication receipt, packaged-device tests, and system integration. These still do not imply local semiconductor-fab closure.

The open-source [LibreLane flow](https://github.com/librelane/librelane) is a current candidate for reproducible RTL-to-GDS implementation. Pin its exact version, PDK, constraints, tools and every output; use only the portion supported by the target profile. NIST's 2026 semiconductor manufacturing review highlights that process control, in-line metrology, advanced packaging and the wider manufacturing ecosystem remain central, so the long-term path is not merely "generate GDS and the rest is solved" ([NIST/CIRP review](https://doi.org/10.1016/j.cirp.2026.04.115)). NASA TRL guidance is also a useful reminder to distinguish concept, breadboard validation, prototype and operational evidence ([NASA TRL](https://www.nasa.gov/directorates/somd/space-communications-navigation-program/technology-readiness-levels/)).

## Frozen synthetic corpus

The JSON corpus holds **13 synthetic cases**. Every number in it is an invented example for testing the benchmark contract, not an actual measurement.

Coverage:
- an identical clone adds reproduction capability but no compute-performance gain;
- a test fixture improves qualification capability without implying faster computing;
- exact HDC RTL simulation provides correctness evidence only;
- a faster accelerator kernel can lose end-to-end because data transfer and setup dominate;
- a measured end-to-end HDC gain may pass the proposed win gate;
- a throughput gain that breaks protected p99 latency is rejected or conditional;
- synthesis estimates do not become measured board power;
- GDS/MPW submission is not fabricated silicon;
- externally fabricated/tested silicon is not local fab closure;
- stale metrology blocks qualification;
- losing protected offline recovery blocks unconditional successor promotion;
- unknown tooling/inventory remains unresolved;
- reproducible RTL artifacts do not imply physical hardware or performance.

The corpus preserves independent values: workload utility/correctness, throughput, latency p50/p95/p99, energy, memory, FPGA resources/temperature, capability gain/loss, external dependencies, build outcomes, repairs, and unresolved evidence. It does not sum them.

## First accelerator win gate (proposal, freeze before measuring)

For a single 16,384-bit HDC workload:
- zero unexplained exact-output mismatches across the frozen test corpus;
- compare against the exact #5805 baseline on the same workload and input;
- count end-to-end device transfers, preprocessing, batching, and output conversion;
- preregister one or more target objectives;
- initial hypothesis: at least **2× end-to-end throughput OR 30% lower measured system energy** for equivalent task output, with every protected latency, thermal, resource/area and correctness constraint still passing;
- report a negative result if the thresholds are not met.

The 2×/30% thresholds are proposed for this experiment, not universal scientific criteria. They must be frozen before candidate measurements or revised transparently with justification. Estimated power from a synthesis report is not system-energy measurement.

## Promotion rules

A successor can earn one of these bounded classifications:
- `reproduction_gain_only`;
- `accepted_capability_gain`;
- `design_correctness_only`;
- `measured_workload_advantage`;
- `external_design_submission_only`;
- `externally_fabricated_hardware_evidence`;
- `no_system_level_acceleration_advantage`;
- `projected_only`;
- `reproducible_design_artifact_only`;
- `unresolved`;
- `reject_or_conditional_pareto_only`.

The label is not a single readiness score. It names what the evidence supports. Actual frontier and envelope decisions should be derived by canonical #5782/#5783 owners.

## Required evidence receipt

Every candidate measurement should bind:
- baseline and candidate source SHA;
- exact workload/input corpus SHA;
- board/module revision;
- compiler, RTL tools, FPGA toolchain or PDK version;
- constraints and configuration;
- raw logs, output/artifact hashes and warnings;
- CPU/GPU/FPGA settings and measurement boundary;
- exact-vs-approximate arithmetic designation;
- thermal/latency/resource hard constraints;
- import and external-service dependencies;
- accepted builds, failed attempts, rework and operator interventions;
- independent review disposition and unresolved claims.

## Qualification boundary

The corpus is an E0 synthetic contract. It has not been evaluated by a Rust gate implementation, FPGA device, ASIC shuttle or packaged chip. No measured accelerator improvement is claimed by adding this file. Real successor promotion requires the exact canonical baseline, frozen RTL fixtures, verified hardware measurements and independent receipts.

## Research calibration update — 2026-10-17

Two external references sharpen this benchmark's measurement rules and establish a credible research fork beyond exact digital logic.

- **MLPerf Tiny v1.4** is a useful benchmark-method reference: compare the same task/data/quality target, keep latency and task quality separate, and state the energy measurement boundary. Its 2026 results note that some entries measure only the inference engine while others include startup, peripherals, firmware and I/O; those values are not interchangeable. Sources: [MLPerf Tiny v1.4 results](https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/) and [MLPerf Inference: Tiny benchmark overview](https://mlcommons.org/benchmarks/inference-tiny/).

- **Physical analogue HDC is a separate, promising track.** A Nature Communications paper published 28 July 2026 reports a memristive in-memory HDC system for 21-language classification; its reported simulation accuracy (98.87%) and hardware result (95.24%) differ, showing why hardware nonidealities and hardware-aware processing matter. That paper's dimensions, task, representation, device and measurement boundaries are not Symthaea results and must not be copied into this project's synthetic corpus as if measured here. Source: [Nature Communications (2026)](https://www.nature.com/articles/s41467-026-76067-5).

### Benchmark-design consequences

1. **Maintain two qualification tracks.** Exact digital HDC requires bit-identical BinaryHV outputs. Approximate / analogue / in-memory computing needs a separately versioned representation, preregistered task-quality floor and explicit robustness/noise/calibration criteria; it cannot inherit an exact-equivalence PASS.
2. **Freeze task and data before tuning.** Bind the source revision, input corpus, data split, preprocessing/postprocessing, target quality and measurement protocol. Keep a closed comparison track with fixed semantics and a separate innovation track where representations may change only under explicit utility constraints.
3. **Run a dimension/resource sweep.** Treat 1,024-, 4,096- and 16,384-bit profiles as different experiments. The 1,024-bit starting point is a useful comparator inspired by the external study, not a claim that its model is interchangeable with Symthaea. Measure any task-quality loss alongside memory, data movement, latency, energy and robustness.
4. **Measure the whole deployed path.** Report core-only and end-to-end values separately. The end-to-end boundary should include encoding, host/device transfer or DMA, preprocessing, setup, batching, output conversion, memory movement, and—where relevant—ADC/DAC, peripheral overhead, write/verify, calibration and update costs. Only the preregistered end-to-end boundary can satisfy a system-advantage gate.
5. **Expose device variation.** For analogue or approximate candidates, report per-device/run quality, drift/noise sensitivity, calibration age and repeatability. Use confidence intervals or distribution summaries where sample sizes support them; a good average must not hide unacceptable tails or failing devices.
6. **Do not change gates after observing results.** The current 2× end-to-end throughput OR 30% lower measured system-energy target remains a Symthaea-specific experimental hypothesis, not an MLPerf rule or universal scientific standard. Freeze it and hard constraints before measurement, or report the outcome as exploratory / Pareto trade-off.

These references improve methodology, not Symthaea performance claims. The synthetic fixtures remain synthetic, and external published measurements remain labelled as external evidence.


## First useful kernel: exact streaming Hamming distance

The first experimentally useful HDC kernel should be more than XOR bind in isolation. Hamming distance combines XOR, popcount and cross-lane accumulation to produce an exact integer used by similarity search/retrieval. Prototype source is tracked by [NEURO-SILICON-001B-E #7731](https://github.com/Luminous-Dynamics/symthaea/issues/7731), with the Rust-oracle differential gate in [#7733](https://github.com/Luminous-Dynamics/symthaea/issues/7733). The current streaming prototype remains uncompiled/unqualified until an exact-head simulator receipt exists.

For the same 16,384-bit input pair, compare declared lane profiles of 64, 128, 256, 512 and 16,384 bits (256, 128, 64, 32 and 1 transfers per vector, respectively). These counts are arithmetic consequences of widths, not measured performance. Freeze the exact input/output corpus and require integer distance equality in every profile. Then measure separately: vector comparisons/s, cycles/vector, p50/p95/p99 latency, sustained backpressured throughput, state/resource/timing reports, host transfers, system energy/comparison, and retrieval/task quality. Simulation cycles and synthesis estimates cannot satisfy the physical system-energy gate.

This kernel also provides a useful falsification test: a wider lane may reduce transfer count but enlarge combinational logic or routing, while a narrow lane may improve implementation feasibility but lose throughput to per-beat and host/protocol overhead. No lane width should be selected as best before pinned synthesis and then physical-board measurement.
## Dynamical HDC-LTC track: 16K–256K ContinuousHV, not just packed BinaryHV

Symthaea's active dynamical architecture is the in-core `symthaea-core::hdc::hdc_ltc_unified` path. The 16,384-bit packed BinaryHV RTL is one exact operator experiment, not a substitute for the liquid system's float32 state, learned gates, effective time constants, and trajectory evolution.

The repository's resolution-extension branch already defines exploratory 131,072- and 262,144-dimensional ContinuousHV operator evidence. The hardware successor program should therefore include a separate dynamical track at 16K, 32K, 64K, 128K and 256K. Do not promote >64K to a production default until the operator, transition, trajectory, resource and task-quality owners qualify it.

Canonical owners to compose:
- [#6560](https://github.com/Luminous-Dynamics/symthaea/issues/6560) / [#6561](https://github.com/Luminous-Dynamics/symthaea/issues/6561): state-continuous resolution transitions and trajectory matrix.
- [#6596](https://github.com/Luminous-Dynamics/symthaea/pull/6596), [#6619](https://github.com/Luminous-Dynamics/symthaea/issues/6619), [#6630](https://github.com/Luminous-Dynamics/symthaea/pull/6630): 16K–256K ContinuousHV operator evidence and fail-closed dependencies.
- [#6631](https://github.com/Luminous-Dynamics/symthaea/issues/6631): resource admissibility after trajectory qualification.
- [#7374](https://github.com/Luminous-Dynamics/symthaea/issues/7374): bounded iterative time-step API and numerical validation.
- [#3348](https://github.com/Luminous-Dynamics/symthaea/issues/3348): manifold-compressed liquid-dynamics controls.
- [#7749](https://github.com/Luminous-Dynamics/symthaea/issues/7749): dynamical HDC-LTC hardware/software co-design matrix.

### Resolution and reference sizes

| ContinuousHV dimensions | One float32 vector | Status in current resolution contract |
|---:|---:|---|
| 16,384 | 64 KiB | Baseline tier |
| 32,768 | 128 KiB | Baseline tier |
| 65,536 | 256 KiB | Baseline tier |
| 131,072 | 512 KiB | Exploratory |
| 262,144 | 1 MiB | Exploratory |

These are one-vector representation sizes, not full neuron/network working sets, cache traffic, measured energy or peak allocations. Multiply by the actual state/weight/gate/tau vector counts from the frozen active configuration, with checked arithmetic.

### Dynamical work to benchmark

- Fixed and irregular `dt`, ordinary evolution and long-horizon sequences.
- Current fused/closed-form path versus the bounded iterative path, with a fine-step RK4 reference where applicable.
- State, equilibrium, effective tau and gate distortion; per-step and accumulated trajectory divergence.
- High-resolution reference versus direct low-dimensional evolution and explicit high→low/evolve/low→high transitions.
- Online adaptation and snapshot/restore continuity; no hidden state reset on a resolution change.
- Resident/temporary memory, bytes moved, allocations, per-update latency, p50/p95/p99, updates/s and system energy under the same boundary.

### Architecture candidates

Compare dense float32 evolution, SIMD/cache-tiled state update, dependency-valid streamed chunks, and manifold/low-rank alternatives with an explicit residual/novelty path. Event-driven skipping is permitted only when a qualified activity/dependency map proves skipped state cannot affect the required output. Keep approximate or analogue representations versioned separately from exact ContinuousHV execution. Do not assume that the static XOR/Hamming RTL path accelerates liquid dynamics.

The acceptance unit is the **trajectory at fixed task quality and explicit resource/latency constraints**, not a single static similarity result. An operator-matrix PASS is a prerequisite, not trajectory qualification or measured performance advantage.
### Stronger paradigm hypothesis: decouple semantic dimension from dynamical state dimension

Do not assume every 128K+ semantic hypervector must be numerically evolved in full at every time step. The architecture question is whether Symthaea can retain a high-dimensional semantic address space while evolving a smaller, task-qualified dynamical state, then lift the result through a projection with a residual/novelty path. This is a research hypothesis owned in part by [NEUROARCH-011 / manifold-compressed liquid dynamics #3348](https://github.com/Luminous-Dynamics/symthaea/issues/3348), not an assumption that low-dimensional state is automatically equivalent.

Benchmark two independent dimensions:

- **Semantic dimension:** 16K / 32K / 64K / 128K / 256K ContinuousHV profile, where required by the task.
- **Local dynamical dimension:** candidate 1K / 4K / 16K / 32K manifold states, selected and evaluated independently of semantic width.

Required controls are full-width dynamics; random projection at a matched local dimension; a deterministic orthogonal/nested projection where applicable; and the proposed manifold projection with an explicit residual/novelty path. A learned task-specific projection comes later, after the non-learned controls. The residual cannot silently carry the entire high-dimensional vector every step in a run claiming compressed dynamics.

Report the task-quality vs local-state-width Pareto frontier, reconstruction error, temporal prediction/control quality, trajectory/tau/gate error, byte traffic, resident and peak temporary memory, projection/lift overhead, and p50/p95/p99 latency. Include cases where compressed dynamics loses so the gate can reject it. High-dimensional semantics may still be needed for binding/retrieval even when local temporal evolution can use fewer coordinates.

External motivation—not a transferable result: a June 2026 Nature Machine Intelligence study on a dual-memory-pathway spiking architecture reports co-design of a compact slow state with a fast event-driven pathway and over 4× throughput / over 5× energy-efficiency improvements in post-layout simulation. It is a different spiking architecture, and those hardware figures are simulation-derived; it supports testing fast/slow separation and locality, not a claim about Symthaea or a reason to force LTC/CfC into spiking semantics ([paper](https://www.nature.com/articles/s42256-026-01255-3)).

## Dynamical HDC-LTC trajectory preregistration (v0.1)

A dedicated, machine-readable protocol and companion document now define candidate gates for the active ContinuousHV dynamical path. The protocol is explicitly **not executed** and its thresholds are proposed targets to freeze before collecting candidate results.

- Protocol JSON: `crates/domains/symthaea-capability-closure/fixtures/dynamical-hdc-ltc-trajectory-protocol-v0.1.json`
- Method: `docs/research/DYNAMICAL_HDC_LTC_TRAJECTORY_PROTOCOL_V0_1.md`
- It keeps numerical integration fidelity, compressed-dynamics task preservation, and end-to-end system improvement as independent gates.
- Initial targets include trajectory NRMSE <= 1e-3 for a refinement-checked numerical reference, <=1 percentage-point classification loss / <=2% control-return loss for compression, and either >=2x end-to-end throughput or >=30% lower measured system energy, with p95/p99 latency regression <=5%.
- The measurement plan calls for at least 5 data seeds, 30 independent measurement blocks/configuration, 1,000 timed invocations/block, and a declared whole-system boundary.
- These are starting acceptance targets, not universal constants, and no target is claimed to have been achieved. No default above 64K is promoted by representability alone.
