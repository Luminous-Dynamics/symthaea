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