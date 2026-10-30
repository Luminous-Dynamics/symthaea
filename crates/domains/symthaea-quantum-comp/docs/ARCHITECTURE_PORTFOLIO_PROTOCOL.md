# QARCH cross-paradigm architecture portfolio protocol

## Purpose

This protocol lets QARCH compare unlike compute paradigms without pretending they share the same physical operations, accuracy model, or maturity. The initial Rust module is a dependency-free vocabulary plus a deliberately conservative system-level Pareto comparator. It is not a workload runner, hardware adapter, statistical engine, or architecture selector.

## Architecture families

| Family | Representative modality | Workload hypotheses to test |
| --- | --- | --- |
| CPU / GPU / NPU / FPGA | Conventional digital | General-purpose, dense vector/matrix, fixed-function and reconfigurable execution |
| Digital neuromorphic | Event-driven digital spiking/local-state | Sparse temporal events, continuous streams, local state and online updates |
| Mixed-signal neuromorphic | Accelerated analog neural dynamics | Low-latency dynamical systems, but with calibration and variability costs |
| Digital processing-in-memory | Digital logic in DRAM/SRAM banks, memory modules or stacked-memory logic layers | Data-movement-bound reductions, matrix/vector operations and bank-local primitives with explicit precision and bank-level parallelism |
| Digital near-memory compute | Digital logic adjacent to memory arrays/modules, but not inside the arrays | Partial reductions, filtering and data-local scheduling where local-link traffic is still accounted for |
| Analog compute-in-memory | RRAM/memristor, PCM, flash/NOR arrays | Data-local matrix/vector operations where conversion and correction costs do not erase array gains |
| Classical photonic linear | Interferometric, diffractive or wavelength-multiplexed optics | High-throughput linear transforms and low-latency optical paths |
| Photonic / physical reservoir | Physical dynamical substrate | Temporal feature expansion and reservoir tasks with cheap readout |
| Gate quantum | Superconducting, trapped-ion, neutral-atom and photonic systems | Specific quantum algorithms/sampling tasks under explicit logical/physical resource accounting |
| Quantum annealing | Annealing / Ising architecture | Formulations that map faithfully to its native optimization model |
| Spintronic / probabilistic | Spin devices and p-bits | Stochastic sampling or specialized low-energy probabilistic inference |
| Reversible / adiabatic | Reversible logic and adiabatic circuits | Workloads where switching energy, clocking and physical overhead make a practical comparison possible |
| Hybrid heterogeneous | Explicit multi-family system | Workloads that benefit from deliberate placement/scheduling across substrates |

Memory-centric designs require separate labels for where computation occurs (in-array, in-module/stack, or adjacent to memory), how it is computed (digital or analog), and the memory substrate (SRAM, DRAM, flash, RRAM, PCM, etc.). Digital HBM/DRAM processing-in-memory is not analog memristor compute-in-memory; near-memory logic is not in-array compute.

For these workloads, report bytes moved at each hierarchy boundary, bank bandwidth/parallelism, precision, host/device transfers, and end-to-end energy/latency. For analog arrays, include ADC/DAC, sensing, calibration, programming, correction and readout. Do not compare an array-only peak with a full-system PIM or GPU result.

An event camera is a sensor modality, not a neuromorphic compute family. Classical photonic matrix multiplication is not quantum photonics. Quantum-inspired software on CPU/GPU remains a classical implementation. A CfC or continuous-time model must not be forced into a spiking-network representation merely to qualify as neuromorphic.

## Reuse external benchmark structures

**NeuroBench** already has a hardware-independent algorithm track and a deployed system track. Its algorithm tasks include few-shot continual learning, vision, motor decoding and chaotic forecasting; its system track reports task correctness, timing and efficiency. QARCH should reuse its harness and compatible tasks, then extend complete-system boundaries where sensor, host, data transfer, calibration or independent verification overhead is absent.

**MLPerf Tiny and Edge** provide common conventional edge workloads with quality targets and latency/throughput rules. MLPerf Tiny also supports optional energy measurements. They are valuable baselines for NPU/GPU/CPU/MCU systems, not automatically a like-for-like comparison for every neuromorphic device.

**QOBLIB** supplies optimization instances; tune classical exact and heuristic baselines independently. **QUOPS** supplies a circuit capability view; use it for quantum hardware where its operation definitions apply. **QED-C and IonQ** are useful application-level quality/time-to-solution comparators.

Do not reimplement an external suite under a new name. Build adapters only when workload, metric, execution boundary and evaluation rules are semantically compatible. State deviations explicitly.

## Two distinct comparison tracks

### Algorithm track

Freeze the algorithm, representation, precision and task semantics. Map the same algorithm onto different execution substrates where possible. The goal is to attribute measured changes to the hardware/runtime path. If an architecture cannot run the same algorithm, mark it unsupported rather than quietly changing the task.

### System co-design track

Allow each architecture to use a competent, architecture-specific algorithm and mapping under matched development, tuning and compute budgets. This estimates the best useful system each family can deliver, not the isolated effect of hardware. Keep algorithm and system results separate.

## Common workload portfolio

Start with the exact canonical Symthaea and public workload owners:

- HDC bind/bundle/permutation and associative recall.
- Event-driven vision and streaming audio, with sensor modality ablations.
- Sparse recurrent/CfC temporal state updates.
- Few-shot continual learning, online adaptation and drift recovery.
- Streaming sensor fusion and latency-bounded edge control.
- QOBLIB combinatorial optimization.
- Dense tensor/matrix inference as a negative control for specialized sparse/event hardware.

Every task needs the same frozen input, objective, constraints, acceptance verifier and task-quality metric for all compared candidates. Include conditions under which the specialized architecture should lose.

## Measurement boundaries

Use an explicit boundary label:

1. Algorithm only: complexity, model footprint and representation properties; no physical energy or wall-system claim.
2. Kernel/device: operation or device execution only.
3. Runtime end-to-end: include host/device transfer, runtime, synchronization, initialization and required data movement.
4. Application end-to-end: include preprocessing, execution, retries, post-processing and independent output verification.
5. Sensor-to-decision: include sensing/readout, device buffering, interface, transport, preprocessing, inference and task output.

A kernel-only result is not compared to an application end-to-end result. Simulation is not compared to a physical prototype as if evidence were equivalent. The Rust comparator rejects mismatched workload IDs, comparison contracts, evidence classes and measurement boundaries.

Algorithm-only observations carry no physical resource envelope and are deliberately excluded from this system-level Pareto comparator. A later algorithm-track comparator can use a different, dimensionally appropriate complexity vector.

## Conservative Pareto relation

Each system-level record contains:
- a quality interval, where higher is better under the exact shared quality contract;
- wall-time interval in seconds;
- energy interval in joules;
- monetary execution-cost interval in USD;
- peak-memory interval in bytes.

For candidate A to robustly dominate B, A's entire quality interval must be no worse than B's and every resource upper bound of A must be no greater than B's lower bound. There must be a strict improvement in at least one dimension. Disjoint preferences produce a tradeoff. Overlapping intervals produce unresolved evidence, not a winner.

This comparator does **not** calculate confidence intervals. The interval method and family-level inference must come from QARCH-015/016/017 and the separately preregistered measurement campaign. A point estimate alone should not be placed into an interval with invented uncertainty.

The first version requires all four resource dimensions. If any dimension is not measured, a complete resource comparison is withheld rather than imputing a favorable value. Architecture-specific metrics (QPU shots, synapse updates, spike counts, optical conversions, ADC/DAC operations, device writes, calibration time, etc.) belong in supplementary records and must not be mistaken for universally comparable units.

## Why neuromorphic is a priority, not a universal winner

Intel's Hala Point is a 1.15-billion-neuron, 128-billion-synapse Loihi 2 system, but its published maximum system power is 2.6 kW and it includes more than 2,300 embedded x86 processors for ancillary computation. Those specifications illustrate why raw neuron count or core efficiency cannot replace an end-to-end task measurement.

Neuromorphic systems have plausible strengths in sparse/event-driven processing, streaming temporal inference, online adaptation and low-duty edge tasks. They can lose on dense/batched matrix workloads, unsupported operators, data conversion, host communication, learning/write overhead, calibration, and task-quality requirements. Those are benchmark hypotheses, not results.

For neuromorphic work, reuse NEURO-SILICON-001 (#5804), the frozen workload/baseline tranche (#5805), and NeuroBench; do not create a duplicate SNN/HDC stack. Sensor/event work stays with SENSE-EVENT (#5822/#5829).

## First cross-paradigm stress matrix

Build negative controls before any headline comparison:

- sparse event stream at low event rate: event-driven overhead can outweigh saved work;
- event burst and flicker: event volume can explode and saturate links/buffers;
- dense matrix workload: GPU/NPU may dominate neuromorphic or event-driven systems;
- analog array precision/noise: task quality may fail or require costly digital correction;
- ADC/DAC and host transfer: analog-CIM advantage may disappear at system boundary;
- photonic conversion and calibration: optical-kernel advantage may disappear after electro-optic interfaces;
- quantum compilation/control/shot/retry/decoding: QPU execution may not dominate total time/cost;
- online learning: device write endurance/energy or weight-update cost may dominate;
- regime shift: a static architecture assignment may incur high regret;
- simulator mismatch: hardware-simulated winner fails against held-out physical measurements.

## Task-conditioned architecture selection

Only after the measured capability atlas exists, compare:
- best single static architecture;
- transparent hand-coded routing rules;
- learned workload-to-architecture router;
- an empirical oracle with access to the measured Pareto surface.

Condition on event sparsity/burstiness, data locality, precision, latency tail, duty cycle, online update frequency, memory pressure, drift/noise and provisioning cost. Measure routing regret, constraint violations, switching/migration overhead, cold-start costs and abstention. The router must abstain outside measured applicability envelopes.

## Acceptance targets

- Exact task, workload fingerprint and verifier identity for every confirmatory record.
- Explicit evidence class and measurement boundary for every record.
- Complete resource vector before system-level Pareto claims.
- No universal architecture score.
- No ranking of incompatible workloads, contracts, evidence classes or boundaries.
- NeuroBench/MLPerf/QOBLIB/QUOPS adapters only where semantics match.
- Publish tradeoffs, unresolved comparisons, negative results and resource omissions.
- Apply QARCH-015 paired gates, QARCH-016 family-level inference and QARCH-017 exposure policy before generalized advantage claims.
- Measure routing regret only against an empirically observed oracle frontier.

## Claim ceiling

This module is a data contract and conservative comparison primitive. Its tests and synthetic fixture do not constitute physical measurements, system benchmarking, statistical significance, or evidence that any family is superior.


## Constraint-aware architecture selection (QARCH-019B)

The routing primitive is a deterministic selector over **already measured** observations. It is not a learned model, an online scheduler, or a claim that automatic routing is already useful.

A request freezes the workload kind and fingerprint, comparison-contract digest, evidence class, measurement boundary, minimum quality, four hard resource ceilings, and one explicit lexicographic priority. A candidate is eligible only when:

- its workload, comparison contract, evidence class and measurement boundary exactly match the request;
- the lower bound of its quality interval meets the minimum quality threshold;
- the upper bound of every resource interval is within its hard ceiling;
- the observation passes the architecture-portfolio validation contract.

The selector distinguishes definite failures from unresolved intervals. For example, a latency interval wholly above the ceiling is a measured limit violation; an interval that straddles the ceiling is simply not proven feasible. Either excludes the candidate, but the report preserves the distinction. The selector abstains when no candidate proves all requirements.

Among eligible candidates, the caller chooses one lexicographic resource priority (latency, energy, monetary cost, or memory). Remaining resources provide deterministic tie-breaks, followed by the guaranteed quality lower bound and stable candidate identifiers. This is intentionally not a weighted scalar score: the priority is visible and the hard limits remain hard constraints.

### What this selector does not do

- It does not estimate the performance of an unmeasured architecture.
- It does not learn a router or generalize beyond an exact workload/contract identity.
- It does not switch devices at runtime.
- It does not include migration, cold-start, cache-warmup, provisioning, or transition costs unless those are measured inside the supplied resource envelope.
- It does not treat a simulation result as interchangeable with a prototype or production result.

The next empirical stage is to populate a capability atlas from canonical Symthaea tasks and compatible NeuroBench/MLPerf/QOBLIB suites, then compare this transparent constrained selector against best-static, hand-coded, learned and empirical-oracle policies. Only then should we report routing regret, feasibility violations, switching cost, cold-start penalty and out-of-domain abstention. The first goal is **zero hard-constraint violations on the measured operating region**, not maximum average reward.

All unit-test examples are synthetic logic checks. They do not represent hardware, measured energy, or architecture advantage.


## Evidence-bound atlas admission (QARCH-020)

Each normalized architecture observation can be paired with an evidence receipt that binds:

- the observation ID, exact workload fingerprint and comparison-contract fingerprint;
- the benchmark suite/version and upstream case ID;
- source/document, runner (when available), fixture, run manifest, environment, raw output, interval analysis and summary digests;
- evidence origin (direct measurement, published result, simulation execution or analytical reference);
- executor/device/toolchain identities where disclosed;
- attempted/failed run counts and declared aggregation and interval methods.

The evidence-bound comparison and selection entry points validate every receipt before admitting observations. A single malformed receipt rejects the supplied atlas slice; candidate feasibility exclusions still appear in the ordinary routing report after receipt admission succeeds.

### Origin is not evidence maturity

The observation's `evidence_class` answers *what kind of evidence is represented* (simulation, hardware emulation, physical prototype, production system or algorithmic reference). The receipt's `origin` answers *where this particular record came from* (direct run, published result, executed simulation or analytical reference). These fields are both retained because a published paper can describe a physical prototype, a simulation or an analytical method; it is not a local reproduction merely because its result is cited.

Direct measurement requires a device, toolchain and runner identity. Published physical/production results require the reported device identity, but undisclosed runner/toolchain values remain unknown. Simulations cannot be promoted to physical evidence. Analytical references cannot enter the system-level resource comparator.

A SHA-256 digest is content identity only. It does not authenticate the submitter, certify the meter, prove chain of custody or prove that a publication's reported measurement is correct. Cryptographic attestation and trusted measurement capture are separate capabilities.

All fixture hashes and identities are fabricated test values. They do not refer to actual devices, benchmark submissions or measurements.
