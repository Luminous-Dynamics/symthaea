# Robotics Physical-Time and Energy Benchmark Contract v1

**Status:** proposed qualification contract; targets, not measured results.  
**Scope:** cognitive substrate modulation, simulator/plant time, embodiment timing, and substrate energy-rate telemetry.  
**Related:** [#7635](https://github.com/Luminous-Dynamics/symthaea/issues/7635), [#1284](https://github.com/Luminous-Dynamics/symthaea/issues/1284), [PR #7634](https://github.com/Luminous-Dynamics/symthaea/pull/7634), [PR #7666](https://github.com/Luminous-Dynamics/symthaea/pull/7666).

## Paradigm shift: measure the boundary, not the proxy

A cognitive backend's modeled speed, CfC integration scale, scheduler cadence, simulator time, transport latency, and electrical energy are different quantities. A benchmark must not use one as a proxy for another without an explicit, tested model.

The key metamorphic question is:

> Holding the physical model/profile, initial state, random seed, and applied command trace fixed, does changing only the cognitive substrate change the plant's time advancement?

The physical-time answer must be **no**. A faster cognitive backend may reduce decision latency or make more command opportunities available; it cannot silently change the physical seconds represented by an identical simulator step. When commands differ, compare plant-time invariants separately from the resulting trajectory.

ROS 2 control's clock model is a useful boundary reference: hardware-oriented read/write periods use steady/monotonic time, while ROS time is used in message/simulation semantics. See [ROS 2 control clock documentation](https://control.ros.org/rolling/doc/ros2_control/controller_manager/doc/clocks.rst) and the [ROS time design](https://github.com/ros2/design/blob/gh-pages/articles/130_ros_time.md).

## Required benchmark matrix

| ID | Contract | Procedure | Acceptance target |
|---|---|---|---|
| PT-1 | Substrate metamorphism | Replay the same fixed plant profile, initial state, seed, and applied command trace across biological, silicon, and photonic cognitive substrate configurations. Compare each transition's plant dt and cumulative model-time. | Exact equality of plant dt and cumulative model-time across arms. Any trajectory difference outside this fixed-command test is assessed as a command-policy effect, not silently attributed to time. |
| PT-2 | Cognitive-time separation | Change cognitive tau factor / CfC integration scale while holding explicit simulator step timing constant. | Zero changes to requested/applied plant dt caused solely by cognitive tau. |
| PT-3 | Clock integrity | Replay valid, equal, reversed, stale, paused, zero/uninitialized, and cross-domain timestamp cases. Preserve both source timestamp and receive/arrival timestamp when both exist. | Invalid or ambiguous timing cannot become a plausible synthetic dt. Each rejected/incomplete case has a stable reason code and a negative-control test. Simulation-time jumps follow the declared simulator profile; monotonic hardware elapsed time never follows wall-clock adjustment. |
| PT-4 | Energy dimensionality | Use a fixed cycle-energy trace and injected monotonic event times. Compare calculated average model power to the sum of modeled joules divided by the same closed window duration. Include uneven cycle cadence. | Relative error <= 1e-9 for deterministic unit-level arithmetic when inputs are finite and representable; tolerance and units are explicit. No cumulative-joule-versus-watts comparisons. |
| PT-5 | Invalid budget fail-closed | First enabled tick with negative, NaN, positive-infinity, or negative-infinity watt budget; repeat with a previously open rate window. | 100% fail closed on that tick; no invalid config receives a grace period while waiting for a measurement window. |
| PT-6 | Window and telemetry provenance | Test first-cycle startup after long idle, exact window boundaries, windows longer than one second, disabled/re-enabled accounting, and serialized snapshots missing newly added fields. | Idle-before-first-cycle excluded; every reported watt sample states its actual elapsed interval and is labelled as a model estimate, not measured electrical power. Compatibility defaults are tested. |
| PT-7 | Runtime timing profile | For each explicitly named robot/backend/profile, record cycle period, p50/p95/p99/p99.9/max cycle latency, p99.9 jitter, deadline misses, CPU/GPU load, thermal state, command age, and sensor age over a declared workload and run duration. | No universal latency number is claimed across unlike robots. Each profile has an owner-approved deadline budget; report tail latency and misses, not only mean/throughput. Zero misses is a qualification result only for the specific tested hardware/software/workload, not a hard-real-time proof. |
| PT-8 | Energy calibration | Collect an external power-meter or qualified hardware-counter trace, synchronized to workload and thermal state, across idle, nominal, peak, and transition workloads. Fit on a training set and evaluate on held-out runs. | Until calibrated, expose only modeled joules/cycle and model-estimated watts. A proposed calibration gate is <=10% mean absolute percentage error on held-out steady workloads, with uncertainty and out-of-distribution cases reported; otherwise mark the model uncalibrated rather than claiming device power accuracy. |
| PT-9 | Replay and evidence | Pin repo SHA, backend/profile, config, seed, initial state, command trace hash, timing source, workload identifier, build/features, and corpus version. Preserve negative controls. | Deterministic simulation replay yields identical plant-time evidence and, where the backend claims determinism, identical transition/capture hashes. Missing provenance is an incomplete result, not a pass. |
| PT-10 | Fault and disturbance response | Inject bounded sensor dropout/staleness, clock reset or epoch change, missed command deadline, actuator rejection, and plant disturbance in simulation with pinned seeds and negative controls. Record detect-to-safe-action latency, time to recover, intervention count, and safety-envelope violations. | Every out-of-policy timing record is rejected; no stale command is reported as fresh. Response deadlines are profile-owned and explicit. Zero unsafe actions is required on the declared test corpus, but does not constitute a proof of safety or a hard-realtime guarantee. |
| PT-11 | Adaptation agility | Start from a declared baseline task and environment, then request a task/tool/obstacle change without changing the plant-time authority. Measure time-to-reconfigure, task success, recovery time, human interventions, constraint violations, and repeatability across randomized but pinned scenarios. | Report distributions and baselines rather than one aggregate score. Faster re-tasking must not relax timing provenance, safety constraints, or evidence requirements. |
| PT-12 | Invalid energy-model inputs and dimensional isolation | Inject NaN, positive/negative infinity, and negative modeled joules per cycle; exercise cumulative-total and observation-window overflow. Vary cognitive tau over 0.5, 1.0 and 2.0 while holding the modeled substrate energy/cycle constant. | Invalid or overflowing energy evidence fails closed without accounting the invalid sample. The modeled joules/cycle remain invariant to cognitive tau; wall-time rates use observed elapsed time. NaN must never bypass a comparison because IEEE comparisons with NaN are false. |

## Reporting rules

Every published result must distinguish the following metrics; do not combine them into a single "speed" number:

- cognitive cycles/s and end-to-end decision latency;
- controller update cadence and deadline/jitter distribution;
- requested and actually applied plant dt, cumulative simulation/model time, and wall-time execution ratio;
- sensor sample age, receive age, command age, and transport latency;
- modeled joules/cycle, total modeled joules, model-estimated average watts, and measured electrical energy/power (if instrumented), plus invalid-model-input rejection counts;
- success/task quality, safety-envelope violations, time-to-safe-action after faults, recovery time, and task reconfiguration/agility.

Report p50, p95, p99, p99.9, maximum, sample count, duration, missed deadlines, and relevant uncertainty. Pin the workload and distinguish warm-up from scored runs. Never infer hard real-time guarantees from a finite test alone.

## Substrate tau sign contract

In the substrate manager, operation_speed() is seconds per operation. Lower values mean faster operation. Under log10(biological_seconds_per_op / substrate_seconds_per_op), faster substrate configurations yield a factor above 1, equal yields 1, and slower configurations yield a factor below 1, subject to the configured clamp. Modeled joules/cycle come from the substrate's energy-per-operation estimate and operation count; cognitive tau is not an energy multiplier. Actual average modeled watts divide completed modeled joules by observed monotonic wall duration. This sign contract does **not** apply automatically to unrelated FEP, thermal, or controller-local factors that happen to share the tau_factor name.

Each call site must document whether its factor changes cognitive integration dynamics, prediction horizon, controller internal response, scheduling, or physical plant time. Physical plant time is never inferred from cognitive tau alone.

## Verification record template

Every run should capture these fields in a machine-readable receipt:

- repository SHA, binary/test target, and feature set;
- robot or simulator profile, hardware, and clock source;
- config SHA-256, seed, initial-state hash, command-trace hash, timing-corpus version;
- sample count, wall duration, deadline-miss count, p50/p95/p99/p99.9/max latency;
- plant-time invariance result and negative-control results;
- modeled energy joules, estimated power watts, and measured energy joules or null;
- calibration dataset/uncertainty or null;
- runner status and exact head SHA.

A queued/skipped runner, missing receipt, or run performed at a different commit SHA is not a pass.

## Methodology references

- [ROS 2 REP-2014: Benchmarking performance in ROS 2](https://reps.openrobotics.org/rep-2014/) is an informational methodology proposal (its current status is **Rejected**, so it is cited as guidance, not a normative standard). It recommends realistic known workloads, quantitative performance evaluation, grey-box tracing, and reproducible reporting. Symthaea should preserve those ideas even when benchmarks target Rust internals rather than ROS nodes.
- [NIST Agility Performance of Robotic Systems](https://www.nist.gov/programs-projects/agility-performance-robotic-systems) emphasizes metrics and test methods for responding to unexpected events, re-tasking, datasets, and combined virtual/real testbeds. PT-10 and PT-11 extend the contract beyond raw cycle speed toward failure recovery and operational agility.
- [NIST Performance Assessment Framework for Robotic Systems](https://www.nist.gov/programs-projects/performance-assessment-framework-robotic-systems) motivates composing perception, mobility, dexterity, and safety measures into task-level capability assessments rather than collapsing system quality into one throughput number.
