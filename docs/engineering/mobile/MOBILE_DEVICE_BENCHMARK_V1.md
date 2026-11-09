# MOBILE-DEVICE-BENCHMARK-V1: sovereign personal-device qualification

**Status:** preregistered research protocol; not a result  
**Parent:** [MOBILE-DEVICE-001 #8413](https://github.com/Luminous-Dynamics/symthaea/issues/8413)  
**Device architecture parent:** [ENG-DEVICE-000 #5670](https://github.com/Luminous-Dynamics/symthaea/issues/5670) / [PR #5672](https://github.com/Luminous-Dynamics/symthaea/pull/5672)  
**Protocol date:** 2026-10-22  
**Version:** 1.0

## 1. Research question

Can a Symthaea/Soma runtime perform the same useful, user-approved local tasks with materially less energy than a fixed mobile configuration, without unacceptable regressions in task quality, latency, thermal behavior, reliability, or privacy?

The primary metric is **joules per successfully completed task under a predeclared task mix**. Model parameter count, target loop frequency, theoretical TOPS, median microbenchmark latency, and battery percentage alone are not substitutes for this metric.

The proposed first hypothesis is:

> On the same handset, runtime, model artifact, and task corpus, an adaptive/event-driven policy reduces energy per successfully completed task by at least 20% relative to a matched fixed-profile baseline, while absolute task success is no more than 2 percentage points lower and interactive p95 latency is no more than 10% worse.

This is a preregistered target, not a result. Before the first scored run, record the hypothesis, test configuration, corpus digest, and analysis plan. Any change after scored data is observed must create a new version and preserve the old result.

## 2. Paradigm shifts to test

### 2.1 Useful work per joule

Optimize completed, correct tasks rather than raw cycles, tokens, TOPS, or model size. A policy that uses less energy because it fails more often is not an improvement. Report both total joules and incremental joules above the matched idle baseline.

### 2.2 Event-driven intelligence rather than perpetual polling

Use explicit wake/sleep transitions and workload-triggered compute. Validate that requested target rates are actually enforced by the host scheduler; a state machine that returns a frequency target is not proof that the underlying process runs at that frequency.

### 2.3 Quality-bounded model and accelerator routing

Select among the CPU, DSP/NPU, compact local model, Symthaea cognitive loop, and deterministic tools only when each candidate is actually supported on the measured device. Record the selected execution path and exact artifact hashes. Do not compare different accelerators, models, quantizations, or operating-system builds as if one scheduling change caused the difference.

### 2.4 Memory and thermals as first-class budgets

Treat resident memory, KV cache, context length, sustained performance, skin/device thermal state, and cooldown as hard design constraints. A model that nominally fits by weight size but fails once its cache or runtime allocations are included is not a usable candidate.

### 2.5 Privacy as a hard constraint, not a weighted score

Energy savings cannot compensate for unauthorized network egress, sensor access, tool execution, or permission escalation. Test negative controls with network tracing and permission denial/revocation.

### 2.6 Evidence-carrying adaptation

Every measured run must identify the device, build, policy, model, scenario, operating conditions, and raw output. Preserve failures, uncertainty, and regressions. Configuration declarations, synthetic replay, host benchmarks, and physical handset measurements remain distinct evidence classes.

### 2.7 Longevity as part of device performance

The later hardware program should optimize useful service-years, replaceable wear components, update support, repair time, and total cost of ownership—not just day-one speed. This is a separate hardware/lifecycle qualification track and is not proven by the runtime benchmark.

## 3. Existing implementation: facts versus assumptions

The source review found these relevant paths on `main`:

- `src/cognitive_loop/config/consciousness.rs`: `ConsciousnessProfile::Mobile` specifies a 20 Hz target, 128 CfC neurons, an energy budget, and thermal adaptation. These settings do not demonstrate physical-device energy or battery life.
- `crates/domains/symthaea-soma/src/engine.rs`: Soma composes Spore with sensors, haptics, metabolism, BLE, desktop sync, battery/charging and thermal state, and optional LiteRT/Prism paths.
- `crates/domains/symthaea-soma/src/metabolism.rs`: the state machine names Sleep/Drowsy/Alert/Focused target rates of 0.5/5/20/50 Hz. The benchmark must trace how these states affect host scheduling and skip-gates instead of assuming that all target rates are enforced.
- `crates/domains/symthaea-soma/src/bin/soma_bench.rs`: the current benchmark times 100 cycles after a short warm-up and prints latency statistics. It is a useful cycle-level microbenchmark, not a full task-energy, sustained-thermal, or all-day battery benchmark. Device thermal comments in that program remain assumptions until independently measured.
- Soma's default `SporeConfig` and the cognitive loop's `ConsciousnessProfile::Mobile` are separate configurations. Record and benchmark them separately; do not give them one shared profile label.

No claim in this document says Symthaea is conscious, certified, more efficient than a commercial phone, or ready for manufacturing.

## 4. Baseline matrix

| ID | Configuration | Purpose / guardrail |
|---|---|---|
| `CTRL-IDLE` | Assistant disabled, otherwise matched environment | Measures device/app baseline; not counted as a task-serving policy |
| `COG-MOBILE-FIXED` | Existing cognitive-loop `Mobile` profile | Freeze exact config and build digest |
| `SOMA-DEFAULT` | Existing `SomaConfig::default()` / default Spore profile | Separate from `COG-MOBILE-FIXED` |
| `SOMA-METABOLISM` | Current metabolism state machine and skip-gates | Use only after tracing actual scheduler effects; state transitions alone do not prove host throttling |
| `ADAPTIVE-V1` | Proposed event-driven scheduler/policy | Candidate; no assumed benefit before measurement |
| `MLPERF-MOBILE` | MLPerf Mobile v6.0 reference workload where supported | Exact task, runtime, model, quantization, delegate, and accelerator must be recorded |
| `MATCHED-LOCAL-LLM` | Strongest reproducible open local model/runtime for the chosen task on the same device | Include only when executable and licensed for the experiment; keep CPU/NPU results distinct |

Each comparison should change one intended factor at a time. If the model/runtime changes, treat that as a separate experiment or use a factorial design. Never compare energy results from incompatible power-measurement methods.

## 5. Frozen scenario corpus

The JSON fixture at `fixtures/mobile_device_scenario_corpus_v1.json` defines the first 12 scenario identifiers and expected invariants. The corpus covers:

- sustained background/idle residency and short foreground bursts;
- a valid multi-step local task and correct tool call;
- unsupported or irrelevant tool requests requiring refusal rather than fabricated execution;
- disconnected/offline operation;
- low-battery, charging, night-mode, and screen-off transitions;
- thermal pressure and cooldown/recovery;
- process pause, interrupt, restart and resume;
- denied/revoked sensor permissions;
- attempted outbound requests under an offline/no-share policy;
- missing, malformed, or stale telemetry.

The first executable phase should use deterministic synthetic inputs to test policy logic, configuration identity, telemetry validity, and fail-closed behavior. These runs establish only software-level properties. Device behavior must then be tested on a physical handset using the same frozen scenario identities.

## 6. Test protocol

### Phase A — freeze configuration

Record the repository commit, build flags, compiler/toolchain, app package and signing identity, model artifact URL/license/hash, quantization, inference runtime and delegate, device/SoC/RAM, Android/iOS build, thermal/energy policy, scenario corpus hash, and scoring code hash. If any semantic input changes, create a new run generation.

### Phase B — control conditions

1. Use the same physical device for paired policy comparisons. Record battery health and starting charge; target the same charge interval for each paired run.
2. Use a controlled ambient environment and fixed display brightness/refresh settings where the scenario allows. Record any unavailable controls rather than pretending they are equal.
3. Disable radios for offline compute scenarios; test networked scenarios separately with a captured, declared network configuration.
4. Randomize or interleave baseline/candidate order. Let the device return to the declared thermal starting band between runs. Record actual temperature/thermal indicators when accessible.
5. Run short interactive tasks and a separate sustained-load test of at least 30 minutes. Short cycle benchmarks cannot stand in for sustained behavior.
6. Use at least 30 paired repetitions for the initial per-condition estimate where feasible. Report per-device raw samples and 95% confidence intervals. If uncertainty is too large to decide against a threshold, label the result inconclusive and collect more data; do not round the result into a pass.

### Phase C — energy

Preferred: calibrated, task-appropriate power/energy instrumentation with method and calibration metadata. If the handset exposes a trustworthy rail/power measurement through supported tooling, preserve its semantics and sampling resolution. If only OS battery accounting or charge percentage is available, mark it as a coarse proxy and report it separately; do not convert a percentage delta into precise joules without a validated calibration model.

Report:

- joules per successful task;
- incremental joules above matched idle;
- joules per idle minute;
- energy and success rate over a fixed mixed-use trace;
- failed tasks and their energy cost.

Do not report a battery-life improvement unless a repeatable, representative end-to-end usage profile demonstrates it.

### Phase D — latency, memory, and reliability

Report median, p95 and maximum end-to-end latency. For local LLM paths, additionally report prompt-prefill time, time to first token, decode tokens/second, completion latency, context length, stop/truncation state, and peak resident memory including cache where observable. Track skipped/missed cycles, deadline misses, crash/restart, cancellation, and recovery correctness. State the timer source and measurement resolution.

### Phase E — thermal and privacy

Record thermal state over the whole sustained run, observed throttling, performance drift, and cooldown/recovery. Compare at the same ambient and workload conditions.

For every privacy case, record requested permissions, actual permission state, sensor/API access, attempted and completed network destinations/bytes, and tool invocations. Any unauthorized sensor read, outbound transmission, tool action, or permission escalation is a hard failure regardless of performance.

## 7. Initial decision gates

These are engineering targets to validate or reject, not measured status:

- **Energy:** at least 20% lower joules per successfully completed task than the matched fixed-profile baseline; the 95% confidence interval must exclude no improvement.
- **Utility:** absolute task success no more than 2 percentage points lower on the frozen corpus; report refusal correctness separately.
- **Responsiveness:** interactive p95 completion latency no more than 10% worse than baseline unless the task-specific contract explicitly permits delayed/background completion.
- **Privacy:** zero unauthorized egress, sensor access, actions, or privilege changes in the negative-control battery. One violation fails the privacy gate.
- **Thermal:** no worse severe thermal-state duration or sustained-throttle behavior without a preregistered, explicitly user-visible trade-off.
- **Robustness:** missing/stale telemetry, unqualified runtime/model combinations, or interrupted execution must trigger an explicit conservative fallback; they may not be silently interpreted as a healthy nominal state.
- **Reproducibility:** every result identifies exact inputs, emits raw measurements, and preserves failed runs.

A single supported handset can establish an initial device-scoped pilot. A generalized industry-performance claim requires at least three physical units per tested model family and at least two SoC families, with comparable methods and uncertainty reporting.

## 8. Industry comparison plan

1. **MLPerf Mobile v6.0** ([release](https://mlcommons.org/2026/06/mlperf-mobile-v6/)): includes Llama 3.2 1B/3B and Llama 3.1 8B on-device evaluations using TinyMMLU/IFEval, CPU execution where the device has enough memory, and selected NPU acceleration. Reuse its standard workloads when supported. The exact supported model/delegate path is part of the result.
2. **Artificial Analysis Mobile Device Benchmark Set** ([methodology](https://artificialanalysis.ai/methodology/mobile-device-benchmark-set)): evaluates quantized mobile-suitable model builds using five quality evaluations (tool calling, instruction following, knowledge/non-hallucination, scientific reasoning, and math) and separately measures real-phone inference speed/latency/memory. Match its model build and procedure for direct comparisons. Add our separate energy, privacy, idle and sustained-thermal dimensions; do not claim its quality score alone proves our system is better.
3. **MLPerf Tiny** ([benchmark](https://mlcommons.org/working-groups/benchmarks/tiny/)): useful if we later deploy microcontroller-scale sensing/wakeword functions. Its optional energy measurement and fixed quality targets are a pattern for reproducible low-power benchmarks.
4. **EU smartphone/tablet ecodesign requirements** ([European Commission overview](https://energy-efficient-products.ec.europa.eu/product-list/smartphones-and-tablets_en)): include battery endurance, spare-parts/disassembly, and operating-system update requirements in the separate hardware/lifecycle scorecard.

An honest “better than SOTA” result must state its frontier and its axes. We could beat a device on task energy while losing on image quality, radio efficiency, repairability, or camera performance. Publish Pareto comparisons and per-axis results; do not collapse incomparable metrics into a single marketing score.

## 9. What would constitute a genuine advance?

A substantive result would show, on a real handset, that the adaptive policy completes the same frozen tasks at comparable quality with lower measured energy, and that its wins persist under sustained load and adversarial privacy cases. Stronger follow-on work would show the gain across device families, validate predicted versus measured thermal/power behavior, and demonstrate a repair/update policy that extends useful service life without sacrificing security.

Until those studies run, the right status is **hypothesis and qualification plan**, not a claim of superiority.

## 10. References

- MLCommons, *MLPerf Mobile v6.0: New GenAI Benchmarks for On-Device LLMs* (2026-06-15): <https://mlcommons.org/2026/06/mlperf-mobile-v6/>
- Artificial Analysis, *Mobile Device Benchmark Set Methodology*: <https://artificialanalysis.ai/methodology/mobile-device-benchmark-set>
- MLCommons, *MLPerf Tiny*: <https://mlcommons.org/working-groups/benchmarks/tiny/>
- European Commission, *Smartphones and Tablets — Ecodesign*: <https://energy-efficient-products.ec.europa.eu/product-list/smartphones-and-tablets_en>
- Android Open Source Project, *Verified Boot*: <https://source.android.com/docs/security/features/verifiedboot>
