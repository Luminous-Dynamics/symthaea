# HARDSEC-018 — PMU Metrology, Counterfactual Acquisition and Observer Effects

**Status:** Rust acquisition auditor, versioned schemas, and synthetic known-answer fixtures authored; no real hardware acquisition campaign is claimed  
**Issue:** [#9470](https://github.com/Luminous-Dynamics/symthaea/issues/9470)  
**Parent benchmark:** [HARDSEC-017 / PR #9468](https://github.com/Luminous-Dynamics/symthaea/pull/9468)

## 1. Thesis: the sensor is part of the experiment

A performance-counter detector is only as trustworthy as its measurement path. Simultaneous event scheduling, limited programmable counters, event constraints, multiplexing, software scaling, kernel/firmware event mapping and observer overhead can change the values being learned or evaluated.

The 2026 physical HARPY-V / XuanTie C910 study reports that concurrent collection with Linux perf perturbed some counter values by more than 300%, motivating separate counter runs. Its selected multi-process normalized scenario reports 98.56% accuracy while raw counts fall to 60%; these are specific to that platform and protocol, not a universal guarantee. See [Pou et al. (2026)](https://doi.org/10.1186/s42400-026-00662-8).

Linux `perf_event_open` exposes `time_enabled` and `time_running`. When an event is multiplexed, the kernel/tool may estimate a full-run count by scaling the raw value by `time_enabled / time_running`. This is an estimate, not a directly observed full-time count, and does not remove systematic interference or phase bias. See the [Linux perf_event_open manual](https://www.man7.org/linux/man-pages/man2/perf_event_open.2.html) and [perf's multiplexing guide](https://perfwiki.github.io/main/tutorial/).

The RISC-V SBI PMU extension exposes per-HART counters, hardware and firmware events, raw event encodings and an interface for Linux perf compatibility. Implementations have finite counter capacity and platform-specific event support; event names/encodings must not be assumed portable across implementations. See the [RISC-V SBI PMU specification](https://docs.riscv.org/reference/sbi/v1.0.0/ext-pmu.html).

## 2. Counterfactual acquisition matrix

Each randomized matched block runs the same fixed work unit under four conditions:

1. `baseline_no_pmu`: workload throughput/latency without PMU reads.
2. `singleton_per_event`: each named event measured separately.
3. `grouped_co_scheduled`: a preregistered subset that fits the declared programmable counter capacity.
4. `grouped_overcommitted`: a preregistered set larger than declared capacity, with multiplex scheduling explicitly observed.

The plan records the exact platform, CPU/HART, PMU, architecture, firmware/SBI, kernel, perf version/configuration, event encoding/semantic/unit, binary digest, work unit, fixed work count, counter capacity, randomized-order seed and thresholds. Each trial records block ID, condition, order index, work units, wall-time, PMU read overhead and per-event raw count, enabled/running time, samples and lost samples.

The design has three distinct comparisons:

- **Event co-scheduling effect:** paired per-event raw count per fixed work unit, grouped versus singleton in the same block.
- **Observer effect:** paired throughput/latency, PMU-enabled conditions versus no-PMU baseline in the same block.
- **Multiplexing estimate error:** singleton count versus `raw_count × time_enabled / time_running` for an overcommitted group. This result is explicitly an estimate and is never eligible as a direct confirmatory count.

The first two are descriptive paired perturbation measures. This small auditor does not calculate confidence intervals or infer causality by itself. A randomized acquisition protocol is necessary for causal interpretation, but the declared seed/order fields are caller-supplied and do not prove randomization occurred.

## 3. Executable contract

Inputs:

- Plan schema: `schemas/hardsec-pmu-metrology-plan-v1.schema.json`
- Trial schema: `schemas/hardsec-pmu-observations-v1.schema.json`
- Plan fixture: `tests/fixtures/hardsec/pmu-metrology-v1/plan.json`
- Synthetic observations: `tests/fixtures/hardsec/pmu-metrology-v1/observations.json`
- Auditor: `examples/hardsec_pmu_metrology_audit.rs`

Fail-closed checks:

- Unique event IDs and unique declared encodings; ambiguous aliases are rejected from this v1 contract.
- Every observed event and encoding must match the plan exactly.
- `time_enabled > 0`, `time_running > 0`, and `time_running ≤ time_enabled`.
- Lost samples cannot exceed total samples.
- Each matched block must contain the no-PMU baseline, one singleton trial per event, one exact co-scheduled event set and one exact overcommitted event set.
- Every trial uses the exact fixed work-unit denominator and platform/workload/protocol IDs.
- Order indices must be unique and contiguous; every trial uses the preregistered randomization seed.
- The co-scheduled group cannot exceed programmable counter capacity and each event must meet the declared running-fraction gate.
- The overcommitted set must contain every declared event, exceed capacity and record at least one event whose running time is less than enabled time.
- Sample loss and PMU read overhead are gated independently; their limits are preregistered in the plan.
- Matched block count must reach the preregistered minimum for descriptive quantiles.

The receipt binds exact plan and observation bytes by SHA-256; it is unsigned and written without overwriting an existing file. It leaves effective sample size null and independence unestablished. Quantiles are descriptive summaries across the supplied matched blocks, not population confidence intervals.

## 4. Run

```bash
cargo test --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-pmu-metrology-audit \
  --example hardsec_pmu_metrology_audit

cargo run --locked \
  --manifest-path tools/hardsec-adaptive-evasion-audit-harness/Cargo.toml \
  --features hardsec-pmu-metrology-audit \
  --example hardsec_pmu_metrology_audit -- \
  --plan tests/fixtures/hardsec/pmu-metrology-v1/plan.json \
  --observations tests/fixtures/hardsec/pmu-metrology-v1/observations.json \
  --receipt /tmp/hardsec-018-receipt.json
```

Fifteen authored Rust tests cover the known answer, invalid running times, event aliasing, work-unit mismatch, duplicated trials, condition imbalance, missing multiplex evidence, wrong randomization seed, unpaired blocks, sample-loss errors and evidence-domain mismatch.

The fixture intentionally contains two synthetic matched blocks. It exercises the contract only; it does not represent C910/HARPY-V, GEM5 or any measured physical platform.

## 5. Gate semantics and claim ceiling

A co-scheduled event is eligible under this plan only when its p95 absolute paired deviation is within the preregistered threshold and every paired running fraction meets the minimum. A workload condition passes its observer gate only when the p95 throughput degradation relative to the no-PMU baseline stays within the declared limit. A failure means the acquisition condition is not qualified for the named workload/platform; it does not imply malicious behavior.

Overcommitted counts are always reported as `multiplex_scaled_estimate_not_observed_count` and `confirmatory_eligible=false`, even if their estimate happens to agree with singleton counts. Scaling does not repair counter co-scheduling interference, event semantic mismatch, workload phase drift or missing event constraints.

This receipt validates caller-declared identities and supplied trial arithmetic. It does not authenticate silicon, PMU firmware, SBI, kernel/perf builds, event semantics, workload artifacts or randomization. It does not establish independent blocks, causal detection, detector performance, cross-platform transfer or hardware security certification.
