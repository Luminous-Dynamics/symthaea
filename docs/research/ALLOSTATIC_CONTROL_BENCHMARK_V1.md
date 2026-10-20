# Capability-Preserving Allostatic Control Benchmark v1

Status: experimental specification + dependency-free synthetic harness; not yet locally compiled or empirically qualified.

## Research question

Does explicitly separating immediate stress, cumulative allostatic load, epistemic uncertainty, and future viability improve closed-loop task performance under delayed costs and repeated disturbances, compared with reactive and task-only baselines?

This is deliberately narrower than a claim that stress should replace the loss function. The proposed architecture keeps task and prediction losses as learning signals, represents regulatory state explicitly, and uses a separate viability monitor for essential constraints.

## Paradigm shift

Move from one scalar objective to a measured, constrained control problem:

1. **Task quality** — progress toward the external task.
2. **Instantaneous stress** — current regulatory demand.
3. **Cumulative exposure** — area under the burden curve, with explicit step duration.
4. **Epistemic uncertainty** — uncertainty about state/forecast, not a multiplier that erases estimated severity.
5. **Future viability** — projected resource and load margins under candidate actions.
6. **Future capability** — the ability to recover, switch strategies, and complete later tasks.

No single weighted score can hide a safety violation. Report task reward, breach counts, exposure, peak, resource cost, recovery, and conservatism separately. A later Pareto or constrained-selection layer may compare trade-offs; v1 does not assign normative weights or claim a universally optimal policy.

## Existing Symthaea work this builds on

- [Issue #268](https://github.com/Luminous-Dynamics/symthaea/issues/268): separate epistemic precision from viability burden.
- [Issue #269](https://github.com/Luminous-Dynamics/symthaea/issues/269): separate mean projected burden from cumulative exposure and other temporal measures.
- [Issue #3431](https://github.com/Luminous-Dynamics/symthaea/issues/3431): predict the state of the computational/embodied substrate before constraints are crossed.
- Existing [AllostaticStress benchmark](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-psych-bench/src/benchmarks/neuromod/allostatic_stress.rs): transmitter-baseline and simplified chronic-stress/recovery dynamics.

The new crate is independent of cognitive-loop internals so we can falsify the decision rule before wiring it into live cognition. It is an engineering abstraction, not a model of subjective distress, human physiology, or consciousness.

## Harness

Location: tools/allostatic-control-bench/

The crate has no third-party dependencies. The executable emits a row-per-episode CSV for 5 scenario families × 32 deterministic seeds × 6 controller conditions (960 rows).

### Scenario families

| Scenario | Purpose |
|---|---|
| Calm | Detect needless conservatism and false alarms |
| Acute | Distinguish a brief high-demand pulse from a chronic problem |
| Chronic | Test accumulation, sustained regulation, and recovery |
| Repeated | Test incomplete recovery between disturbances |
| Regime shift | Test forecast uncertainty and changed operating conditions |

These profiles are synthetic and intentionally small. They are not an industry benchmark, a physiological calibration, or evidence of real-world safety.

### Controller conditions

| Condition | Information/action rule | Role |
|---|---|---|
| Task-only | Maximize immediate task progress, rest only near depletion | Loss-only baseline |
| Stress-reactive | Rest based on current stress and a resource floor | Instantaneous-response baseline |
| Allostatic-reactive | Rest based on accumulated load and a resource floor | Cumulative-load baseline |
| Predictive viability | Use a noisy forecast, its confidence, projected reserve and projected load | Candidate method |
| Shuffled-forecast control | Same decision structure with a time-shifted forecast | Tests whether useful forecast content matters |
| Oracle predictive | Predictive method given the simulator's true short-horizon forecast | Diagnostic upper bound only; not deployable |

The oracle is not a fair competitor and must never be reported as a deployable baseline. The shuffled forecast is a negative control. Neither should count as independent evidence that predictive control generalizes.

### Reported metrics

- Task reward and number of push/maintenance steps.
- Cumulative exposure (discrete area under allostatic load), mean load, and peak load.
- Unsafe ticks and missed-limit decisions under the harness's declared synthetic limits.
- False-alarm maintenance ticks under calm, apparently safe conditions.
- Resource cost, episode failure indicator, and time from the last disturbance to a declared recovered state.
- Recovery not observed by the end of the horizon is emitted as NA (right-censored), not fabricated as a successful recovery.

A missed-limit counter is an early-warning diagnostic; it is not a proof that a runtime safety constraint was enforced. This first simulator does not yet contain an independent action shield or formal barrier certificate.

## Measurement contract

Three semantics must remain separate:

- Intensity / normalized mean: sum of samples divided by the declared number of samples.
- Cumulative exposure: sum of nonnegative burden × declared abstract time step. It grows with longer exposure at constant burden.
- Epistemic uncertainty: 1 − confidence in this toy harness. This is only a simple diagnostic proxy, not a calibrated probabilistic uncertainty model.

The Rust unit tests check that extending constant burden changes cumulative exposure but not the mean, changing confidence alone does not erase viability burden, equal means can conceal different peaks, and a predictive action can differ from a reactive one before current stress rises.

Abstract model steps are not seconds. Cross-environment aggregation requires explicit time-step normalization. Missing forecasts, invalid model states, and unavailable evidence must remain explicit rather than being silently interpreted as safety.

## Baselines and current research context

This is not yet an SOTA comparison. Relevant external methods must be tested in the same environments and under identical budgets:

- Homeostatic reinforcement learning: [Neurocomputational theories of homeostatic control](https://www.sciencedirect.com/science/article/pii/S1571064519301009).
- Adaptive reward design: [IJCAI 2026 survey](https://www.ijcai.org/proceedings/2026/863).
- Control-theoretic safety regulators: [CTR-Net, 2025](https://www.sciencedirect.com/science/article/pii/S2667305325001231).
- Model predictive control and reinforcement-learning combinations: [RL-MPC benchmark study](https://doi.org/10.1016/j.apenergy.2021.118346).
- Practical comparison caution: a real-world building-energy study reported similar costs for simple rule-based, tree, and MPC methods while its RL approach was still training; more complex policies do not automatically win ([2024 study](https://www.sciencedirect.com/science/article/pii/S2666546824001149)).

The v1 harness implements none of these external algorithms. The next comparison should use independently reviewed implementations of rule-based control, MPC, homeostatic RL, and a constrained/safe RL method. Compare on common scenario traces, observations, action spaces, tuning budgets, and compute budgets. Reuse established benchmark environments when practical rather than claiming this custom simulator is an accepted SOTA suite.

## Qualification protocol

### Stage 0 — semantics and reproducibility

1. Run formatting, tests, and Clippy on the exact commit.
2. Independently regenerate the CSV twice and compare byte digests.
3. Add mutation/negative controls that deliberately break exposure normalization, merge confidence into burden, or scramble the forecast. Each weakened contract must be detected by a test.
4. Bind the result receipt to the exact Git commit, command, scenario version, seeds, and output digest.

### Stage 1 — held-out synthetic tournament

- Keep the 32 development seeds separate from a newly generated and frozen held-out seed set.
- Use paired scenario/seed comparisons across all policies.
- Report per-scenario distributions, median and 95th percentile, worst case, failures, and uncertainty intervals; do not report only a pooled mean.
- Bootstrap paired differences at the episode level (and at the scenario-family level where feasible); publish raw CSV, aggregation code, and frozen seeds.
- Run ablations: remove cumulative load, ignore forecast confidence, replace forecast by the shuffled control, and disable recovery.
- Test observation degradation, forecast bias, missing forecasts, slower recovery, resource budget changes, and unannounced regime changes.
- Add a separate action-shield experiment: measured controller performance is not a substitute for enforcement of safety constraints.

### Stage 2 — external baseline tournament

Use common environments and compare with:
1. Hand-designed threshold/rule-based control.
2. Model-predictive control with a declared model and horizon.
3. Homeostatic RL with explicit drives.
4. Constrained/safe RL or a control-barrier-function approach.
5. The task-only and stress-only ablations.

Use the strongest tuned non-oracle baseline as the main comparator, not only the weakest strawman. Freeze training and tuning budgets before examining held-out results.

### Stage 3 — substrate/embodiment relevance

Only after the synthetic tournament, connect the model to validated telemetry such as memory pressure, queue depth, thermal headroom, battery reserve, network quality, actuator health, and calibration drift. Each adapter must declare units, sampling intervals, sensor uncertainty, limits, and provenance. Do not map abstract stress onto physical danger without evidence.

## Proposed advancement gates (targets, not results)

The candidate should advance only if it meets all applicable gates on frozen held-out tasks:

- **Safety:** zero observed hard-limit violations in the declared constrained test envelope, with a separate uncertainty bound and clear statement that this is not a universal guarantee.
- **Burden:** at least 25% lower 95th-percentile unsafe exposure than the strongest reactive baseline, with the paired 95% interval excluding zero improvement.
- **Benign performance:** no more than 3% task-reward loss against task-only control in the calm regime.
- **False alarms:** no more than a 5-percentage-point increase in calm-regime false-alarm rate.
- **Recovery:** lower recovery exposure or time without hiding right-censored episodes.
- **Forecast value:** predictive controller must outperform the shuffled-forecast negative control on held-out regime changes; otherwise the forecast mechanism has not been shown useful.
- **Compute:** quantify p50/p95 decision latency and memory overhead; a provisional target is under 10% decision-path overhead versus the same base controller, subject to the actual runtime budget.
- **Robustness:** no one scenario family may regress severely while the pooled average improves.

These thresholds are proposed screening gates, not values drawn from a standard or evidence that the prototype already achieves them. The 25% improvement target must not be tuned post hoc on held-out seeds.

## Limitations and stop conditions

- The harness is a deterministic toy environment with hand-chosen dynamics and thresholds.
- Its forecasts and failure rules are synthetic; there are no human data, physical-device telemetry, calibrated probabilities, formal proofs, or external baseline implementations in v1.
- The measured allostatic variable is not inherently a feeling, reward, suffering, or goal.
- A lower synthetic burden score alone does not establish better intelligence or consciousness.
- If the candidate only improves its own weighted score, fails under small threshold perturbations, or becomes needlessly conservative, do not integrate it.
- Do not replace Symthaea's existing loss/learning objectives or modify the qualified interoception v0.1 semantics as part of this experiment. Keep the prototype isolated until evidence justifies a separately versioned integration.

## Reproduction

From the repository root:

    cargo fmt --manifest-path tools/allostatic-control-bench/Cargo.toml -- --check
    cargo test --manifest-path tools/allostatic-control-bench/Cargo.toml
    cargo clippy --manifest-path tools/allostatic-control-bench/Cargo.toml --all-targets -- -D warnings
    cargo run --release --manifest-path tools/allostatic-control-bench/Cargo.toml > allostatic-control-v1.csv

The authoring environment used for this initial draft had no Rust toolchain available. The commands above are therefore instructions for exact-head qualification, not claimed execution results.
