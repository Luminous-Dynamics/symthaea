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
7. **Viable option breadth** — how many actions the current condition is actually authorized to choose remain inside the declared one-step viability region, reported as both a time-average and an episode minimum. Policy-only conditions count Push and Maintain; shielded conditions may also count EmergencyRecover.

No single weighted score can hide a safety violation. Report task reward, breach counts, exposure, peak, resource cost, recovery, and conservatism separately. A later Pareto or constrained-selection layer may compare trade-offs; v1 does not assign normative weights or claim a universally optimal policy.

## Existing Symthaea work this builds on

- [Issue #268](https://github.com/Luminous-Dynamics/symthaea/issues/268): separate epistemic precision from viability burden.
- [Issue #269](https://github.com/Luminous-Dynamics/symthaea/issues/269): separate mean projected burden from cumulative exposure and other temporal measures.
- [Issue #3431](https://github.com/Luminous-Dynamics/symthaea/issues/3431): predict the state of the computational/embodied substrate before constraints are crossed.
- [Issue #8243](https://github.com/Luminous-Dynamics/symthaea/issues/8243): follow-up experiment for model mismatch, noisy/missing state telemetry, and robust shield evaluation.
- Existing [AllostaticStress benchmark](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-psych-bench/src/benchmarks/neuromod/allostatic_stress.rs): transmitter-baseline and simplified chronic-stress/recovery dynamics.

The new crate is independent of cognitive-loop internals so we can falsify the decision rule before wiring it into live cognition. It is an engineering abstraction, not a model of subjective distress, human physiology, or consciousness.

## Harness

Location: tools/allostatic-control-bench/

The crate has no third-party dependencies. The episode executable emits 1,920 rows: 5 scenario families × 32 deterministic seeds × 6 controller policies × 2 shield settings. A second Rust binary emits a distributional summary and same-seed paired shield deltas with deterministic percentile-bootstrap intervals.

### Scenario families

| Scenario | Purpose |
|---|---|
| Calm | Detect needless conservatism and false alarms |
| Acute | Distinguish a brief high-demand pulse from a chronic problem |
| Chronic | Test accumulation, sustained regulation, and recovery |
| Repeated | Test incomplete recovery between disturbances |
| Regime shift | Test forecast uncertainty and changed operating conditions |

These profiles are synthetic and intentionally small. They are not an industry benchmark, a physiological calibration, or evidence of real-world safety. The candidate forecaster receives an exogenous, noisy warning stream: acute, chronic, and repeated-disturbance profiles can supply lead signals, while the regime-shift warning deliberately remains at baseline through the shift. The candidate controller does not read realized future loads; only the explicitly labeled oracle condition does. Forecast confidence is independently varied and controls forecast-noise scale. This isolates decision-rule value from perfect future information, but does not test how a real forecasting model learns or generalizes.

### Controller conditions

| Condition | Information/action rule | Role |
|---|---|---|
| Task-only | Maximize immediate task progress, rest only near depletion | Loss-only baseline |
| Stress-reactive | Rest based on current stress and a resource floor | Instantaneous-response baseline |
| Allostatic-reactive | Rest based on accumulated load and a resource floor | Cumulative-load baseline |
| Predictive viability | Use a noisy external warning stream, its confidence, projected reserve and projected load | Candidate method |
| Shuffled-forecast control | Apply the same controller to a deterministic permutation of noisy forecast-value/confidence pairs | Removes temporal alignment while preserving the set of warning observations |
| Oracle predictive | Predictive method given the simulator's true future short-horizon load | Diagnostic upper bound only; not deployable |

The oracle is not a fair competitor and must never be reported as a deployable baseline. The shuffled forecast is an offline negative control: it preserves the generated warning observations but permutes their time order, so it is not a deployable policy. Neither should count as independent evidence that predictive control generalizes.

### Policy × shield factorial

Each of the six policies is evaluated with the shield disabled and enabled, giving 12 condition cells per scenario/seed. The shield is separate from policy selection and checks the candidate action against the same declared one-step simulator transition model:

- viability limits: next resource reserve ≥ 0.15 and next allostatic load ≤ 0.88;
- if the proposed action is predicted to cross a limit, try `Maintain`;
- if `Maintain` is also predicted to cross a limit, try the task-interrupting `EmergencyRecover`;
- if none of the available actions is predicted to satisfy both limits, execute the best-effort recovery action and count the event as `no_safe_action_ticks`.

The shield and simulator deliberately share the same synthetic transition function. This removes implementation drift between the prediction used for vetoing and the transition used for scoring, but it also means the shield is **model-perfect by construction**. Zero violations in shielded synthetic runs is a semantic regression test—not evidence of safety under model error, uncertain state estimates, physical dynamics, or external tasks. The follow-up [issue #8243](https://github.com/Luminous-Dynamics/symthaea/issues/8243) requires perturbing shield-model parameters and state estimates, and comparing against independently implemented control-barrier/MPC filters.

### Reported metrics

- Task reward and number of push/maintenance steps.
- Cumulative exposure (discrete area under allostatic load), mean load, and peak load.
- Unsafe ticks, per-episode violation indicator, and missed-limit proposals (counterfactual raw-policy actions), all under the harness's declared synthetic limits.
- **Violation exposure:** sum across ticks of normalized reserve shortfall `max(0, 0.15 - reserve) / 0.15` plus normalized load excess `max(0, load - 0.88) / 0.12`.
- **Peak violation severity:** maximum single-tick normalized severity across the two limits; conditional mean over violating episodes is reported as `NA` when no episode violates.
- Shield veto count, no-safe-action count, and emergency-recovery steps, so avoided hazards and the cost of the shield remain visible.
- False-alarm maintenance count, number of pre-action false-alarm opportunities, and false-alarm rate given opportunity. The denominator is measured before action so controller behavior cannot change whether a timestep counts as an opportunity; a zero denominator is reported as `NA`.
- Resource cost, minimum reserve, terminal reserve, and terminal allostatic load as separate future-option proxies.
- Viable action-set capacity: mean safe one-step action count, per-episode minimum safe-action count, and the worst episode's minimum across seeds. Counts respect action authority: unshielded policies can select Push/Maintain; the shielded condition may also invoke EmergencyRecover. These measure local option breadth under the toy model, not long-horizon reachability or value of future options.
- Episode failure indicator and time from the last disturbance to a declared recovered state.
- The summary reports observed recovery count and mean among observed recoveries **alongside the right-censoring rate**; it does not treat unrecovered episodes as successful or silently use the observed-only mean as an unconditional estimate. It also reports mean viable action breadth and the worst episode-minimum viable-action count across the seed panel.
- Recovery not observed by the end of the horizon is emitted as NA (right-censored), not fabricated as a successful recovery.

A missed-limit counter counts raw policy proposals whose predicted next state crosses a limit, even when the shield vetoes the proposal. It must not be misreported as a realized violation. The shield is a model-perfect one-step filter for the toy simulator; it is not a formal barrier certificate and does not qualify a real runtime safety boundary.

## Measurement contract

Three semantics must remain separate:

- Intensity / normalized mean: sum of samples divided by the declared number of samples.
- Cumulative exposure: sum of nonnegative burden × declared abstract time step. It grows with longer exposure at constant burden.
- Epistemic uncertainty: 1 − confidence in this toy harness. This is only a simple diagnostic proxy, not a calibrated probabilistic uncertainty model.

The Rust unit tests check duration-vs-mean semantics, reject non-finite exposure evidence, separate confidence from viability burden, test warning permutation and an unannounced regime shift, exercise a shield veto/recovery choice, measure viable action-set breadth, and assert that shielded episodes preserve one-step viability under the exact synthetic model. The report binary also checks the exact 28-column episode schema, option-capacity ranges, trajectory/counter consistency, CSV violation-flag consistency, duplicate condition/seed rejection, recovery right-censoring, nearest-rank tail estimates, and deterministic paired-bootstrap calculations. The generated summary has 26 columns; the paired shield-delta report has 18.

Abstract model steps are not seconds. Cross-environment aggregation requires explicit time-step normalization. Missing forecasts, invalid model states, and unavailable evidence must remain explicit rather than being silently interpreted as safety.

## Baselines and current research context

This is not yet an SOTA comparison. Relevant external methods must be tested in the same environments and under identical budgets. Use established suites where the state/action mapping is defensible rather than inventing scores that cannot be compared with prior work:

- Homeostatic reinforcement learning: [Neurocomputational theories of homeostatic control](https://www.sciencedirect.com/science/article/pii/S1571064519301009).
- Adaptive reward design: [IJCAI 2026 survey](https://www.ijcai.org/proceedings/2026/863).
- Control-theoretic safety regulators: [CTR-Net, 2025](https://www.sciencedirect.com/science/article/pii/S2667305325001231).
- Model predictive control and reinforcement-learning combinations: [2026 MPC/RL synthesis survey](https://doi.org/10.1016/j.arcontrol.2026.101045) and [RL-MPC benchmark study](https://doi.org/10.1016/j.apenergy.2021.118346).
- [Safe-Control-Gym](https://github.com/learnsyslab/safe-control-gym) provides constrained/disturbed CartPole and quadrotor settings with reference controllers including PID, LQR, MPC, PPO/SAC, and safety filters. It is a strong external robotics/control comparison candidate.
- [Safety-Gymnasium](https://safety-gymnasium.readthedocs.io/en/stable/introduction/about_safety_gymnasium.html) provides safe-RL environments with explicit reward and cost signals. Its cost/violation reporting is a better comparison pattern than a single reward score.
- The September 2026 preprint [Evaluation Metrics for Safe Reinforcement Learning / SafeRLEval](https://arxiv.org/abs/2609.15315) argues that expected cumulative cost alone hides violation frequency, severity, and consistency across tasks/bounds. Follow its distributional and per-constraint reporting principle: pair aggregate safety cost with violation counts/severity, worst-case/tail metrics, and task/bound-specific results.
- Practical comparison caution: a real-world building-energy study reported similar costs for simple rule-based, tree, and MPC methods while its RL approach was still training; more complex policies do not automatically win ([2024 study](https://www.sciencedirect.com/science/article/pii/S2666546824001149)).

The v1 harness implements none of these external algorithms. The next comparison should use independently reviewed implementations of rule-based control, MPC, homeostatic RL, and a constrained/safe RL method. Compare on common scenario traces, observations, action spaces, tuning budgets, and compute budgets. For a first external tournament, use Safe-Control-Gym and/or Safety-Gymnasium where a valid adapter is possible, and add allostatic delayed-cost tasks as a clearly labeled supplementary suite. Keep external benchmark execution in an isolated research/evaluation adapter; do not add a Python runtime dependency to production Symthaea merely to run comparison baselines. Our custom toy simulator is not an accepted SOTA suite.

## Qualification protocol

### Stage 0 — semantics and reproducibility

1. Run formatting, tests, and Clippy on the exact commit.
2. Independently regenerate the CSV twice and compare byte digests.
3. Add mutation/negative controls that deliberately break exposure normalization, merge confidence into burden, or scramble the forecast. Each weakened contract must be detected by a test.
4. Bind the result receipt to the exact Git commit, command, scenario version, seeds, and output digest.

### Stage 1 — held-out synthetic tournament

- Treat the shipped 32-seed panel as development/qualification data, not final held-out evidence; freeze a separate held-out seed panel and new disturbance variants before tuning.
- Use `allostatic-control-report` for the per-condition distribution summary and paired shield effects. Its 4,000-resample percentile bootstrap is deterministic and seed-paired; its intervals quantify variation over this synthetic seed panel only, not unseen scenario families.
- Use paired scenario/seed comparisons across all policies.
- Report per-scenario distributions, median and 95th percentile, worst case, failures, frequency/severity of limit violations, and uncertainty intervals; do not report only a pooled mean.
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
- **Burden:** at least 25% lower 95th-percentile normalized violation exposure than the strongest reactive baseline, with the paired 95% interval excluding zero improvement.
- **Benign performance:** no more than 3% task-reward loss against the strongest reactive baseline that satisfies the declared viability limits in the calm regime. Report task-only reward separately as a throughput reference, even if task-only violates those limits.
- **False alarms:** no more than a 5-percentage-point increase in calm-regime false-alarm rate, calculated as false-alarm actions divided by pre-action opportunities.
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

## Reproduction and provenance

From the repository root, on a Rust-enabled checkout:

    bash tools/allostatic-control-bench/qualify.sh <exact-tested-commit>

The output schema is now versioned in Rust as `EPISODE_CSV_HEADER`; the analyzer rejects any noncanonical header, out-of-range option-capacity metrics, and inconsistent episode summaries. The repository includes a path-filtered [Allostatic Control Benchmark workflow](https://github.com/Luminous-Dynamics/symthaea/blob/research/capability-preserving-allostasis-v1/.github/workflows/allostatic-control-bench.yml) intended to run this same qualification on the exact PR head and preserve artifacts. The workflow is present in the branch, but this authoring environment cannot execute Rust locally; no passing runner result is asserted. The qualification script checks that the checkout matches the supplied commit; runs formatting, locked tests, and Clippy; executes the episode benchmark twice; rejects non-byte-identical episode and report outputs; checks complete scenario × seed × policy × shield coverage; and writes a SHA-256 provenance receipt for all outputs and source inputs. By default it produces `allostatic-control-v1.csv`, `allostatic-control-v1-summary.csv`, `allostatic-control-v1-paired-deltas.csv`, and `allostatic-control-v1.csv.receipt.txt`. Pass a second argument to choose a different episode CSV path; the summary and paired filenames follow that base name. Keep all outputs and the receipt together when publishing results. The script intentionally does not decide whether the benchmark passes the scientific advancement gates; those require analysis of the output.

The authoring environment used for this draft has no Rust toolchain available. The qualification script and Rust tests have therefore **not** been executed here; the repository query at the time of authoring also returned no workflow run or check status for the current PR head. This specification and prototype are not qualified until a Rust-enabled environment runs the script on the exact PR head and the receipt is independently checked.
