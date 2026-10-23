# Standalone RoboDojo contract gate

This dependency-free Rust crate is the first runnable, simulator-independent test layer for the Symthaea × RoboDojo-RC work. It checks episode boundaries, observation shape/finiteness/order, stale action rejection, action dimension and numeric bounds, prediction-error-triggered replanning, and latched stop behavior. It also provides partial-progress trial records, task-by-variant summaries that retain invalid-run/timeout counts, deterministic bootstrap confidence intervals, and paired baseline-versus-candidate comparisons stratified by task.

## Run without Isaac Sim or the root workspace

~~~sh
cargo test --locked --manifest-path experiments/robodojo-contract/Cargo.toml
cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml
cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml -- --summarize experiments/robodojo-contract/examples/synthetic_trials.tsv
cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml -- --compare experiments/robodojo-contract/examples/synthetic_trials.tsv baseline symthaea_sidecar
cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml -- --calibrate experiments/robodojo-contract/examples/synthetic_forecasts.tsv experiments/robodojo-contract/examples/synthetic_trials.tsv
~~~

The analysis commands demonstrate the CLI using **synthetic fixture data only**. They are not RoboDojo-RC measurements. The nested workspace and lockfile are intentional: contract tests should not have to resolve or compile Symthaea's large workspace, download a policy checkpoint, start Isaac Sim, or wait for its runners.

## What approval means

An approved proposal is only valid against this crate's declared vector dimensions and per-coordinate numeric limits, for the current episode and latest accepted observation. It is **not** a kinematics check, collision check, workspace guarantee, policy-quality measure, or physical safety certificate. Robot-specific adapters must add those constraints where the appropriate state/telemetry is available.

Prediction error is a caller-defined scalar and must be calibrated for the selected policy. A high value suppresses action approval and requests replanning; repeated high values latch a stop. It is not used as proof of safety, consciousness, or task competence.

The calibration module also exposes an expected-value intervention rule: it compares the estimated success-probability improvement from replanning, multiplied by an explicit failure-cost estimate, against the intervention cost. If the current success probability is below a declared continue threshold and no justified intervention remains, it returns StopAndEscalate rather than authorizing continuation. All costs must use the same utility unit and probabilities must be calibrated on held-out data. This is a research primitive, not a safety controller, and it has no direct actuator authority.

The `active_sensing` module adds a value-of-information decision for acquiring another observation. It prices expected task-success improvement and reduction in unsafe-event probability against sensing energy and latency, using one explicit utility scale. A mandatory risk/confidence gate takes priority over optional expected utility: if the required sensor is missing, its posterior forecast is unavailable, the risk remains above the declared bound, or sensing cannot fit the remaining resource budget, the result is `StopAndEscalate` rather than pretending the current evidence is sufficient. When no hard gate requires more evidence, sensing is selected only when its estimated net benefit meets the declared minimum.

The sensing function has no sensor or actuator I/O and does not establish that its forecasts are calibrated. Its probability and cost inputs must come from a preregistered evaluation, and the caller must re-evaluate after the new observation while retaining its independent collision, workspace, interlock, and actuator checks. Eleven synthetic unit tests cover value-positive sensing, low-value sensing, risk-bound recovery, unavailable/missing evidence, over-budget cases, and invalid inputs. These fixtures test decision arithmetic only; they are not robot performance measurements.

Research grounding: Krishna, Hu and Jayaraman, *The Value of Sensory Information to a Robot* (ICLR 2025), empirically studies sensing value by withholding task-relevant observations at different states and finds that sensing criticality depends strongly on task stochasticity and policy. This module is a new cost/risk-gated decision primitive; it does not reproduce that study or prove it improves any policy. The benchmark should compare no-extra-sensing, fixed-rate sensing, uncertainty-triggered sensing, and this cost-sensitive rule with matched trials and held-out conditions. Source: https://proceedings.iclr.cc/paper_files/paper/2025/hash/e1126028d9f1f69c13571ec462084d31-Abstract-Conference.html.

The `sensing_benchmark` module provides matched-trial accounting for five arms: no-extra-sensing, fixed-rate, uncertainty-triggered, value-of-information, and placebo-cost. It requires exactly one record per arm in each scenario/seed/split/evidence cell. It reports completion and scored progress separately, calibrated Brier score when forecasts exist, sensing-only energy and latency with measured/missing counts, observation requests/acquisitions, total-episode resources, safety events, and paired comparisons against fixed-rate sensing. Family-cluster bootstrap intervals are withheld when there are fewer than five independent scenario families or too few bootstrap replicates. Missing sensing-cost observations are never imputed as zero, and any recorded safety violation marks the audit as failed while preserving the trial in the report.

Eight synthetic tests exercise full five-arm matching, missing-arm rejection, safety-violation reporting, missing-energy handling, order invariance, progress/completion consistency, acquisition/request consistency, and missing forecasts. The arm records are generated fixture data, not a RoboDojo result. The output is a measurement contract; no arm is claimed to outperform another until real paired episodes run under a preregistered setup.

## Trial TSV schema

The CLI requires this exact tab-separated header, in this order:

~~~text
task_id variant_id trial_id progress completed valid_run timed_out failure_class elapsed_seconds model_calls action_steps estimated_cost_usd safety_events
~~~

Fields are separated by tabs (the display above uses spaces). IDs should be stable slugs/identifiers without tabs. The progress field is a number in [0,1], or NA when no valid final judge score exists. Booleans must be true or false. An attempt with progress=NA must set valid_run=false and completed=false; it still counts as an attempted trial, timeout/runtime/cost observation, and non-completion. The failure_class must be one of none, timeout, thermal_stop, server_error, invalid_action, safety_stop, or other. timed_out=true must agree with failure_class=timeout. Invalid runs need an explicit non-none failure class; a timeout can still be a valid, partially graded run.

- --summarize emits task-by-variant rows with attempts, validity, full completions, rubric mean and bootstrap interval, elapsed time, model calls, action steps, estimated cost, timeouts, thermal stops, server errors, invalid actions, safety stops, other failures, and safety events.
- --compare requires exactly matching task/trial IDs across the chosen baseline and candidate; unpaired runs are rejected. It reports paired progress delta with a task-stratified bootstrap interval, plus completion/valid-run-rate, runtime/cost/safety, and individual operational failure-class deltas.
- Bootstrap count defaults to 2,000 and seed to 42. Reuse the same input and seed to reproduce the interval.

## Probability forecasts and calibration

The optional forecast TSV uses this exact header:

~~~text
task_id variant_id trial_id predicted_success_probability
~~~

The probability must be emitted **before** the task outcome is known. The forecast file deliberately does not contain an outcome label: --calibrate joins each forecast to the trial ledger by task, variant, and trial ID, deriving full completion and run validity from the authoritative trial row. The sets of IDs must match exactly; missing, extra, or duplicate rows are rejected instead of silently biasing calibration. Partial rubric progress remains in the trial ledger and is not relabeled as binary success. Invalid, interrupted, or timed-out runs count as non-completions for end-to-end reliability.

Run --calibrate FORECASTS.tsv TRIALS.tsv [BIN_COUNT] [HIGH_CONFIDENCE_THRESHOLD]. It reports Brier score, log loss, fixed-bin expected calibration error (ECE), failure rate among forecasts above the chosen confidence threshold, and area under a confidence-ranked risk/coverage curve. Lower Brier, log loss, ECE, and risk/coverage area are generally better. The high-confidence failure rate is especially relevant to a recovery gate: a system that predicts 90% success should not fail often on the trials it labels that confidently.

Do not tune the confidence threshold on held-out test data. Split calibration and evaluation by scenes/tasks or a preregistered trial set where feasible; report both the ungated base policy and the gated system. These are decision-quality diagnostics, **not** safety certificates or proof of consciousness. The synthetic forecast fixture exists only to test file parsing and metric arithmetic.

## Scope boundary

The statistics layer treats a valid partial grade as scored progress and a missing grade as an invalid run. Invalid runs remain in the attempted-trial and reliability denominators; they are excluded only from the mean of graded progress. Paired progress deltas require the same task/trial IDs on both sides and macro-average tasks equally before bootstrapping so an overrepresented easy task cannot dominate the comparison.

This crate currently validates the integration boundary and scoring logic only. It does not connect to RoboDojo-RC, XPolicyLab, a pretrained policy, a simulator, or actuators. Passing these tests is not a RoboDojo-RC task result. The next integration step is to write the thin XPolicyLab adapter and run a simulator/physical evaluation before comparing baseline versus sidecar task progress.

## Shift manifest and robustness audit

Compare a pinned baseline and candidate within each pre-registered perturbation stratum:

~~~sh
cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml -- --shift-audit \
  experiments/robodojo-contract/examples/synthetic_trials.tsv \
  experiments/robodojo-contract/examples/synthetic_shift_plan.tsv baseline symthaea_sidecar

cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml -- --sensing-audit \
  experiments/robodojo-contract/examples/synthetic_sensing_trials.tsv
~~~

The exact shift-plan header is:

~~~text
task_id trial_id condition_id family severity seed perturbation_spec_sha256
~~~

Fields are tab-separated. Each task/trial pair has one manifest assignment shared by both variants. Families: nominal, appearance, geometry, lighting, occlusion, dynamics, sensor_latency, instruction_semantics, object_state, tool_failure. Severities: clean, mild, moderate, severe. The SHA-256 identifies the immutable perturbation configuration artifact; this harness validates the digest format but does not dereference or recompute that artifact.

The audit rejects duplicate manifest keys, malformed digests, unmatched or missing rows, unpaired trials, and unexpected variants. Results are kept separate by family, severity, and exact spec digest. They include paired progress deltas/intervals where sample size permits, completion and valid-run deltas, runtime/cost, safety events, and operational failure-class deltas. Sparse strata emit NA intervals rather than implying precision.

The shift-plan fixture is synthetic test data, not an RC benchmark result. Physical evaluation must preserve the canonical configuration payload beside its digest, measure realized perturbations where possible, retain unsuccessful attempts, and pre-register conditions and stopping rules.

## Active-sensing benchmark ledger

Run `--sensing-audit experiments/robodojo-contract/examples/synthetic_sensing_trials.tsv [BOOTSTRAPS] [SEED]` to compare these preregistered arms using matched scenario/seed/evidence cells:

- `no_extra_sensing`
- `fixed_rate`
- `uncertainty_triggered`
- `value_of_information`
- `placebo_cost`

The exact tab-separated header is:

~~~text
scenario_id scenario_family task_id trial_id seed split evidence_class arm progress valid_run completed safety_violation safety_event_count sensor_requests sensor_acquisitions sensing_energy_j sensing_latency_ms total_energy_j elapsed_ms predicted_success_probability
~~~

The fields are tab-separated, in the exact order above. Numeric measurements may use `NA` only when unmeasured; the audit leaves them missing rather than treating them as zero. Every scenario/seed/split/evidence group must contain exactly one row for all five arms. Progress and completion must agree; acquisitions cannot exceed requests; probabilities and measurements are validated for finite ranges. Any declared safety violation produces a failed qualification result but remains in the printed ledger.

The audit prints per-arm valid-run/completion/progress/calibration/resource metrics and paired candidate-minus-fixed-rate comparisons. Progress uncertainty is bootstrapped by scenario family, not individual rows. Fewer than five families or fewer than 100 replicates suppresses the interval; this is a screening limitation, not evidence of equivalence. The included 30-row example is synthetic arithmetic smoke data only. It demonstrates parsing, pairing, missing-cost handling and report formatting; it is not a RoboDojo score.
