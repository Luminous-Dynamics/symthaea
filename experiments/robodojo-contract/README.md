# Standalone RoboDojo contract gate

This dependency-free Rust crate is the first runnable, simulator-independent test layer for the Symthaea × RoboDojo-RC work. It checks episode boundaries, observation shape/finiteness/order, stale action rejection, action dimension and numeric bounds, prediction-error-triggered replanning, and latched stop behavior. It also provides partial-progress trial records, task-by-variant summaries that retain invalid-run/timeout counts, deterministic bootstrap confidence intervals, and paired baseline-versus-candidate comparisons stratified by task.

## Run without Isaac Sim or the root workspace

~~~sh
cargo test --locked --manifest-path experiments/robodojo-contract/Cargo.toml
cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml
cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml -- --summarize experiments/robodojo-contract/examples/synthetic_trials.tsv
cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml -- --compare experiments/robodojo-contract/examples/synthetic_trials.tsv baseline symthaea_sidecar
cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml -- --calibrate experiments/robodojo-contract/examples/synthetic_forecasts.tsv
~~~

The analysis commands demonstrate the CLI using **synthetic fixture data only**. They are not RoboDojo-RC measurements. The nested workspace and lockfile are intentional: contract tests should not have to resolve or compile Symthaea's large workspace, download a policy checkpoint, start Isaac Sim, or wait for its runners.

## What approval means

An approved proposal is only valid against this crate's declared vector dimensions and per-coordinate numeric limits, for the current episode and latest accepted observation. It is **not** a kinematics check, collision check, workspace guarantee, policy-quality measure, or physical safety certificate. Robot-specific adapters must add those constraints where the appropriate state/telemetry is available.

Prediction error is a caller-defined scalar and must be calibrated for the selected policy. A high value suppresses action approval and requests replanning; repeated high values latch a stop. It is not used as proof of safety, consciousness, or task competence.

## Trial TSV schema

The CLI requires this exact tab-separated header, in this order:

~~~text
task_id variant_id trial_id progress completed valid_run timed_out elapsed_seconds model_calls action_steps estimated_cost_usd safety_events
~~~

Fields are separated by tabs (the display above uses spaces). IDs should be stable slugs/identifiers without tabs. The progress field is a number in [0,1], or NA when no valid final judge score exists. Booleans must be true or false. An attempt with progress=NA must set valid_run=false and completed=false; it still counts as an attempted trial, timeout/runtime/cost observation, and non-completion.

- --summarize emits task-by-variant rows with attempts, validity, full completions, rubric mean and bootstrap interval, elapsed time, model calls, action steps, estimated cost, timeouts and safety events.
- --compare requires exactly matching task/trial IDs across the chosen baseline and candidate; unpaired runs are rejected. It reports paired progress delta with a task-stratified bootstrap interval, plus completion/valid-run-rate deltas and runtime/cost/safety deltas.
- Bootstrap count defaults to 2,000 and seed to 42. Reuse the same input and seed to reproduce the interval.

## Probability forecasts and calibration

The optional forecast TSV uses this exact header:

~~~text
task_id variant_id trial_id predicted_success_probability success valid_run
~~~

The probability must be emitted **before** the task outcome is known. success=true means the benchmark's full completion criterion was achieved; partial rubric progress remains in the trial TSV and is not silently relabeled as binary success. An invalid, interrupted, or timed-out run uses success=false and valid_run=false, so confidence in a task is judged against end-to-end reliability.

Run --calibrate FORECASTS.tsv [BIN_COUNT] [HIGH_CONFIDENCE_THRESHOLD]. It reports Brier score, log loss, fixed-bin expected calibration error (ECE), failure rate among forecasts above the chosen confidence threshold, and area under a confidence-ranked risk/coverage curve. Lower Brier, log loss, ECE, and risk/coverage area are generally better. The high-confidence failure rate is especially relevant to a recovery gate: a system that predicts 90% success should not fail often on the trials it labels that confidently.

Do not tune the confidence threshold on held-out test data. Split calibration and evaluation by scenes/tasks or a preregistered trial set where feasible; report both the ungated base policy and the gated system. These are decision-quality diagnostics, **not** safety certificates or proof of consciousness. The synthetic forecast fixture exists only to test file parsing and metric arithmetic.

## Scope boundary

The statistics layer treats a valid partial grade as scored progress and a missing grade as an invalid run. Invalid runs remain in the attempted-trial and reliability denominators; they are excluded only from the mean of graded progress. Paired progress deltas require the same task/trial IDs on both sides and macro-average tasks equally before bootstrapping so an overrepresented easy task cannot dominate the comparison.

This crate currently validates the integration boundary and scoring logic only. It does not connect to RoboDojo-RC, XPolicyLab, a pretrained policy, a simulator, or actuators. Passing these tests is not a RoboDojo-RC task result. The next integration step is to write the thin XPolicyLab adapter and run a simulator/physical evaluation before comparing baseline versus sidecar task progress.
