# RoboDojo × Symthaea: staged external manipulation evaluation

**Status:** research plan; no RoboDojo score or performance improvement has been measured by this document.

**Primary target:** [RoboDojo-RC Tier 1](https://robocurve.org/opus-5-5-robodojo-rc-tier-1/) — Robocurve's adapted, physical-robot evaluation of six RoboDojo tasks. It changes physical setups and instructions and uses task-specific partial-progress rubrics, so its scores must not be treated as interchangeable with scores from the original RoboDojo benchmark.

**Complementary target:** the [official RoboDojo benchmark](https://github.com/RoboDojo-Benchmark/RoboDojo) and [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) for simulator-scale evaluation, adapter plumbing, and broader generalization/memory/precision/long-horizon/open-instruction coverage.

## Why this is worth doing

RoboDojo-RC Tier 1 uses bimanual I2RT YAM arms with parallel-jaw grippers, absolute end-effector pose control, three camera views (top and both wrists), 224×224 frames, a recorded 10 Hz control frequency, a 40-model-call budget, a 900-step cap, and a 25% speed cap. Its six tasks are Pack And Pour Fruit, Stack Bowls, Store In Safe, Cap Pen, Stand Up Bottles, and Sort Objects. The published September cohort used 20 trials per task and model and human-scored progress from 0 to 1. See the [RC methodology and results](https://robocurve.org/opus-5-5-robodojo-rc-tier-1/) for its exact rubrics and stated limitations.

Robocurve's October 15 [Robotics Capability Index analysis](https://robocurve.org/rci/) pooled 1,055 graded trials across 11 configurations on five tasks; Stand Up Bottles was excluded from that fit because later cohorts had many invalid trials related to overheating and API timeouts. That exclusion must not hide the reliability issue from Symthaea's report: retain all six tasks and report task-specific valid-run rate, timeouts, thermal stops, and missing grades separately.

Symthaea already has a robotics bridge and a platform microbenchmark example. Those are useful starting points, but neither by itself establishes general-purpose tabletop manipulation competence. The current bridge consumes proprioceptive-style observations and modulates a hypervector; RoboDojo policies must update state from observation dictionaries, reset episode state, and emit robot-specific action chunks with valid arm and gripper dimensions. We must prove that mapping instead of assuming it.

## Core hypothesis: use Symthaea as a measurable cognitive sidecar first

Do not start by claiming Symthaea replaces a complete vision-language-action (VLA) policy. First take a reproducible, task-capable XPolicyLab policy and compare:

1. **Frozen baseline:** pretrained policy, checkpoint and configuration pinned.
2. **Baseline + Symthaea sidecar:** the same policy plus a bounded, observable cognitive state module.
3. **Ablations:** remove one sidecar capability at a time to find which component causes any measured change.
4. **Placebo control:** equal integration and latency overhead without meaningful memory/recovery signals, where feasible.

The initial sidecar should focus on capabilities Symthaea can test independently of raw pixel understanding: task/episode memory, temporal context, prediction-error tracking, confidence/uncertainty signals, structured subgoal state, and requests for a shorter action horizon or recovery/replanning when the serving API supports them. Any sidecar output must be typed, bounded, logged, and mapped to a supported action or policy-server operation.

Keep the baseline's perception and primary actuator mapping unchanged during the first experiment. This makes attribution more credible and avoids confusing a new controller, new training data, and a new cognitive mechanism in the same comparison.

Do not use Phi or an unvalidated “consciousness level” as a safety certificate. Safety decisions must rely on explicit, testable constraints: action schema and numeric checks, workspace/joint limits when exposed, collision/contact telemetry when available, timeouts, and a fail-closed safe-stop path. Treat consciousness-related quantities, if retained for research, as exploratory predictors rather than proof of competence or safety.

## Work plan

### Stage 0 — pin the benchmark and adapters

For RoboDojo-RC, record the exact task rubric/revision, Inspect Robots harness version, model/tool configuration, action budget, speed cap, setup photographs or placement annotations, robot/camera/software versions, and any run artifact IDs. The published Tier 1 report is a useful reference cohort, not a matched baseline for Symthaea unless the same robot, setup, rubric, policy budget, and conditions can be reproduced.

Use the XPolicyLab demo policy only for the official RoboDojo offline/debug wiring check. It returns correctly shaped zero actions and is a protocol smoke test, **not** a useful manipulation-success baseline. For physical RoboDojo-RC execution, first confirm access to an I2RT YAM/compatible rig through an authorized operator or remote evaluation service. The contract harness can be developed without that access, but cannot itself generate physical benchmark evidence.

### Stage 1 — contract tests with no simulator

Create small deterministic tests for the policy boundary and Rust-side state logic. These should run without Isaac Sim or model weights:

- reset clears all episode-specific state and prevents leakage between episodes;
- the same pinned observation trace and seed produce repeatable outputs where deterministic mode is intended;
- action keys, vector lengths, dtypes, and action chunk lengths match the selected embodiment and action mode;
- all numeric action values are finite and within known configured bounds;
- quaternion-like pose fields, when used, are normalized within a documented tolerance;
- missing, stale, malformed, or out-of-order observations fail closed rather than creating arbitrary actions;
- timeout, empty action chunk, server restart, and policy reset behavior are tested explicitly;
- the sidecar can be disabled so the baseline path remains an exact comparison.

Keep simulator assets and external checkpoints optional for this test layer.

### Stage 2 — RoboDojo-RC Tier 1 evaluation plan

The RC-specific task matrix is:

| RC Tier 1 task | Critical partial-progress signals | Symthaea hypothesis to test |
|---|---|---|
| Pack And Pour Fruit | Objects contained at once; bowl lifted with contents; fruit retained in target bowl | Persistent subgoal/mass-containment state reduces loss of earlier task progress |
| Stack Bowls | Overlap versus stable two-/three-bowl stack | Action-chunk length control and post-contact recovery improve final stable completion |
| Store In Safe | Objects stored and safe lid closed | Episode memory and subgoal tracking reduce omitted-object/unfinished-closure failures |
| Cap Pen | Both objects grasped; contact; actual cap attached and retained | Precision/recovery policy separates contact from success and retries only under bounded conditions |
| Stand Up Bottles | Grasp/lift and number standing upright | Uncertainty-aware grasp/reorientation reduces failed attempts; report thermal/timeouts explicitly |
| Sort Objects | Correct objects in each of two baskets; no cross-contamination | Persistent category-to-container bindings reduce swap/memory errors |

Keep each task's published rubric as discrete, observable milestones; do not replace it with a model's self-reported completion. The score is partial credit on a 0–1 scale. Also report exact completion rate (score 1.0), valid-run rate, timeouts, safety/thermal stops, model-call count, wall-clock duration, action steps, and estimated cost. These explain why a partial-progress mean changed.

The published September 2026 cohort provides context—not a guaranteed matched baseline: across 120 trials per model, GPT-6 Astra averaged 36.7% progress with 5 full completions, Opus 5.5 averaged 36.0% with 1, and Opus 5 averaged 19.9% with 2. The [October RCI analysis](https://robocurve.org/rci/) later excluded Stand Up Bottles from its capability fit because of invalid trials in later cohorts. We should not optimize against those historical means without matched setups, versions, and budgets.

The first useful experiment is not a six-task full run. First create a deterministic trace/contract test, then run a small task pilot using the same underlying vision/action model with and without Symthaea. Prioritize **Store In Safe** for persistent subgoals and **Cap Pen** for a genuine precision/contact distinction; include Stack Bowls as a simpler calibration task. Include Stand Up Bottles in reliability reporting even if some trials are invalid.

Use paired physical trials where setup resets are repeatable, but randomize run order and blind the grader to the condition. A practical screening target is at least 20 trials per condition per task to line up with the published cohort; treat this as a starting floor, not a guarantee of adequate statistical power. Bootstrap confidence intervals by task and report the cross-task macro-average. If robot access is unavailable, finish and test offline contract/evaluation tooling and label RC physical results as not run; do not substitute simulator progress for physical RC scores.

### Stage 2b — complementary official simulation pilot

Separately use official RoboDojo's [Stack Bowls](https://robodojo-benchmark.com/doc/sim-tasks/stack-bowls/) and task definitions for generalization, memory, precision, long-horizon, and open-vocabulary coverage. Pin the exact official task/config revision. Keep these results in a separate section from RoboDojo-RC, because adapted physical tasks, rubrics, and hardware are not directly comparable to simulation tasks.

Use common task instances and paired seeds for baseline versus sidecar. Start with 10 seeds per task for screening; expand the sample if confidence intervals are inconclusive. Do not choose tasks after observing which ones favor Symthaea.

First confirm that the chosen pretrained policy is supported, its checkpoint is available under the license terms, its resource needs fit the available machine, and the environment uses a pinned XPolicyLab revision. Choose the strongest reproducible compatible baseline with available checkpoint/config artifacts—not necessarily the highest aggregate leaderboard entry. Then run baseline-only and sidecar variants under the same budget.

### Stage 3 — full simulation sweep

If the pilot clears the integration gates and shows a credible signal, evaluate the full eligible simulation suite. Report task-level results as well as macro-averages across capability dimensions so easy or overrepresented tasks do not dominate the aggregate.

Run component ablations, common-seed paired comparisons, and predeclared robustness tests. Keep training data, fine-tuning steps, model checkpoints, task budgets, and environment settings identical unless the experiment explicitly studies that variable.

### Stage 4 — sim-to-real only after a safety review

Physical evaluation is a separate gate, not an automatic consequence of a good simulation score. Require an explicitly reviewed command envelope, a reachable emergency stop, conservative speed/force limits, operator supervision, and a successful dry run with a known baseline. Do not connect an experimental sidecar to physical actuators until the action mapping and shutdown behavior are independently verified.

## Metrics and proposed targets

These are **proposed acceptance targets**, not results or claims about current Symthaea performance.

| Measure | What to publish | Proposed gate or stretch target |
|---|---|---|
| Protocol correctness | Valid resets, observations, action schemas, chunks, and server lifecycle | 100% on deterministic contract tests; zero malformed or non-finite actions |
| Task success | Success rate per task and per capability dimension, with 95% confidence intervals | Stretch: +5 percentage points in overall macro-average and +10 points on a preregistered target dimension, with the paired 95% interval excluding zero |
| Non-target regressions | Per-dimension difference versus the frozen baseline | No dimension regresses by more than 2 percentage points without an explicit trade-off analysis |
| Generalization | Performance on held-out layouts/objects/conditions available in the benchmark | Report the baseline-to-shift drop; target at least a 25% relative reduction in the drop, if the perturbation protocol permits a fair comparison |
| Memory | Tasks where earlier observations/actions affect later decisions; reset isolation | Show a preregistered memory-task gain and zero episode-state leakage across resets |
| Recovery | Success and time-to-recovery after benchmark-supported recoverable disturbances or injected faults | Report the full distribution; stretch: reduce median recovery time by 25% without increasing unsafe contacts |
| Precision | Task success plus placement/alignment errors where ground truth is exposed | Success must not improve by trading away declared tolerances or safety constraints |
| Long horizon | Completion rate, interventions, and failure point by step/chunk | Report survival/completion curves, not only final averages |
| Runtime | End-to-end policy latency (p50/p95), deadline misses, CPU/GPU memory, and power/energy if measurable | p95 under 80% of the configured action deadline as a target; zero unhandled deadline misses |
| Safety | Invalid actions, limit violations, collisions/unsafe contacts when telemetry exists, safe-stop behavior | Zero invalid actions and zero unhandled limit violations in the pilot; report every observed contact/failure, not only averages |
| Data efficiency | Additional training data and fine-tuning budget | First test should require no benchmark-specific retraining of the frozen baseline; report any extra data or learning explicitly |

For success-rate comparisons, use paired task/seed results and bootstrap confidence intervals. Report raw counts and confidence intervals alongside percentages. These numeric targets are deliberately ambitious and should be revised only before—not after—looking at test results.

## Ablation matrix

Keep the base manipulation policy constant and change one item at a time:

- Base policy only.
- Base + persistent HDC/episodic memory.
- Base + temporal/CfC state.
- Base + uncertainty or prediction-error signal.
- Base + structured recovery/replanning request.
- Base + explicit safety governor.
- Full sidecar.
- Placebo sidecar with comparable call overhead, if implementable without changing policy behavior.

A module earns inclusion only if it delivers repeatable benefit, has a measured cost, and does not weaken the safety gate. If only the memory tasks improve, that is a useful scoped result—not evidence of universal robotics improvement.

## Required artifact for every run

Write one machine-readable record per episode, plus an aggregate summary. Preserve at least:

- benchmark/repository commit and task/config digest;
- XPolicyLab commit, Symthaea commit, adapter version, and checkpoint/model digest;
- task identifier, capability dimension, embodiment, action mode, and seed;
- environment/runtime/software versions and available hardware identifiers;
- baseline/variant/ablation label and exact configuration digest;
- success outcome, termination/failure reason, elapsed simulated time, action count, and recovery count where defined;
- policy latency p50/p95, deadline misses, resource measurements, and safety events when available;
- output artifact paths and SHA-256 hashes.

Do not commit large licensed assets or checkpoints. Keep experiment scripts and metadata reproducible, and document how a reviewer can recreate the run. Only call results “official RoboDojo leaderboard results” after the upstream evaluation/submission process accepts them.

## Paradigm shifts worth testing

1. **Cognition as a sidecar, not a monolith.** Preserve strong task-specific perception and manipulation while testing whether persistent symbolic/hyperdimensional state adds useful memory, recovery, and uncertainty handling.
2. **Measure the cost of intelligence.** Every gain is paired with latency, memory, power, and action-safety cost. A small controller that matches task success and improves recovery per watt may be more useful than a larger model.
3. **Benchmark recovery and calibration, not only success.** Two methods can have the same success rate while differing in whether they notice uncertainty, recover from errors, or make unsafe corrections.
4. **Treat internal claims as hypotheses.** Use ablations and baseline controls; do not infer robotics competence from a high internal metric, a synthetic simulator run, or a correlation.
5. **Make the benchmark falsifiable.** Pin versions, preregister tasks and thresholds, retain failure artifacts, and publish negative findings.

## Immediate runnable work

A dependency-free nested Rust crate now lives at **experiments/robodojo-contract/**. Its 11 unit tests cover episode identity/reset, state dimensions and finiteness, step ordering, stale action rejection, action dimensions and range checks, prediction-error replanning, latched stops, configuration rejection, and deterministic trace behavior. A small executable exercises accepted action → replan request → latched stop.

Run locally with:

```sh
cargo test --locked --manifest-path experiments/robodojo-contract/Cargo.toml
cargo run --locked --manifest-path experiments/robodojo-contract/Cargo.toml
```

This nested workspace is intentionally independent of the main Symthaea workspace and Isaac Sim, so these contract tests can be run before GPU/simulator provisioning. Passing them would show only that the contract gate behaves as specified; it would not demonstrate RoboDojo task competence or physical safety. The action validator is a bounded boundary check, not a kinematics or collision checker.

The existing **examples/robotics_benchmark_suite.rs** is a platform timing/stability microbenchmark, not a RoboDojo task-success evaluation. Its current implementation also had two compile blockers: it pushed into an immutable **results** binding and used **Instant** without importing it. This branch applies the minimal source correction. Neither the crate nor the root example has been compiled in this tool session because a local Rust toolchain is unavailable; their run status remains unverified and no test pass is claimed.

## Upstream references

- RoboDojo official benchmark: https://github.com/RoboDojo-Benchmark/RoboDojo
- RoboDojo documentation: https://robodojo-benchmark.com/doc/
- XPolicyLab shared policy interface: https://github.com/XPolicyLab/XPolicyLab
- XPolicyLab reference adapter: https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/demo_policy
- Symthaea robotics bridge: ../src/embodiment/robotics_bridge.rs
- Symthaea existing platform microbenchmark: ../examples/robotics_benchmark_suite.rs
