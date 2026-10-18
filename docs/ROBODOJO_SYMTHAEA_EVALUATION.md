# RoboDojo × Symthaea: staged external manipulation evaluation

**Status:** research plan; no RoboDojo score or performance improvement has been measured by this document.

**Working target:** the official RoboDojo benchmark at https://github.com/RoboDojo-Benchmark/RoboDojo. A separate project or release specifically called “RoboDojo-RC” has not been independently identified. If RC names a different repository or task suite, pin that source before implementation.

## Why this is worth doing

RoboDojo separates an evaluation client from policy integration. Its published benchmark contains 42 simulation tasks and 18 real-robot tasks, organized around generalization, memory, precision, long-horizon execution, and open-vocabulary instructions. XPolicyLab owns the policy-side adapter contract and supports the RoboDojo client.

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

Record the exact RoboDojo commit, XPolicyLab commit, Symthaea commit, benchmark task/config revisions, checkpoint digest, and simulator/runtime versions. Read the current official task and configuration docs before choosing tasks. “RoboDojo-RC” remains an unresolved name until a canonical URL or revision is supplied.

Use XPolicyLab's demo policy only for the documented offline/debug wiring check. It returns correctly shaped zero actions and is a protocol smoke test, **not** a useful manipulation-success baseline.

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

### Stage 2 — a small, preregistered simulation pilot

Use this candidate matrix for the pilot, then pin the official task/config revision and verify each capability label before running:

| Capability | Candidate tasks | What it probes |
|---|---|---|
| Generalization | [Stack Bowls](https://robodojo-benchmark.com/doc/sim-tasks/stack-bowls/); [Sweep Blocks](https://robodojo-benchmark.com/doc/sim-tasks/sweep-blocks/) | Placement versus coordinated tool handoff and scene/task variation |
| Memory | [Imitate Sorting Sequence](https://robodojo-benchmark.com/doc/sim-tasks/imitate-sorting-sequence/); [Press By Number](https://robodojo-benchmark.com/doc/sim-tasks/press-by-number/) | Retaining an observed sequence or numbers while executing later actions |
| Precision | [Insert Tubes](https://robodojo-benchmark.com/doc/sim-tasks/insert-tubes/); [Insert Key](https://robodojo-benchmark.com/doc/sim-tasks/insert-key/) | Alignment and insertion tolerance |
| Long-Horizon | [Organize Table](https://robodojo-benchmark.com/doc/sim-tasks/organize-table/); [Put Bottles Into Dustbin](https://robodojo-benchmark.com/doc/sim-tasks/put-bottles-into-dustbin/) | Multi-step completion and accumulated-error sensitivity |
| Open-vocabulary | [Stack Blocks by Language](https://robodojo-benchmark.com/doc/sim-tasks/stack-blocks-by-language/); [Classify Objects by Language](https://robodojo-benchmark.com/doc/sim-tasks/classify-objects-by-language/) | Grounding instruction phrases into object selection and action |

The smallest useful preflight is **Stack Bowls** for task/action wiring, then **Imitate Sorting Sequence** for an explicitly memory-dependent task, then **Insert Tubes** for precision. The XPolicyLab demo adapter documents Stack Bowls as a wiring example but emits zero actions, so it validates the protocol only; it is not a success-rate baseline. Verify the current task taxonomy before freezing the matrix because benchmark tasks and variants can evolve.

Use common task instances and paired seeds for baseline versus sidecar. Start with 10 seeds per task for screening; expand the sample if confidence intervals are inconclusive. Do not choose tasks after observing which ones favor Symthaea.

First confirm that the chosen pretrained policy is supported, its checkpoint is available under the license terms, its resource needs fit the available machine, and the environment uses a pinned XPolicyLab revision. Choose the strongest reproducible compatible baseline with available checkpoint/config artifacts—not necessarily the highest aggregate leaderboard entry, since policy scores can vary substantially by task and capability dimension. Then run baseline-only and sidecar variants under the same budget.

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
