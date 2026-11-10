# Paired outcome-transition diagnostics for robotics shift evaluation

**Status:** implemented as a generic Rust analysis extension; awaiting exact-head hosted validation. The fixture tests are synthetic. This is not a RoboCasa-Shift reproduction or robot-performance result.

## Why completion-rate deltas are not enough

A paired clean/perturbed study needs to report more than the difference in aggregate success rates. If one clean success becomes a failure while one clean failure becomes a success, the net change is zero, but two episode outcomes changed. A net ΔSR therefore does not measure outcome sensitivity.

RoboCasa-Shift makes this concrete. Its public formal report describes a Camera Orientation L1 condition with ΔSR = 0.000 but 4 Success→Failure and 4 Failure→Success pairs (8/80 changed outcomes). It also shows that Camera Orientation L2 and Gaussian Blur L1 have the same ΔSR (0.188) while the stage depth of failure differs: among success→failure cases, 19.0% versus 36.4% fail before the first tracked milestone, with mean normalized progress 0.437 versus 0.356. This motivates keeping task outcome and execution-depth diagnostics separate.

Reference implementation and records:
- https://github.com/HaochengWang23/RoboCasa-Shift
- https://arxiv.org/abs/2610.27861
- https://huggingface.co/datasets/HaochengWang23/robocasa-shift

The published formal design selects four clean-qualified RoboCasa365 tasks, evaluates 20 matched base episodes for each, and runs a clean condition plus 12 single-factor perturbations: illumination, camera orientation, robot-base initialization, and Gaussian blur at three severities. The formal report contains 1,040 rollouts. Its primary policy is GR00T N1.5 Composite-Seen; this is a specific external comparator, not a universal industry leaderboard.

## Outcome-state contract

For each baseline/candidate pair, classify each arm independently:

- **Success:** valid run and official completion.
- **Failure:** valid, scored run that did not complete.
- **Unresolved:** invalid/unscored run. This is not silently treated as a task failure.

The audit emits the complete matrix:

| Baseline \ Candidate | Success | Failure | Unresolved |
|---|---:|---:|---:|
| Success | SS | SF | SU |
| Failure | FS | FF | FU |
| Unresolved | US | UF | UU |

Derived values:
- `resolved_outcome_pairs = SS + SF + FS + FF`
- `changed_resolved_outcome_pairs = SF + FS`
- `unresolved_outcome_pairs = SU + FU + US + UF + UU`

The existing completion-rate delta remains useful and is reported separately. The matrix is not a replacement for progress, calibration, resource, operational-fault, or safety metrics.

## Implementation

The Rust `shift` module now reports all nine cells per exact `(family, severity, perturbation_spec_sha256)` stratum. The shift CLI emits these columns alongside existing progress and operational deltas. The classification retains invalid runs as unresolved. Two regression tests cover:
1. Bidirectional Success↔Failure flips with net completion delta exactly zero.
2. An invalid baseline paired with a successful candidate, retained as Unresolved→Success rather than Failure→Success.

The report is deterministic because pair identity is matched before stratum aggregation, and rows are grouped in ordered maps. The perturbation digest is a declared identifier; this contract does not prove the simulator actually applied the perturbation.

## What is deliberately not implemented here

RoboCasa-Shift's StageTracker uses task-specific simulator-state predicates. Our generic RoboDojo trial record currently has no stage-roster or milestone-receipt fields, so this patch does **not** claim to reproduce first-incomplete-stage or normalized milestone progress. The next native adapter must supply a frozen ordered milestone roster and evaluator-side observations/receipts. It must preserve:
- official success independently from milestone completion;
- stage order, applicability, unresolved/missing milestones and evidence references;
- first incomplete milestone and normalized progress;
- source state/config digests and per-episode seed identity.

Do not synthesize milestone labels from the policy's own narrative or use them as a substitute for the official judge.

## Evaluation targets

For the first native comparison, freeze the same policy/checkpoint, action executor, task seeds, initial states and evaluator across clean and shifted arms. Report:
- clean success and progress (the competence gate);
- shift-specific paired ΔSR and the 3×3 transition matrix;
- stage/milestone progress and first incomplete stage;
- invalid/unresolved attempts, runtime faults and safety events;
- per-condition energy, latency, action/model-call budgets;
- paired confidence intervals clustered by base episode and macro-averaged by task.

Only claim an improvement if the candidate preserves clean competence, reduces harmful Success→Failure transitions, does not merely increase unresolved attempts, and improves the task/resource Pareto frontier across held-out shift families. Report Failure→Success transitions as well; do not optimize robustness by hiding useful recovery. Preregister the clean-competence threshold and shift levels before evaluation.

The next benchmark ladder is complementary, not a merged score:
- **ActiveArena:** active visual evidence acquisition and memory under ID/OOD.
- **RoboCasa-Mem:** environment/process memory and behavior-conditioned evaluation.
- **RoboCasa-Shift:** matched distribution shifts and stage-aware failure diagnostics.
- **RoboDojo / RoboDojo-RC:** manipulation capability, partial progress and physical confirmation.

Never combine their task-success denominators or declare a composite SOTA score without a separately preregistered cross-benchmark protocol.
