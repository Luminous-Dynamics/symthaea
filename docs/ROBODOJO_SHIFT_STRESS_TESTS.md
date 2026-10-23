# RoboDojo × Symthaea: shift-robust manipulation protocol

**Status:** research protocol and proposed targets only. No physical-robot results are claimed here.

## Paradigm shift: invariance and sensitivity

A robust manipulator should preserve intent when appearance, wording, camera view, or object placement changes without changing the task; it should change its plan when target identity, relation, destination, or constraints change. These are distinct capabilities. Over-invariance can make a robot persist at the wrong goal; over-sensitivity can provoke needless replanning.

Related directions include VLA-Replica's separate in-distribution/out-of-distribution evaluations, BAS-VLA's task-preserving versus semantic-breaking interventions, and the Robotics-Oriented Evaluation Protocol's emphasis on separating task failure from execution-interface faults and missing recovery. They motivate tests, not evidence that a Symthaea sidecar will reproduce published results.

References:
- https://irvlutd.github.io/VLAReplica/
- https://github.com/NEBULIS-Lab/Beyond-Appearance-Shifts
- https://www.mdpi.com/1424-8220/26/15/4757
- https://robocurve.org/opus-5-5-robodojo-rc-tier-1/
- https://robocurve.org/rci/

## Shift taxonomy to preregister

| Family | Task-preserving perturbation | Semantic-breaking condition |
|---|---|---|
| Appearance | color/texture, benign distractors | target visually resembles a distractor |
| Geometry | object placement, camera pose, legal workspace offset | target/destination relation changes |
| Lighting / occlusion | shadows, exposure, partial view | task stays fixed; test calibrated uncertainty |
| Dynamics | friction, mass, contact compliance, grip slip | physical constraints explicitly change |
| Sensor / interface | observation latency, dropped frame, bounded tool timeout | persistent fault must escalate, not loop indefinitely |
| Language / semantics | paraphrase preserving the task graph | target swap, reversed relation, changed destination |
| Object state | legal pose/rotation perturbation | required final state changes |
| Tool failure | recoverable delay, server error, rejected action | repeated failure requires escalation |

Unsafe physical faults are out of scope. Use simulation for severe conditions until robot-specific limits and stop paths are independently validated.

For each task/trial, preregister a condition ID, family, severity, seed, and SHA-256 of the canonical perturbation configuration. Save the canonical payload as well: a digest is an identity key, not proof that the shift was correctly applied. Record realized conditions as well as intended ones.

## Arms and controls

1. Frozen task-capable policy with checkpoint/configuration pinned.
2. The same policy plus a bounded Symthaea sidecar (temporal state, subgoal tracking, prediction-error monitoring, bounded replan request).
3. Fixed-threshold replan control with comparable latency budget.
4. Calibration-trained decision rule, with thresholds fit on a separate calibration split.
5. Placebo sidecar/latency control where feasible.
6. Leave-one-capability-out ablations.

Pair trial IDs across arms by task, seed, and condition. Keep vision, action mapping, speed limits, and setup constant unless explicitly declared as the intervention. Randomize trial order, blind graders to arm, and retain every timeout, fault, missing grade, and safety event. Never choose test conditions after looking at model results.

## Measurements

Primary: task-macro mean rubric progress, with full completion shown separately.

Always report valid-run rate and missing grades, timeouts, thermal stops, server faults, invalid actions, safety stops, recovery rate, replan/retry burden, median and tail latency, policy calls, action steps, estimated cost, and safety events. For forecasts, report Brier score, log loss, ECE with bin definitions, high-confidence failure rate, and risk/coverage curves on held-out trials.

For each shift, report (a) the candidate's progress delta relative to its own nominal condition and (b) paired candidate-minus-baseline differences under the exact same shift. Report task-level results to prevent a strong easy task from hiding regressions. Separate task-preserving shifts from semantic-breaking changes.

## Proposed targets (not measured results)

- Contract integrity: 100% rejection in the declared fault-injection suite for stale/mismatched actions, invalid dimensions/non-finite values, bounds violations, duplicate keys, and missing/unmatched shift assignments.
- Task utility: aim for +10 percentage points in task-macro partial progress; report completion separately and require paired uncertainty intervals to support claims.
- Mild task-preserving shifts: target no more than 10 percentage points progress degradation from nominal; report moderate and severe strata separately.
- Semantic shifts: measure correct plan changes against explicit target/relation swaps; continuing the old plan must not count as robustness.
- Calibration: ECE at or below 0.05 and high-confidence failure rate below 5% for forecasts at or above 0.90 are hypotheses only; require enough held-out samples and useful confidence intervals.
- Operational reliability: target at least 99% valid runs in a pilot. Faults must remain in overall attempt denominators.
- Contract safety: zero unapproved contract violations in the declared software fault-injection suite. Zero observed physical incidents alone is not proof of safety.

A sidecar must not pass on mean progress alone. Publish latency/cost overhead, high-confidence failures, severe-stratum regressions, all operational failure classes, and a predeclared maximum tolerated per-task regression.

## Sample-size and stopping plan

Screen many perturbations in simulation first. Then run a small physical pilot only after the robot adapter's independent limits and stop path are tested. Confirm a short list of high-value conditions with matched trials, blinded grades, and a declared maximum sample budget. Choose either a fixed sample size justified by pilot variance or a predeclared precision-based sequential rule; do not stop when results merely become favorable. Wide intervals mean inconclusive evidence, not equivalence.

Keep official RoboDojo simulation, RoboDojo-RC physical adapted tasks, and results from other embodiments in separate tables. Cross-platform comparisons need explicit task, setup, safety-envelope, scoring, and inclusion-rule equivalence.

## Implementation sequence

1. Validate a shift manifest whose exact task/trial keys match the paired trial ledger.
2. Report per-family/severity/config-digest deltas and failure classes.
3. Add adapter-side perturbation controls and provenance receipts.
4. Run simulator screening, then preregister a small physical confirmation campaign.
5. Publish all trials, including failed and ungraded attempts, with uncertainty.


## Active-sensing value-of-information experiment

The standalone Rust module `experiments/robodojo-contract/src/active_sensing.rs` now provides a pure decision primitive for choosing whether to acquire another observation. It estimates the expected change in task/risk utility and subtracts the declared energy and latency costs of sensing. This is an experimental adapter boundary—not a sensor driver, collision checker, or movement authorization.

### Paired arms

Use the same task, seed, setup, and perturbation assignments for each arm:

1. **No-extra-sensing control:** continue from the current accepted observation unless an independent hard gate requires stopping.
2. **Fixed-rate sensing:** request an additional observation at a preregistered interval, independent of state uncertainty.
3. **Uncertainty-triggered sensing:** request a sensor observation when the calibrated success/risk estimate crosses a frozen threshold.
4. **Cost-sensitive value-of-information:** request an observation when its estimated net benefit exceeds a frozen utility threshold, with hard risk/confidence gates overriding optional cost optimization.
5. **Placebo-cost control:** run the decision logic with matched latency/accounting overhead but without using the observation's task-relevant posterior, where feasible.

Keep policy checkpoint, vision/action adapter, action budget, and the set of available sensor channels fixed. Randomize arm order within paired blocks. Freeze the forecasting model, calibration split, utility units, resource budget, safety thresholds, and hypotheses before the held-out test. Do not tune the cost or confidence thresholds on test results.

### Required measurements

Record per episode: sensor requests and successful acquisitions, observation freshness, task progress/completion, expected and realized unsafe events, success/risk probabilities before and after sensing, observation energy, latency, replanning count, total episode energy/time, interventions, safe stops/escalations, and all missing/invalid runs. Report these by task, shift family/severity, evidence class, and whether sensing was optional or required by a declared gate.

Primary comparison: paired task-macro progress/completion and episode resource cost versus fixed-rate sensing. Secondary: calibration (Brier/log loss/ECE), unsafe-risk-gate decisions, risk/coverage, observations per successful task, and sensing energy per useful-work unit. Do not convert a hard safety gate into a weighted average metric.

### Negative and adversarial controls

- No-value observation: the additional sensor provides no change to a task-relevant estimate; optional sensing should not spend resources without a justified benefit.
- Required risk evidence available: if the current risk forecast violates the gate and the observation can restore it inside the declared bound, the policy should request it even if optional ROI is small.
- Required sensor unavailable, posterior forecast missing, risk remains too high after sensing, or resource budget insufficient: stop/escalate; never silently treat unknown as clear.
- Conflicting/stale observation: require explicit resolution or remain unresolved; do not average away a critical hazard.
- Forecast shift: perturb calibration under held-out lighting, occlusion, sensor delay, terrain and tool-fault families; report when the active-sensing rule itself becomes miscalibrated.
- Decision-order invariance: replay identical inputs and seed with scenario rows permuted; canonical decisions and receipts must be invariant.

### Proposed screening targets (not achieved)

- Zero cases in the declared synthetic adversarial suite where an above-limit risk proceeds without an admissible observation or safe escalation.
- At least 20% lower *sensing-only* energy-plus-latency cost than fixed-rate sensing, while task-success non-inferiority is within a preregistered 2 percentage-point margin; report separate physical units as well as any explicitly declared common utility.
- No more than 2% false required-sensing/escalation decisions on held-out in-domain controls where the required disposition is run and sufficient evidence is already available.
- Report results only when there are enough independent scenario families for a useful paired interval; otherwise label them screening or inconclusive.
- Track the cost of a wrong confident continue decision separately from an unnecessary observation. A low average cost must not hide a single violation of a hard gate.

These are prospective hypotheses. The 2025 ICLR study *The Value of Sensory Information to a Robot* found sensing criticality varies with task stochasticity and policy, and specifically motivates controlled observation-withholding and sensing-budget comparisons. This code and these proposed arms do not reproduce that study or claim an improvement. Reference: https://proceedings.iclr.cc/paper_files/paper/2025/hash/e1126028d9f1f69c13571ec462084d31-Abstract-Conference.html.
