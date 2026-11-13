# MCR² v15 — Counterfactual Reflex Value Benchmark v0.1

A deterministic, dependency-free synthetic experiment asking whether a predictive reflex **causes** a better outcome than a reactive threshold controller when both encounter the same frozen exogenous disturbance trace.

## Why this follows v14

v14 established a conservative eligibility reference: a proposal may be sent to independent projection only when its qualified time-to-boundary lower bound exceeds the full response path and reserve. v15 adds a distinct question: even when a predictor is eligible, does acting on it improve the outcome enough to justify its compute and intervention cost—and does the true predictor beat a matched sham signal?

## Arms

- `no_action`: baseline plant trajectory.
- `reactive_threshold`: triggers only after measured margin crosses a fixed threshold.
- `predictive_reflex`: uses the scenario's observable precursor and v14-style qualification checks.
- `shuffled_precursor_control`: negative control using a deliberately shuffled precursor timestamp.
- `evaluator_oracle`: clairvoyant timing upper bound; never deployable and excluded from promotion comparisons.

Every arm replays the same scenario's exogenous hazard onset and loss-per-tick trace. Actions may change the margin trajectory. The paired unit is the scenario episode, not a control tick.

## Run

```bash
python3 run_benchmark.py
python3 -m unittest discover -v
```

Outputs `report.json`. Scenario parameters, response times, compute cost and intervention cost are synthetic integer units. They are not physical seconds, joules, forces or actuator measurements.

## Current synthetic result

The 12 authored scenarios produce 4/12 safe episodes for predictive reflex, 3/12 for reactive threshold and 5/12 for the shuffled-precursor negative control. Predictive beats reactive on one paired scenario and loses on none (+1/12 paired net lift). Against the sham, predictive wins zero and loses one (-1/12 paired net lift). Therefore this v0.1 corpus does **not** establish that the true precursor signal contributes value beyond a sham trigger; on this authored set the sham does better. See the generated report and interpretation note.

This is intentionally retained as a negative control. It exposes the need for source-bound signal identity, randomized sham construction within disturbance strata and a larger held-out corpus before any causal claim.

## Critical limitations

- One-dimensional scalar-margin abstraction; not a validated physics model.
- Twelve authored scenarios are insufficient for population-level claims.
- No confidence interval or SOTA claim is supported by this fixed branch-coverage corpus.
- The evaluator oracle is an upper bound only.
- Units are synthetic and cannot be converted to physical joules or seconds.
- No simulator, ROS 2, PLC, HIL, sensor, actuator or robot was operated.
- No actuator authority is provided.

## Follow-on

Extend only after an exact task/plant is selected under Symthaea #11880 and instrumented under #4814. Bind scenarios to measured plant dynamics and real sensor/actuator timing; retain reactive, sham, stale-model and evaluator-only oracle controls. Run held-out disturbance families with episode-level uncertainty and preserve every adverse/aborted run.
