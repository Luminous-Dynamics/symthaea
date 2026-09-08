# Spark Scenario Missing-Prediction Boundary v1 — design / qualification contract

Status: staged architecture/correctness contract only. No product implementation. No scientific or action authority.

Tracks: #909

Base audited: `2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`.

## Current defect

`greedy_sequence()` simulates the current MAP hypothesis world after selecting an experiment.

When the selected design contains an `ExpectedOutcome` for the MAP hypothesis, it synthesizes an observation from that prediction and runs `belief.update()`.

When the MAP hypothesis has no prediction, current main instead creates an artificial observation with rate `-1.0` so that it falls outside every predicted class, then uses that synthetic datum in a Bayesian update.

Thus missing model information can alter posterior beliefs and all subsequent planning steps.

## Required theorem

```text
PredictionUnavailable
    != SimulatedObservation
    != NoClassMatchedObservation
    != Counterevidence
```

Missing predictive coverage must not be converted into evidence.

## Proposed result semantics

A future scenario simulator should return an explicit typed result such as:

```text
ScenarioPredictionResultV1 {
    Simulated {
        assumed_hypothesis,
        observation,
        model_profile,
    },
    PredictionUnavailable {
        assumed_hypothesis,
        experiment_ref,
    },
    PredictionInvalid {
        reason,
    },
    ObservationModelUnavailable {
        required_channels,
    },
}
```

Only `Simulated` may feed a synthetic Bayesian update.

## Fail-closed rollout behavior

Preferred v1 behavior for `PredictionUnavailable`:

1. do not construct `ObservedOutcome`;
2. do not call `HypothesisBelief::update()`;
3. preserve the pre-step belief unchanged;
4. stop the scenario branch, or explicitly skip the candidate under a declared analysis-only policy;
5. emit a machine-readable reason in the scenario/decision receipt.

Do not silently substitute a default, sentinel, midpoint, background value, zero, negative value, or uniform likelihood.

## Real negative observations remain separate

This contract does not declare all negative background-subtracted neutron-rate measurements impossible.

If a later SCI-008 measurement model allows a real or simulated negative value, that value must carry a genuine observation/predictive-model lineage.

A numeric value of `-1.0` is not a valid sentinel for epistemic absence.

## Required current-main negative control

Future executable qualification should construct:

- a belief whose MAP hypothesis is H_missing;
- a design with machine-readable predictions for at least one other hypothesis but no prediction for H_missing;
- enough outcome classes for the current updater to change relative weights.

Before repair, require the fixture to demonstrate:

1. `greedy_sequence()` reaches the missing-prediction fallback;
2. a synthetic `-1.0` observation is constructed;
3. `belief.update()` changes the posterior or downstream entropy/selection trajectory despite no H_missing prediction existing.

After repair require:

1. scenario result is `PredictionUnavailable`;
2. no `ObservedOutcome` is constructed for that branch;
3. belief is byte/semantically unchanged across the unavailable step;
4. no later sequence step is derived from fabricated evidence;
5. the reason is retained in the rollout/planning receipt;
6. real/simulated observations remain distinguishable from unavailable predictions.

## Relationship to #868

#868: missing prediction must not become invented **likelihood**.

This contract: missing prediction must not become invented **observation**.

Together they establish:

```text
absence of predictive model
    -> explicit coverage deficiency
    -> no probability or evidence invented by the common planner
```

## Relationship to #892 and #906

A MAP-scenario rollout is only defined where the assumed scenario supplies a predictive model for the selected experiment.

A future scenario receipt should bind:

```text
assumed_hypothesis
experiment_ref
prediction_status
synthetic_observation?  // only when prediction_status == Simulated
pre_step_belief
post_step_belief?
```

The ordinary one-step real-evidence planner does not need to manufacture future outcomes at all.

## Non-claims

This contract does not calibrate Spark likelihoods, choose a full measurement model, establish MAP rollout optimality, authorize physical experiments, or claim every existing experiment has complete predictions.
