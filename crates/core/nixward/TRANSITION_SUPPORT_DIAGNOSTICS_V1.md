# NixWard transition support and prequential error diagnostics

**Status:** non-breaking diagnostic API. The existing prediction vector and action-selection behavior remain unchanged.

## Support states

`NixWorldModel::predict_state_with_diagnostics` distinguishes:

- `NoExamples`: the action category has no learned transition observations. For compatibility, the returned numeric vector is still the current state, but it is only an identity placeholder—not an evidence-supported no-op.
- `FewerThanThreeExamples`: the action category has one or two observations.
- `AtLeastThreeExamples`: the existing `has_learned` sample-count heuristic is met.

The three-example threshold is not a confidence guarantee and must not be treated as calibration.

## Prequential delta error

For every observation after the first, the implementation compares the new observed state delta against the existing mean delta **before** incorporating the new observation. It records the running mean squared L2 residual and the count of available residual observations.

This is a descriptive prequential training-stream diagnostic. It is not:
- a calibrated predictive variance;
- a probability of success;
- a prediction interval;
- evidence that a new environment/mechanism will behave similarly.

It should be calibrated against held-out action × context × mechanism clusters before a planner uses it as a risk estimate.

## Compatibility and claim boundaries

- `predict_state` delegates to the new API and returns the same ContinuousHV value as before.
- Unknown actions still return the current state through the legacy method, but callers can now inspect `NoExamples`.
- No expected-free-energy score, action ranking, safety gate or execution behavior is changed.
- This PR does not create CfC/HDC-LTC transition prediction.
- No prediction diagnostic grants execution authority.

## Follow-on experimental design

Compare identity/no-op, action mean-delta and future context-conditioned models on:
- unknown actions versus known no-op actions;
- same action category with different parameters;
- context and environment-family shifts;
- failed, partial, timed-out and unknown outcomes;
- held-out mechanisms and action sequences.

Report support status, prequential error, held-out response distortion, false no-change rate, abstention, downstream regret and independent-cluster uncertainty separately. A future policy may choose to abstain on unsupported transitions, but that policy needs its own matched evaluation and authorization boundary.
