# NixWard active-inference transition-support diagnostics

**Status:** additive observability API; no EFE or action-order change.

## What it adds

`ScoredActionDiagnostics` pairs:
- the existing `ScoredAction` (same EFE, pragmatic, epistemic, episodic and rationale fields);
- the `TransitionPredictionDiagnostics` that describes the evidence behind its predicted next state.

`NixActiveInference::score_action_with_diagnostics` reports one action. `score_candidates_with_diagnostics` reports all standard candidates sorted by the same ascending expected-free-energy comparison as the legacy scorer.

## Why the distinction matters

The current score formula gives epistemic value 1.0 to actions that do not meet the three-observation heuristic and 0.1 to actions that do. The world model returns the current state for an unseen action as a numeric fallback. The new result makes the evidence status explicit alongside that unchanged score:

- no examples;
- fewer than three examples;
- at least three examples.

This does not say that a curious action is safe or that a three-example model is calibrated. Epistemic value is not execution authority or permission to explore.

## Compatibility

The existing `ScoredAction` and `ActionPlan` layouts are unchanged. Existing candidate ordering still calls the legacy wrapper, which delegates to the diagnostics implementation and discards the extra metadata. The new diagnostic list is an additional API for callers that need to inspect support.

Tests compare the full score components and candidate order between the legacy and diagnostic methods for both supported and unsupported actions.

## Not yet implemented

- No EFE or curiosity weighting changes.
- No action authorization or safety gate changes.
- No autonomous probing or physical experimentation.
- No abstention policy.
- No calibrated confidence interval.

A later policy experiment may compare curiosity-driven safe probes, abstention and ordinary action selection, but it must use a preregistered matched test and independent authorization boundary.
