# Visual Observation Freshness v1

## Purpose

VIS-004F separates **last actually observed** from **still plausibly present**.

A predictor may support object permanence through an occlusion, but prediction confidence must never make a stale entity look continuously observed.

## Lifecycle

The initial vocabulary is:

- `Visible`
- `PartiallyOccluded`
- `Unobserved`
- `OccludedPredicted`
- `Lost`
- `Retired`

`Unobserved` is intentionally distinct from `OccludedPredicted`. Failure to see an entity does not by itself establish that an occluder explains the absence or that a prediction supports continued presence.

## Last-observed rule

`last_observed` may change only through `refresh_observation(...)` using current `VisualOrigin::Inferred` perception evidence that cites the exact new `VisualObservationRef`.

The following cannot refresh `last_observed`:

- remembered evidence;
- predicted evidence;
- simulated evidence;
- counterfactual evidence;
- confidence alone.

A new observation must also be temporally forward relative to the prior observation. Retrograde frame/timestamp transitions fail closed.

## Prediction rule

`apply_occlusion_prediction(...)` accepts only `VisualOrigin::Predicted` evidence whose observation lineage includes the entity's current `last_observed`.

Applying a prediction never changes `last_observed`.

A prediction may support `OccludedPredicted` while the entity is unseen. It cannot resurrect `Lost` into observed continuity.

## Freshness policy

`ObservationFreshnessPolicy` defines:

- `unobserved_after_us`
- `lost_after_us`

with the invariant:

```text
0 <= unobserved_after_us < lost_after_us
```

Freshness is evaluated against another concrete `VisualObservationRef`, used only as an evidence-bearing clock instant. The reference frame does not imply that the subject entity appears in it.

If the elapsed age reaches the unobserved horizon:

- with prediction support -> `OccludedPredicted`;
- without prediction support -> `Unobserved`.

If the elapsed age reaches the lost horizon:

- state becomes `Lost` regardless of prediction confidence.

## Clock semantics

Freshness uses the clock-domain rules already defined by `VisualObservationRef`.

- same-stream monotonic/device-local observations can be ordered;
- Unix-epoch observations can be numerically ordered across streams;
- incompatible/unspecified clock domains fail closed;
- reference observations that precede `last_observed` are rejected.

The presence of a numeric timestamp is not enough to establish comparability.

## Terminal retirement

`Retired` is terminal in v1. Observation refresh, prediction, absence marking, and freshness evaluation reject attempts to mutate a retired entity.

## Negative controls

Qualification demonstrates that:

1. prediction does not refresh `last_observed`;
2. prediction confidence `1.0` cannot prevent transition to `Lost`;
3. absence without prediction becomes `Unobserved`, not `OccludedPredicted`;
4. remembered evidence cannot refresh the observation marker;
5. a newer real observation refreshes state and clears stale prediction support;
6. incompatible clock domains fail closed;
7. retrograde observations cannot replace the last real observation.

## Integration into VIS-004D

The eventual `VisualEntityBelief` should compose `EntityObservationFreshness` rather than maintaining independent booleans such as `seen`, `occluded`, or `predicted_visible`.

Prediction state remains separable from historical observation state, so the UI can display both:

```text
last actually seen: 1.4 s ago
current state: occluded_predicted
prediction confidence: 0.92
```

without implying continuous observation.

## Nonclaims

VIS-004F does not establish calibrated object-presence probabilities, physical occlusion cause, world pose, cross-camera synchronization quality, identity correctness, navigation, targeting, manipulation, or motor authority.
