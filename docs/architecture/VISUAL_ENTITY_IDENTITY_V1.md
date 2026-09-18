# VIS-004A — Visual Entity Identity v1

## Purpose

VIS-004A separates four identity concepts that must not collapse into one another:

1. an object hypothesis inside one concrete observation;
2. a tracker-local continuity handle;
3. a belief-layer entity hypothesis that may span multiple tracks;
4. an identity asserted by an external system.

The governing rule is:

```text
observation-local hypothesis != track != entity hypothesis != external identity assertion
```

None of these is a canonical physical-world identity in this tranche.

## Why this is needed

The current `ObjectMemory` assigns monotonic numeric track IDs and matches new object hypotheses using appearance-HV similarity plus spatial/velocity gating. That is useful tracking machinery, but a successful association is still an inference. A numeric track ID is not a globally stable identity and an `identity_hv` is a representation of track history, not proof that two observations contain the same physical object.

VIS-004A creates a vocabulary that allows later world-model code to preserve that distinction explicitly.

## Reference levels

### `ObservationEntityRef`

Scoped to one exact `VisualObservationRef` and a nonzero local object-hypothesis ID.

The same local ID in a different observation is a different reference.

### `VisualTrackRef`

Contains:

- `tracker_namespace`
- `track_id`

`track_id` may be zero because the existing allocator starts at zero. The namespace must be nonzero.

A raw numeric track ID must never be interpreted outside its tracker namespace.

### `VisualEntityHypothesisRef`

A belief-layer entity handle:

- `belief_namespace`
- `entity_id`

This can eventually span multiple tracker handles, but remains a hypothesis rather than canonical world identity.

### `ExternalEntityAssertionRef`

Stores an external authority/source name plus an identifier value.

Examples could include inventory IDs, simulation IDs exposed through an allowed interface, or future authenticated external registries. The type records only that the identifier was asserted. It does not establish authenticity, correctness, uniqueness, or equivalence with Symthaea's own entity hypothesis.

## Associations

`EntityAssociation` records evidence that two references support either:

- `SupportsSameEntity`; or
- `SupportsDistinctEntity`.

Associations require `VisualEvidence` whose origin is `Inferred` or `Remembered`.

Direct `Observed` evidence is rejected because identity equivalence is an interpretation of observations rather than a raw sensor datum.

`Predicted`, `Simulated`, and `Counterfactual` evidence is also rejected from the historical association state. Those origins may propose future/hypothetical relations elsewhere, but must not silently rewrite historical identity.

Self-associations are rejected.

## Wire validation

Invariant-bearing references use private fields plus validating deserialization. Malformed namespaces, empty IDs, oversized external identifiers, and invalid associations are rejected while decoding.

## Explicit nonclaims

VIS-004A does not establish:

- canonical physical-world identity;
- person identity;
- biometric identity;
- cross-camera re-identification accuracy;
- tracker correctness;
- semantic class correctness;
- external-ID authenticity;
- world pose;
- robotics authority.

## Follow-up

VIS-004B should build a bounded competing-hypothesis set over these references, where multiple class/identity explanations may coexist and probabilities/confidences cannot be silently normalized into certainty.

VIS-004C should add an append-only belief-revision record connecting every major entity-belief change to the evidence that caused it.
