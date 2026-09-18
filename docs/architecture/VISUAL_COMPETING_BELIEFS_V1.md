# VIS-004B — Visual Competing Beliefs v1

## Purpose

VIS-004B prevents weak or ambiguous perceptual evidence from being silently normalized into certainty.

Two distinct questions receive two distinct belief-set types:

- semantic class: "what kind of thing might this entity be?"
- identity: "which existing entity, if any, might this track correspond to?"

These questions never share one probability simplex.

## Explicit unknown mass

Every set contains candidate mass plus `unassigned_mass`.

The constructor requires the total to equal 1 within a small floating-point tolerance and never rescales values.

Example:

```text
cup        0.18
container  0.12
unknown    0.70
```

This stays exactly that distribution. It is not converted into `cup=0.60, container=0.40`.

Complete abstention is valid:

```text
candidates = []
unassigned_mass = 1.0
```

## Semantic class belief

`SemanticClassBeliefSet` is scoped to one `VisualEntityHypothesisRef` and one explicit vocabulary identity.

Each candidate has:

- validated label;
- explicit `BeliefMass`;
- evidence.

Duplicate labels are rejected.

## Track identity belief

`TrackIdentityBeliefSet` is scoped to one `VisualTrackRef`.

Initial candidate values are:

- `ExistingEntity(VisualEntityHypothesisRef)`
- `NewEntity`

Anything not represented by candidates remains in `unassigned_mass`.

Duplicate identity candidates are rejected.

## Evidence rule

Candidate support accepts only `VisualEvidence::Inferred` or `VisualEvidence::Remembered`.

Direct `Observed` support is rejected because semantic class and identity are interpretations of observation.

`Predicted`, `Simulated`, and `Counterfactual` evidence cannot enter historical belief state.

## Selection non-authority

Candidate ranking is diagnostic only. A downstream consumer must inspect `unassigned_mass` and the full candidate distribution; the existence of a largest candidate does not make that candidate established, action-authorizing, or safe to persist as fact.

## Wire validation

`BeliefMass`, candidate evidence, duplicate detection, vocabulary/label text, and mass partitioning are revalidated during deserialization.

## Explicit nonclaims

VIS-004B does not establish calibrated probabilities, semantic correctness, identity correctness, Bayesian optimality, canonical object identity, world pose, or robotics authority.

## Follow-up

VIS-004C adds an append-only belief-revision ledger. Every semantic/identity distribution change should be attributable to the observations/evidence that caused it, while predicted/simulated state remains segregated from historical belief updates.
