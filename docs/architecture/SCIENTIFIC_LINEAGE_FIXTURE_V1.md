# Scientific lineage fixture v1

This fixture defines the first cross-domain lineage contract for the shared
scientific substrate. It is deliberately synthetic: it demonstrates typed
identity and provenance boundaries, not physical performance.

## Canonical lineage

```
theorem --constrains--> physical_model --instantiates--> simulation
simulation --produces--> prediction --supports--> scientific_claim
process --produces--> material --has_property--> predicted_property
experiment --observes--> observation --quantifies--> uncertainty
```

## Evidence boundaries

- A theorem constrains a model; it does not establish that the physical model matches reality.
- A simulation produces a prediction; a prediction is not a measurement.
- A process produces a material candidate; the candidate is not automatically qualified.
- An experiment produces an observation; uncertainty remains an explicit object.
- `supports` is epistemic and must never become a deterministic qualification invalidation edge.
- Qualification is a separate projection over explicitly admitted evidence.

## Required invariants

1. Every object has a stable `EngineeringObjectId`.
2. Every relation has a stable relation kind and endpoint semantics.
3. Prediction and observation remain distinct object kinds.
4. Predicted and measured properties remain distinct.
5. Uncertainty remains explicit rather than being folded into a scalar score.
6. Epistemic relations remain outside deterministic engineering closure.
7. Removing a model, parameter set, process route, toolchain, or calibration input invalidates only projections that actually depend on it.
8. Historical lineage remains addressable after currentness changes.

## Next implementation step

Encode this fixture in a graph-level test using the relation endpoint oracle,
then add a projection test demonstrating that the scientific DKG can be richer
than the bounded CP-04 qualification DAG without amplifying authority.
