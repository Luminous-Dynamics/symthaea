# symthaea-chemosensation-bridge

Evidence-preserving adapter between `symthaea-chemosensation` and Symthaea's root multimodal cognition.

## Boundary

This crate owns the one-way composition seam between the chemical sensing domain and the root `symthaea` package. It does not move chemical sensing into the root package and does not make the root package depend on `symthaea-chemosensation`.

```text
ChemicalModalBridgeInput
        |
        v
ChemicalRootProjector
  validates evidence / spaces / timestamps / geometry
        |
        v
ChemicalRootProjection
        +
ChemicalRootContentLineage
        |
        v
ModalLineageReceipt
        +
ModalInput
        |
        v
LineagedModalInput
```

## Identity is not activation

`Olfactory` and `Gustatory` can be named at the root boundary, but this crate does not add them to the default root channel set or amodal convergence topology. A chemical input can therefore be observed and retain exact lineage while still being excluded from root fusion.

Chemesthesis remains distinct and has no mapping here because the chemical domain does not yet expose a genuine chemesthetic transduction target.

## Evidence semantics

The root lineage preserves four roles:

1. raw chemical evidence bundle identity;
2. ContinuousHV encoding-space identity;
3. projection-policy identity;
4. BinaryHV output-space identity.

The bridge carries the chemical aggregate's conflict-adjusted confidence into root cognition and retains the complete `ChemicalRootProjection` alongside the generic input so agreement and projection-quality diagnostics are not discarded.

Content identity is not authenticity, trust, or subjective experience.

## Validation intent

The crate's tests require that:

- chemical target IDs agree with root stable modality IDs;
- projection validation runs before root input construction;
- timestamp, confidence, BinaryHV, and lineage roles survive handoff;
- an olfactory input is visible as current-cycle evidence while unconfigured;
- an unconfigured chemical input cannot alter legacy visual fusion;
- gustation never aliases chemesthesis or a derived flavor representation.

The bridge should remain draft until both the root multimodal prerequisite stack and chemical projection/evidence stack have executable CI evidence.
