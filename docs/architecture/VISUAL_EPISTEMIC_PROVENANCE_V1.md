# Visual Epistemic Provenance v1

Status: implementation contract for VIS-000.

## Governing invariant

```text
Observed != Remembered != Inferred != Predicted != Simulated != Counterfactual
```

No transformation may strengthen a visual claim's epistemic origin. In particular, generated or recalled visual state may never self-promote to `Observed`.

## Direct observation boundary

Only a concrete visual sensor/capture boundary may construct `VisualEvidence::observed(...)`. Direct observation evidence carries a `VisualObservationRef` containing source identity, frame identity, and capture time.

All non-observed visual origins must carry no direct-observation slot. Remembered and inferred claims must retain at least one original observation reference. Predicted, simulated, and counterfactual state may be grounded in observation lineage but remains generated state regardless of confidence.

## Confidence semantics

Confidence is finite and bounded to `[0, 1]`. Confidence expresses uncertainty within the declared origin; it never changes the origin itself. A `Simulated` claim at confidence `1.0` remains simulated.

## Trust-boundary validation

`VisualEvidence` must be revalidated after deserialization or any untrusted transport boundary. Validation rejects:

- observed claims without a direct observation reference;
- non-observed claims carrying the direct-observation slot;
- remembered/inferred claims without observation lineage;
- duplicate parent observation references;
- non-finite or out-of-range confidence.

## Authority boundary

This contract is epistemic metadata only. It grants no motor, robotics, camera-motion, actuation, targeting, execution, or other physical authority.

## Compatibility strategy

VIS-000 introduces the provenance vocabulary without changing existing `VisionManifold`, foveation, object-memory, scene-memory, mental-movie, or cognitive-loop behavior. Later tranches should thread this type through those boundaries one at a time, with migration adapters and negative controls.

## Follow-up sequence

1. VIS-001 — structured foveation/entity evidence with provenance.
2. VIS-002 — provider-neutral dense visual encoder evidence.
3. VIS-003 — region/box/mask-aware object geometry.
4. VIS-004 — persistent object-centric visual belief state.
5. VIS-005 — action-conditioned visual prediction, explicitly `Predicted`/`Counterfactual`.
6. VIS-006 — non-actuating epistemic look proposals.
7. VIS-008 — Symtropy hidden-ground-truth perception crucible.
8. VIS-009 — Visual Mind observatory rendering origin/provenance distinctly.

## Nonclaims

VIS-000 does not improve recognition accuracy, tracking quality, world-model fidelity, visual memory, mental imagery, robotics performance, or scientific understanding. It establishes the type-level vocabulary and validation rules required to measure and compose those later capabilities without collapsing observation and imagination into one representation.
