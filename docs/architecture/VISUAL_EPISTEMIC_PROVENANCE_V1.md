# Visual Epistemic Provenance v1

Status: implementation contract for VIS-000, refined by `VISUAL_OBSERVATION_IDENTITY_V2.md`.

## Governing invariant

```text
Observed != Remembered != Inferred != Predicted != Simulated != Counterfactual
```

No transformation may strengthen a visual claim's epistemic origin. In particular, generated or recalled visual state may never self-promote to `Observed`.

## Direct observation boundary

Only a concrete visual sensor/capture boundary may construct `VisualEvidence::observed(...)`.

A direct observation carries a `VisualObservationRef` containing:

- stable source identity;
- an explicit stream epoch so restarts/reconnections cannot reuse identity accidentally;
- frame identity scoped to that exact stream;
- capture time;
- an explicit capture-clock domain.

The stronger stream/clock semantics are frozen in `VISUAL_OBSERVATION_IDENTITY_V2.md`.

All non-observed visual origins must carry no direct-observation slot. Remembered and inferred claims must retain at least one original observation reference. Predicted, simulated, and counterfactual state may be grounded in observation lineage but remains generated state regardless of confidence.

## Confidence semantics

Confidence is finite and bounded to `[0, 1]`. Confidence expresses uncertainty within the declared origin; it never changes the origin itself. A `Simulated` claim at confidence `1.0` remains simulated.

## Trust-boundary validation

`VisualEvidence` is validated during deserialization and may also be revalidated after other trust-boundary crossings. Validation rejects:

- observed claims without a direct observation reference;
- non-observed claims carrying the direct-observation slot;
- remembered/inferred claims without observation lineage;
- duplicate parent observation references;
- non-finite or out-of-range confidence.

Nested stream identity is likewise fail-closed: zero source identity or zero stream epoch cannot deserialize as a valid stream reference.

## Time semantics

Numeric timestamps are not assumed mutually comparable merely because they use microseconds. Cross-stream or cross-device temporal reasoning must respect the declared capture clock and, when bounded synchronization matters, consume separately qualified time-integrity evidence.

## Authority boundary

This contract is epistemic metadata only. It grants no motor, robotics, camera-motion, actuation, targeting, execution, or other physical authority.

## Compatibility strategy

VIS-000 introduces the provenance vocabulary without changing existing `VisionManifold`, foveation, object-memory, scene-memory, mental-movie, or cognitive-loop behavior. Later tranches should thread this type through those boundaries one at a time, with migration adapters and negative controls.

## Follow-up sequence

1. VIS-000R — restart-safe observation/clock identity.
2. VIS-001 — structured foveation/entity evidence with provenance.
3. VIS-002 — provider-neutral dense visual encoder evidence.
4. VIS-003 — region/box/mask-aware object geometry.
5. VIS-004 — persistent object-centric visual belief state.
6. VIS-005 — action-conditioned visual prediction, explicitly `Predicted`/`Counterfactual`.
7. VIS-006 — non-actuating epistemic look proposals.
8. VIS-008 — Symtropy hidden-ground-truth perception crucible.
9. VIS-009 — Visual Mind observatory rendering origin/provenance distinctly.

## Nonclaims

VIS-000 does not improve recognition accuracy, tracking quality, world-model fidelity, visual memory, mental imagery, robotics performance, clock synchronization, or scientific understanding. It establishes the type-level vocabulary and validation rules required to measure and compose those later capabilities without collapsing observation and imagination into one representation.
