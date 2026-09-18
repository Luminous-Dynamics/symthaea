# Structured Foveation Evidence v1

Status: VIS-001 stacked on VIS-000.

## Purpose

Bind the existing asynchronous ventral/foveation result to explicit epistemic provenance without breaking the existing `FoveationResult` API or slowing the dorsal vision loop.

## Semantic boundary

A ventral recognition result is an **inference over observed pixels**, not a direct observation itself.

```text
camera/screen frame
      ↓
VisualObservationRef              origin = Observed at capture boundary
      ↓ crop + ventral model
FoveationResult
      ↓ VIS-001 adapter
StructuredFoveationEvidence       origin = Inferred
```

Even recognition confidence `1.0` does not change the origin to `Observed`.

## Source identity

Structured evidence requires an explicit non-zero `VisualSourceId` supplied by the caller that owns the capture stream. VIS-001 does not invent a source identity from frame number, crop coordinates, model output, or wall-clock time.

The resulting VIS-000 `VisualObservationRef` is exactly:

```text
(source_id, source_frame_id, source_timestamp_us)
```

from the completed `FoveationResult` plus the caller-owned source identity.

## Compatibility

VIS-001 intentionally leaves these existing surfaces unchanged:

- `FoveationRequest`;
- `FoveationResult`;
- `RecognizedContent`;
- `FoveationManager` dispatch/backpressure behavior;
- ventral routing;
- GWT injection behavior;
- the high-rate VisionManifold/dorsal loop.

Consumers may opt into `StructuredFoveationEvidence::from_result(...)` incrementally.

## Preserved semantics

The adapter retains:

- request identity;
- semantic HDC vector;
- recognized content;
- patch/grid location;
- source frame and source capture time;
- processing latency;
- source-patch velocity used for delayed-result compensation.

## Failure behavior

Construction fails closed when:

- source identity is absent/zero;
- recognition confidence is NaN, infinite, or outside `[0,1]` through VIS-000 validation;
- VIS-000 provenance construction fails for any future reason.

## Authority boundary

Structured foveation evidence is descriptive perception evidence only. It grants no action, camera motion, motor, robotics, targeting, execution, or other physical authority.

## Nonclaims

VIS-001 does not add a learned visual encoder, improve recognition quality, establish object identity, establish world coordinates, add bounding boxes/masks, or prove visual-model calibration. It only establishes a provenance-bearing structured bridge over the current semantic result.

## Next refinements

- VIS-003 should add region/box/mask geometry rather than pretending patch centroids are full object detections.
- A later source-boundary tranche should propagate stable `VisualSourceId` from camera/screen/phone capture owners instead of requiring an adapter caller to supply it manually.
- Model/backend identity and calibration receipts should be added before recognition confidence is used as assurance evidence.
