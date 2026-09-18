# Soma Visual Observation Source v1

Status: implementation contract for VIS-001S.

## Purpose

VIS-001S connects the provenance vocabulary from VIS-000R and the capture-bound foveation evidence from VIS-001R to a real capture owner: Soma screen vision.

The governing rule is conservative:

```text
raw pixels alone are not enough to claim Observed provenance
```

A direct screen observation exists only when the capture owner supplies all of:

- a validated `VisualStreamRef`;
- an explicit `VisualCaptureClock`;
- the acquisition timestamp for the concrete frame;
- the frame bytes processed under that identity.

## Two APIs, two claims

`ScreenVisionBridge::process_frame(...)` remains the compatibility path. It accepts raw pixels and uses an internal nominal frame counter for local scheduling. Its `ScreenPerception::observation` is always `None`, even if the bridge was configured with a source.

`ScreenVisionBridge::process_observed_frame(...)` is the provenance-bearing path. It requires a bridge created with `new_with_observation_source(...)`, constructs the exact `VisualObservationRef`, returns it in `ScreenPerception`, and passes the same reference into `FoveationManager::on_observed_frame(...)`.

Configuring a source does not silently upgrade compatibility calls. The caller must deliberately choose the observed API.

## Clock separation

Two clocks have different jobs and must not be conflated:

```text
capture-owner acquisition clock -> evidence identity
process-local nominal clock      -> foveation cooldown/backpressure scheduling
```

The acquisition clock may be Unix epoch, stream-monotonic, device-local, or unspecified according to VIS-000R. Its label does not prove synchronization accuracy, drift, offset, or authenticity.

The process-local clock is not serialized into observation provenance and is not presented as a sensor timestamp.

## Stream lifecycle

`VisualStreamRef` includes both source identity and stream epoch. Capture owners must issue a new stream epoch when a logical capture stream restarts in a way that may reuse frame counters or local timestamps.

Soma does not infer restart boundaries from pixel content.

## Foveation propagation

For the observed path:

```text
capture owner
  -> ScreenObservationSource
  -> process_observed_frame(captured_at_us)
  -> VisualObservationRef
  -> ScreenPerception.observation
  -> FoveationManager::on_observed_frame
  -> FoveationRequest.source_observation
  -> FoveationResult.source_observation
  -> StructuredFoveationEvidence
  -> VisualEvidence::Inferred
```

The final semantic result remains an inference about an observed crop. It does not become `Observed` merely because it was produced from direct sensor evidence.

## Failure behavior

`process_observed_frame(...)` fails closed when no observation source was configured.

Frame/provenance disagreement at the foveation boundary is delegated to VIS-001R validation and returns a `ScreenVisionObservationError::FoveationBinding(...)` rather than silently rewriting identity.

A failed provenance admission must not advance the bridge frame counter.

## Compatibility

The existing `process_frame(...)` API remains available and keeps its previous raw-pixel behavior, except that the output now makes its epistemic status explicit through `observation: None`.

No anonymous input is assigned a source ID, clock domain, stream epoch, or acquisition timestamp by guesswork.

## Qualification target

The focused VIS-001S lane must run against one exact head and cover:

- rustfmt;
- compile/check of `symthaea-vision-manifold`, `symthaea-foveation`, and `symthaea-soma`;
- tests for those crates;
- strict Clippy for those crates.

Qualification of VIS-000R or VIS-001R does not automatically qualify this descendant.

## Nonclaims

VIS-001S does not establish:

- capture-device authenticity;
- timestamp synchronization quality;
- cross-device ordering;
- exact backend/model identity;
- semantic calibration;
- object identity;
- metric world pose;
- actuation authority;
- robotics safety.

It establishes only that a capture owner can explicitly bind a Soma screen frame to typed observation provenance without upgrading anonymous raw-pixel paths.

## Follow-up

The next generic boundary should introduce a typed vision-frame injection envelope for the main cognitive service. The existing `inject_vision_frame(Vec<u8>)` API should remain unproven for compatibility; a provenance-bearing sibling API may carry dimensions/channels, `VisualObservationRef`, and acquisition semantics without inventing metadata for legacy callers.
