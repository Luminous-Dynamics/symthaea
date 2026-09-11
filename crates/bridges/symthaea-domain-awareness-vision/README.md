# symthaea-domain-awareness-vision

Evidence-only bridge between `symthaea-vision-manifold` and
`symthaea-domain-awareness`.

The bridge exists to preserve an important boundary: a visual tracker can produce
useful physical evidence without becoming an authority system.

## What it does

- converts `TrackedObject` grid state into normalized `VisualTrackEvidence`
- publishes visual persistence as a canonical `ObservationEnvelope`
- requires an explicit external calibration result before publishing bearing evidence
- evaluates camera health from reviewed deployment thresholds
- evaluates track evidence maturity from freshness, persistence, independent physical
  sources, independent modalities, and identity evidence

## What it deliberately does not do

- infer world-space range or position from image coordinates
- invent camera intrinsics or field of view
- classify a visual track as hostile
- infer intent from identity
- allow multiple algorithms fed by one physical camera to count as independent sensors
- turn confidence, risk, or track assurance into actuation authority

## Track assurance

`TrackAssuranceLevel` is an evidence-maturity state only:

```text
Unassessed
    -> Tentative
    -> Persistent
    -> Corroborated
    -> IdentityEvidenceSupported
```

A track reaches `Corroborated` only when the deployment policy's persistence,
independent-physical-source, and independent-modality requirements are met.
Identity support additionally requires a `Known` epistemic state and fresh accepted
observations referenced by the identity hypothesis.

`TrackAssuranceReport::grants_physical_authority()` is intentionally always `false`.

## Camera assurance

There are no built-in safety-critical image-quality thresholds. A deployment must
supply an explicit `CameraAssurancePolicy` covering obstruction, saturation, focus,
and environmental visibility. Loss of timing integrity or calibration makes a camera
`suspect`; stopped frames make it `unavailable`; image-quality degradation can only
reduce the sensor-health confidence ceiling.

## Spatial evidence

The vision manifold's object tracker operates in patch/grid coordinates. The bridge
therefore publishes only normalized image-plane evidence by default. Bearing evidence
requires a separately calibrated `BearingProjection` and records its calibration
reference in the resulting observation.

A future calibration adapter can map stereo/multi-camera geometry into richer
kinematic observations, but that adapter should carry explicit calibration uncertainty,
model identity, and divergence evidence rather than silently converting pixels to meters.

## Verification

```bash
cargo test -p symthaea-domain-awareness-vision
```

This crate contains no weapon-control, firing, interception, jamming, or engagement
optimization logic.
