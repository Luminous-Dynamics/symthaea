# Visual Spatial Evidence v1

## Purpose

VIS-004E prevents visual coordinates with different meanings from becoming interchangeable merely because they are represented by similar numbers.

The initial boundary distinguishes:

- exact image-plane coordinates;
- exact patch-grid cells;
- dimensionless relative depth;
- calibrated camera-frame bearing.

It intentionally does **not** define metric range, camera-frame XYZ, or world-frame XYZ.

## Governing rule

```text
pixel location
  != patch-grid location
  != relative depth
  != camera bearing
  != metric range
  != camera-frame position
  != world-frame position
```

No confidence value can bridge these semantic levels.

## Image planes

`ImagePlaneRef` carries a nonzero source-owned namespace plus either:

- `RawSensor`; or
- `UndistortedRectified`.

Two rasters with identical dimensions are not coordinate-compatible unless the complete `ImageFrameRef` is equal. The frame reference includes the exact `VisualObservationRef`, image dimensions, and image-plane identity.

This means equal numeric `(x, y)` values from different observations are not silently comparable.

## Point localization

`ImagePointEvidence` is an inferred/remembered localization claim bound to one exact image frame. It carries per-axis one-sigma pixel uncertainty.

Pixel-distance comparison fails unless both points belong to the exact same observation/plane/raster.

Direct `Observed` evidence is rejected for localization because the sensor frame is observed but the selected object point is an interpretation of that frame.

## Patch-grid coordinates

`PatchGridCellRef` derives grid rows/columns from the exact image dimensions and patch size. It rejects zero patch sizes and out-of-range cells.

The grid cell remains image-space support; it is not a detector box, a bearing, or a world-space cell.

## Relative depth

The current VisionManifold stereo representation is explicitly dimensionless `0 = near, 1 = far`. VIS-004E preserves that meaning as `PatchRelativeDepthEvidence`.

Initial source vocabulary:

- `StereoDisparity`
- `MonocularRelative`
- `ProviderRelative`

A relative depth of `0.25` means only a location on the provider's normalized near/far scale. It does not mean `0.25 m`, inverse meters, disparity pixels, or any other physical unit.

No conversion from `PatchRelativeDepthEvidence` to metric range exists in this tranche.

## Camera optical frames and bearings

`CameraOpticalFrameRef` scopes an optical-frame namespace to a concrete `VisualStreamRef`. Equal numeric frame namespaces from different capture streams are unrelated.

`CameraBearingEvidence` carries:

- exact source observation;
- exact camera optical frame;
- nonzero calibration namespace;
- azimuth in radians;
- elevation in radians;
- one-sigma angular uncertainty;
- observation-backed inferred/remembered evidence.

The observation stream must exactly match the camera optical frame's stream.

Bearing evidence has no range field and no world-position conversion.

## Relationship to PR #1744

PR #1744 already established the correct conservative geometry rule: an explicit pinhole calibration may convert an already-undistorted image coordinate into **camera-frame bearing only**, with calibration provenance and uncertainty, while refusing range/world-position invention.

VIS-004E does not replace that geometry math. A follow-up adapter should modernize #1744 onto the current typed observation/stream/clock model:

```text
VisualObservationRef
  + ImageFrameRef / ImagePointEvidence
  + qualified calibration
        -> CameraBearingEvidence
```

That adapter must bind the calibration to the exact optical frame and observation timestamp/clock semantics rather than relying only on an unrelated string frame label or bare millisecond number.

## Historical-state epistemic rule

Historical spatial claims accept only `VisualEvidence::Inferred` or `VisualEvidence::Remembered` and must cite the exact source observation.

The following are rejected:

- `Observed` as though localization itself were raw sensor truth;
- `Predicted`;
- `Simulated`;
- `Counterfactual`.

Generative spatial state belongs in future prediction/counterfactual structures and cannot rewrite the historical entity state.

## Wire boundary

Invariant-bearing VIS-004E structures are constructor-only and Serialize-only in v1. Validating deserialization is intentionally deferred rather than allowing generic serde construction to bypass dimensions, namespaces, evidence lineage, or frame compatibility checks.

## Negative controls

Qualification includes tests showing that:

1. equal pixel numbers from different observations are not comparable;
2. relative stereo depth remains explicitly non-metric;
3. a camera bearing is bound to the same capture stream as its observation;
4. bearing has no metric range capability;
5. spatial evidence must cite the exact source observation;
6. prediction cannot enter historical depth/bearing state;
7. direct `Observed` evidence cannot masquerade as an inferred localization claim.

## Required follow-ups

Before metric/world positions are introduced:

1. qualify a metric range provider with explicit units and uncertainty;
2. define typed camera extrinsics/pose with provenance and validity interval;
3. define temporal compatibility between observation, range, and pose evidence;
4. propagate uncertainty through the transform chain;
5. reject transform composition across incompatible coordinate frames or clock domains;
6. prove that prediction cannot become observation during coordinate transforms.

Only then should a type such as `CameraPoint3Meters` or `WorldPoint3Meters` exist.

## Nonclaims

VIS-004E makes no claim of metric depth, SLAM/world pose, camera extrinsic calibration, cross-camera registration, object identity, navigation, targeting, manipulation, or motor authority.
