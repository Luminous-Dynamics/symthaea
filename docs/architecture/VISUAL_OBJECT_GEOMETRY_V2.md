# Visual Object Geometry v2

Status: implementation contract for VIS-003.

## Purpose

VIS-003 introduces an honest geometry vocabulary between Symthaea's current patch/centroid object representation and future detector, segmentation, and pose backends.

The governing rule is:

```text
having spatial extent does not imply having detector-box evidence
```

The existing tracker remains a centroid/patch-cluster system. VIS-003 does not relabel it as a conventional bounding-box multi-object tracker.

## Geometry levels

`VisualObjectGeometry` distinguishes five representation levels:

1. `CentroidPoint` — a point claim only. No object-area claim exists.
2. `PatchClusterExtent` — the envelope of HDC patches that supported an object hypothesis. This is useful spatial support, but it is not a detector-produced bounding box.
3. `DetectorBox` — a rectangle explicitly emitted by a detector/localizer.
4. `SegmentationMask` — exact binary foreground support represented as validated RLE.
5. `Keypoints` — named landmark positions with per-keypoint confidence.

These levels describe what evidence exists, not a ranking of quality.

## Coordinate contract

Geometry lives in image pixel coordinates:

- origin: top-left pixel corner;
- +x: right;
- +y: down;
- rectangles are half-open: `[left, right) × [top, bottom)`;
- centroids may be sub-pixel positions;
- all geometry is validated against the concrete frame width and height.

No camera-frame bearing, depth, metric pose, or world coordinate is implied by this module.

## Patch-cluster adapter

`VisualObjectGeometry::from_patch_hypothesis(...)` converts the existing `ObjectHypothesis` patch membership into:

- a validated pixel centroid;
- sorted unique supporting patch indices;
- patch-grid extent;
- a clipped pixel envelope.

The envelope is typed `PatchClusterExtent`, never `DetectorBox`.

## Metric eligibility

`GeometryMetricEligibility` states which metric families the representation can honestly support:

| Geometry | Centroid distance | Patch overlap | Box IoU | Mask IoU | Keypoint distance |
| --- | --- | --- | --- | --- | --- |
| Centroid point | yes | no | no | no | no |
| Patch-cluster extent | yes | yes | **no** | no | no |
| Detector box | yes | no | yes | no | no |
| Segmentation mask | yes | no | no | yes | no |
| Keypoints | yes | no | no | no | yes |

A false eligibility flag means the representation does not establish the required geometry. It is not a low performance score.

In particular, a patch-cluster envelope must never be evaluated or advertised as a detector box merely because it can be drawn as a rectangle.

## Epistemic contract

`InferredObjectGeometry` accepts only valid `VisualEvidence` whose origin is `Inferred`.

A box, mask, patch envelope, or keypoint set is an interpretation of sensor observations. It is not the raw observation itself, even at confidence `1.0`.

This preserves the VIS-000/VIS-001 invariant:

```text
Observed sensor evidence -> Inferred geometry
Observed != Inferred
```

## Uncertainty

Every geometry estimate carries `GeometryUncertainty`:

- confidence in `[0, 1]`;
- optional 1-sigma centroid uncertainty in pixels;
- optional 1-sigma extent uncertainty in pixels.

Non-finite, negative, or out-of-range uncertainty values fail construction.

No claim is made that a backend's confidence is statistically calibrated unless that backend is separately qualified for calibration.

## Segmentation-mask contract

`SegmentationMaskRle` uses sorted, non-overlapping row-major runs. Construction rejects:

- zero-sized frames;
- empty foreground support;
- zero-length runs;
- out-of-bounds runs;
- overlapping or out-of-order runs;
- arithmetic overflow.

Bounds and centroid are derived from the validated foreground support.

## Wire boundary

Invariant-bearing geometry types are constructor-only and Serialize-only in this tranche.

VIS-003 deliberately does **not** derive `Deserialize` for validated masks, uncertainty envelopes, geometry support, complete geometry, or inferred geometry. A future wire format must implement validating deserialization rather than bypassing constructors.

Simple leaf/value vocabulary such as `PixelPoint`, `PixelRect`, `MaskRun`, `GeometryKind`, and metric-eligibility metadata may remain ordinary serde values, but they do not by themselves constitute a validated geometry claim.

## Evaluation nonclaims

VIS-003 does not establish or report:

- HOTA;
- IDF1;
- MOTA;
- detector AP;
- box IoU performance;
- mask IoU performance;
- keypoint AP/PCK;
- re-identification accuracy;
- object permanence quality.

Those metrics require a real producer and benchmark whose evidence representation is eligible for the metric.

The existing centroid-tracking benchmark remains valid as the historical baseline and must not be retrospectively renamed as box-level MOT.

## Integration sequence

After VIS-003 qualifies:

1. expose patch-cluster geometry alongside current `ObjectHypothesis` without changing tracker decisions;
2. add a detector/provider interface that can emit genuine `DetectorBox` evidence;
3. add optional mask/keypoint providers;
4. evolve object memory to retain geometry history and uncertainty;
5. only then add representation-appropriate MOT/detection/segmentation benchmark lanes;
6. feed qualified geometry into the persistent visual world model.

## Authority / nonclaims

Geometry evidence grants no camera movement, motor, targeting, manipulation, navigation, robotics, or other actuation authority. It also does not infer metric depth or world pose from image coordinates.
