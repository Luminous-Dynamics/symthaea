# symthaea-camera-geometry

Evidence-first calibrated camera geometry for conservative conversion from normalized image-plane coordinates to **camera-frame bearing only**.

The crate intentionally does **not** infer:

- range
- world position
- physical velocity
- target identity
- intent
- physical authority

## Calibration contract

`PinholeCalibration` requires explicit:

- calibration id and durable calibration evidence reference
- exact optical `camera_frame_id`
- calibrated image dimensions
- focal lengths and principal point
- one-sigma reprojection/calibration error floor
- validated ray-radius envelope
- validity interval

Input coordinates are assumed to be in the **already-undistorted calibrated image plane**. Raw distorted pixels must be rectified by a separately qualified process before use.

The domain-awareness adapter requires the observation context's coordinate frame to match the calibration's `camera_frame_id` exactly. A numerically valid calibration cannot therefore be silently reused under another optical frame label.

## Projection

A validated normalized image point `(u, v)` is mapped to a unit optical ray and then to camera-frame azimuth/elevation. Pixel-localization uncertainty and the declared calibration error floor are conservatively combined into an angular uncertainty bound.

If the calibration is stale, malformed, outside its validated image/ray envelope, missing evidence, or bound to a different camera frame, publication fails closed.

## Verification

```bash
cargo test -p symthaea-camera-geometry
cargo test -p symthaea-domain-awareness-camera-geometry
```
