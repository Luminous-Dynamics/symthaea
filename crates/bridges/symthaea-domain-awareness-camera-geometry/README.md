# symthaea-domain-awareness-camera-geometry

Adapter from `symthaea-camera-geometry` into the existing evidence-only vision/domain-awareness bridge.

The adapter takes:

- normalized `VisualTrackEvidence`
- an explicit `PinholeCalibration`
- localization uncertainty in pixels
- observation timing/health/provenance

and produces a domain-awareness `Measurement::Bearing` only after the camera calibration passes its validity and ray-envelope checks.

It deliberately does **not** produce:

- range
- world position
- physical velocity
- identity
- intent
- physical authority

All calibration evidence references are propagated into the resulting observation.

## Verification

```bash
cargo test -p symthaea-domain-awareness-camera-geometry
```
