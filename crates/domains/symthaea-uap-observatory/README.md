# symthaea-uap-observatory

A std-only first milestone for calibration-aware, provenance-preserving
sensor evidence. It is intentionally a measurement-quality layer, not an
object classifier or a theory of UAP propulsion.

## First milestone

- Typed sensor modality and coordinate frame.
- Distinct measurement payloads for bearings, range-bearing, ECEF position,
  RF spectra and magnetic vectors.
- Explicit measurement uncertainty; rejects NaN, infinity, non-positive
  standard deviations, negative range and invalid elevation.
- TAI nanosecond timestamps and explicit per-sensor clock offset / uncertainty.
- Calibration validity windows, modality/frame matching and nonzero SHA-256
  digests for the raw calibration artefact.
- Raw observation SHA-256 references and duplicate sequence rejection.
- Evidence completeness summary that distinguishes uncalibrated,
  mixed-calibration, single-sensor and multi-sensor-calibrated bundles.

Uncalibrated records are retained and marked as such. They are not deleted,
silently repaired or converted into anomalous-object claims.


## Bearing-only triangulation (new prototype)

The `triangulation` module adds a multi-sensor ECEF bearing solution:

- Sensor origins expressed in ECEF metres.
- Active sensor-body-to-ECEF quaternion.
- Azimuth/elevation bearings and per-axis one-sigma angular uncertainty.
- Least-squares intersection initialization.
- Uncertainty-weighted Gauss-Newton refinement in angular residual space.
- Linearized ECEF covariance, chi-square residual, degrees of freedom,
  angular RMS, geometric condition number and iteration count.
- Fail-closed rejection of parallel/ill-conditioned geometry, duplicate
  sensors, invalid quaternions/uncertainties, vertical azimuth singularities,
  behind-sensor solutions and non-convergence.

Sensor-body convention is +X forward, +Y right, +Z up. Azimuth is measured
from +X toward +Y; elevation is positive toward +Z. Sensor rotations are active
body-to-ECEF quaternions.

The covariance is a local linearized approximation to the angular
least-squares model. It does not yet propagate sensor-pose, clock, refraction,
boresight drift or correlated calibration covariance. Observations must
already be synchronized to a common epoch; this module does not perform
time alignment, platform-motion smoothing, multi-target association or
radar-range fusion.

### Current independent synthetic reference

A separate Python implementation was tested on a fixed four-sensor geometry
and 1,000 independently perturbed bearing sets. At nominal per-axis angular
standard deviations of roughly 80–120 microradians, the 3D position RMSE was
about 3.1 cm and the local Gaussian 95% covariance ellipsoid covered about
94.4% of the synthetic target positions. These are reference-design numbers,
not Rust execution or real-instrument results; the complete protocol and
limitations are in `docs/research/uap-bearing-triangulation-reference-v0.1.md`.

## Run

```bash
cargo test -p symthaea-uap-observatory
```

## Epistemic boundary

A multi-sensor-calibrated bundle only means that its records satisfy the
structural calibration contract. It does not establish correct triangulation,
track association, extraordinary motion, object identity or non-human origin.
The next milestones must implement calibrated frame transforms, timestamp
synchronization, uncertainty propagation, blind known-object benchmarks and
conventional-explanation controls before any anomaly conclusion.
