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

## Fisher-information sensor placement (new prototype)

The triangulation module now exposes:

- `bearing_fisher_information`: local 3x3 position Fisher-information
  matrix at an assumed target point.
- `rank_sensor_placements`: scores candidate sensor origins/orientations
  by D-optimal log(det(FIM)), A-optimal covariance trace, condition number
  and the local Cramer-Rao covariance lower bound.
- Candidate observations are predicted using the candidate's active
  sensor-body-to-ECEF quaternion and declared per-axis angular uncertainty.

- `rank_sensor_placements_over_target_region`: robust finite-scenario design
  over a weighted set of plausible target locations. It reports prior-weighted
  expected D/A-optimal scores and worst-case information/covariance scores,
  ranking first by worst-case D-optimality.
- Existing sensor poses are reprojected toward each hypothesized target before
  recomputing the local Fisher matrix; the nominal measured bearing is not
  reused as if it were valid at every target hypothesis.

The target-region API is still a finite-scenario approximation. It does not
prove a continuous-region guarantee, and currently rejects a sensor geometry
if it is unobservable at any supplied target hypothesis. The current objective
is minimax log(det(FIM)), followed by expected log(det(FIM)); this is distinct
from pure Bayesian D-optimality and should be compared against expected-FIM,
A-optimal, E-optimal, and minimax CRLB designs.

This turns triangulation into a sensor-design question: given an assumed
target region and current observations, which sensor placement is expected
to add the most independent geometric information?

This is a local design calculation at one assumed target point, not a global
optimal deployment solution. It currently treats the target point and
candidate angular noise as known. It omits uncertainty over target location,
pose/boresight, clock synchronization, visibility, terrain/occlusion,
atmospheric propagation, weather, cost, and platform constraints. It must
not recommend real sensor placement without those constraints and an
independent geometric reference.

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
