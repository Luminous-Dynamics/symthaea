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

The point-target API is a local design calculation, not a global deployment
solution. The target-region API handles a finite weighted scenario set, not a
continuous-region guarantee. Both currently treat candidate pose and angular
noise as known. They omit pose/boresight covariance, clock synchronization,
visibility, terrain/occlusion, atmospheric propagation, weather, cost, and
platform constraints. They must not recommend real sensor placement without
those constraints and independent geometric references.

### Current independent synthetic reference

A separate Python implementation was tested on a fixed four-sensor geometry
and 1,000 independently perturbed bearing sets. At nominal per-axis angular
standard deviations of roughly 80–120 microradians, the 3D position RMSE was
about 3.1 cm and the local Gaussian 95% covariance ellipsoid covered about
94.4% of the synthetic target positions. These are reference-design numbers,
not Rust execution or real-instrument results; the complete protocol and
limitations are in `docs/research/uap-bearing-triangulation-reference-v0.1.md`.

## Multi-target evaluation metric (new reference)

The `tracking_metrics` module implements a frame-level GOSPA metric with
global optimal assignment, cutoff localization cost, and separate cardinality
penalties for missed and false targets. The dynamic programme solves the
assignment exactly on the smaller side (up to 20 targets); it fails closed
above that bound rather than substituting a greedy algorithm.

The output reports:

- total GOSPA distance;
- matched localization cost before the p-root;
- number of missed and false targets;
- separate missed-target and false-target penalty contributions;
- number of matched targets.

This is a frame-level metric oracle, not a multi-target tracker. It does not
implement track filtering, data association or classification. Track-set
temporal evaluation is implemented separately in the trajectory_metrics
module as OSPA(2); it uses one fixed track assignment across the evaluation
window and therefore detects identity swaps that frame-level GOSPA misses.
OSPA(2) does not add a distinct switch-cost term. The 2026 T-GOSPA family is
a separate metric with an explicit track-switch penalty.

## Track-set temporal metric: OSPA(2) (new reference)

The trajectory_metrics module implements fixed-window OSPA(2):

- Each truth/estimate track pair is scored using a weighted temporal L_q
  aggregation of cutoff spatial error. A frame where both are absent costs
  zero; a frame where exactly one exists costs the cutoff.
- The resulting pairwise track distances are assigned globally across the
  whole truth and estimate track sets. The assignment does not change frame
  by frame, so track identity switches cannot receive a perfect score merely
  because the unordered point sets match at every frame.
- Unmatched track sets receive the OSPA(2) cardinality penalty. Uniform time
  weights are the default; callers can provide nonnegative custom weights.

The implementation caps the smaller assignment side at 20 tracks, validates
window lengths and finite positions, and reports localization and cardinality
contributions separately. It is an evaluation metric, not a tracker. It does
not implement the explicit switch-cost term in T-GOSPA.

Run the metric tests with:

    cargo test --locked -p symthaea-uap-observatory --lib trajectory_metrics

### UAP track benchmark contract

Any future tracker should be compared on a sealed known-object corpus using
at least:

1. Frame GOSPA with frozen p, cutoff c and alpha; report its localization,
   missed-target and false-target components separately.
2. Trajectory-level OSPA(2)/T-GOSPA or a documented equivalent, including
   identity switches, fragmentation and track continuity.
3. NEES/NIS consistency and empirical 50/90/95/99% uncertainty coverage.
4. False tracks per sensor-hour, missed-track rate and detection probability
   at a predeclared false-alarm rate.
5. Track initiation delay, termination delay, localization RMSE and runtime.
6. Ablations for sensor calibration, timestamp errors, parallax, boresight
   drift, occlusion, multipath, missing modalities and adversarial negatives.
7. Matched EKF/UKF/ESKF, JPDA/MHT or random-finite-set baselines where the
   measurement model warrants them. A baseline must use the same input data,
   sensor information and truth set.

A GOSPA score is not directly comparable across different cutoffs, orders,
alpha values, units or truth sets. Freeze those parameters and publish all
three error components; do not report only the scalar distance.

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
