# Robust bearing-sensor placement — independent reference v0.2

Date: 2026-11-07
Status: independent NumPy reference computed; Rust implementation and regression tests committed; pinned Rust execution pending lockfile reconciliation.

## Why v0.2

The v0.1 prototype ranked placements at one assumed target point. This is useful for
a local geometry check, but can overfit a nominal location. The v0.2 design evaluates
a finite target-region prior and reports both expected and worst-case local
information. Existing sensor bearings are re-predicted from their sensor poses at
each target hypothesis; the measured angle at the nominal point is not reused at
other target points.

This follows the broader optimal-placement literature's use of CRLB-based
objectives while adding an explicit finite-region robustness criterion. It also
addresses the validation gap in much of the sensor-deployment literature: simulated
performance must be tied to reproducible geometry, noise assumptions, common
baselines, and hardware or field validation before deployment claims are made.

## Synthetic scenario

All coordinates are Cartesian metres in an ECEF-like frame. All sensors use
identity sensor-body-to-ECEF rotations, +X forward, +Y right, +Z up, and independent
per-axis one-sigma angular uncertainty of 1e-4 rad. Candidate sensor poses and
angular calibration are assumed exact.

Existing sensors:

| Sensor | Origin (m) |
|---|---|
| existing-north | [0, 0, 0] |
| existing-east | [100, 0, 0] |

Candidate placements:

| Candidate | Origin (m) |
|---|---|
| candidate-north-far | [-100, 0, 0] |
| candidate-south | [0, 100, 0] |
| candidate-high | [0, 0, 100] |

Target hypotheses and relative prior weights:

| Hypothesis | Target (m) | Weight |
|---|---|---:|
| H1 | [50, 40, 200] | 0.5 |
| H2 | [70, 30, 180] | 0.3 |
| H3 | [35, 55, 230] | 0.2 |

Weights are normalized by their sum. The same hypotheses and weights are used for
all candidate placements.

## Local Fisher-information model

For a hypothesized target p and sensor origin s, predict the unit direction
d=(p-s)/||p-s||, azimuth alpha and elevation beta. Let t_alpha and t_beta denote
the ECEF-rotated azimuth and elevation tangent vectors. The local measurement
Jacobian rows are

    j_alpha = t_alpha / (range cos(beta))
    j_beta  = t_beta / range

For independent Gaussian angular errors with stated standard deviations, the local
Fisher information is

    F(p) = sum_i [j_alpha^T j_alpha / sigma_alpha^2
                 + j_beta^T  j_beta  / sigma_beta^2]

The local Cramer-Rao covariance lower bound is C(p)=F(p)^-1. We report:

- D-optimal information: log(det(F)); larger is better.
- A-optimal variance proxy: trace(C); smaller is better.
- Condition number of F; smaller indicates more balanced local observability.

Robust aggregation:

    expected_D = sum_h normalized_weight_h * log(det(F_h))
    worst_D    = min_h log(det(F_h))
    expected_A = sum_h normalized_weight_h * trace(C_h)
    worst_A    = max_h trace(C_h)

The ordering is lexicographic: maximize worst_D, then expected_D, then minimize
worst_A. This prioritizes minimum information over the declared scenario set rather
than allowing a high average to hide one poor target region. It is not the only
defensible objective; a deployment comparison must include Bayesian expected-FIM,
expected log-det, A-optimal, E-optimal/minimum-eigenvalue, and cost-constrained
alternatives.

## Independent numerical reference

A separate NumPy implementation, evaluating predicted bearings at each target
hypothesis, produced:

| Candidate | Expected log(det F) | Worst log(det F) | Expected trace(C) m² | Worst trace(C) m² | Worst condition number |
|---|---:|---:|---:|---:|---:|
| candidate-high | 28.8199643352 | 27.5072652997 | 5.9374699840e-4 | 1.1343564893e-3 | 51.8602781 |
| candidate-south | 27.5166009832 | 26.9387900958 | 1.5260747542e-3 | 2.4002975201e-3 | 114.4628513 |
| candidate-north-far | 27.4060657909 | 26.5126217892 | 1.1413374809e-3 | 1.6244280014e-3 | 48.9426193 |

In this geometry and finite prior, candidate-high wins all reported D- and
A-optimal comparisons. This is not a universal result for elevated sensors; it
depends on the target-region shape, baseline, orientation, error model and costs.
The highest worst-case condition number belongs to candidate-south. Condition
number is therefore useful as a complementary diagnostic, not a substitute for a
decision objective.

## Rust implementation

Implemented in
`crates/domains/symthaea-uap-observatory/src/triangulation.rs`:

- `TargetHypothesis` represents a target location and relative prior weight.
- `rank_sensor_placements_over_target_region` returns expected and worst-case
  D/A-optimal metrics.
- Existing sensor poses are reprojected toward each target hypothesis before
  calculating the local information matrix.
- Weights are normalized; non-finite or non-positive weights fail closed.
- Singular/behind-sensor geometry at any supplied hypothesis fails closed rather
  than silently dropping a difficult region.
- `sensor_placement_design` prints both single-point and target-region rankings.

Tests pin the candidate-high expected/worst-case log determinants and covariance
traces to the independent values above. This is source-level test intent; the pinned
Rust test has not yet completed because the exact workspace lockfile is being
regenerated and validated.

## Limitations

This is a local Fisher/CRLB design under a finite target-hypothesis prior. It does
not include:
- continuous-region integration or a guarantee between supplied hypotheses;
- uncertainty over target location beyond the finite weighted scenarios;
- sensor position, orientation, boresight, timing and calibration covariance;
- correlated azimuth/elevation noise or non-Gaussian outliers;
- field of view, occlusion, terrain, weather, refraction, multipath or sensor range;
- sensor movement, platform dynamics, power, access, security or lifecycle cost;
- multi-target association or track-level filtering/smoothing.

A target-region design should eventually integrate pose and calibration nuisance
parameters, account for visibility and non-detection, and optimize a trajectory or
sensor set under an explicit deployment budget.

## Next benchmark protocol

1. **Independent implementation:** compare Rust FIM, determinant, covariance and
   rankings against a separately maintained Python/NumPy reference over a seeded
   corpus of non-coplanar and near-degenerate geometries.
2. **Monte Carlo calibration:** generate 10,000+ noisy multi-sensor observations
   per scenario; compare empirical position covariance to reported local CRLB and
   measure empirical 50/90/95/99% coverage, bias and normalized estimation error.
3. **Noise misspecification:** vary angular variance, introduce correlated
   azimuth/elevation errors, heavy tails, boresight bias, sensor pose uncertainty,
   timestamp offset and clock drift.
4. **Geometry stress:** sweep target range, altitude, sensor baselines, crossing
   angles, field-of-view boundaries, collinear/coplanar arrangements and missing
   sensors.
5. **Objective ablation:** compare minimax log-det, expected log-det, expected FIM,
   A-optimal, E-optimal and random placement under identical candidate sets.
6. **Cost-constrained deployment:** include per-sensor cost, range, field of view,
   occlusion, energy, communication reliability and maintenance; report Pareto
   fronts, not one unqualified winner.
7. **Blind known-object test:** evaluate sealed trajectories with independent
   truth for aircraft, drones and satellites, including non-anomalous controls and
   sensor-artifact negatives. Report false-track and false-anomaly rates.
8. **Reproducibility:** publish raw inputs, calibration lineage, time standards,
   seeds, toolchain, source SHA, result digests and exact commands.

## Comparison to current practice

Bearing triangulation, multi-sensor fusion, EKF/UKF/factor-graph tracking and
CRLB-driven placement are established approaches. The intended differentiator is
not a new claim that Fisher information is novel. It is a reproducible,
evidence-qualified pipeline that couples sensor-pose provenance, fail-closed
geometry, uncertainty-aware local inference, region-robust active sensing and
blind negative controls. A superiority claim requires controlled comparisons
against these established baselines on shared data.

NASA's UAP Independent Study emphasized calibration, sensor metadata, time and
location context, multiple well-calibrated modalities, and explicit evidence
thresholds; it also noted that sensor artifacts can explain some apparent events.
NIST's metrological traceability guidance requires an unbroken calibration chain
with documented uncertainty at each link. These are design requirements, not
optional annotations.

## References

- Xu, S. et al. (2025), "Optimal Sensor Placement for Target Localization in IoT
  Systems: A Cramér-Rao Bound Perspective", IEEE Internet of Things Magazine 8(5),
  120-126. DOI: 10.1109/MIOT.2025.3575317.
- NASA UAP Independent Study Team (2023), Final Report,
  https://science.nasa.gov/uap/ .
- NIST, "Metrological Traceability: Frequently Asked Questions and NIST Policy",
  https://www.nist.gov/metrology/metrological-traceability .
- NIST Technical Note 1297, Guidelines for Evaluating and Expressing the
  Uncertainty of NIST Measurement Results.

## Scientific boundary

The result demonstrates a mathematical sensor-placement prototype on one small
synthetic scenario. It does not demonstrate an improvement on field sensors,
does not establish any anomalous object, and must not be used to classify UAP.
The next credible milestone is independent Rust/Python agreement followed by
Monte Carlo covariance-coverage testing and blind known-object trajectories.
