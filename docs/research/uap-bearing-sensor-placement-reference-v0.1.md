# Bearing sensor placement — independent Fisher-information reference v0.1

Date: 2026-11-07
Status: independent Python 3D Fisher-information calculation; Rust tests and example pending pinned toolchain execution.

## Synthetic scenario

Assumed target:

    p = [50, 40, 200] metres

Existing bearing sensors:

| Sensor | ECEF-like origin (m) |
|---|---|
| existing-north | [0, 0, 0] |
| existing-east | [100, 0, 0] |

Candidate placements:

| Candidate | ECEF-like origin (m) |
|---|---|
| candidate-north-far | [-100, 0, 0] |
| candidate-south | [0, 100, 0] |
| candidate-high | [0, 0, 100] |

All candidate orientations are identity rotations, using +X forward, +Y right,
+Z up. Each bearing has per-axis one-sigma angular uncertainty of 1e-4 rad.
The target and candidate poses are treated as known exactly.

## Method

For each sensor at origin s, define the predicted unit direction
d=(p-s)/||p-s||, azimuth alpha and elevation beta. The local angular
Jacobian rows are

    j_alpha = t_alpha / (range cos(beta))
    j_beta  = t_beta / range

where t_alpha and t_beta are the ECEF-rotated azimuth/elevation tangent basis
vectors. The local Fisher information is

    F = sum_i (j_alpha^T j_alpha / sigma_alpha^2
             + j_beta^T  j_beta  / sigma_beta^2)

The local Cramer-Rao covariance lower bound is C=F^-1. Candidate scores are:

- D-optimal: log(det(F)); higher is better.
- A-optimal: trace(C); lower is better.
- Condition number: lambda_max(F)/lambda_min(F); lower indicates more
  balanced observability.

A separate Python/NumPy calculation produced:

| Candidate | log(det(F)) | trace(C) m² | condition number | Czz m² |
|---|---:|---:|---:|---:|
| candidate-high | 28.7783051731 | 5.5417087438e-4 | 26.5782274 | 4.93814456e-4 |
| candidate-south | 27.4602895696 | 1.4857709113e-3 | 62.1714362 | 1.42628740e-3 |
| candidate-north-far | 27.3624532904 | 1.1228056913e-3 | 37.1506485 | 1.04834500e-3 |

The candidate above the existing baseline has the largest information
determinant and smallest covariance trace in this scenario. The result is
geometry-specific, not a universal claim that elevated sensors are always
best. D-optimality and A-optimality can disagree in other layouts and under
different costs/constraints.

## Rust implementation

Implemented in
`crates/domains/symthaea-uap-observatory/src/triangulation.rs`:

- `bearing_fisher_information` computes local FIM at a supplied target point.
- `rank_sensor_placements` predicts each candidate bearing from its origin
  and body-to-ECEF quaternion, computes FIM and CRLB, and ranks by D-optimal
  information gain with A-optimal covariance trace as a tie-breaker.
- Duplicate candidate IDs and collisions with observed sensor IDs fail closed.
- The example
  `cargo run -p symthaea-uap-observatory --example sensor_placement_design`
  emits a machine-readable CSV ranking.

Tests pin the high candidate's log determinant, covariance trace and vertical
variance to these independent Python values. This is source-level test intent;
the Rust test has not run on the pinned toolchain.

## Limitations and next gates

The local FIM/CRLB assumes a fixed target point and known candidate pose and
angular calibration. It does not include:
- target location prior or target-region integration;
- pose, boresight and extrinsic calibration covariance;
- clock offset/drift or asynchronous observation;
- correlated azimuth/elevation noise;
- visibility, occlusion, atmospheric refraction, multipath or weather;
- sensor range limits, platform motion, deployment cost, power or access.

Next benchmark:
1. Sweep candidate baselines, elevation separation, crossing angles and
   target ranges over a declared 3D region.
2. Compare D-optimal, A-optimal, E-optimal (minimum eigenvalue), and robust
   worst-case criteria under target-region and pose uncertainty.
3. Monte Carlo the full nonlinear estimator, compare empirical covariance
   coverage against CRLB predictions, and include biased calibration controls.
4. Optimize sensor subsets under fixed budget/cost and visibility constraints.
5. Blind-test on known aircraft/drone/satellite tracks with independent truth.

## Scientific boundary

This demonstrates a local, synthetic information-geometry calculation. It
does not show real sensor improvement, prove an anomalous target exists, or
classify any UAP. It is a design tool that should improve measurement
observability before any physical anomaly is inferred.
