# UAP bearing triangulation — independent synthetic reference v0.1

Date: 2026-11-07
Status: independent Python reference experiment; Rust implementation and tests pending pinned toolchain execution.

## Geometry and protocol

A synthetic target was fixed at ECEF-like Cartesian coordinates

    p_true = [50.0, 40.0, 200.0] metres

Four sensors were placed at

| Sensor | Origin (m) |
|---|---|
| north | [0, 0, 0] |
| east | [100, 0, 0] |
| south | [0, 100, 5] |
| west | [100, 100, 10] |

Each sensor measured azimuth and elevation under the sensor-body convention
+X forward, +Y right, +Z up. Nominal one-sigma angular errors per axis were
100, 120, 80 and 110 microradians respectively.

A Python reference implemented:
1. Normalized sensor-to-target directions and azimuth/elevation observations.
2. The line-intersection least-squares normal equations
   sum(I - d_i d_i^T) p = sum(I - d_i d_i^T) o_i.
3. A weighted Gauss-Newton refinement using local tangent-plane angular
   residuals and their position Jacobians.
4. The local covariance approximation C = (J^T W J)^-1.

A deterministic NumPy generator was initialized with seed 20261107. One
thousand independent trial sets were produced by adding independent Gaussian
azimuth/elevation perturbations to each sensor's true bearing. The solver was
re-run for each trial using each sensor's stated angular standard deviation.

## Results

| Metric | Independent Python result |
|---|---:|
| Number of Monte Carlo trials | 1,000 |
| 3D position RMSE | 0.03110 m |
| Mean X position error | -0.000125 m |
| Mean Y position error | +0.000109 m |
| Mean Z position error | -0.000474 m |
| Empirical X error SD | 0.00478 m |
| Empirical Y error SD | 0.00464 m |
| Empirical Z error SD | 0.03039 m |
| Mean predicted covariance diag | [2.3273e-5, 2.2344e-5, 9.0074e-4] m^2 |
| Empirical coverage of nominal 95% 3D Gaussian ellipsoid | 94.4% |
| Mean Mahalanobis squared error | 2.9706 |
| Median Mahalanobis squared error | 2.3049 |
| 95th percentile Mahalanobis squared error | 8.0245 |
| Mean iterations | 2.278 |

The expected 95% chi-square threshold for three dimensions is approximately
7.815. The empirical 94.4% coverage falls inside the programme's initial
93–97% synthetic interval-coverage target.

The geometry is deliberately favourable and the noise model is deliberately
simple. This does not demonstrate accuracy under real sensor pose errors,
clock drift, atmospheric refraction, multipath, correlated boresight errors,
rolling-shutter effects or platform-motion uncertainty.

## Conditioning and geometry

The unperturbed four-ray line-intersection normal matrix had a condition
number of about 8.21. The final angular information matrix had a condition
number of about 41.95. The corresponding predicted position uncertainty was
strongly anisotropic: vertical uncertainty was substantially larger than
horizontal uncertainty because the sensors had limited vertical baseline.

That is an important result in itself: a system should expose geometry and
conditioning rather than report a target position without its uncertainty.

## Implemented Rust model

The Rust prototype is in
`crates/domains/symthaea-uap-observatory/src/triangulation.rs`.

It provides:
- ECEF sensor origins and sensor-body-to-ECEF quaternions;
- azimuth/elevation bearing measurements and separate one-sigma angular errors;
- least-squares line intersection as an initializer;
- weighted Gauss-Newton refinement in local angular residual coordinates;
- local linearized covariance, residual chi-square, degrees of freedom,
  RMS tangent-plane angular residual, condition number and iteration count;
- failure on duplicate sensor IDs, invalid quaternions, malformed errors,
  near-vertical azimuth singularities, behind-sensor solutions, degenerate
  geometry and non-convergence.

The covariance is a local linear approximation, not a fully propagated
sensor/clock/trajectory covariance. The implementation expects synchronized
observations and validated sensor poses from upstream.

## Next validation gates

1. Run the Rust crate tests with the pinned Rust 1.96.0 toolchain.
2. Compare Rust output against the Python seeds and target coordinates.
3. Add synthetic quaternion-rotated sensor frames and ECEF/ENU transform tests.
4. Add clock offset, clock drift and asynchronous observation controls.
5. Add Monte Carlo pose and boresight uncertainty, including correlated errors.
6. Test baselines and crossing angles from well-conditioned through
   deliberately degenerate; publish error and condition-number curves.
7. Add sensor-motion/parallax and platform-trajectory uncertainty.
8. Add blinded known-object tracks with aircraft, balloons, drones, birds,
   satellites, flares and atmospheric phenomena before any anomaly claim.
9. Evaluate false-positive rates with confidence intervals on a sealed set;
   preserve an explicit unresolved outcome.

## Scientific boundary

The experiment demonstrates a synthetic multi-ray geometry calculation.
It does not demonstrate that a real UAP has been triangulated, that a
reported acceleration is physical, or that any target has extraordinary
origin or propulsion. No classification is made by this module.
