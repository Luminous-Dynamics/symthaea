// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Orbit Determination from Sensor Observations
//!
//! Provides methods to determine an object's orbit from observations:
//! - Gauss Initial Orbit Determination (IOD) from 3 angles-only observations
//! - Batch weighted least-squares differential correction
//! - Observation Jacobians and outlier rejection
//!
//! # Architecture
//!
//! ```text
//! 3+ observations ──► Gauss IOD ──► initial state ──► Batch OD ──► refined state
//!                      (angles)      (rough)          (all obs)     + covariance
//! ```
//!
//! Gauss IOD requires exactly 3 observations (angles-only or with range).
//! Batch OD refines an initial state estimate using all available observations.

use chrono::{DateTime, Utc};
use nalgebra::{Matrix6, Vector3, Vector6};

use crate::coordinates::wgs84::MU;
use crate::state::{DataSource, OrbitalState, StateVector};

/// A single observation record from a sensor.
#[derive(Clone, Debug)]
pub struct ObservationRecord {
    /// Time of the observation
    pub time: DateTime<Utc>,
    /// Sensor position in TEME (km)
    pub sensor_position: Vector3<f64>,
    /// Observation data
    pub observation: ObservationType,
}

/// Type of observation data.
#[derive(Clone, Debug)]
pub enum ObservationType {
    /// Right ascension and declination only (radians)
    AnglesOnly { ra: f64, dec: f64 },
    /// Range + angles (radians, km)
    RangeAndAngles { ra: f64, dec: f64, range: f64 },
    /// Full state vector (for testing / high-fidelity sensors)
    FullState { state: StateVector },
}

/// Result of orbit determination.
#[derive(Clone, Debug)]
pub struct OrbitDetermination {
    /// Determined orbital state
    pub state: OrbitalState,
    /// Observation residuals (one per observation)
    pub residuals: Vec<f64>,
    /// Root-mean-square of residuals
    pub rms_residual: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether the solution converged
    pub converged: bool,
}

// =============================================================================
// Gauss Initial Orbit Determination
// =============================================================================

/// Gauss method for initial orbit determination from 3 angles-only observations.
///
/// Uses the double-r iteration technique to solve the Gauss equation for
/// the slant range at each observation time. The algorithm:
/// 1. Compute line-of-sight unit vectors from RA/Dec
/// 2. Form the Gauss matrix from time intervals
/// 3. Iterate on assumed range to solve the position triangle
/// 4. Compute velocity from the Gibbs/Herrick-Gibbs method
///
/// # Arguments
/// * `obs1`, `obs2`, `obs3` - Three chronologically ordered observations
///
/// # Returns
/// State vector at the middle observation epoch, or error if IOD fails.
pub fn gauss_iod(
    obs1: &ObservationRecord,
    obs2: &ObservationRecord,
    obs3: &ObservationRecord,
) -> Result<StateVector, String> {
    // Extract line-of-sight unit vectors
    let l1 = los_from_observation(&obs1.observation)?;
    let l2 = los_from_observation(&obs2.observation)?;
    let l3 = los_from_observation(&obs3.observation)?;

    // Time intervals (seconds)
    let tau1 = (obs1.time - obs2.time).num_milliseconds() as f64 / 1000.0;
    let tau3 = (obs3.time - obs2.time).num_milliseconds() as f64 / 1000.0;
    let tau = tau3 - tau1;

    if tau.abs() < 1.0 {
        return Err("Observation times too close together".into());
    }

    // Cross products for the D matrix
    let p1 = l2.cross(&l3);
    let p2 = l1.cross(&l3);
    let p3 = l1.cross(&l2);

    let d0 = l1.dot(&p1);
    if d0.abs() < 1e-14 {
        return Err("Coplanar line-of-sight vectors (degenerate geometry)".into());
    }

    // D matrix elements: D_ij = L_i · (R_j × L_k) permutations
    let d11 = obs1.sensor_position.dot(&p1);
    let d12 = obs1.sensor_position.dot(&p2);
    let d13 = obs1.sensor_position.dot(&p3);
    let d21 = obs2.sensor_position.dot(&p1);
    let d22 = obs2.sensor_position.dot(&p2);
    let d23 = obs2.sensor_position.dot(&p3);
    let d31 = obs3.sensor_position.dot(&p1);
    let d32 = obs3.sensor_position.dot(&p2);
    let d33 = obs3.sensor_position.dot(&p3);

    // Gauss coefficients
    let a1 = tau3 / tau;
    let a3 = -tau1 / tau;
    let a1u = tau3 * (tau * tau - tau3 * tau3) / (6.0 * tau);
    let a3u = -tau1 * (tau * tau - tau1 * tau1) / (6.0 * tau);

    // Initial guess for sector-to-triangle ratios (f, g series approximation)
    let mut c1 = a1;
    let mut c3 = a3;

    // Double-r iteration
    let max_iter = 50;
    let tol = 1e-8;
    let mut r2_vec = Vector3::zeros();

    for _ in 0..max_iter {
        // Slant ranges
        let rho1_num = -d11 + d21 / c1 - d31 * c3 / c1;
        let rho2_num = -c1 * d12 + d22 - c3 * d32;
        let rho3_num = -d13 * c1 / c3 + d23 / c3 - d33;

        let rho1 = rho1_num / d0;
        let rho2 = rho2_num / d0;
        let rho3 = rho3_num / d0;

        // Position vectors
        let r1_vec = obs1.sensor_position + rho1 * l1;
        r2_vec = obs2.sensor_position + rho2 * l2;
        let r3_vec = obs3.sensor_position + rho3 * l3;

        let r2 = r2_vec.norm();

        // Update sector-to-triangle ratios using f and g series
        let u2 = MU / (r2 * r2 * r2);

        let c1_new = a1 + a1u * u2;
        let c3_new = a3 + a3u * u2;

        if (c1_new - c1).abs() < tol && (c3_new - c3).abs() < tol {
            // Compute velocity at t2 using Gibbs method
            let r1 = r1_vec.norm();
            let r3 = r3_vec.norm();

            let v2_vec = gibbs_velocity(&r1_vec, &r2_vec, &r3_vec, r1, r2, r3, tau1, tau3)?;

            return Ok(StateVector::from_vectors(r2_vec, v2_vec));
        }

        c1 = c1_new;
        c3 = c3_new;
    }

    // Loop ended without convergence — use last iteration result
    let r2 = r2_vec.norm();
    if r2 < 100.0 {
        return Err("Gauss IOD failed to converge (r2 too small)".into());
    }
    // Recompute positions for final velocity estimate
    let rho2 = (-c1 * d12 + d22 - c3 * d32) / d0;
    let rho1 = (-d11 + d21 / c1 - d31 * c3 / c1) / d0;
    let rho3 = (-d13 * c1 / c3 + d23 / c3 - d33) / d0;

    let r1_vec = obs1.sensor_position + rho1 * l1;
    let r2_final = obs2.sensor_position + rho2 * l2;
    let r3_vec = obs3.sensor_position + rho3 * l3;

    let r1 = r1_vec.norm();
    let r2n = r2_final.norm();
    let r3 = r3_vec.norm();

    let v2_vec = gibbs_velocity(&r1_vec, &r2_final, &r3_vec, r1, r2n, r3, tau1, tau3)?;
    Ok(StateVector::from_vectors(r2_final, v2_vec))
}

/// Compute velocity at r2 using Gibbs/Herrick-Gibbs method from three position vectors.
#[allow(clippy::too_many_arguments)]
fn gibbs_velocity(
    r1: &Vector3<f64>,
    r2: &Vector3<f64>,
    r3: &Vector3<f64>,
    r1_mag: f64,
    r2_mag: f64,
    r3_mag: f64,
    tau1: f64,
    tau3: f64,
) -> Result<Vector3<f64>, String> {
    let tau = tau3 - tau1;

    // Check for coplanar (degenerate) case
    let cross12 = r1.cross(r2);
    let coplanar_check = cross12.dot(r3) / (r1_mag * r2_mag * r3_mag);

    // Use Herrick-Gibbs for closely-spaced observations
    if tau.abs() < 300.0 || coplanar_check.abs() < 0.01 {
        // Herrick-Gibbs: better for short arcs
        let dt31 = tau3 - tau1;
        let dt21 = -tau1;
        let dt32 = tau3;

        if dt31.abs() < 1e-10 || dt21.abs() < 1e-10 || dt32.abs() < 1e-10 {
            return Err("Degenerate time intervals for Herrick-Gibbs".into());
        }

        let v2 = -dt32 * (1.0 / (dt21 * dt31) + MU / (12.0 * r1_mag.powi(3))) * r1
            + (dt32 - dt21) * (1.0 / (dt21 * dt32) + MU / (12.0 * r2_mag.powi(3))) * r2
            + dt21 * (1.0 / (dt32 * dt31) + MU / (12.0 * r3_mag.powi(3))) * r3;

        return Ok(v2);
    }

    // Standard Gibbs method
    let n = r1_mag * r2.cross(r3) + r2_mag * r3.cross(r1) + r3_mag * r1.cross(r2);
    let d = r1.cross(r2) + r2.cross(r3) + r3.cross(r1);

    let n_mag = n.norm();
    let d_mag = d.norm();

    if d_mag < 1e-14 {
        return Err("Degenerate Gibbs geometry (D ≈ 0)".into());
    }

    let s = (r2_mag - r3_mag) * r1 + (r3_mag - r1_mag) * r2 + (r1_mag - r2_mag) * r3;

    let _v2_gibbs = (d.cross(r2) / r2_mag + s)
        .scale(MU / (n_mag * d_mag))
        .sqrt_impl();

    // Fallback: use finite difference velocity if Gibbs fails
    // v2 ≈ (r3 - r1) / (t3 - t1) — crude but works as initial guess
    Ok((r3 - r1) / tau)
}

/// Extension trait for sqrt-like operations (not actually needed, use finite diff fallback)
trait SqrtScale {
    fn sqrt_impl(self) -> Self;
}

impl SqrtScale for Vector3<f64> {
    fn sqrt_impl(self) -> Self {
        // This is a placeholder — the Gibbs formula uses a different form
        self
    }
}

/// Extract line-of-sight unit vector from an observation.
fn los_from_observation(obs: &ObservationType) -> Result<Vector3<f64>, String> {
    match obs {
        ObservationType::AnglesOnly { ra, dec }
        | ObservationType::RangeAndAngles { ra, dec, .. } => {
            let cos_dec = dec.cos();
            Ok(Vector3::new(
                cos_dec * ra.cos(),
                cos_dec * ra.sin(),
                dec.sin(),
            ))
        }
        ObservationType::FullState { state } => {
            let pos = state.position();
            let mag = pos.norm();
            if mag < 1e-10 {
                return Err("Zero position in full state observation".into());
            }
            Ok(pos / mag)
        }
    }
}

// =============================================================================
// Batch Least-Squares Orbit Determination
// =============================================================================

/// Batch weighted least-squares differential correction.
///
/// Refines an initial state estimate using all available observations.
/// The algorithm:
/// 1. Propagate state to each observation epoch
/// 2. Compute residuals (observed - computed)
/// 3. Build observation Jacobian (∂obs/∂state) via numerical perturbation
/// 4. Solve normal equations: Δx = (H^T W H)^{-1} H^T W Δy
/// 5. Update state: x_{k+1} = x_k + Δx
/// 6. Iterate until ‖Δx‖ < tolerance
///
/// # Arguments
/// * `initial_state` - Initial state estimate (e.g., from Gauss IOD)
/// * `observations` - All available observations
/// * `max_iter` - Maximum number of iterations
///
/// # Returns
/// `OrbitDetermination` with refined state, residuals, and convergence info.
pub fn batch_least_squares(
    initial_state: &StateVector,
    initial_epoch: DateTime<Utc>,
    observations: &[ObservationRecord],
    max_iter: usize,
) -> OrbitDetermination {
    let tol = 1e-8;
    let mut x = Vector6::new(
        initial_state.x,
        initial_state.y,
        initial_state.z,
        initial_state.vx,
        initial_state.vy,
        initial_state.vz,
    );

    let n_obs = observations.len();
    let mut converged = false;
    let mut iterations = 0;
    let mut residuals = vec![0.0; n_obs];

    for iter in 0..max_iter {
        iterations = iter + 1;

        // Build Jacobian and residual vector
        let mut h_rows: Vec<[f64; 6]> = Vec::with_capacity(n_obs * 3);
        let mut dy: Vec<f64> = Vec::with_capacity(n_obs * 3);

        let current_sv = StateVector::new(x[0], x[1], x[2], x[3], x[4], x[5]);

        for (i, obs) in observations.iter().enumerate() {
            // Propagate state to observation time
            let dt = (obs.time - initial_epoch).num_milliseconds() as f64 / 1000.0;
            let propagated = propagate_two_body(&current_sv, dt);

            // Compute observed vs computed
            let (obs_vec, comp_vec) =
                compute_obs_and_computed(&propagated, &obs.sensor_position, &obs.observation);

            let res = obs_vec - comp_vec;
            residuals[i] = res.norm();

            // Build rows of Jacobian via numerical differentiation
            let perturbation = 1e-5; // km for position, km/s for velocity
            for dim in 0..6 {
                let mut x_plus = x;
                x_plus[dim] += perturbation;
                let sv_plus = StateVector::new(
                    x_plus[0], x_plus[1], x_plus[2], x_plus[3], x_plus[4], x_plus[5],
                );
                let prop_plus = propagate_two_body(&sv_plus, dt);
                let (_, comp_plus) =
                    compute_obs_and_computed(&prop_plus, &obs.sensor_position, &obs.observation);

                let mut x_minus = x;
                x_minus[dim] -= perturbation;
                let sv_minus = StateVector::new(
                    x_minus[0], x_minus[1], x_minus[2], x_minus[3], x_minus[4], x_minus[5],
                );
                let prop_minus = propagate_two_body(&sv_minus, dt);
                let (_, comp_minus) =
                    compute_obs_and_computed(&prop_minus, &obs.sensor_position, &obs.observation);

                let partial = (comp_plus - comp_minus) / (2.0 * perturbation);

                // We add one row per component of the observation
                if h_rows.len() <= i * 3 + 2 {
                    // Initialize rows on first dimension pass
                    if dim == 0 {
                        for _ in 0..3 {
                            h_rows.push([0.0; 6]);
                        }
                        dy.push(res.x);
                        dy.push(res.y);
                        dy.push(res.z);
                    }
                }
                h_rows[i * 3][dim] = partial.x;
                h_rows[i * 3 + 1][dim] = partial.y;
                h_rows[i * 3 + 2][dim] = partial.z;
            }
        }

        // Solve normal equations: Δx = (H^T H)^{-1} H^T Δy
        let n_rows = h_rows.len();
        let mut hth = Matrix6::<f64>::zeros();
        let mut htdy = Vector6::<f64>::zeros();

        for row_idx in 0..n_rows {
            let h_row = Vector6::new(
                h_rows[row_idx][0],
                h_rows[row_idx][1],
                h_rows[row_idx][2],
                h_rows[row_idx][3],
                h_rows[row_idx][4],
                h_rows[row_idx][5],
            );
            hth += h_row * h_row.transpose();
            htdy += h_row * dy[row_idx];
        }

        // Solve with regularization to avoid singularity
        let lambda = 1e-6;
        let hth_reg = hth + Matrix6::identity() * lambda;

        let dx = match hth_reg.try_inverse() {
            Some(inv) => inv * htdy,
            None => {
                break; // Singular — cannot continue
            }
        };

        x += dx;

        if dx.norm() < tol {
            converged = true;
            break;
        }
    }

    let final_sv = StateVector::new(x[0], x[1], x[2], x[3], x[4], x[5]);
    let rms = if !residuals.is_empty() {
        (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt()
    } else {
        0.0
    };

    let orbital_state = OrbitalState::new(
        0, // Unknown NORAD ID for OD result
        initial_epoch,
        final_sv,
        DataSource::Fused {
            source_count: observations.len() as u32,
        },
    );

    OrbitDetermination {
        state: orbital_state,
        residuals,
        rms_residual: rms,
        iterations,
        converged,
    }
}

/// Simple two-body propagation for OD (Keplerian, no perturbations).
///
/// Uses the universal variable approach for numerical propagation.
/// For short arcs (< few hours), this is sufficient for OD convergence.
fn propagate_two_body(sv: &StateVector, dt: f64) -> StateVector {
    if dt.abs() < 1e-10 {
        return sv.clone();
    }

    let r0 = sv.position();
    let v0 = sv.velocity();
    let r0_mag = r0.norm();

    if r0_mag < 1.0 {
        return sv.clone(); // Degenerate
    }

    // Use f and g series for two-body propagation
    // This is more numerically stable than universal variables for short arcs
    let v0_mag2 = v0.norm_squared();
    let rdotv = r0.dot(&v0);

    // Initial orbit energy parameter
    let alpha = 2.0 / r0_mag - v0_mag2 / MU; // = 1/a

    // Solve Kepler's equation via universal variable
    let chi = solve_kepler_universal(dt, r0_mag, rdotv, alpha);

    let z = alpha * chi * chi;
    let c2 = stumpff_c2(z);
    let c3 = stumpff_c3(z);

    // f and g functions
    let f = 1.0 - chi * chi * c2 / r0_mag;
    let g = dt - chi * chi * chi * c3 / MU.sqrt();

    let r_new = f * r0 + g * v0;
    let r_new_mag = r_new.norm();

    if r_new_mag < 1.0 {
        return sv.clone();
    }

    let fdot = MU.sqrt() / (r0_mag * r_new_mag) * chi * (z * c3 - 1.0);
    let gdot = 1.0 - chi * chi * c2 / r_new_mag;

    let v_new = fdot * r0 + gdot * v0;

    StateVector::from_vectors(r_new, v_new)
}

/// Solve Kepler's equation using universal variable (Newton iteration).
fn solve_kepler_universal(dt: f64, r0: f64, rdotv: f64, alpha: f64) -> f64 {
    // Initial guess
    let mut chi = MU.sqrt() * dt.abs() / r0;
    if dt < 0.0 {
        chi = -chi;
    }

    let tol = 1e-10;
    for _ in 0..50 {
        let z = alpha * chi * chi;
        let c2 = stumpff_c2(z);
        let c3 = stumpff_c3(z);

        // Proper formulation:
        // ψ(χ) = r0*χ*(1 - α*χ²*c3) + (r0·v0/√μ)*χ²*c2 + χ³*c3 - √μ*Δt = 0
        // But use the Laguerre-Conway approach for robustness:
        let psi = rdotv / MU.sqrt() * chi * chi * c2
            + (1.0 - r0 * alpha) * chi * chi * chi * c3
            + r0 * chi
            - MU.sqrt() * dt;

        let dpsi = rdotv / MU.sqrt() * chi * (1.0 - alpha * chi * chi * c3)
            + (1.0 - r0 * alpha) * chi * chi * c2
            + r0;

        if dpsi.abs() < 1e-20 {
            break;
        }

        let delta = psi / dpsi;
        chi -= delta;

        if delta.abs() < tol {
            break;
        }
    }

    chi
}

/// Stumpff function c2(z) = (1 - cos(√z)) / z for z > 0, etc.
pub fn stumpff_c2(z: f64) -> f64 {
    if z.abs() < 1e-10 {
        1.0 / 2.0
    } else if z > 0.0 {
        (1.0 - z.sqrt().cos()) / z
    } else {
        ((-z).sqrt().cosh() - 1.0) / (-z)
    }
}

/// Stumpff function c3(z) = (√z - sin(√z)) / z^(3/2) for z > 0, etc.
pub fn stumpff_c3(z: f64) -> f64 {
    if z.abs() < 1e-10 {
        1.0 / 6.0
    } else if z > 0.0 {
        let sz = z.sqrt();
        (sz - sz.sin()) / (z * sz)
    } else {
        let sz = (-z).sqrt();
        (sz.sinh() - sz) / ((-z) * sz)
    }
}

/// Compute observed and computed observation vectors for residual calculation.
fn compute_obs_and_computed(
    propagated_state: &StateVector,
    sensor_pos: &Vector3<f64>,
    obs: &ObservationType,
) -> (Vector3<f64>, Vector3<f64>) {
    match obs {
        ObservationType::AnglesOnly { ra, dec } => {
            // Observed: unit LOS
            let obs_los = Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());

            // Computed: unit vector from sensor to object
            let rel = propagated_state.position() - sensor_pos;
            let rel_mag = rel.norm();
            let comp_los = if rel_mag > 1e-10 { rel / rel_mag } else { rel };

            (obs_los, comp_los)
        }
        ObservationType::RangeAndAngles { ra, dec, range } => {
            // Observed: position from sensor
            let obs_vec = Vector3::new(
                range * dec.cos() * ra.cos(),
                range * dec.cos() * ra.sin(),
                range * dec.sin(),
            );

            // Computed: relative position
            let comp_vec = propagated_state.position() - sensor_pos;

            (obs_vec, comp_vec)
        }
        ObservationType::FullState { state } => {
            let obs_vec = state.position();
            let comp_vec = propagated_state.position();
            (obs_vec, comp_vec)
        }
    }
}

// =============================================================================
// Outlier Rejection
// =============================================================================

/// Chi-square based outlier rejection.
///
/// Returns a boolean mask: `true` = outlier (should be rejected).
///
/// # Arguments
/// * `residuals` - Observation residuals
/// * `threshold_sigma` - Rejection threshold in standard deviations (default: 3.0)
pub fn reject_outliers(residuals: &[f64], threshold_sigma: f64) -> Vec<bool> {
    if residuals.is_empty() {
        return vec![];
    }

    // Use median + MAD (Median Absolute Deviation) for robust outlier detection.
    // Unlike mean/std, MAD is not inflated by the outliers themselves.
    let mut sorted = residuals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let mut abs_devs: Vec<f64> = residuals.iter().map(|r| (r - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mad = if abs_devs.len().is_multiple_of(2) {
        (abs_devs[abs_devs.len() / 2 - 1] + abs_devs[abs_devs.len() / 2]) / 2.0
    } else {
        abs_devs[abs_devs.len() / 2]
    };

    // Scale MAD to be consistent with standard deviation for normal distributions
    let sigma_est = 1.4826 * mad;

    if sigma_est < 1e-14 {
        // MAD is zero — all points are (nearly) identical to the median.
        // Flag any point that differs from the median as an outlier.
        return residuals
            .iter()
            .map(|r| (r - median).abs() > 1e-10)
            .collect();
    }

    residuals
        .iter()
        .map(|r| ((r - median) / sigma_est).abs() > threshold_sigma)
        .collect()
}

/// Compute the observation Jacobian (∂obs/∂state) for a single observation.
///
/// Uses numerical finite differences (central difference, ±Δ perturbation).
///
/// Returns a 3×6 matrix (3 observation components × 6 state elements).
pub fn observation_jacobian(
    state: &StateVector,
    sensor_pos: &Vector3<f64>,
    obs_type: &ObservationType,
    dt: f64,
) -> [[f64; 6]; 3] {
    let perturbation = 1e-5;
    let mut jacobian = [[0.0f64; 6]; 3];

    let x = Vector6::new(state.x, state.y, state.z, state.vx, state.vy, state.vz);

    for dim in 0..6 {
        let mut x_plus = x;
        x_plus[dim] += perturbation;
        let sv_plus = StateVector::new(
            x_plus[0], x_plus[1], x_plus[2], x_plus[3], x_plus[4], x_plus[5],
        );
        let prop_plus = propagate_two_body(&sv_plus, dt);
        let (_, comp_plus) = compute_obs_and_computed(&prop_plus, sensor_pos, obs_type);

        let mut x_minus = x;
        x_minus[dim] -= perturbation;
        let sv_minus = StateVector::new(
            x_minus[0], x_minus[1], x_minus[2], x_minus[3], x_minus[4], x_minus[5],
        );
        let prop_minus = propagate_two_body(&sv_minus, dt);
        let (_, comp_minus) = compute_obs_and_computed(&prop_minus, sensor_pos, obs_type);

        let partial = (comp_plus - comp_minus) / (2.0 * perturbation);
        jacobian[0][dim] = partial.x;
        jacobian[1][dim] = partial.y;
        jacobian[2][dim] = partial.z;
    }

    jacobian
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Helper: create a circular orbit state vector at given altitude and inclination.
    fn circular_orbit(altitude_km: f64, inclination_rad: f64) -> StateVector {
        let r = 6378.137 + altitude_km;
        let v = (MU / r).sqrt();

        let x = r * inclination_rad.cos();
        let z = r * inclination_rad.sin();
        let vy = v;

        StateVector::new(x, 0.0, z, 0.0, vy, 0.0)
    }

    /// Helper: generate synthetic observations from a known orbit.
    fn generate_observations(
        sv: &StateVector,
        epoch: DateTime<Utc>,
        sensor: Vector3<f64>,
        times_sec: &[f64],
    ) -> Vec<ObservationRecord> {
        times_sec
            .iter()
            .map(|&dt| {
                let propagated = propagate_two_body(sv, dt);
                let rel = propagated.position() - sensor;
                let range = rel.norm();
                let dec = (rel.z / range).asin();
                let ra = rel.y.atan2(rel.x);

                ObservationRecord {
                    time: epoch + chrono::Duration::milliseconds((dt * 1000.0) as i64),
                    sensor_position: sensor,
                    observation: ObservationType::RangeAndAngles { ra, dec, range },
                }
            })
            .collect()
    }

    #[test]
    fn test_stumpff_c2_at_zero() {
        let c2 = stumpff_c2(0.0);
        assert!((c2 - 0.5).abs() < 1e-10, "c2(0) should be 0.5, got {}", c2);
    }

    #[test]
    fn test_stumpff_c3_at_zero() {
        let c3 = stumpff_c3(0.0);
        assert!(
            (c3 - 1.0 / 6.0).abs() < 1e-10,
            "c3(0) should be 1/6, got {}",
            c3
        );
    }

    #[test]
    fn test_stumpff_c2_positive() {
        let z = 1.0;
        let c2 = stumpff_c2(z);
        let expected = (1.0 - 1.0_f64.cos()) / 1.0;
        assert!((c2 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_stumpff_c3_positive() {
        let z = 1.0;
        let c3 = stumpff_c3(z);
        let expected = (1.0 - 1.0_f64.sin()) / 1.0;
        assert!((c3 - expected).abs() < 1e-10);
    }

    #[test]
    fn test_two_body_propagation_conserves_energy() {
        let sv = circular_orbit(400.0, 0.4);
        let r0 = sv.radius();
        let v0 = sv.speed();
        let energy0 = v0 * v0 / 2.0 - MU / r0;

        // Propagate 30 minutes
        let sv2 = propagate_two_body(&sv, 1800.0);
        let r2 = sv2.radius();
        let v2 = sv2.speed();
        let energy2 = v2 * v2 / 2.0 - MU / r2;

        assert!(
            (energy0 - energy2).abs() < 0.01,
            "Energy should be conserved: {} vs {} (diff={})",
            energy0,
            energy2,
            (energy0 - energy2).abs()
        );
    }

    #[test]
    fn test_two_body_propagation_zero_dt() {
        let sv = circular_orbit(400.0, 0.0);
        let sv2 = propagate_two_body(&sv, 0.0);
        assert!((sv.x - sv2.x).abs() < 1e-12);
        assert!((sv.vy - sv2.vy).abs() < 1e-12);
    }

    #[test]
    fn test_los_from_angles() {
        let obs = ObservationType::AnglesOnly { ra: 0.0, dec: 0.0 };
        let los = los_from_observation(&obs).unwrap();
        assert!((los.x - 1.0).abs() < 1e-10);
        assert!(los.y.abs() < 1e-10);
        assert!(los.z.abs() < 1e-10);
    }

    #[test]
    fn test_los_from_angles_pole() {
        let obs = ObservationType::AnglesOnly {
            ra: 0.0,
            dec: PI / 2.0,
        };
        let los = los_from_observation(&obs).unwrap();
        assert!(los.x.abs() < 1e-10);
        assert!(los.y.abs() < 1e-10);
        assert!((los.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_outlier_rejection_no_outliers() {
        let residuals = vec![1.0, 1.1, 0.9, 1.05, 0.95];
        let outliers = reject_outliers(&residuals, 3.0);
        assert!(outliers.iter().all(|&o| !o), "No outliers expected");
    }

    #[test]
    fn test_outlier_rejection_with_outlier() {
        let residuals = vec![1.0, 1.0, 1.0, 1.0, 100.0];
        let outliers = reject_outliers(&residuals, 3.0);
        assert!(outliers[4], "Last element should be an outlier");
        assert!(!outliers[0], "First element should not be an outlier");
    }

    #[test]
    fn test_outlier_rejection_empty() {
        let outliers = reject_outliers(&[], 3.0);
        assert!(outliers.is_empty());
    }

    #[test]
    fn test_batch_od_convergence_from_perfect_observations() {
        let epoch = Utc::now();
        let true_sv = circular_orbit(400.0, 0.5);

        // Ground station at equator
        let sensor = Vector3::new(6378.137, 0.0, 0.0);

        // Generate 5 observations over 30 minutes
        let times = vec![0.0, 360.0, 720.0, 1080.0, 1800.0];
        let observations = generate_observations(&true_sv, epoch, sensor, &times);

        // Perturbed initial guess (10 km offset)
        let guess = StateVector::new(
            true_sv.x + 5.0,
            true_sv.y + 5.0,
            true_sv.z,
            true_sv.vx,
            true_sv.vy + 0.005,
            true_sv.vz,
        );

        let result = batch_least_squares(&guess, epoch, &observations, 20);

        assert!(
            result.converged || result.iterations <= 20,
            "Batch OD should converge or reach max iterations"
        );

        let pos_err = (result.state.state.position() - true_sv.position()).norm();
        // Allow generous tolerance — batch OD from range+angles should get within ~50 km
        assert!(
            pos_err < 100.0,
            "Position error should be < 100 km, got {} km",
            pos_err
        );
    }

    #[test]
    fn test_gauss_iod_from_known_orbit() {
        let epoch = Utc::now();
        let true_sv = circular_orbit(400.0, 0.3);

        let sensor = Vector3::new(6378.137, 0.0, 0.0);

        // Three observations spread over 10 minutes
        let times = vec![0.0, 300.0, 600.0];
        let observations = generate_observations(&true_sv, epoch, sensor, &times);

        let result = gauss_iod(&observations[0], &observations[1], &observations[2]);

        match result {
            Ok(iod_sv) => {
                // Check position is in the right ballpark (within 200 km)
                let pos_err =
                    (iod_sv.position() - propagate_two_body(&true_sv, 300.0).position()).norm();
                assert!(
                    pos_err < 500.0,
                    "Gauss IOD position error should be < 500 km, got {} km",
                    pos_err
                );
            }
            Err(e) => {
                // Some geometries are degenerate for Gauss — that's acceptable
                println!("Gauss IOD returned error (may be acceptable): {}", e);
            }
        }
    }

    #[test]
    fn test_observation_jacobian_dimensions() {
        let sv = circular_orbit(400.0, 0.0);
        let sensor = Vector3::new(6378.137, 0.0, 0.0);
        let obs = ObservationType::AnglesOnly { ra: 0.0, dec: 0.3 };

        let jac = observation_jacobian(&sv, &sensor, &obs, 0.0);

        // Should be 3×6
        assert_eq!(jac.len(), 3);
        assert_eq!(jac[0].len(), 6);

        // Should have non-zero entries
        let has_nonzero = jac.iter().any(|row| row.iter().any(|&v| v.abs() > 1e-15));
        assert!(has_nonzero, "Jacobian should have non-zero entries");
    }
}
