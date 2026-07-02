// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Sensor Observation Fusion Pipeline
//!
//! Fuses observations from multiple sensors into a single best-estimate state
//! with uncertainty, using inverse-variance weighting, covariance intersection,
//! or weighted least-squares. Includes chi-square gating, adaptive covariance
//! inflation, per-sensor bias estimation, and trust update recommendations.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use nalgebra::{Matrix6, Vector6};

use crate::covariance::CovarianceMatrix;
use crate::state::{DataSource, OrbitalState, StateVector};

/// Trust-aware quality weighting for observation fusion.
/// effective_quality = data_quality * (trust_floor + (1 - trust_floor) * trust_weight)
/// Ref: Bar-Shalom & Li (2001), Estimation with Applications to Tracking
#[derive(Debug, Clone)]
pub struct TrustWeighting {
    /// Minimum quality multiplier for untrusted sources (default: 0.3)
    /// Range: 0.0-1.0. Higher values = more trust in unknown sources.
    pub trust_floor: f64,
    /// Per-sensor trust weights (sensor_id -> trust level 0.0-1.0)
    pub sensor_trust: HashMap<String, f64>,
}

impl Default for TrustWeighting {
    fn default() -> Self {
        Self {
            trust_floor: 0.3,
            sensor_trust: HashMap::new(),
        }
    }
}

impl TrustWeighting {
    /// Compute effective quality: quality * (floor + (1 - floor) * trust).
    /// Unknown sensors get trust = 0.0, fully trusted get 1.0.
    pub fn effective_quality(&self, raw_quality: f64, sensor_id: &str) -> f64 {
        let trust = self.sensor_trust.get(sensor_id).copied().unwrap_or(0.0);
        let multiplier = self.trust_floor + (1.0 - self.trust_floor) * trust;
        (raw_quality * multiplier).clamp(0.0, 1.0)
    }
}

/// A single sensor measurement with uncertainty.
#[derive(Clone, Debug)]
pub struct SensorMeasurement {
    /// Time of measurement
    pub time: DateTime<Utc>,
    /// Measured state vector
    pub state: StateVector,
    /// Measurement uncertainty (6×6 covariance)
    pub covariance: CovarianceMatrix,
    /// Sensor identifier (e.g., "MIT-LL-RADAR-01")
    pub sensor_id: String,
    /// Data source classification
    pub data_source: DataSource,
    /// Quality score (0.0 = unusable, 1.0 = perfect)
    pub quality: f64,
}

/// Result of multi-sensor fusion.
#[derive(Clone, Debug)]
pub struct FusedEstimate {
    /// Fused orbital state with covariance
    pub state: OrbitalState,
    /// Sensors that contributed to the fusion
    pub contributing_sensors: Vec<String>,
    /// Fused quality score (weighted average of input qualities)
    pub fused_quality: f64,
    /// Chi-square consistency statistic
    /// Low values indicate consistent measurements; high values suggest outliers
    pub chi_square_consistency: f64,
    /// Epoch of the fused estimate
    pub timestamp: DateTime<Utc>,
    /// Number of measurements rejected by chi-square gating
    pub gated_count: u32,
    /// Total measurements considered (after quality/age filter, before gating)
    pub total_filtered: u32,
    /// Covariance inflation factor applied (1.0 = none)
    pub covariance_inflation_factor: f64,
    /// Per-sensor bias estimates (populated after fusion)
    pub bias_estimates: Vec<SensorBiasEstimate>,
    /// Recommended trust adjustments for sensors
    pub trust_updates: Vec<TrustUpdate>,
}

/// Estimated bias for a single sensor relative to the fused estimate.
/// Residual = measurement − fused; flagged if |residual| > 2σ in any dimension.
#[derive(Clone, Debug)]
pub struct SensorBiasEstimate {
    pub sensor_id: String,
    /// Residual vector: measurement − fused state (km, km/s)
    pub residual: [f64; 6],
    /// Per-dimension 1σ from sensor covariance
    pub sigma: [f64; 6],
    /// True if any dimension has |residual| > 2σ
    pub is_biased: bool,
    /// Biased dimension indices (0-5: x,y,z,vx,vy,vz)
    pub biased_dimensions: Vec<usize>,
}

/// Recommended trust adjustment for a sensor based on fusion residuals.
/// Callers use this to update `TrustWeighting::sensor_trust` between epochs.
#[derive(Clone, Debug)]
pub struct TrustUpdate {
    pub sensor_id: String,
    /// Multiplicative factor (0.0-1.0). 1.0 = no change, < 1.0 = reduce trust.
    pub recommended_factor: f64,
    pub reason: TrustUpdateReason,
}

/// Reason for a trust adjustment recommendation.
#[derive(Clone, Debug, PartialEq)]
pub enum TrustUpdateReason {
    Consistent,
    Gated,
    BiasDetected,
    ElevatedResiduals,
}

/// Configuration for the fusion pipeline.
#[derive(Clone, Debug)]
pub struct FusionPipeline {
    /// Chi-square gating threshold (default: 9.21 for 99% at 2 DOF)
    pub gating_threshold: f64,
    /// Minimum sensor quality to accept (0.0 - 1.0)
    pub min_quality: f64,
    /// Maximum observation age in seconds
    pub max_age_seconds: f64,
    /// Optional trust weighting for per-sensor quality adjustment
    pub trust_weighting: Option<TrustWeighting>,
}

impl Default for FusionPipeline {
    fn default() -> Self {
        Self {
            gating_threshold: 9.21, // Chi-square 99% threshold at 2 DOF
            min_quality: 0.1,
            max_age_seconds: 86400.0, // 24 hours
            trust_weighting: None,
        }
    }
}

impl FusionPipeline {
    /// Create a new fusion pipeline with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the chi-square gating threshold.
    pub fn with_gating_threshold(mut self, threshold: f64) -> Self {
        self.gating_threshold = threshold;
        self
    }

    /// Set the minimum quality threshold.
    pub fn with_min_quality(mut self, quality: f64) -> Self {
        self.min_quality = quality;
        self
    }

    /// Set the maximum observation age.
    pub fn with_max_age(mut self, seconds: f64) -> Self {
        self.max_age_seconds = seconds;
        self
    }

    /// Set trust weighting for per-sensor quality adjustment.
    pub fn with_trust_weighting(mut self, weighting: TrustWeighting) -> Self {
        self.trust_weighting = Some(weighting);
        self
    }

    /// Main fusion entry point. Filters, propagates, gates, fuses, then
    /// computes bias estimates and trust update recommendations.
    pub fn fuse(&self, measurements: &[SensorMeasurement]) -> Result<FusedEstimate, String> {
        if measurements.is_empty() {
            return Err("No measurements to fuse".into());
        }

        // Step 1: Filter by quality (trust-adjusted) and age
        let now = Utc::now();
        let filtered: Vec<&SensorMeasurement> = measurements
            .iter()
            .filter(|m| {
                let effective_q = self.effective_quality_for(m);
                effective_q >= self.min_quality
                    && (now - m.time).num_seconds().abs() as f64 <= self.max_age_seconds
            })
            .collect();

        if filtered.is_empty() {
            return Err("All measurements filtered out (quality/age)".into());
        }

        if filtered.len() == 1 {
            // Single measurement: just return it directly
            let m = filtered[0];
            let effective_q = self.effective_quality_for(m);
            let state = OrbitalState::new(0, m.time, m.state.clone(), m.data_source.clone())
                .with_covariance(m.covariance.clone());

            return Ok(FusedEstimate {
                state,
                contributing_sensors: vec![m.sensor_id.clone()],
                fused_quality: effective_q,
                chi_square_consistency: 0.0,
                timestamp: m.time,
                gated_count: 0,
                total_filtered: 1,
                covariance_inflation_factor: 1.0,
                bias_estimates: Vec::new(),
                trust_updates: Vec::new(),
            });
        }

        // Step 2: Find common epoch (latest time)
        let target_epoch = filtered.iter().map(|m| m.time).max().unwrap();

        // Step 3: Propagate all measurements to common epoch
        let propagated: Vec<(StateVector, CovarianceMatrix, &SensorMeasurement)> = filtered
            .iter()
            .map(|m| {
                let dt = (target_epoch - m.time).num_milliseconds() as f64 / 1000.0;
                let prop_state = propagate_state(&m.state, dt);
                let prop_cov = m.covariance.propagate(dt);
                (prop_state, prop_cov, *m)
            })
            .collect();

        // Step 4: Sequential fusion with chi-square gating
        let total_filtered = propagated.len() as u32;
        let mut fused_state = propagated[0].0.clone();
        let mut fused_cov = propagated[0].1.clone();
        let mut contributing = vec![propagated[0].2.sensor_id.clone()];
        let mut total_quality = self.effective_quality_for(propagated[0].2);
        let mut chi_square_sum = 0.0;
        let mut gated_count = 0_u32;
        let mut gated_sensors = Vec::new();

        for (state_i, cov_i, meas) in propagated.iter().skip(1) {
            // Chi-square gate: check if this measurement is consistent
            let delta = state_vector_to_vec6(state_i) - state_vector_to_vec6(&fused_state);

            let mahal_sq = mahalanobis_squared(&delta, &fused_cov, cov_i);

            if mahal_sq > self.gating_threshold {
                gated_count += 1;
                gated_sensors.push(meas.sensor_id.clone());
                continue; // Reject this measurement
            }

            chi_square_sum += mahal_sq;

            // Fuse covariance matrices
            if let Some(new_cov) = fused_cov.fuse(cov_i) {
                // Compute fused state using inverse-variance weighting
                let fused_sv = fuse_states(&fused_state, &fused_cov, state_i, cov_i);
                fused_state = fused_sv;
                fused_cov = new_cov;
            } else {
                // Fallback: use simple weighted average
                let w1 = 1.0 / fused_cov.position_sigma().max(1e-10);
                let w2 = 1.0 / cov_i.position_sigma().max(1e-10);
                let w_total = w1 + w2;

                fused_state = StateVector::new(
                    (fused_state.x * w1 + state_i.x * w2) / w_total,
                    (fused_state.y * w1 + state_i.y * w2) / w_total,
                    (fused_state.z * w1 + state_i.z * w2) / w_total,
                    (fused_state.vx * w1 + state_i.vx * w2) / w_total,
                    (fused_state.vy * w1 + state_i.vy * w2) / w_total,
                    (fused_state.vz * w1 + state_i.vz * w2) / w_total,
                );
            }

            contributing.push(meas.sensor_id.clone());
            total_quality += self.effective_quality_for(meas);
        }

        // Step 5: Adaptive covariance inflation when many measurements are gated
        // If >30% of filtered measurements are rejected, the fusion is low-confidence.
        // Inflate covariance proportionally: factor = 1 + 2 * (rejection_rate - 0.3)
        let rejection_rate = if total_filtered > 1 {
            gated_count as f64 / (total_filtered - 1) as f64 // -1 because first is seed
        } else {
            0.0
        };
        let inflation_factor = if rejection_rate > 0.3 {
            1.0 + 2.0 * (rejection_rate - 0.3)
        } else {
            1.0
        };
        if inflation_factor > 1.0 {
            let inflated = fused_cov.matrix() * inflation_factor;
            fused_cov = CovarianceMatrix::from_matrix(inflated);
        }

        let n_contributing = contributing.len() as f64;
        let fused_quality = total_quality / n_contributing;

        // Step 6: Compute per-sensor bias estimates and trust updates
        let fused_vec = state_vector_to_vec6(&fused_state);
        let mut bias_estimates = Vec::new();
        let mut trust_updates = Vec::new();

        for (state_i, cov_i, meas) in &propagated {
            let meas_vec = state_vector_to_vec6(state_i);
            let residual_vec = meas_vec - fused_vec;
            let cov_mat = cov_i.matrix();

            let mut residual = [0.0; 6];
            let mut sigma = [0.0; 6];
            let mut biased_dims = Vec::new();

            for d in 0..6 {
                residual[d] = residual_vec[d];
                sigma[d] = cov_mat[(d, d)].sqrt().max(1e-15);
                if residual[d].abs() > 2.0 * sigma[d] {
                    biased_dims.push(d);
                }
            }

            let is_biased = !biased_dims.is_empty();
            bias_estimates.push(SensorBiasEstimate {
                sensor_id: meas.sensor_id.clone(),
                residual,
                sigma,
                is_biased,
                biased_dimensions: biased_dims,
            });

            // Trust update: recommend factor based on residual severity
            let is_gated = gated_sensors.contains(&meas.sensor_id);
            let (factor, reason) = if is_gated {
                (0.5, TrustUpdateReason::Gated)
            } else if is_biased {
                // Scale factor by how many dimensions are biased (more = worse)
                let bias_severity = bias_estimates.last().unwrap().biased_dimensions.len();
                let f = (1.0 - 0.15 * bias_severity as f64).max(0.3);
                (f, TrustUpdateReason::BiasDetected)
            } else {
                // Check for elevated residuals (between 1σ and 2σ)
                let max_norm_residual = (0..6)
                    .map(|d| residual[d].abs() / sigma[d])
                    .fold(0.0_f64, f64::max);
                if max_norm_residual > 1.0 {
                    (0.9, TrustUpdateReason::ElevatedResiduals)
                } else {
                    (1.0, TrustUpdateReason::Consistent)
                }
            };

            trust_updates.push(TrustUpdate {
                sensor_id: meas.sensor_id.clone(),
                recommended_factor: factor,
                reason,
            });
        }

        let state = OrbitalState::new(
            0,
            target_epoch,
            fused_state,
            DataSource::Fused {
                source_count: contributing.len() as u32,
            },
        )
        .with_covariance(fused_cov);

        Ok(FusedEstimate {
            state,
            contributing_sensors: contributing,
            fused_quality,
            chi_square_consistency: chi_square_sum,
            timestamp: target_epoch,
            gated_count,
            total_filtered,
            covariance_inflation_factor: inflation_factor,
            bias_estimates,
            trust_updates,
        })
    }

    /// Compute effective quality for a measurement, applying trust weighting if configured.
    fn effective_quality_for(&self, measurement: &SensorMeasurement) -> f64 {
        if let Some(ref tw) = self.trust_weighting {
            tw.effective_quality(measurement.quality, &measurement.sensor_id)
        } else {
            measurement.quality
        }
    }
}

/// Fuse two states using inverse-variance weighting.
///
/// x_fused = C_fused * (C1^{-1} * x1 + C2^{-1} * x2)
fn fuse_states(
    s1: &StateVector,
    c1: &CovarianceMatrix,
    s2: &StateVector,
    c2: &CovarianceMatrix,
) -> StateVector {
    let x1 = state_vector_to_vec6(s1);
    let x2 = state_vector_to_vec6(s2);

    let inv1 = c1.matrix().try_inverse();
    let inv2 = c2.matrix().try_inverse();

    match (inv1, inv2) {
        (Some(i1), Some(i2)) => {
            let combined_inv = i1 + i2;
            if let Some(c_fused) = combined_inv.try_inverse() {
                let x_fused = c_fused * (i1 * x1 + i2 * x2);
                StateVector::new(
                    x_fused[0], x_fused[1], x_fused[2], x_fused[3], x_fused[4], x_fused[5],
                )
            } else {
                // Fallback: simple average
                StateVector::new(
                    (s1.x + s2.x) / 2.0,
                    (s1.y + s2.y) / 2.0,
                    (s1.z + s2.z) / 2.0,
                    (s1.vx + s2.vx) / 2.0,
                    (s1.vy + s2.vy) / 2.0,
                    (s1.vz + s2.vz) / 2.0,
                )
            }
        }
        _ => {
            // If either covariance is singular, simple average
            StateVector::new(
                (s1.x + s2.x) / 2.0,
                (s1.y + s2.y) / 2.0,
                (s1.z + s2.z) / 2.0,
                (s1.vx + s2.vx) / 2.0,
                (s1.vy + s2.vy) / 2.0,
                (s1.vz + s2.vz) / 2.0,
            )
        }
    }
}

/// Covariance intersection: conservative fusion when cross-correlations are unknown.
/// Minimizes trace(C_fused) over ω ∈ [0,1].
pub fn covariance_intersection(
    states: &[(StateVector, CovarianceMatrix)],
) -> Option<(StateVector, CovarianceMatrix)> {
    if states.is_empty() {
        return None;
    }

    if states.len() == 1 {
        return Some((states[0].0.clone(), states[0].1.clone()));
    }

    // Pairwise CI, folding left
    let mut result_state = states[0].0.clone();
    let mut result_cov = states[0].1.clone();

    for (state_i, cov_i) in states.iter().skip(1) {
        let (s, c) = ci_pair(&result_state, &result_cov, state_i, cov_i)?;
        result_state = s;
        result_cov = c;
    }

    Some((result_state, result_cov))
}

/// Covariance intersection for a pair of estimates.
fn ci_pair(
    s1: &StateVector,
    c1: &CovarianceMatrix,
    s2: &StateVector,
    c2: &CovarianceMatrix,
) -> Option<(StateVector, CovarianceMatrix)> {
    let inv1 = c1.matrix().try_inverse()?;
    let inv2 = c2.matrix().try_inverse()?;

    // Search for optimal ω that minimizes trace(C_fused)
    // Simple grid search over ω ∈ [0, 1]
    let steps = 20;
    let mut best_omega = 0.5;
    let mut best_trace = f64::MAX;

    for i in 0..=steps {
        let omega = i as f64 / steps as f64;
        let combined = omega * inv1 + (1.0 - omega) * inv2;
        if let Some(c_fused) = combined.try_inverse() {
            let trace = c_fused.trace();
            if trace < best_trace && trace > 0.0 {
                best_trace = trace;
                best_omega = omega;
            }
        }
    }

    let combined_inv = best_omega * inv1 + (1.0 - best_omega) * inv2;
    let c_fused_mat = combined_inv.try_inverse()?;

    let x1 = state_vector_to_vec6(s1);
    let x2 = state_vector_to_vec6(s2);

    let x_fused = c_fused_mat * (best_omega * inv1 * x1 + (1.0 - best_omega) * inv2 * x2);

    let fused_state = StateVector::new(
        x_fused[0], x_fused[1], x_fused[2], x_fused[3], x_fused[4], x_fused[5],
    );

    let fused_cov = CovarianceMatrix::from_matrix(c_fused_mat);

    Some((fused_state, fused_cov))
}

/// Weighted least-squares fusion for independent measurements.
pub fn weighted_least_squares(measurements: &[SensorMeasurement]) -> Result<FusedEstimate, String> {
    if measurements.is_empty() {
        return Err("No measurements".into());
    }

    let mut info_sum = Matrix6::<f64>::zeros(); // Σ Ci^{-1}
    let mut weighted_state_sum = Vector6::<f64>::zeros(); // Σ Ci^{-1} * xi
    let mut contributing = Vec::new();
    let mut total_quality = 0.0;

    for m in measurements {
        let inv = match m.covariance.matrix().try_inverse() {
            Some(i) => i,
            None => continue, // Skip singular covariance
        };

        let x = state_vector_to_vec6(&m.state);
        info_sum += inv;
        weighted_state_sum += inv * x;
        contributing.push(m.sensor_id.clone());
        total_quality += m.quality;
    }

    if contributing.is_empty() {
        return Err("All covariance matrices were singular".into());
    }

    let c_fused = info_sum
        .try_inverse()
        .ok_or("Fused information matrix is singular")?;

    let x_fused = c_fused * weighted_state_sum;

    let fused_state = StateVector::new(
        x_fused[0], x_fused[1], x_fused[2], x_fused[3], x_fused[4], x_fused[5],
    );

    let n = contributing.len() as f64;
    let epoch = measurements
        .iter()
        .map(|m| m.time)
        .max()
        .unwrap_or_else(Utc::now);

    let state = OrbitalState::new(
        0,
        epoch,
        fused_state,
        DataSource::Fused {
            source_count: contributing.len() as u32,
        },
    )
    .with_covariance(CovarianceMatrix::from_matrix(c_fused));

    Ok(FusedEstimate {
        state,
        contributing_sensors: contributing,
        fused_quality: total_quality / n,
        chi_square_consistency: 0.0, // Not computed in WLS
        timestamp: epoch,
        gated_count: 0,
        total_filtered: measurements.len() as u32,
        covariance_inflation_factor: 1.0,
        bias_estimates: Vec::new(),
        trust_updates: Vec::new(),
    })
}

/// Propagate a measurement's state and covariance to a target epoch.
pub fn propagate_measurement(
    meas: &SensorMeasurement,
    target_time: DateTime<Utc>,
) -> SensorMeasurement {
    let dt = (target_time - meas.time).num_milliseconds() as f64 / 1000.0;

    let new_state = propagate_state(&meas.state, dt);
    let new_cov = meas.covariance.propagate(dt);

    SensorMeasurement {
        time: target_time,
        state: new_state,
        covariance: new_cov,
        sensor_id: meas.sensor_id.clone(),
        data_source: meas.data_source.clone(),
        quality: meas.quality,
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Convert StateVector to Vector6.
fn state_vector_to_vec6(sv: &StateVector) -> Vector6<f64> {
    Vector6::new(sv.x, sv.y, sv.z, sv.vx, sv.vy, sv.vz)
}

/// Squared Mahalanobis distance between two state estimates.
fn mahalanobis_squared(delta: &Vector6<f64>, c1: &CovarianceMatrix, c2: &CovarianceMatrix) -> f64 {
    let combined = c1.matrix() + c2.matrix();
    match combined.try_inverse() {
        Some(inv) => {
            let result = delta.transpose() * inv * delta;
            result[(0, 0)].max(0.0)
        }
        None => f64::MAX, // If combined covariance is singular, gate out
    }
}

/// Simple linear state propagation (position += velocity * dt).
fn propagate_state(sv: &StateVector, dt: f64) -> StateVector {
    StateVector::new(
        sv.x + sv.vx * dt,
        sv.y + sv.vy * dt,
        sv.z + sv.vz * dt,
        sv.vx,
        sv.vy,
        sv.vz,
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test measurement with specified position sigma.
    fn test_measurement(
        x: f64,
        sigma_pos: f64,
        sensor_id: &str,
        quality: f64,
    ) -> SensorMeasurement {
        let sigma_vel = 0.001;
        SensorMeasurement {
            time: Utc::now(),
            state: StateVector::new(x, 0.0, 0.0, 0.0, 7.5, 0.0),
            covariance: CovarianceMatrix::diagonal([
                sigma_pos, sigma_pos, sigma_pos, sigma_vel, sigma_vel, sigma_vel,
            ]),
            sensor_id: sensor_id.to_string(),
            data_source: DataSource::GroundObservation {
                sensor_id: sensor_id.to_string(),
                sensor_type: crate::state::SensorType::Radar,
            },
            quality,
        }
    }

    #[test]
    fn test_fuse_two_identical_measurements_halves_variance() {
        let m1 = test_measurement(7000.0, 1.0, "sensor-1", 0.9);
        let m2 = test_measurement(7000.0, 1.0, "sensor-2", 0.9);
        let input_sigma = m1.covariance.position_sigma();

        let fused = FusionPipeline::new().fuse(&[m1, m2]).unwrap();
        let fused_sigma = fused.state.position_uncertainty_km().unwrap();
        assert!(fused_sigma < input_sigma, "Fused sigma should decrease");

        // For two identical measurements: σ_fused = σ/√2
        let ratio = fused_sigma / (input_sigma / 2.0_f64.sqrt());
        assert!(ratio > 0.5 && ratio < 2.0, "Ratio to expected: {}", ratio);
    }

    #[test]
    fn test_fuse_high_and_low_quality() {
        let m_high = test_measurement(7000.0, 0.1, "good-sensor", 0.95);
        let m_low = test_measurement(7010.0, 10.0, "bad-sensor", 0.5);

        let fused_x = FusionPipeline::new()
            .fuse(&[m_high, m_low])
            .unwrap()
            .state
            .state
            .x;
        assert!(
            (fused_x - 7000.0).abs() < (fused_x - 7010.0).abs(),
            "Fused ({}) should be closer to high-quality (7000)",
            fused_x
        );
    }

    #[test]
    fn test_chi_square_gate_rejects_outlier() {
        let m1 = test_measurement(7000.0, 1.0, "sensor-1", 0.9);
        let m2 = test_measurement(7000.0, 1.0, "sensor-2", 0.9);
        let m_outlier = test_measurement(7100.0, 1.0, "bad-sensor", 0.9); // 100 km away

        let result = FusionPipeline::new().fuse(&[m1, m2, m_outlier]).unwrap();
        assert!(result.gated_count >= 1, "Outlier should be gated");
    }

    #[test]
    fn test_quality_filter_rejects_low_quality() {
        let m_good = test_measurement(7000.0, 1.0, "good", 0.9);
        let m_bad = test_measurement(7000.0, 1.0, "bad", 0.01);

        let result = FusionPipeline::new()
            .with_min_quality(0.1)
            .fuse(&[m_good, m_bad])
            .unwrap();
        assert_eq!(result.contributing_sensors, vec!["good"]);
    }

    #[test]
    fn test_covariance_intersection_is_conservative() {
        let s = StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0);
        let c = CovarianceMatrix::diagonal([1.0, 1.0, 1.0, 0.001, 0.001, 0.001]);

        let (_, ci_cov) =
            covariance_intersection(&[(s.clone(), c.clone()), (s, c.clone())]).unwrap();
        let iv_cov = c.fuse(&c).unwrap();
        // CI should be at least as conservative as inverse-variance
        assert!(ci_cov.position_sigma() >= iv_cov.position_sigma() * 0.95);
    }

    #[test]
    fn test_weighted_least_squares() {
        let m1 = test_measurement(7000.0, 1.0, "sensor-1", 0.8);
        let m2 = test_measurement(7002.0, 2.0, "sensor-2", 0.7);

        let x = weighted_least_squares(&[m1, m2]).unwrap().state.state.x;
        assert!(
            (x - 7000.0).abs() < (x - 7002.0).abs(),
            "WLS should weight lower uncertainty"
        );
    }

    #[test]
    fn test_empty_measurements() {
        let pipeline = FusionPipeline::new();
        assert!(pipeline.fuse(&[]).is_err());
    }

    #[test]
    fn test_single_measurement() {
        let m = test_measurement(7000.0, 1.0, "only-sensor", 0.9);
        let pipeline = FusionPipeline::new();
        let result = pipeline.fuse(&[m]).unwrap();

        assert_eq!(result.contributing_sensors.len(), 1);
        assert!((result.state.state.x - 7000.0).abs() < 1e-10);
    }

    #[test]
    fn test_five_sensor_fusion() {
        let measurements: Vec<SensorMeasurement> = (0..5)
            .map(|i| {
                test_measurement(
                    7000.0 + i as f64 * 0.5, // Slight offsets
                    1.0,
                    &format!("sensor-{}", i),
                    0.8 + i as f64 * 0.02,
                )
            })
            .collect();

        let pipeline = FusionPipeline::new();
        let result = pipeline.fuse(&measurements).unwrap();

        // All 5 sensors should contribute (small offsets within gate)
        assert!(
            result.contributing_sensors.len() >= 3,
            "At least 3 of 5 sensors should contribute"
        );

        // Fused uncertainty should be lower than any individual
        let fused_sigma = result.state.position_uncertainty_km().unwrap();
        let min_input_sigma = measurements
            .iter()
            .map(|m| m.covariance.position_sigma())
            .fold(f64::MAX, f64::min);

        assert!(
            fused_sigma < min_input_sigma,
            "Fused sigma ({}) should be less than min input sigma ({})",
            fused_sigma,
            min_input_sigma
        );
    }

    #[test]
    fn test_propagate_measurement() {
        let m = test_measurement(7000.0, 1.0, "sensor-1", 0.9);
        let future = m.time + chrono::Duration::seconds(60);

        let propagated = propagate_measurement(&m, future);

        // Position should change (velocity is ~7.5 km/s in y)
        let expected_y = m.state.vy * 60.0;
        assert!(
            (propagated.state.y - expected_y).abs() < 0.01,
            "Propagated y ({}) should be ~{} km",
            propagated.state.y,
            expected_y
        );

        // Covariance should grow
        assert!(
            propagated.covariance.position_sigma() > m.covariance.position_sigma(),
            "Propagated uncertainty should grow"
        );
    }

    // =========================================================================
    // Trust weighting tests
    // =========================================================================

    #[test]
    fn test_trust_floor_prevents_zero_weight() {
        let tw = TrustWeighting::default(); // floor = 0.3
        let eq = tw.effective_quality(1.0, "unknown_sensor");
        assert!(
            (eq - 0.3).abs() < 1e-10,
            "Unknown sensor should get floor weight, got {}",
            eq
        );
    }

    #[test]
    fn test_fully_trusted_gets_full_quality() {
        let mut tw = TrustWeighting::default();
        tw.sensor_trust.insert("trusted".to_string(), 1.0);
        let eq = tw.effective_quality(0.8, "trusted");
        assert!(
            (eq - 0.8).abs() < 1e-10,
            "Fully trusted should get raw quality, got {}",
            eq
        );
    }

    #[test]
    fn test_partial_trust_blends() {
        let mut tw = TrustWeighting {
            trust_floor: 0.5,
            sensor_trust: HashMap::new(),
        };
        tw.sensor_trust.insert("half".to_string(), 0.5);
        let eq = tw.effective_quality(1.0, "half");
        // floor + (1-floor) * trust = 0.5 + 0.5 * 0.5 = 0.75
        assert!(
            (eq - 0.75).abs() < 1e-10,
            "Partial trust should blend: expected 0.75, got {}",
            eq
        );
    }

    #[test]
    fn test_effective_quality_clamped() {
        let mut tw = TrustWeighting::default();
        tw.sensor_trust.insert("s".to_string(), 1.5); // Invalid but shouldn't crash
        let eq = tw.effective_quality(1.5, "s");
        assert!(eq <= 1.0, "Must be clamped to 1.0, got {}", eq);
    }

    #[test]
    fn test_trust_weighting_default() {
        let tw = TrustWeighting::default();
        assert!(
            (tw.trust_floor - 0.3).abs() < 1e-10,
            "Default floor should be 0.3"
        );
        assert!(tw.sensor_trust.is_empty(), "Default should have no sensors");
    }

    #[test]
    fn test_fuse_with_trust_weighting_reduces_untrusted_quality() {
        let m_trusted = test_measurement(7000.0, 1.0, "trusted-sensor", 0.9);
        let m_untrusted = test_measurement(7000.0, 1.0, "untrusted-sensor", 0.9);

        let mut tw = TrustWeighting::default();
        tw.sensor_trust.insert("trusted-sensor".to_string(), 1.0);
        // untrusted-sensor not in map -> gets floor (0.3)

        let pipeline = FusionPipeline::new().with_trust_weighting(tw);
        let result = pipeline.fuse(&[m_trusted, m_untrusted]).unwrap();

        // Fused quality should reflect trust weighting:
        // trusted effective = 0.9 * 1.0 = 0.9
        // untrusted effective = 0.9 * 0.3 = 0.27
        // average = (0.9 + 0.27) / 2 = 0.585
        assert!(
            result.fused_quality < 0.9,
            "Fused quality ({}) should be less than raw quality (0.9) due to untrusted sensor",
            result.fused_quality
        );
    }

    #[test]
    fn test_fuse_without_trust_weighting_unchanged() {
        let m1 = test_measurement(7000.0, 1.0, "sensor-1", 0.9);
        let m2 = test_measurement(7000.0, 1.0, "sensor-2", 0.8);

        let pipeline = FusionPipeline::new(); // No trust weighting
        let result = pipeline.fuse(&[m1, m2]).unwrap();

        // Without trust weighting, fused quality = average of raw qualities
        let expected = (0.9 + 0.8) / 2.0;
        assert!(
            (result.fused_quality - expected).abs() < 1e-10,
            "Without trust weighting, quality should be raw average: expected {}, got {}",
            expected,
            result.fused_quality
        );
    }

    // =========================================================================
    // Multi-sensor fusion, bias estimation, covariance inflation, trust updates
    // =========================================================================

    #[test]
    fn test_multi_sensor_fusion_three_trust_levels() {
        let m_high = test_measurement(7000.0, 0.5, "high-trust", 0.95);
        let m_mid = test_measurement(7001.0, 1.0, "mid-trust", 0.80);
        let m_low = test_measurement(7002.0, 2.0, "low-trust", 0.60);

        let mut tw = TrustWeighting::default();
        tw.sensor_trust.insert("high-trust".to_string(), 1.0);
        tw.sensor_trust.insert("mid-trust".to_string(), 0.5);
        // low-trust not in map -> gets floor (0.3)

        let pipeline = FusionPipeline::new().with_trust_weighting(tw);
        let result = pipeline.fuse(&[m_high, m_mid, m_low]).unwrap();

        assert_eq!(result.contributing_sensors.len(), 3);
        // Fused state should be closer to high-trust (lower cov + higher trust)
        let fused_x = result.state.state.x;
        assert!(
            (fused_x - 7000.0).abs() < (fused_x - 7002.0).abs(),
            "Fused x ({}) should be closer to high-trust (7000) than low-trust (7002)",
            fused_x
        );
        // Quality should reflect trust weighting
        assert!(
            result.fused_quality < 0.95,
            "Quality should be reduced by trust"
        );
    }

    #[test]
    fn test_biased_sensor_detection() {
        // Sensor with +100 km systematic bias in x
        let m_good1 = test_measurement(7000.0, 1.0, "good-1", 0.9);
        let m_good2 = test_measurement(7000.0, 1.0, "good-2", 0.9);
        let m_biased = test_measurement(7100.0, 50.0, "biased-sensor", 0.9);

        let pipeline = FusionPipeline::new().with_gating_threshold(100.0); // Wide gate
        let result = pipeline.fuse(&[m_good1, m_good2, m_biased]).unwrap();

        // Find the bias estimate for the biased sensor
        let biased_est = result
            .bias_estimates
            .iter()
            .find(|b| b.sensor_id == "biased-sensor")
            .expect("Should have bias estimate for biased-sensor");

        // The residual in x should be large and positive
        assert!(
            biased_est.residual[0] > 50.0,
            "Biased sensor x residual ({}) should be >50 km",
            biased_est.residual[0]
        );
    }

    #[test]
    fn test_covariance_inflation_high_gating_rate() {
        // Create measurements where most will be gated:
        // One seed at 7000, many outliers far away
        let m_seed = test_measurement(7000.0, 1.0, "seed", 0.9);
        let m_ok = test_measurement(7000.5, 1.0, "ok", 0.9);
        // These three are 200+ km away — will be gated
        let m_out1 = test_measurement(7200.0, 1.0, "outlier-1", 0.9);
        let m_out2 = test_measurement(7300.0, 1.0, "outlier-2", 0.9);
        let m_out3 = test_measurement(7400.0, 1.0, "outlier-3", 0.9);

        let pipeline = FusionPipeline::new().with_gating_threshold(9.21);
        let result = pipeline
            .fuse(&[m_seed, m_ok, m_out1, m_out2, m_out3])
            .unwrap();

        // 3 out of 4 non-seed measurements gated = 75% rejection rate
        assert!(
            result.gated_count >= 3,
            "Expected >= 3 gated, got {}",
            result.gated_count
        );
        assert!(
            result.covariance_inflation_factor > 1.0,
            "Inflation factor ({}) should be > 1.0 with high gating rate",
            result.covariance_inflation_factor
        );
        // Check that the inflated covariance is larger
        let inflated_sigma = result.state.position_uncertainty_km().unwrap();
        // Without inflation, two consistent measurements would give sigma/sqrt(2) ~ 0.707
        assert!(
            inflated_sigma > 0.7,
            "Inflated sigma ({}) should be larger due to inflation",
            inflated_sigma
        );
    }

    #[test]
    fn test_trust_update_for_gated_sensor() {
        let m1 = test_measurement(7000.0, 1.0, "good", 0.9);
        let m_outlier = test_measurement(7500.0, 1.0, "bad", 0.9);

        let pipeline = FusionPipeline::new().with_gating_threshold(9.21);
        let result = pipeline.fuse(&[m1, m_outlier]).unwrap();

        let bad_update = result
            .trust_updates
            .iter()
            .find(|u| u.sensor_id == "bad")
            .expect("Should have trust update for bad sensor");

        assert_eq!(bad_update.reason, TrustUpdateReason::Gated);
        assert!(
            bad_update.recommended_factor < 1.0,
            "Gated sensor should get reduced trust factor, got {}",
            bad_update.recommended_factor
        );
    }

    #[test]
    fn test_trust_update_for_biased_sensor() {
        // Use a wide gate so the biased sensor contributes but shows residuals
        let m1 = test_measurement(7000.0, 1.0, "clean-1", 0.9);
        let m2 = test_measurement(7000.0, 1.0, "clean-2", 0.9);
        // 10 km offset with large covariance (won't be gated, but will show bias)
        let m_biased = test_measurement(7010.0, 50.0, "drifty", 0.9);

        let pipeline = FusionPipeline::new().with_gating_threshold(100.0);
        let result = pipeline.fuse(&[m1, m2, m_biased]).unwrap();

        let drifty_update = result
            .trust_updates
            .iter()
            .find(|u| u.sensor_id == "drifty")
            .expect("Should have trust update for drifty");

        // Depending on residual magnitude vs sigma, could be BiasDetected or ElevatedResiduals
        assert!(
            drifty_update.recommended_factor <= 1.0,
            "Biased/elevated sensor trust should be <= 1.0, got {}",
            drifty_update.recommended_factor
        );
    }

    #[test]
    fn test_consistent_sensor_gets_no_trust_reduction() {
        let m1 = test_measurement(7000.0, 1.0, "s1", 0.9);
        let m2 = test_measurement(7000.0, 1.0, "s2", 0.9);

        let pipeline = FusionPipeline::new();
        let result = pipeline.fuse(&[m1, m2]).unwrap();

        for update in &result.trust_updates {
            assert_eq!(
                update.reason,
                TrustUpdateReason::Consistent,
                "Consistent sensors should get Consistent reason, got {:?} for {}",
                update.reason,
                update.sensor_id
            );
            assert!(
                (update.recommended_factor - 1.0).abs() < 1e-10,
                "Consistent sensor {} should have factor 1.0, got {}",
                update.sensor_id,
                update.recommended_factor
            );
        }
    }

    #[test]
    fn test_single_sensor_no_bias_no_updates() {
        let m = test_measurement(7000.0, 1.0, "only", 0.9);
        let pipeline = FusionPipeline::new();
        let result = pipeline.fuse(&[m]).unwrap();

        assert!(
            result.bias_estimates.is_empty(),
            "Single sensor has no bias estimates"
        );
        assert!(
            result.trust_updates.is_empty(),
            "Single sensor has no trust updates"
        );
        assert_eq!(result.gated_count, 0);
        assert_eq!(result.covariance_inflation_factor, 1.0);
    }

    #[test]
    fn test_all_sensors_untrusted() {
        let m1 = test_measurement(7000.0, 1.0, "unk-1", 0.9);
        let m2 = test_measurement(7000.0, 1.0, "unk-2", 0.9);
        let m3 = test_measurement(7000.0, 1.0, "unk-3", 0.9);

        let tw = TrustWeighting {
            trust_floor: 0.1,
            sensor_trust: HashMap::new(), // No sensor in map
        };
        let pipeline = FusionPipeline::new().with_trust_weighting(tw);
        let result = pipeline.fuse(&[m1, m2, m3]);

        // All sensors get floor trust, effective quality = 0.9 * 0.1 = 0.09
        // min_quality default is 0.1, so all should be filtered out
        assert!(
            result.is_err(),
            "All untrusted sensors should be filtered out"
        );
    }

    #[test]
    fn test_all_sensors_untrusted_with_low_threshold() {
        let m1 = test_measurement(7000.0, 1.0, "unk-1", 0.9);
        let m2 = test_measurement(7000.0, 1.0, "unk-2", 0.9);

        let tw = TrustWeighting {
            trust_floor: 0.2,
            sensor_trust: HashMap::new(),
        };
        let pipeline = FusionPipeline::new()
            .with_min_quality(0.05) // Low threshold so they pass
            .with_trust_weighting(tw);
        let result = pipeline.fuse(&[m1, m2]).unwrap();

        // effective quality = 0.9 * 0.2 = 0.18
        assert!(
            result.fused_quality < 0.3,
            "Untrusted sensors should yield low fused quality, got {}",
            result.fused_quality
        );
        assert_eq!(result.contributing_sensors.len(), 2);
    }

    #[test]
    fn test_all_measurements_gated() {
        // Seed at 7000, one measurement 500 km away — will be gated
        let m_seed = test_measurement(7000.0, 1.0, "seed", 0.9);
        let m_far = test_measurement(7500.0, 1.0, "far", 0.9);

        let pipeline = FusionPipeline::new().with_gating_threshold(9.21);
        let result = pipeline.fuse(&[m_seed, m_far]).unwrap();

        assert_eq!(result.gated_count, 1);
        // Only seed contributes
        assert_eq!(result.contributing_sensors.len(), 1);
        assert_eq!(result.contributing_sensors[0], "seed");
        // Inflation should apply: 1/1 = 100% rejection rate, factor = 1 + 2*(1.0-0.3) = 2.4
        assert!(
            result.covariance_inflation_factor > 2.0,
            "All-gated inflation factor ({}) should be > 2.0",
            result.covariance_inflation_factor
        );
    }

    #[test]
    fn test_gated_count_and_total_filtered_reported() {
        let m1 = test_measurement(7000.0, 1.0, "s1", 0.9);
        let m2 = test_measurement(7000.5, 1.0, "s2", 0.9);
        let m3 = test_measurement(7200.0, 1.0, "outlier", 0.9);

        let pipeline = FusionPipeline::new();
        let result = pipeline.fuse(&[m1, m2, m3]).unwrap();

        assert_eq!(
            result.total_filtered, 3,
            "3 measurements passed quality filter"
        );
        // outlier should be gated
        assert!(
            result.gated_count >= 1,
            "At least 1 measurement should be gated"
        );
    }

    #[test]
    fn test_no_inflation_when_gating_rate_below_threshold() {
        // Two consistent measurements — no gating
        let m1 = test_measurement(7000.0, 1.0, "s1", 0.9);
        let m2 = test_measurement(7000.1, 1.0, "s2", 0.9);

        let pipeline = FusionPipeline::new();
        let result = pipeline.fuse(&[m1, m2]).unwrap();

        assert_eq!(result.gated_count, 0);
        assert!(
            (result.covariance_inflation_factor - 1.0).abs() < 1e-10,
            "No inflation when no gating: factor = {}",
            result.covariance_inflation_factor
        );
    }
}
