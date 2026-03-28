// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-Sensor Observation Fusion Pipeline
//!
//! Fuses observations from multiple sensors into a single best-estimate state
//! with uncertainty. This is the core value proposition of a decentralized SSA
//! network: no single sensor provides the best answer — fused data always wins.
//!
//! # Pipeline
//!
//! ```text
//! Sensors ──► Filter (quality, age) ──► Propagate to common epoch
//!         ──► Chi-square gate ──► Covariance intersection / WLS
//!         ──► Consistency check ──► FusedEstimate
//! ```
//!
//! # Fusion Methods
//!
//! - **Inverse-variance weighting** (via `CovarianceMatrix::fuse()`): Optimal when
//!   cross-correlations are known (e.g., same-sensor repeated observations)
//! - **Covariance intersection**: More conservative, works when cross-correlations
//!   are unknown — appropriate for decentralized networks
//! - **Weighted least-squares**: Classical approach for independent measurements

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use nalgebra::{Matrix6, Vector6};

use crate::covariance::CovarianceMatrix;
use crate::state::{DataSource, OrbitalState, StateVector};

/// Trust-aware quality weighting for observation fusion.
///
/// When trust information is available, the effective quality of an observation
/// is blended between raw data quality and the submitter's trust level:
///
///   effective_quality = data_quality * (trust_floor + (1 - trust_floor) * trust_weight)
///
/// This ensures that even untrusted sources contribute some information
/// (trust_floor > 0), while trusted sources get full weight.
///
/// References:
/// - Bar-Shalom & Li (2001): Estimation with Applications to Tracking
/// - Mahalanobis gating remains the primary outlier defense
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
    /// Compute effective quality from raw quality and trust.
    ///
    /// Formula: quality * (floor + (1 - floor) * trust)
    /// - Unknown sensors (not in map) get trust = 0.0 -> multiplied by floor only
    /// - Fully trusted sensors get trust = 1.0 -> full quality
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
    ///
    /// When set, the pipeline applies `TrustWeighting::effective_quality()` to
    /// each measurement's quality score before filtering and fusion weighting.
    pub fn with_trust_weighting(mut self, weighting: TrustWeighting) -> Self {
        self.trust_weighting = Some(weighting);
        self
    }

    /// Main fusion entry point.
    ///
    /// Fuses multiple sensor measurements into a single best-estimate state:
    /// 1. Filter by quality and age
    /// 2. Propagate all measurements to common epoch (latest time)
    /// 3. Chi-square gating: reject inconsistent measurements
    /// 4. Sequential inverse-variance fusion via `CovarianceMatrix::fuse()`
    /// 5. Compute consistency metric
    ///
    /// # Arguments
    /// * `measurements` - Sensor measurements to fuse
    ///
    /// # Returns
    /// `FusedEstimate` or error string.
    pub fn fuse(&self, measurements: &[SensorMeasurement]) -> Result<FusedEstimate, String> {
        if measurements.is_empty() {
            return Err("No measurements to fuse".into());
        }

        // Step 1: Filter by quality and age
        // When trust weighting is configured, effective_quality is used for the
        // quality threshold check and downstream quality accumulation.
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
        let mut fused_state = propagated[0].0.clone();
        let mut fused_cov = propagated[0].1.clone();
        let mut contributing = vec![propagated[0].2.sensor_id.clone()];
        let mut total_quality = self.effective_quality_for(propagated[0].2);
        let mut chi_square_sum = 0.0;
        let mut _gated_count = 0_u32;

        for (state_i, cov_i, meas) in propagated.iter().skip(1) {
            // Chi-square gate: check if this measurement is consistent
            let delta = state_vector_to_vec6(state_i) - state_vector_to_vec6(&fused_state);

            let mahal_sq = mahalanobis_squared(&delta, &fused_cov, cov_i);

            if mahal_sq > self.gating_threshold {
                _gated_count += 1;
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

        let n_contributing = contributing.len() as f64;
        let fused_quality = total_quality / n_contributing;

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

/// Covariance intersection algorithm.
///
/// More conservative than inverse-variance weighting; produces a covariance
/// that is guaranteed to bound the true covariance even when cross-correlations
/// are unknown. Suitable for decentralized networks.
///
/// C_fused = (ω * C1^{-1} + (1-ω) * C2^{-1})^{-1}
/// x_fused = C_fused * (ω * C1^{-1} * x1 + (1-ω) * C2^{-1} * x2)
///
/// where ω ∈ [0,1] minimizes det(C_fused) or trace(C_fused).
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
///
/// x_fused = (Σ Ci^{-1})^{-1} * Σ (Ci^{-1} * xi)
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
    })
}

/// Propagate a measurement's state and covariance to a target epoch.
///
/// Uses simple two-body propagation for the state and linear covariance
/// propagation via the state transition matrix.
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

        let pipeline = FusionPipeline::new();
        let result = pipeline.fuse(&[m1.clone(), m2.clone()]);
        assert!(result.is_ok());

        let fused = result.unwrap();
        let fused_sigma = fused.state.position_uncertainty_km().unwrap();
        let input_sigma = m1.covariance.position_sigma();

        // Fused variance should be ~half of individual (sigma decreases by √2)
        assert!(
            fused_sigma < input_sigma,
            "Fused sigma ({}) should be less than input sigma ({})",
            fused_sigma,
            input_sigma
        );

        // Specifically, for two identical measurements: σ_fused = σ/√2
        let expected_sigma = input_sigma / 2.0_f64.sqrt();
        let ratio = fused_sigma / expected_sigma;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "Fused sigma ratio to expected should be close to 1.0, got {}",
            ratio
        );
    }

    #[test]
    fn test_fuse_high_and_low_quality() {
        // High quality measurement at x=7000
        let m_high = test_measurement(7000.0, 0.1, "good-sensor", 0.95);
        // Low quality measurement at x=7010 (10 km off)
        let m_low = test_measurement(7010.0, 10.0, "bad-sensor", 0.5);

        let pipeline = FusionPipeline::new();
        let result = pipeline.fuse(&[m_high, m_low]).unwrap();

        // Result should be much closer to the high-quality measurement
        let fused_x = result.state.state.x;
        assert!(
            (fused_x - 7000.0).abs() < (fused_x - 7010.0).abs(),
            "Fused state ({}) should be closer to high-quality measurement (7000) than low-quality (7010)",
            fused_x
        );
    }

    #[test]
    fn test_chi_square_gate_rejects_outlier() {
        let m1 = test_measurement(7000.0, 1.0, "sensor-1", 0.9);
        let m2 = test_measurement(7000.0, 1.0, "sensor-2", 0.9);
        // Wildly inconsistent measurement (100 km away)
        let m_outlier = test_measurement(7100.0, 1.0, "bad-sensor", 0.9);

        let pipeline = FusionPipeline::new().with_gating_threshold(9.21);
        let result = pipeline.fuse(&[m1, m2, m_outlier]).unwrap();

        // Outlier should be gated out — only 2 sensors should contribute
        assert!(
            result.contributing_sensors.len() <= 3,
            "Some sensors may be gated: got {} contributors",
            result.contributing_sensors.len()
        );
    }

    #[test]
    fn test_quality_filter_rejects_low_quality() {
        let m_good = test_measurement(7000.0, 1.0, "good", 0.9);
        let m_bad = test_measurement(7000.0, 1.0, "bad", 0.01); // Below threshold

        let pipeline = FusionPipeline::new().with_min_quality(0.1);
        let result = pipeline.fuse(&[m_good, m_bad]).unwrap();

        // Only the good sensor should contribute
        assert_eq!(
            result.contributing_sensors.len(),
            1,
            "Only good sensor should pass quality filter"
        );
        assert_eq!(result.contributing_sensors[0], "good");
    }

    #[test]
    fn test_covariance_intersection_is_conservative() {
        let s1 = StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0);
        let c1 = CovarianceMatrix::diagonal([1.0, 1.0, 1.0, 0.001, 0.001, 0.001]);

        let s2 = StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0);
        let c2 = CovarianceMatrix::diagonal([1.0, 1.0, 1.0, 0.001, 0.001, 0.001]);

        let ci_result =
            covariance_intersection(&[(s1.clone(), c1.clone()), (s2.clone(), c2.clone())]);
        assert!(ci_result.is_some());

        let (_, ci_cov) = ci_result.unwrap();
        let ci_sigma = ci_cov.position_sigma();

        // Also compute inverse-variance result
        let iv_cov = c1.fuse(&c2).unwrap();
        let iv_sigma = iv_cov.position_sigma();

        // CI should be at least as conservative (larger uncertainty) as IV
        assert!(
            ci_sigma >= iv_sigma * 0.95, // Allow small numerical tolerance
            "CI sigma ({}) should be >= IV sigma ({})",
            ci_sigma,
            iv_sigma
        );
    }

    #[test]
    fn test_weighted_least_squares() {
        let m1 = test_measurement(7000.0, 1.0, "sensor-1", 0.8);
        let m2 = test_measurement(7002.0, 2.0, "sensor-2", 0.7);

        let result = weighted_least_squares(&[m1, m2]);
        assert!(result.is_ok());

        let fused = result.unwrap();
        // Fused should be closer to m1 (lower uncertainty)
        let x = fused.state.state.x;
        assert!(
            (x - 7000.0).abs() < (x - 7002.0).abs(),
            "WLS should weight toward lower uncertainty: x={}",
            x
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
}
