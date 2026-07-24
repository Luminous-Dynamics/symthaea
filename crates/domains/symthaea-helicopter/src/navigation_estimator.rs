// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed navigation fusion and health gating.
//!
//! The estimator accepts timestamped position/velocity observations from
//! independent sources, propagates the previous state to measurement time,
//! rejects implausible innovations, and fuses accepted measurements with a
//! scalar covariance update. It remains intentionally small enough for the
//! Rust-native simulator while exposing the contracts a future EKF/UKF backend
//! must preserve.

use crate::navigation_consistency::{
    NavigationConsistencyEvidence, NavigationConsistencyMonitor, NavigationConsistencySample,
};

/// Navigation state estimate (position + velocity + uncertainty).
#[derive(Debug, Clone, PartialEq)]
pub struct HelicopterNavigationEstimate {
    /// Position in world/local frame (x, y, z), meters.
    pub position: [f64; 3],
    /// Velocity in world/local frame (vx, vy, vz), m/s.
    pub velocity: [f64; 3],
    /// Position variance (trace-like scalar approximation), m².
    pub position_variance: f64,
    /// Monotonic measurement timestamp in seconds.
    pub timestamp_s: f64,
}

/// Origin of a navigation observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationSource {
    External,
    Gnss,
    VisualOdometry,
    Inertial,
    RadarAltimeter,
}

impl NavigationSource {
    const COUNT: usize = 5;

    const fn index(self) -> usize {
        match self {
            Self::External => 0,
            Self::Gnss => 1,
            Self::VisualOdometry => 2,
            Self::Inertial => 3,
            Self::RadarAltimeter => 4,
        }
    }

    const fn from_index(index: usize) -> Self {
        match index {
            0 => Self::External,
            1 => Self::Gnss,
            2 => Self::VisualOdometry,
            3 => Self::Inertial,
            _ => Self::RadarAltimeter,
        }
    }

    /// Whether the source can independently constrain absolute position.
    const fn is_absolute(self) -> bool {
        !matches!(self, Self::Inertial)
    }
}

/// Errors that prevent a navigation estimate from being used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationEstimateError {
    Unavailable,
    NonFiniteMeasurement,
    NegativeVariance,
    TimeWentBackwards,
    InvalidHealthConfig,
    InvalidFusionConfig,
    QueryTimeBeforeMeasurement,
    Stale,
    ExcessiveUncertainty,
    InnovationRejected,
    SourceQuarantined,
    InsufficientIndependentSources,
    ConsistencyUnreliable,
}

/// Tuning for prediction and innovation gating.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationFusionConfig {
    /// Position uncertainty accumulated during dead reckoning, m²/s.
    pub process_noise_m2_per_s: f64,
    /// Maximum three-dimensional innovation measured in standard deviations.
    pub max_position_innovation_sigma: f64,
    /// Numerical floor for all accepted variances.
    pub minimum_variance: f64,
}

impl Default for NavigationFusionConfig {
    fn default() -> Self {
        Self {
            process_noise_m2_per_s: 1.0,
            max_position_innovation_sigma: 6.0,
            minimum_variance: 1.0e-6,
        }
    }
}

impl NavigationFusionConfig {
    pub fn validate(&self) -> Result<(), NavigationEstimateError> {
        if !self.process_noise_m2_per_s.is_finite()
            || self.process_noise_m2_per_s < 0.0
            || !self.max_position_innovation_sigma.is_finite()
            || self.max_position_innovation_sigma <= 0.0
            || !self.minimum_variance.is_finite()
            || self.minimum_variance <= 0.0
        {
            return Err(NavigationEstimateError::InvalidFusionConfig);
        }
        Ok(())
    }
}

/// Source-level integrity policy. Repeated innovation failures quarantine only
/// the offending source instead of poisoning the fused estimate or silently
/// disabling every navigation input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationIntegrityConfig {
    pub max_consecutive_rejections: u32,
    pub quarantine_duration_s: f64,
    pub source_freshness_s: f64,
    pub minimum_independent_sources: usize,
}

impl Default for NavigationIntegrityConfig {
    fn default() -> Self {
        Self {
            max_consecutive_rejections: 3,
            quarantine_duration_s: 5.0,
            source_freshness_s: 2.0,
            minimum_independent_sources: 1,
        }
    }
}

impl NavigationIntegrityConfig {
    pub fn validate(&self) -> Result<(), NavigationEstimateError> {
        if self.max_consecutive_rejections == 0
            || !self.quarantine_duration_s.is_finite()
            || self.quarantine_duration_s <= 0.0
            || !self.source_freshness_s.is_finite()
            || self.source_freshness_s < 0.0
            || self.minimum_independent_sources == 0
            || self.minimum_independent_sources > NavigationSource::COUNT - 1
        {
            return Err(NavigationEstimateError::InvalidFusionConfig);
        }
        Ok(())
    }
}

/// Public evidence for one navigation source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationSourceStatus {
    pub source: NavigationSource,
    pub accepted_updates: u64,
    pub rejected_updates: u64,
    pub consecutive_rejections: u32,
    pub last_accepted_s: Option<f64>,
    pub quarantined_until_s: Option<f64>,
}

impl NavigationSourceStatus {
    fn new(source: NavigationSource) -> Self {
        Self {
            source,
            accepted_updates: 0,
            rejected_updates: 0,
            consecutive_rejections: 0,
            last_accepted_s: None,
            quarantined_until_s: None,
        }
    }

    fn is_quarantined_at(&self, now_s: f64) -> bool {
        self.quarantined_until_s.is_some_and(|until| now_s < until)
    }
}

/// Independent-source observability summary for authority gating.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationObservability {
    pub fresh_sources: usize,
    pub fresh_absolute_sources: usize,
    pub usable: bool,
    pub reason: Option<NavigationEstimateError>,
}

/// Operational acceptance thresholds for a navigation estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationHealthConfig {
    /// Maximum age of the latest accepted observation at the time of use.
    pub max_age_s: f64,
    /// Maximum accepted position variance.
    pub max_position_variance: f64,
}

impl Default for NavigationHealthConfig {
    fn default() -> Self {
        Self {
            max_age_s: 0.5,
            max_position_variance: 25.0,
        }
    }
}

impl NavigationHealthConfig {
    pub fn validate(&self) -> Result<(), NavigationEstimateError> {
        if !self.max_age_s.is_finite()
            || self.max_age_s < 0.0
            || !self.max_position_variance.is_finite()
            || self.max_position_variance < 0.0
        {
            return Err(NavigationEstimateError::InvalidHealthConfig);
        }
        Ok(())
    }
}

/// Health summary suitable for telemetry and authority gates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NavigationHealth {
    pub age_s: f64,
    pub position_variance: f64,
    pub usable: bool,
    pub reason: Option<NavigationEstimateError>,
}

/// Counters retained for evidence and sensor-quality monitoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NavigationFusionStats {
    pub accepted_updates: u64,
    pub rejected_updates: u64,
    pub last_source: Option<NavigationSource>,
}

/// Measurement-backed estimator facade with deterministic fusion.
#[derive(Debug, Clone)]
pub struct HelicopterNavigationEstimator {
    latest: Option<HelicopterNavigationEstimate>,
    fusion: NavigationFusionConfig,
    integrity: NavigationIntegrityConfig,
    stats: NavigationFusionStats,
    source_status: [NavigationSourceStatus; NavigationSource::COUNT],
    consistency: NavigationConsistencyMonitor,
}

impl Default for HelicopterNavigationEstimator {
    fn default() -> Self {
        Self {
            latest: None,
            fusion: NavigationFusionConfig::default(),
            integrity: NavigationIntegrityConfig::default(),
            stats: NavigationFusionStats::default(),
            source_status: std::array::from_fn(|index| {
                NavigationSourceStatus::new(NavigationSource::from_index(index))
            }),
            consistency: NavigationConsistencyMonitor::default(),
        }
    }
}

impl HelicopterNavigationEstimator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_fusion_config(
        fusion: NavigationFusionConfig,
    ) -> Result<Self, NavigationEstimateError> {
        Self::with_configs(fusion, NavigationIntegrityConfig::default())
    }

    pub fn with_configs(
        fusion: NavigationFusionConfig,
        integrity: NavigationIntegrityConfig,
    ) -> Result<Self, NavigationEstimateError> {
        fusion.validate()?;
        integrity.validate()?;
        Ok(Self {
            latest: None,
            fusion,
            integrity,
            stats: NavigationFusionStats::default(),
            source_status: std::array::from_fn(|index| {
                NavigationSourceStatus::new(NavigationSource::from_index(index))
            }),
            consistency: NavigationConsistencyMonitor::default(),
        })
    }

    pub fn fusion_stats(&self) -> NavigationFusionStats {
        self.stats
    }

    pub fn source_status(&self, source: NavigationSource) -> NavigationSourceStatus {
        self.source_status[source.index()]
    }

    pub fn consistency_evidence(&self) -> NavigationConsistencyEvidence {
        self.consistency.evidence()
    }

    fn register_rejection(&mut self, source: NavigationSource, timestamp_s: f64) {
        self.stats.rejected_updates = self.stats.rejected_updates.saturating_add(1);
        let status = &mut self.source_status[source.index()];
        status.rejected_updates = status.rejected_updates.saturating_add(1);
        status.consecutive_rejections = status.consecutive_rejections.saturating_add(1);
        if timestamp_s.is_finite()
            && status.consecutive_rejections >= self.integrity.max_consecutive_rejections
        {
            status.quarantined_until_s = Some(timestamp_s + self.integrity.quarantine_duration_s);
        }
    }

    fn register_acceptance(&mut self, source: NavigationSource, timestamp_s: f64) {
        self.stats.accepted_updates = self.stats.accepted_updates.saturating_add(1);
        self.stats.last_source = Some(source);
        let status = &mut self.source_status[source.index()];
        status.accepted_updates = status.accepted_updates.saturating_add(1);
        status.consecutive_rejections = 0;
        status.last_accepted_s = Some(timestamp_s);
        status.quarantined_until_s = None;
    }

    /// Submit a generic fused or sensor-derived navigation measurement.
    pub fn update_measurement(
        &mut self,
        position: [f64; 3],
        velocity: [f64; 3],
        position_variance: f64,
        timestamp_s: f64,
    ) -> Result<(), NavigationEstimateError> {
        self.update_from_source(
            NavigationSource::External,
            position,
            velocity,
            position_variance,
            timestamp_s,
        )
    }

    /// Predict the current estimate to a later monotonic time without changing
    /// the timestamp of the last accepted sensor observation.
    fn predict_from(
        &self,
        estimate: &HelicopterNavigationEstimate,
        timestamp_s: f64,
    ) -> Result<HelicopterNavigationEstimate, NavigationEstimateError> {
        if !timestamp_s.is_finite() {
            return Err(NavigationEstimateError::NonFiniteMeasurement);
        }
        if timestamp_s < estimate.timestamp_s {
            return Err(NavigationEstimateError::TimeWentBackwards);
        }
        let dt = timestamp_s - estimate.timestamp_s;
        let mut predicted = estimate.clone();
        for axis in 0..3 {
            predicted.position[axis] += estimate.velocity[axis] * dt;
        }
        predicted.position_variance = (estimate.position_variance
            + self.fusion.process_noise_m2_per_s * dt)
            .max(self.fusion.minimum_variance);
        predicted.timestamp_s = timestamp_s;
        Ok(predicted)
    }

    /// Submit a source-labelled observation. Measurements are innovation-gated
    /// before covariance-weighted fusion, so a single finite but implausible
    /// GNSS/vision jump cannot silently become the vehicle position.
    pub fn update_from_source(
        &mut self,
        source: NavigationSource,
        position: [f64; 3],
        velocity: [f64; 3],
        position_variance: f64,
        timestamp_s: f64,
    ) -> Result<(), NavigationEstimateError> {
        self.fusion.validate()?;
        self.integrity.validate()?;
        if timestamp_s.is_finite()
            && self.source_status[source.index()].is_quarantined_at(timestamp_s)
        {
            let _ = self.consistency.observe(NavigationConsistencySample {
                source,
                normalized_innovation_sq: None,
                estimate_variance_m2: self.latest.as_ref().map_or(1.0, |e| e.position_variance),
                measurement_variance_m2: position_variance.max(self.fusion.minimum_variance),
                accepted: false,
            });
            self.stats.rejected_updates = self.stats.rejected_updates.saturating_add(1);
            let status = &mut self.source_status[source.index()];
            status.rejected_updates = status.rejected_updates.saturating_add(1);
            return Err(NavigationEstimateError::SourceQuarantined);
        }
        if !position.iter().all(|v| v.is_finite())
            || !velocity.iter().all(|v| v.is_finite())
            || !position_variance.is_finite()
            || !timestamp_s.is_finite()
        {
            let _ = self.consistency.observe(NavigationConsistencySample {
                source,
                normalized_innovation_sq: None,
                estimate_variance_m2: self.latest.as_ref().map_or(1.0, |e| e.position_variance),
                measurement_variance_m2: 1.0,
                accepted: false,
            });
            self.register_rejection(source, timestamp_s);
            return Err(NavigationEstimateError::NonFiniteMeasurement);
        }
        if position_variance < 0.0 {
            let _ = self.consistency.observe(NavigationConsistencySample {
                source,
                normalized_innovation_sq: None,
                estimate_variance_m2: self.latest.as_ref().map_or(1.0, |e| e.position_variance),
                measurement_variance_m2: 0.0,
                accepted: false,
            });
            self.register_rejection(source, timestamp_s);
            return Err(NavigationEstimateError::NegativeVariance);
        }

        let measurement_variance = position_variance.max(self.fusion.minimum_variance);
        let (fused, consistency_nis) = if let Some(previous) = self.latest.as_ref() {
            if timestamp_s < previous.timestamp_s {
                let _ = self.consistency.observe(NavigationConsistencySample {
                    source,
                    normalized_innovation_sq: None,
                    estimate_variance_m2: previous.position_variance,
                    measurement_variance_m2: measurement_variance,
                    accepted: false,
                });
                self.register_rejection(source, timestamp_s);
                return Err(NavigationEstimateError::TimeWentBackwards);
            }
            let predicted = self.predict_from(previous, timestamp_s)?;
            let innovation = [
                position[0] - predicted.position[0],
                position[1] - predicted.position[1],
                position[2] - predicted.position[2],
            ];
            let innovation_norm_sq = innovation.iter().map(|v| v * v).sum::<f64>();
            let innovation_variance = predicted.position_variance + measurement_variance;
            let normalized_innovation_sq =
                innovation_norm_sq / innovation_variance.max(self.fusion.minimum_variance);
            let gate_sq = self.fusion.max_position_innovation_sigma.powi(2)
                * innovation_variance.max(self.fusion.minimum_variance);
            if innovation_norm_sq > gate_sq {
                let _ = self.consistency.observe(NavigationConsistencySample {
                    source,
                    normalized_innovation_sq: Some(normalized_innovation_sq),
                    estimate_variance_m2: predicted.position_variance,
                    measurement_variance_m2: measurement_variance,
                    accepted: false,
                });
                self.register_rejection(source, timestamp_s);
                return Err(NavigationEstimateError::InnovationRejected);
            }

            let gain = predicted.position_variance / innovation_variance;
            let mut fused = predicted;
            for axis in 0..3 {
                fused.position[axis] += gain * innovation[axis];
                fused.velocity[axis] += gain * (velocity[axis] - fused.velocity[axis]);
            }
            fused.position_variance =
                ((1.0 - gain) * fused.position_variance).max(self.fusion.minimum_variance);
            (fused, Some(normalized_innovation_sq))
        } else {
            (
                HelicopterNavigationEstimate {
                    position,
                    velocity,
                    position_variance: measurement_variance,
                    timestamp_s,
                },
                None,
            )
        };

        let _ = self.consistency.observe(NavigationConsistencySample {
            source,
            normalized_innovation_sq: consistency_nis,
            estimate_variance_m2: fused.position_variance,
            measurement_variance_m2: measurement_variance,
            accepted: true,
        });
        self.latest = Some(fused);
        self.register_acceptance(source, timestamp_s);
        Ok(())
    }

    /// Return the latest accepted estimate, or fail closed before first update.
    pub fn estimate(&self) -> Result<&HelicopterNavigationEstimate, NavigationEstimateError> {
        self.latest
            .as_ref()
            .ok_or(NavigationEstimateError::Unavailable)
    }

    /// Return a dead-reckoned copy at `now_s`. The caller must still apply its
    /// health policy; prediction never refreshes the observation age.
    pub fn predicted_estimate_at(
        &self,
        now_s: f64,
    ) -> Result<HelicopterNavigationEstimate, NavigationEstimateError> {
        self.predict_from(self.estimate()?, now_s)
    }

    /// Return an estimate only when it is fresh and sufficiently certain at
    /// the supplied monotonic query time.
    pub fn estimate_at(
        &self,
        now_s: f64,
        health: &NavigationHealthConfig,
    ) -> Result<&HelicopterNavigationEstimate, NavigationEstimateError> {
        health.validate()?;
        if !now_s.is_finite() {
            return Err(NavigationEstimateError::NonFiniteMeasurement);
        }
        if !self.consistency.is_usable() {
            return Err(NavigationEstimateError::ConsistencyUnreliable);
        }
        let estimate = self.estimate()?;
        if now_s < estimate.timestamp_s {
            return Err(NavigationEstimateError::QueryTimeBeforeMeasurement);
        }
        if now_s - estimate.timestamp_s > health.max_age_s {
            return Err(NavigationEstimateError::Stale);
        }
        let predicted_variance = estimate.position_variance
            + self.fusion.process_noise_m2_per_s * (now_s - estimate.timestamp_s);
        if predicted_variance > health.max_position_variance {
            return Err(NavigationEstimateError::ExcessiveUncertainty);
        }
        Ok(estimate)
    }

    /// Summarize navigation health without converting an unusable estimate
    /// into a believable position.
    pub fn health_at(&self, now_s: f64, health: &NavigationHealthConfig) -> NavigationHealth {
        match self.estimate_at(now_s, health) {
            Ok(estimate) => NavigationHealth {
                age_s: now_s - estimate.timestamp_s,
                position_variance: estimate.position_variance
                    + self.fusion.process_noise_m2_per_s * (now_s - estimate.timestamp_s),
                usable: true,
                reason: None,
            },
            Err(reason) => {
                let (age_s, position_variance) = self
                    .latest
                    .as_ref()
                    .map(|estimate| {
                        let age = if now_s.is_finite() {
                            now_s - estimate.timestamp_s
                        } else {
                            f64::NAN
                        };
                        (
                            age,
                            estimate.position_variance
                                + self.fusion.process_noise_m2_per_s * age.max(0.0),
                        )
                    })
                    .unwrap_or((f64::NAN, f64::NAN));
                NavigationHealth {
                    age_s,
                    position_variance,
                    usable: false,
                    reason: Some(reason),
                }
            }
        }
    }

    /// Count fresh, non-quarantined sources. This does not replace the
    /// covariance/age health gate; it prevents one repeatedly failing source
    /// from masquerading as independent navigation observability.
    pub fn observability_at(
        &self,
        now_s: f64,
    ) -> Result<NavigationObservability, NavigationEstimateError> {
        self.integrity.validate()?;
        if !now_s.is_finite() {
            return Err(NavigationEstimateError::NonFiniteMeasurement);
        }
        let mut fresh_sources = 0;
        let mut fresh_absolute_sources = 0;
        for status in &self.source_status {
            let fresh = status.last_accepted_s.is_some_and(|timestamp| {
                now_s >= timestamp
                    && now_s - timestamp <= self.integrity.source_freshness_s
                    && !status.is_quarantined_at(now_s)
            });
            if fresh {
                fresh_sources += 1;
                if status.source.is_absolute() {
                    fresh_absolute_sources += 1;
                }
            }
        }
        let usable = fresh_absolute_sources >= self.integrity.minimum_independent_sources;
        Ok(NavigationObservability {
            fresh_sources,
            fresh_absolute_sources,
            usable,
            reason: (!usable).then_some(NavigationEstimateError::InsufficientIndependentSources),
        })
    }

    /// Invalidate navigation after sensor reset, frame change, or timeout.
    pub fn reset(&mut self) {
        self.latest = None;
        self.stats = NavigationFusionStats::default();
        self.source_status = std::array::from_fn(|index| {
            NavigationSourceStatus::new(NavigationSource::from_index(index))
        });
        self.consistency.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_is_unavailable_before_measurement() {
        let estimator = HelicopterNavigationEstimator::new();
        assert_eq!(
            estimator.estimate(),
            Err(NavigationEstimateError::Unavailable)
        );
    }

    #[test]
    fn first_measurement_initializes_exactly() {
        let mut estimator = HelicopterNavigationEstimator::new();
        estimator
            .update_from_source(
                NavigationSource::Gnss,
                [1.0, 2.0, 30.0],
                [0.1, 0.2, 0.0],
                0.25,
                10.0,
            )
            .unwrap();
        let estimate = estimator.estimate().unwrap();
        assert_eq!(estimate.position, [1.0, 2.0, 30.0]);
        assert_eq!(
            estimator.fusion_stats().last_source,
            Some(NavigationSource::Gnss)
        );
    }

    #[test]
    fn estimator_rejects_invalid_or_backward_measurements() {
        let mut estimator = HelicopterNavigationEstimator::new();
        assert_eq!(
            estimator.update_measurement([0.0; 3], [0.0; 3], -1.0, 1.0),
            Err(NavigationEstimateError::NegativeVariance)
        );
        estimator
            .update_measurement([0.0; 3], [0.0; 3], 1.0, 2.0)
            .unwrap();
        assert_eq!(
            estimator.update_measurement([0.0; 3], [0.0; 3], 1.0, 1.0),
            Err(NavigationEstimateError::TimeWentBackwards)
        );
    }

    #[test]
    fn innovation_gate_rejects_position_jump() {
        let mut estimator = HelicopterNavigationEstimator::new();
        estimator
            .update_measurement([0.0; 3], [0.0; 3], 0.1, 0.0)
            .unwrap();
        assert_eq!(
            estimator.update_from_source(
                NavigationSource::Gnss,
                [1_000.0, 0.0, 0.0],
                [0.0; 3],
                0.1,
                0.1,
            ),
            Err(NavigationEstimateError::InnovationRejected)
        );
        assert_eq!(estimator.fusion_stats().accepted_updates, 1);
        assert_eq!(estimator.fusion_stats().rejected_updates, 1);
        assert_eq!(estimator.estimate().unwrap().position, [0.0; 3]);
    }

    #[test]
    fn compatible_measurements_reduce_uncertainty() {
        let mut estimator = HelicopterNavigationEstimator::new();
        estimator
            .update_measurement([0.0; 3], [1.0, 0.0, 0.0], 4.0, 0.0)
            .unwrap();
        estimator
            .update_from_source(
                NavigationSource::VisualOdometry,
                [1.1, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                1.0,
                1.0,
            )
            .unwrap();
        let estimate = estimator.estimate().unwrap();
        assert!(estimate.position[0] > 1.0 && estimate.position[0] < 1.1);
        assert!(estimate.position_variance < 4.0);
    }

    #[test]
    fn prediction_does_not_refresh_sensor_age() {
        let mut estimator = HelicopterNavigationEstimator::new();
        estimator
            .update_measurement([0.0; 3], [2.0, 0.0, 0.0], 1.0, 5.0)
            .unwrap();
        let predicted = estimator.predicted_estimate_at(7.0).unwrap();
        assert_eq!(predicted.position[0], 4.0);
        let health = NavigationHealthConfig {
            max_age_s: 1.0,
            max_position_variance: 100.0,
        };
        assert_eq!(
            estimator.estimate_at(7.0, &health),
            Err(NavigationEstimateError::Stale)
        );
    }

    #[test]
    fn freshness_and_uncertainty_fail_closed() {
        let mut estimator = HelicopterNavigationEstimator::new();
        estimator
            .update_measurement([1.0, 2.0, 3.0], [0.0; 3], 4.0, 10.0)
            .unwrap();
        let health = NavigationHealthConfig {
            max_age_s: 0.5,
            max_position_variance: 5.0,
        };
        assert!(estimator.estimate_at(10.4, &health).is_ok());
        assert_eq!(
            estimator.estimate_at(10.6, &health),
            Err(NavigationEstimateError::Stale)
        );
    }

    #[test]
    fn repeated_rejections_quarantine_only_the_offending_source() {
        let mut estimator = HelicopterNavigationEstimator::with_configs(
            NavigationFusionConfig::default(),
            NavigationIntegrityConfig {
                max_consecutive_rejections: 2,
                quarantine_duration_s: 5.0,
                ..NavigationIntegrityConfig::default()
            },
        )
        .unwrap();
        estimator
            .update_from_source(
                NavigationSource::VisualOdometry,
                [0.0; 3],
                [0.0; 3],
                0.1,
                0.0,
            )
            .unwrap();
        for timestamp in [0.1, 0.2] {
            assert_eq!(
                estimator.update_from_source(
                    NavigationSource::Gnss,
                    [1_000.0, 0.0, 0.0],
                    [0.0; 3],
                    0.1,
                    timestamp,
                ),
                Err(NavigationEstimateError::InnovationRejected)
            );
        }
        assert_eq!(
            estimator.update_from_source(NavigationSource::Gnss, [0.0; 3], [0.0; 3], 0.1, 1.0,),
            Err(NavigationEstimateError::SourceQuarantined)
        );
        assert!(
            estimator
                .update_from_source(
                    NavigationSource::VisualOdometry,
                    [0.0; 3],
                    [0.0; 3],
                    0.1,
                    1.0,
                )
                .is_ok()
        );
        assert!(
            estimator
                .source_status(NavigationSource::Gnss)
                .quarantined_until_s
                .is_some()
        );
    }

    #[test]
    fn observability_counts_only_fresh_absolute_sources() {
        let mut estimator = HelicopterNavigationEstimator::with_configs(
            NavigationFusionConfig::default(),
            NavigationIntegrityConfig {
                minimum_independent_sources: 2,
                source_freshness_s: 1.0,
                ..NavigationIntegrityConfig::default()
            },
        )
        .unwrap();
        estimator
            .update_from_source(NavigationSource::Gnss, [0.0; 3], [0.0; 3], 1.0, 0.0)
            .unwrap();
        estimator
            .update_from_source(NavigationSource::Inertial, [0.0; 3], [0.0; 3], 1.0, 0.1)
            .unwrap();
        let one = estimator.observability_at(0.2).unwrap();
        assert_eq!(one.fresh_sources, 2);
        assert_eq!(one.fresh_absolute_sources, 1);
        assert!(!one.usable);
        estimator
            .update_from_source(
                NavigationSource::VisualOdometry,
                [0.0; 3],
                [0.0; 3],
                1.0,
                0.2,
            )
            .unwrap();
        assert!(estimator.observability_at(0.3).unwrap().usable);
        assert!(!estimator.observability_at(2.0).unwrap().usable);
    }

    #[test]
    fn repeated_innovation_failures_make_health_fail_closed() {
        let mut estimator = HelicopterNavigationEstimator::new();
        estimator
            .update_from_source(NavigationSource::Gnss, [0.0; 3], [0.0; 3], 1.0, 0.0)
            .unwrap();
        for step in 1..=5 {
            let _ = estimator.update_from_source(
                NavigationSource::VisualOdometry,
                [1_000.0; 3],
                [0.0; 3],
                1.0,
                step as f64 * 0.1,
            );
        }
        assert_eq!(
            estimator.consistency_evidence().state,
            crate::navigation_consistency::NavigationConsistencyState::Unreliable
        );
        assert_eq!(
            estimator.estimate_at(0.6, &NavigationHealthConfig::default()),
            Err(NavigationEstimateError::ConsistencyUnreliable)
        );
    }
}
