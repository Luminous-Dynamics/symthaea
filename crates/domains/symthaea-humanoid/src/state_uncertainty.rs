// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit bounded uncertainty for humanoid state estimation.
//!
//! This module deliberately does **not** claim probabilistic covariance. The
//! current fused estimator exposes deterministic innovation gates, measurement
//! freshness, and contact trust. We preserve those semantics as auditable
//! bounded uncertainty and derive a restrictive epistemic-authority budget from
//! them. A future EKF/factor-graph estimator may add calibrated covariance
//! without changing the meaning of these existing bounds.

use serde::{Deserialize, Serialize};

use crate::execution::HumanoidAuthorityEnvelope;
use crate::state_estimation::{StateEstimatorConfig, StateEstimatorReport};

/// One observed estimator quantity relative to its configured rejection bound.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundedUncertainty {
    pub observed: f64,
    pub rejection_bound: f64,
}

impl BoundedUncertainty {
    pub fn new(observed: f64, rejection_bound: f64) -> Self {
        Self {
            observed,
            rejection_bound,
        }
    }

    /// Fraction of the configured rejection envelope consumed by the observed
    /// innovation. Invalid values fail closed to full utilization.
    pub fn utilization(self) -> f64 {
        if !self.observed.is_finite()
            || !self.rejection_bound.is_finite()
            || self.rejection_bound <= 0.0
            || self.observed < 0.0
        {
            return 1.0;
        }
        (self.observed / self.rejection_bound).clamp(0.0, 1.0)
    }

    /// Remaining normalized margin before the estimator's rejection boundary.
    pub fn remaining_margin(self) -> f64 {
        1.0 - self.utilization()
    }
}

/// Auditable uncertainty envelope associated with one estimator update.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanoidStateUncertaintyEnvelope {
    pub sequence: u64,
    pub accepted: bool,
    pub measurement_age: BoundedUncertainty,
    pub orientation_innovation: BoundedUncertainty,
    pub linear_velocity_innovation: BoundedUncertainty,
    pub maximum_joint_position_innovation: BoundedUncertainty,
    /// Existing contact confidence/freshness gate from the estimator.
    pub contact_trust: f32,
}

impl HumanoidStateUncertaintyEnvelope {
    pub fn from_estimator_report(
        report: StateEstimatorReport,
        config: &StateEstimatorConfig,
    ) -> Self {
        Self {
            sequence: report.sequence,
            accepted: report.accepted,
            measurement_age: BoundedUncertainty::new(
                report.measurement_age_s,
                config.maximum_measurement_age_s,
            ),
            orientation_innovation: BoundedUncertainty::new(
                report.orientation_innovation_rad,
                config.maximum_orientation_innovation_rad,
            ),
            linear_velocity_innovation: BoundedUncertainty::new(
                report.linear_velocity_innovation_mps,
                config.maximum_linear_velocity_innovation_mps,
            ),
            maximum_joint_position_innovation: BoundedUncertainty::new(
                report.maximum_joint_position_innovation_rad,
                config.maximum_joint_position_innovation_rad,
            ),
            contact_trust: report.contact_trust,
        }
    }

    /// Conservative authority available to actions that depend on the current
    /// state estimate. This is intentionally not a probability of correctness.
    /// It is the minimum remaining margin across explicit estimator bounds and
    /// contact trust, suitable only as a restrictive authority input.
    pub fn epistemic_authority(self) -> f32 {
        if !self.accepted || !self.contact_trust.is_finite() {
            return 0.0;
        }
        let contact = self.contact_trust.clamp(0.0, 1.0) as f64;
        let remaining = [
            self.measurement_age.remaining_margin(),
            self.orientation_innovation.remaining_margin(),
            self.linear_velocity_innovation.remaining_margin(),
            self.maximum_joint_position_innovation.remaining_margin(),
            contact,
        ]
        .into_iter()
        .fold(1.0, f64::min);
        remaining.clamp(0.0, 1.0) as f32
    }

    /// Restrict an already established authority envelope with this state
    /// evidence. Existing stricter epistemic limits are preserved.
    pub fn restrict_authority(
        self,
        mut authority: HumanoidAuthorityEnvelope,
    ) -> HumanoidAuthorityEnvelope {
        let state_limit = self.epistemic_authority();
        authority.epistemic = if authority.epistemic.is_finite() {
            authority.epistemic.clamp(0.0, 1.0).min(state_limit)
        } else {
            0.0
        };
        authority
    }

    pub fn fully_within_bounds(self) -> bool {
        self.accepted
            && self.contact_trust.is_finite()
            && self.measurement_age.utilization() < 1.0
            && self.orientation_innovation.utilization() < 1.0
            && self.linear_velocity_innovation.utilization() < 1.0
            && self.maximum_joint_position_innovation.utilization() < 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> StateEstimatorReport {
        StateEstimatorReport {
            sequence: 7,
            measurement_age_s: 0.0,
            dt_s: 0.01,
            contact_trust: 1.0,
            support: crate::contact::BipedSupport::Double,
            orientation_innovation_rad: 0.0,
            linear_velocity_innovation_mps: 0.0,
            maximum_joint_position_innovation_rad: 0.0,
            accepted: true,
        }
    }

    #[test]
    fn zero_innovation_has_full_epistemic_authority() {
        let envelope = HumanoidStateUncertaintyEnvelope::from_estimator_report(
            report(),
            &StateEstimatorConfig::default(),
        );
        assert_eq!(envelope.epistemic_authority(), 1.0);
        assert!(envelope.fully_within_bounds());
    }

    #[test]
    fn half_consumed_orientation_gate_halves_authority() {
        let config = StateEstimatorConfig::default();
        let mut report = report();
        report.orientation_innovation_rad = 0.5 * config.maximum_orientation_innovation_rad;
        let envelope = HumanoidStateUncertaintyEnvelope::from_estimator_report(report, &config);
        assert!((envelope.epistemic_authority() - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn contact_trust_can_be_the_limiting_source() {
        let mut report = report();
        report.contact_trust = 0.3;
        let envelope = HumanoidStateUncertaintyEnvelope::from_estimator_report(
            report,
            &StateEstimatorConfig::default(),
        );
        assert!((envelope.epistemic_authority() - 0.3).abs() < 1.0e-6);
    }

    #[test]
    fn uncertainty_only_restricts_existing_authority() {
        let config = StateEstimatorConfig::default();
        let mut source = report();
        source.orientation_innovation_rad = 0.5 * config.maximum_orientation_innovation_rad;
        let uncertainty = HumanoidStateUncertaintyEnvelope::from_estimator_report(source, &config);

        let admitted = uncertainty.restrict_authority(HumanoidAuthorityEnvelope::fully_admitted());
        assert!((admitted.epistemic - 0.5).abs() < 1.0e-6);

        let already_stricter = uncertainty.restrict_authority(HumanoidAuthorityEnvelope {
            epistemic: 0.2,
            ..HumanoidAuthorityEnvelope::fully_admitted()
        });
        assert!((already_stricter.epistemic - 0.2).abs() < 1.0e-6);
    }

    #[test]
    fn rejected_or_non_finite_evidence_fails_closed() {
        let mut rejected = report();
        rejected.accepted = false;
        let envelope = HumanoidStateUncertaintyEnvelope::from_estimator_report(
            rejected,
            &StateEstimatorConfig::default(),
        );
        assert_eq!(envelope.epistemic_authority(), 0.0);

        let mut invalid = report();
        invalid.contact_trust = f32::NAN;
        let envelope = HumanoidStateUncertaintyEnvelope::from_estimator_report(
            invalid,
            &StateEstimatorConfig::default(),
        );
        assert_eq!(envelope.epistemic_authority(), 0.0);
    }
}
