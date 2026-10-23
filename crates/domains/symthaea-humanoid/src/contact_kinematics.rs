// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed contact-acceleration kinematics.
//!
//! A rigid contact constraint is not generally `J(q) qdd = a*`.  The complete
//! spatial acceleration relation is
//!
//! `J(q) qdd + Jdot(q, qdot) qdot = a*`.
//!
//! The second term is a velocity-dependent bias acceleration.  This module
//! makes that term explicit and fail-closed: callers cannot obtain a qualified
//! contact-acceleration residual without separately supplied bias evidence.
//! In particular, absence of bias evidence is not interpreted as a zero vector.

use serde::{Deserialize, Serialize};

use crate::full_dynamics::SpatialContactJacobian;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContactBiasAccelerationSource {
    /// Directly produced by the simulator/dynamics solver at the same state.
    SimulatorSolver,
    /// Produced by a separately qualified finite-difference or numerical oracle.
    FiniteDifferenceOracle,
    /// Produced by a hardware/system-identification model.
    SystemIdentification,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContactAccelerationBiasV1 {
    pub site_id: String,
    /// Spatial bias acceleration `Jdot(q, qdot) qdot`, angular xyz then linear xyz.
    pub spatial_bias_acceleration: [f64; 6],
    pub source: ContactBiasAccelerationSource,
    pub sampled_at_s: f64,
    pub model_id: String,
}

impl ContactAccelerationBiasV1 {
    pub fn validate(&self) -> bool {
        !self.site_id.trim().is_empty()
            && self
                .spatial_bias_acceleration
                .iter()
                .all(|value| value.is_finite())
            && self.sampled_at_s.is_finite()
            && self.sampled_at_s >= 0.0
            && !self.model_id.trim().is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContactAccelerationError {
    MissingBiasEvidence,
    InvalidJacobian,
    InvalidBiasEvidence,
    SiteMismatch,
    GeneralizedAccelerationCount { expected: usize, actual: usize },
    NonFiniteGeneralizedAcceleration { index: usize },
    NonFiniteTarget,
}

/// Evaluate the complete spatial contact-acceleration residual
///
/// `r = J qdd + Jdot qdot - a*`.
///
/// Bias evidence is mandatory. This intentionally does not provide a helper that
/// silently substitutes zero for `Jdot qdot`; stationary/zero-bias cases should
/// carry explicit evidence containing a zero vector.
pub fn contact_acceleration_residual(
    jacobian: &SpatialContactJacobian,
    generalized_acceleration: &[f64],
    bias: Option<&ContactAccelerationBiasV1>,
    target_spatial_acceleration: [f64; 6],
) -> Result<[f64; 6], ContactAccelerationError> {
    let bias = bias.ok_or(ContactAccelerationError::MissingBiasEvidence)?;
    let nv = generalized_acceleration.len();

    if !jacobian.validate(nv) {
        return Err(ContactAccelerationError::InvalidJacobian);
    }
    if !bias.validate() {
        return Err(ContactAccelerationError::InvalidBiasEvidence);
    }
    if jacobian.site_id != bias.site_id {
        return Err(ContactAccelerationError::SiteMismatch);
    }
    if jacobian.rows.iter().any(|row| row.len() != nv) {
        return Err(ContactAccelerationError::GeneralizedAccelerationCount {
            expected: jacobian.rows[0].len(),
            actual: nv,
        });
    }
    if let Some((index, _)) = generalized_acceleration
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(ContactAccelerationError::NonFiniteGeneralizedAcceleration { index });
    }
    if target_spatial_acceleration
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(ContactAccelerationError::NonFiniteTarget);
    }

    Ok(std::array::from_fn(|axis| {
        dot(&jacobian.rows[axis], generalized_acceleration)
            + bias.spatial_bias_acceleration[axis]
            - target_spatial_acceleration[axis]
    }))
}

/// Maximum absolute component of a spatial contact-acceleration residual.
pub fn maximum_contact_acceleration_residual(residual: [f64; 6]) -> f64 {
    residual.into_iter().map(f64::abs).fold(0.0, f64::max)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jacobian() -> SpatialContactJacobian {
        SpatialContactJacobian {
            site_id: "right_foot".to_string(),
            rows: [
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
                vec![2.0, 0.0],
                vec![0.0, 2.0],
                vec![1.0, -1.0],
            ],
            confidence: 1.0,
        }
    }

    fn bias(values: [f64; 6]) -> ContactAccelerationBiasV1 {
        ContactAccelerationBiasV1 {
            site_id: "right_foot".to_string(),
            spatial_bias_acceleration: values,
            source: ContactBiasAccelerationSource::SimulatorSolver,
            sampled_at_s: 1.0,
            model_id: "synthetic-model-v1".to_string(),
        }
    }

    #[test]
    fn missing_bias_evidence_fails_closed() {
        let error = contact_acceleration_residual(&jacobian(), &[0.0, 0.0], None, [0.0; 6])
            .unwrap_err();
        assert_eq!(error, ContactAccelerationError::MissingBiasEvidence);
    }

    #[test]
    fn nonzero_bias_is_not_erased_by_zero_generalized_acceleration() {
        let evidence = bias([0.1, -0.2, 0.3, -0.4, 0.5, -0.6]);
        let residual = contact_acceleration_residual(
            &jacobian(),
            &[0.0, 0.0],
            Some(&evidence),
            [0.0; 6],
        )
        .unwrap();
        assert_eq!(residual, evidence.spatial_bias_acceleration);
        assert!((maximum_contact_acceleration_residual(residual) - 0.6).abs() < 1.0e-12);
    }

    #[test]
    fn generalized_acceleration_can_cancel_bias_exactly() {
        let evidence = bias([-1.0, -2.0, -3.0, -2.0, -4.0, 1.0]);
        let residual = contact_acceleration_residual(
            &jacobian(),
            &[1.0, 2.0],
            Some(&evidence),
            [0.0; 6],
        )
        .unwrap();
        assert!(residual.iter().all(|value| value.abs() < 1.0e-12));
    }

    #[test]
    fn target_acceleration_is_part_of_the_residual() {
        let evidence = bias([0.0; 6]);
        let qdd = [1.0, 2.0];
        let target = [1.0, 2.0, 3.0, 2.0, 4.0, -1.0];
        let residual = contact_acceleration_residual(
            &jacobian(),
            &qdd,
            Some(&evidence),
            target,
        )
        .unwrap();
        assert!(residual.iter().all(|value| value.abs() < 1.0e-12));
    }

    #[test]
    fn mismatched_site_identity_is_rejected() {
        let mut evidence = bias([0.0; 6]);
        evidence.site_id = "left_foot".to_string();
        let error = contact_acceleration_residual(
            &jacobian(),
            &[0.0, 0.0],
            Some(&evidence),
            [0.0; 6],
        )
        .unwrap_err();
        assert_eq!(error, ContactAccelerationError::SiteMismatch);
    }

    #[test]
    fn non_finite_bias_evidence_is_rejected() {
        let mut evidence = bias([0.0; 6]);
        evidence.spatial_bias_acceleration[3] = f64::NAN;
        let error = contact_acceleration_residual(
            &jacobian(),
            &[0.0, 0.0],
            Some(&evidence),
            [0.0; 6],
        )
        .unwrap_err();
        assert_eq!(error, ContactAccelerationError::InvalidBiasEvidence);
    }
}
