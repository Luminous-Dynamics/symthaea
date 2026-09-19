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

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::floating_base::FloatingBaseDynamicsSnapshot;
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

/// Why a contact-bias set cannot be treated as belonging to a floating-base
/// dynamics snapshot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContactBiasBindingError {
    InvalidDynamicsSnapshot,
    IncompleteBiasSet { expected: usize, actual: usize },
    InvalidBiasEvidence,
    DuplicateBiasSite,
    UnknownBiasSite,
    MissingBiasSite,
    ModelIdentityMismatch,
    SampleTimeMismatch,
    NonFiniteLinearTarget,
}

/// A floating-base dynamics snapshot plus a complete set of contact acceleration
/// bias terms bound to the same model subject and sample instant.
///
/// This wrapper is deliberately additive. `FloatingBaseDynamicsSnapshot`
/// remains the compatibility contract for callers that do not yet claim full
/// contact-acceleration kinematics. Controllers that claim the stronger
/// `J qdd + Jdot qdot = a*` relation should require this wrapper instead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasQualifiedFloatingBaseDynamicsV1 {
    pub dynamics: FloatingBaseDynamicsSnapshot,
    pub contact_biases: Vec<ContactAccelerationBiasV1>,
}

impl BiasQualifiedFloatingBaseDynamicsV1 {
    /// Validate exact subject binding between every contact Jacobian and every
    /// contact acceleration-bias record.
    pub fn validate_binding(&self) -> Result<(), ContactBiasBindingError> {
        if !self.dynamics.validate() {
            return Err(ContactBiasBindingError::InvalidDynamicsSnapshot);
        }
        if self.contact_biases.len() != self.dynamics.contacts.len() {
            return Err(ContactBiasBindingError::IncompleteBiasSet {
                expected: self.dynamics.contacts.len(),
                actual: self.contact_biases.len(),
            });
        }

        let contact_sites = self
            .dynamics
            .contacts
            .iter()
            .map(|contact| contact.site_id.as_str())
            .collect::<BTreeSet<_>>();
        let mut bias_sites = BTreeSet::new();

        for bias in &self.contact_biases {
            if !bias.validate() {
                return Err(ContactBiasBindingError::InvalidBiasEvidence);
            }
            if !bias_sites.insert(bias.site_id.as_str()) {
                return Err(ContactBiasBindingError::DuplicateBiasSite);
            }
            if !contact_sites.contains(bias.site_id.as_str()) {
                return Err(ContactBiasBindingError::UnknownBiasSite);
            }
            if bias.model_id != self.dynamics.model_id {
                return Err(ContactBiasBindingError::ModelIdentityMismatch);
            }
            if !same_sample_time(bias.sampled_at_s, self.dynamics.sampled_at_s) {
                return Err(ContactBiasBindingError::SampleTimeMismatch);
            }
        }

        if self
            .dynamics
            .contacts
            .iter()
            .any(|contact| !bias_sites.contains(contact.site_id.as_str()))
        {
            return Err(ContactBiasBindingError::MissingBiasSite);
        }

        Ok(())
    }

    pub fn bias_for_site(&self, site_id: &str) -> Option<&ContactAccelerationBiasV1> {
        self.contact_biases
            .iter()
            .find(|bias| bias.site_id == site_id)
    }

    /// Equality target for the existing three-axis point-contact solver.
    ///
    /// For desired linear contact acceleration `a*`, the solver row is
    /// `J_linear qdd = a* - (Jdot qdot)_linear`.
    ///
    /// Angular components remain bound in the evidence object for the later 6D
    /// contact-wrench controller, but are not silently imposed on today's 3D
    /// point-force formulation.
    pub fn linear_contact_equality_target(
        &self,
        site_id: &str,
        desired_linear_acceleration: [f64; 3],
    ) -> Result<[f64; 3], ContactBiasBindingError> {
        self.validate_binding()?;
        if desired_linear_acceleration
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(ContactBiasBindingError::NonFiniteLinearTarget);
        }
        let bias = self
            .bias_for_site(site_id)
            .ok_or(ContactBiasBindingError::MissingBiasSite)?;
        Ok(std::array::from_fn(|axis| {
            desired_linear_acceleration[axis] - bias.spatial_bias_acceleration[3 + axis]
        }))
    }
}

fn same_sample_time(left: f64, right: f64) -> bool {
    let tolerance = 1.0e-9 * (1.0 + left.abs().max(right.abs()));
    (left - right).abs() <= tolerance
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
    use crate::floating_base::{FLOATING_BASE_DOF, FloatingBaseDynamicsSnapshot};
    use crate::full_dynamics::DynamicsProvenance;
    use crate::morphology::HumanoidMorphology;

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

    fn floating_snapshot() -> FloatingBaseDynamicsSnapshot {
        let morphology = HumanoidMorphology::Dmc21;
        let nv = FLOATING_BASE_DOF + morphology.num_actuators();
        let mut mass_matrix = vec![0.0; nv * nv];
        for index in 0..nv {
            mass_matrix[index * nv + index] = 1.0 + index as f64 * 0.01;
        }
        FloatingBaseDynamicsSnapshot {
            morphology,
            sampled_at_s: 1.0,
            total_mass_kg: 70.0,
            gravity_world_mps2: [0.0, 0.0, -9.81],
            generalized_velocity_count: nv,
            mass_matrix,
            bias_force: vec![0.0; nv],
            actuator_velocity_indices: (FLOATING_BASE_DOF..nv).collect(),
            torque_limits_nm: vec![100.0; morphology.num_actuators()],
            centroidal_momentum_matrix: vec![0.0; 6 * nv],
            contacts: vec![SpatialContactJacobian {
                site_id: "right_foot".to_string(),
                rows: std::array::from_fn(|_| vec![0.0; nv]),
                confidence: 1.0,
            }],
            provenance: DynamicsProvenance::mujoco_solver_with_morphology_limits(),
            model_id: "synthetic-model-v1".to_string(),
        }
    }

    fn qualified() -> BiasQualifiedFloatingBaseDynamicsV1 {
        BiasQualifiedFloatingBaseDynamicsV1 {
            dynamics: floating_snapshot(),
            contact_biases: vec![bias([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])],
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

    #[test]
    fn complete_bias_set_binds_to_exact_dynamics_subject() {
        let qualified = qualified();
        assert!(qualified.validate_binding().is_ok());
        assert!(qualified.bias_for_site("right_foot").is_some());
    }

    #[test]
    fn missing_bias_site_is_not_a_qualified_model() {
        let mut qualified = qualified();
        qualified.contact_biases.clear();
        assert_eq!(
            qualified.validate_binding(),
            Err(ContactBiasBindingError::IncompleteBiasSet {
                expected: 1,
                actual: 0,
            })
        );
    }

    #[test]
    fn stale_or_cross_model_bias_is_rejected() {
        let mut stale = qualified();
        stale.contact_biases[0].sampled_at_s += 0.01;
        assert_eq!(
            stale.validate_binding(),
            Err(ContactBiasBindingError::SampleTimeMismatch)
        );

        let mut cross_model = qualified();
        cross_model.contact_biases[0].model_id = "other-model".to_string();
        assert_eq!(
            cross_model.validate_binding(),
            Err(ContactBiasBindingError::ModelIdentityMismatch)
        );
    }

    #[test]
    fn duplicate_bias_sites_are_rejected() {
        let mut qualified = qualified();
        qualified.contact_biases.push(qualified.contact_biases[0].clone());
        qualified.dynamics.contacts.push(qualified.dynamics.contacts[0].clone());
        // The underlying dynamics snapshot rejects duplicate contact ids first;
        // retain a direct duplicate-bias test with one declared contact below.
        qualified.dynamics.contacts.pop();
        qualified.contact_biases.pop();
        let duplicate = qualified.contact_biases[0].clone();
        qualified.contact_biases.push(duplicate);
        assert_eq!(
            qualified.validate_binding(),
            Err(ContactBiasBindingError::IncompleteBiasSet {
                expected: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn linear_equality_target_subtracts_velocity_bias() {
        let qualified = qualified();
        let target = qualified
            .linear_contact_equality_target("right_foot", [1.0, 2.0, 3.0])
            .unwrap();
        assert_eq!(target, [0.6, 1.5, 2.4]);
    }

    #[test]
    fn finite_difference_bias_can_be_bound_when_subject_identity_matches() {
        let mut qualified = qualified();
        qualified.contact_biases[0].source = ContactBiasAccelerationSource::FiniteDifferenceOracle;
        assert!(qualified.validate_binding().is_ok());
    }
}
