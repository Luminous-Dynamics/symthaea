// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Per-joint actuation authority projected into contact-wrench space.
//!
//! Coarse physical-health authority intentionally collapses the body to its
//! most restrictive scalar. That is useful for fail-closed execution admission,
//! but it loses *where* actuation capability has degraded. This module retains
//! per-joint torque authority and projects it through the full-order spatial
//! contact Jacobians already supplied by the dynamics layer.
//!
//! For a unit wrench axis `w`, the actuator torque contribution is `J^T w`.
//! Under independent joint torque box limits, the maximum axis-aligned wrench is
//! therefore bounded by the most restrictive `tau_limit / |J_i|`. Applying a
//! retained joint-authority fraction yields a conservative degraded bound.
//!
//! This is deliberately **not** a complete dynamic controllability proof: it
//! does not model simultaneous multi-axis wrench polytopes, friction cones,
//! passive structural load paths, bias/dynamic compensation, contact loss, or
//! task reachability. Those remain separate constraints in the controller and
//! capability stack.

use serde::{Deserialize, Serialize};

use crate::full_dynamics::{
    DynamicsComponentSource, FullRigidBodyDynamicsSnapshot, SpatialContactJacobian,
};
use crate::morphology::HumanoidMorphology;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JointTorqueAuthoritySample {
    /// Whether the producer has valid health/drive evidence for this joint.
    pub evidence_valid: bool,
    /// Fraction of the qualified torque limit currently retained in [0, 1].
    pub retained_torque_fraction: f64,
}

impl JointTorqueAuthoritySample {
    pub fn effective_fraction(self) -> f64 {
        if self.evidence_valid && self.retained_torque_fraction.is_finite() {
            self.retained_torque_fraction.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidJointTorqueAuthorityFrame {
    pub morphology: HumanoidMorphology,
    pub sequence: u64,
    pub sampled_at_s: f64,
    /// Stable producer/profile identity. This is not a display label.
    pub authority_profile_id: String,
    /// Non-zero calibration / actuator-map identity supplied by the backend.
    pub calibration_fingerprint: u64,
    pub joints: Vec<JointTorqueAuthoritySample>,
}

impl HumanoidJointTorqueAuthorityFrame {
    pub fn validate(&self) -> bool {
        self.sampled_at_s.is_finite()
            && !self.authority_profile_id.trim().is_empty()
            && self.authority_profile_id == self.authority_profile_id.trim()
            && self.authority_profile_id.len() <= 256
            && self.calibration_fingerprint != 0
            && self.joints.len() == self.morphology.num_actuators()
            && self.joints.iter().all(|joint| {
                joint.retained_torque_fraction.is_finite()
                    && (0.0..=1.0).contains(&joint.retained_torque_fraction)
            })
    }

    pub fn minimum_retained_fraction(&self) -> f64 {
        if !self.validate() {
            return 0.0;
        }
        self.joints
            .iter()
            .map(|joint| joint.effective_fraction())
            .fold(1.0, f64::min)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContactWrenchControllabilityConfig {
    /// Maximum admitted age of the per-joint authority frame.
    pub maximum_authority_age_s: f64,
    /// Contact Jacobians below this confidence are not used for capability.
    pub minimum_contact_confidence: f64,
    /// Jacobian coefficients at or below this magnitude are treated as zero.
    pub jacobian_epsilon: f64,
}

impl ContactWrenchControllabilityConfig {
    pub fn validate(self) -> bool {
        self.maximum_authority_age_s.is_finite()
            && self.maximum_authority_age_s > 0.0
            && self.minimum_contact_confidence.is_finite()
            && (0.0..=1.0).contains(&self.minimum_contact_confidence)
            && self.jacobian_epsilon.is_finite()
            && self.jacobian_epsilon > 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContactWrenchAxis {
    MomentX,
    MomentY,
    MomentZ,
    ForceX,
    ForceY,
    ForceZ,
}

impl ContactWrenchAxis {
    pub const ALL: [Self; 6] = [
        Self::MomentX,
        Self::MomentY,
        Self::MomentZ,
        Self::ForceX,
        Self::ForceY,
        Self::ForceZ,
    ];

    pub const fn row(self) -> usize {
        match self {
            Self::MomentX => 0,
            Self::MomentY => 1,
            Self::MomentZ => 2,
            Self::ForceX => 3,
            Self::ForceY => 4,
            Self::ForceZ => 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContactWrenchAxisMargin {
    pub axis: ContactWrenchAxis,
    /// False means the supplied Jacobian has no actuated contribution along
    /// this axis, so this module makes no controllability claim for it.
    pub actuated_support_present: bool,
    /// Axis-aligned wrench limit under nominal joint torque limits.
    pub nominal_limit: Option<f64>,
    /// Axis-aligned wrench limit after per-joint retained authority is applied.
    pub retained_limit: Option<f64>,
    /// Retained / nominal limit in [0, 1]. Unsupported axes report zero.
    pub retained_fraction: f64,
    /// Joint that first saturates the retained axis-aligned wrench bound.
    pub limiting_joint: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContactWrenchMarginAssessment {
    pub site_id: String,
    pub contact_confidence: f64,
    pub jacobian_source: DynamicsComponentSource,
    pub actuator_limit_source: DynamicsComponentSource,
    pub axes: Vec<ContactWrenchAxisMargin>,
    /// Minimum retained fraction over axes with actuated support. Zero if no
    /// supported axis is available.
    pub minimum_retained_fraction: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidActuationControllabilityAssessment {
    pub morphology: HumanoidMorphology,
    pub authority_sequence: u64,
    pub authority_age_s: f64,
    pub authority_profile_id: String,
    pub calibration_fingerprint: u64,
    pub dynamics_model_id: String,
    pub sites: Vec<ContactWrenchMarginAssessment>,
}

impl HumanoidActuationControllabilityAssessment {
    pub fn site(&self, site_id: &str) -> Option<&ContactWrenchMarginAssessment> {
        self.sites.iter().find(|site| site.site_id == site_id)
    }

    pub fn minimum_retained_fraction(&self) -> f64 {
        if self.sites.is_empty() {
            0.0
        } else {
            self.sites
                .iter()
                .map(|site| site.minimum_retained_fraction)
                .fold(1.0, f64::min)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActuationControllabilityError {
    InvalidConfig,
    InvalidDynamics,
    InvalidAuthorityFrame,
    MorphologyMismatch,
    NonFiniteEvaluationTime,
    TimestampRegression,
    StaleAuthorityFrame,
}

impl std::fmt::Display for ActuationControllabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidConfig => "contact-wrench controllability config is invalid",
            Self::InvalidDynamics => "full rigid-body dynamics snapshot is invalid",
            Self::InvalidAuthorityFrame => "joint torque authority frame is invalid",
            Self::MorphologyMismatch => "dynamics and joint authority morphology differ",
            Self::NonFiniteEvaluationTime => "controllability evaluation time is non-finite",
            Self::TimestampRegression => "authority frame timestamp is in the future",
            Self::StaleAuthorityFrame => "joint torque authority evidence is stale",
        };
        f.write_str(message)
    }
}

impl std::error::Error for ActuationControllabilityError {}

/// Project current per-joint retained torque authority into each supplied
/// spatial contact Jacobian.
pub fn assess_humanoid_contact_wrench_controllability(
    dynamics: &FullRigidBodyDynamicsSnapshot,
    authority: &HumanoidJointTorqueAuthorityFrame,
    now_s: f64,
    config: ContactWrenchControllabilityConfig,
) -> Result<HumanoidActuationControllabilityAssessment, ActuationControllabilityError> {
    if !config.validate() {
        return Err(ActuationControllabilityError::InvalidConfig);
    }
    if !dynamics.validate() {
        return Err(ActuationControllabilityError::InvalidDynamics);
    }
    if !authority.validate() {
        return Err(ActuationControllabilityError::InvalidAuthorityFrame);
    }
    if dynamics.morphology != authority.morphology {
        return Err(ActuationControllabilityError::MorphologyMismatch);
    }
    if !now_s.is_finite() {
        return Err(ActuationControllabilityError::NonFiniteEvaluationTime);
    }
    if now_s < authority.sampled_at_s {
        return Err(ActuationControllabilityError::TimestampRegression);
    }
    let authority_age_s = now_s - authority.sampled_at_s;
    if authority_age_s > config.maximum_authority_age_s {
        return Err(ActuationControllabilityError::StaleAuthorityFrame);
    }

    let sites = dynamics
        .contacts
        .iter()
        .filter(|contact| contact.confidence >= config.minimum_contact_confidence)
        .map(|contact| assess_site(dynamics, authority, contact, config.jacobian_epsilon))
        .collect::<Vec<_>>();

    Ok(HumanoidActuationControllabilityAssessment {
        morphology: dynamics.morphology,
        authority_sequence: authority.sequence,
        authority_age_s,
        authority_profile_id: authority.authority_profile_id.clone(),
        calibration_fingerprint: authority.calibration_fingerprint,
        dynamics_model_id: dynamics.model_id.clone(),
        sites,
    })
}

fn assess_site(
    dynamics: &FullRigidBodyDynamicsSnapshot,
    authority: &HumanoidJointTorqueAuthorityFrame,
    contact: &SpatialContactJacobian,
    epsilon: f64,
) -> ContactWrenchMarginAssessment {
    let axes = ContactWrenchAxis::ALL
        .into_iter()
        .map(|axis| {
            assess_axis(
                axis,
                &contact.rows[axis.row()],
                &dynamics.torque_limits_nm,
                &authority.joints,
                epsilon,
            )
        })
        .collect::<Vec<_>>();
    let supported = axes
        .iter()
        .filter(|axis| axis.actuated_support_present)
        .map(|axis| axis.retained_fraction)
        .collect::<Vec<_>>();
    let minimum_retained_fraction = if supported.is_empty() {
        0.0
    } else {
        supported.into_iter().fold(1.0, f64::min)
    };
    ContactWrenchMarginAssessment {
        site_id: contact.site_id.clone(),
        contact_confidence: contact.confidence,
        jacobian_source: dynamics.provenance.contact_jacobians,
        actuator_limit_source: dynamics.provenance.actuator_limits,
        axes,
        minimum_retained_fraction,
    }
}

fn assess_axis(
    axis: ContactWrenchAxis,
    jacobian_row: &[f64],
    torque_limits_nm: &[f64],
    joints: &[JointTorqueAuthoritySample],
    epsilon: f64,
) -> ContactWrenchAxisMargin {
    let mut nominal_limit = f64::INFINITY;
    let mut retained_limit = f64::INFINITY;
    let mut limiting_joint = None;
    let mut supported = false;

    for (joint_index, ((coefficient, torque_limit), authority)) in jacobian_row
        .iter()
        .zip(torque_limits_nm.iter())
        .zip(joints.iter())
        .enumerate()
    {
        let coefficient = coefficient.abs();
        if coefficient <= epsilon {
            continue;
        }
        supported = true;
        let nominal_candidate = torque_limit / coefficient;
        let retained_candidate = torque_limit * authority.effective_fraction() / coefficient;
        nominal_limit = nominal_limit.min(nominal_candidate);
        if retained_candidate < retained_limit {
            retained_limit = retained_candidate;
            limiting_joint = Some(joint_index);
        }
    }

    if !supported || !nominal_limit.is_finite() || nominal_limit <= 0.0 {
        return ContactWrenchAxisMargin {
            axis,
            actuated_support_present: false,
            nominal_limit: None,
            retained_limit: None,
            retained_fraction: 0.0,
            limiting_joint: None,
        };
    }

    let retained_limit = if retained_limit.is_finite() {
        retained_limit.max(0.0)
    } else {
        0.0
    };
    ContactWrenchAxisMargin {
        axis,
        actuated_support_present: true,
        nominal_limit: Some(nominal_limit),
        retained_limit: Some(retained_limit),
        retained_fraction: (retained_limit / nominal_limit).clamp(0.0, 1.0),
        limiting_joint,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::full_dynamics::{
        DynamicsFidelity, DynamicsProvenance, FullRigidBodyDynamicsSnapshot,
        SpatialContactJacobian,
    };

    fn dynamics() -> FullRigidBodyDynamicsSnapshot {
        let morphology = HumanoidMorphology::Dmc21;
        let n = morphology.num_actuators();
        let mut mass = vec![0.0; n * n];
        for i in 0..n {
            mass[i * n + i] = 1.0;
        }
        let mut rows: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; n]);
        for row in 0..6 {
            rows[row][row] = 1.0;
        }
        FullRigidBodyDynamicsSnapshot {
            morphology,
            sampled_at_s: 1.0,
            total_mass_kg: 70.0,
            gravity_world_mps2: [0.0, 0.0, -9.81],
            mass_matrix: mass,
            bias_torque_nm: vec![0.0; n],
            torque_limits_nm: vec![100.0; n],
            centroidal_momentum_matrix: vec![0.0; 6 * n],
            contacts: vec![SpatialContactJacobian {
                site_id: "right_foot".into(),
                rows,
                confidence: 1.0,
            }],
            fidelity: DynamicsFidelity::SolverDerived,
            provenance: DynamicsProvenance::mujoco_solver_with_morphology_limits(),
            model_id: "controllability-test-model-v1".into(),
        }
    }

    fn authority() -> HumanoidJointTorqueAuthorityFrame {
        let morphology = HumanoidMorphology::Dmc21;
        HumanoidJointTorqueAuthorityFrame {
            morphology,
            sequence: 7,
            sampled_at_s: 1.0,
            authority_profile_id: "drive-health-test-v1".into(),
            calibration_fingerprint: 11,
            joints: vec![
                JointTorqueAuthoritySample {
                    evidence_valid: true,
                    retained_torque_fraction: 1.0,
                };
                morphology.num_actuators()
            ],
        }
    }

    fn config() -> ContactWrenchControllabilityConfig {
        ContactWrenchControllabilityConfig {
            maximum_authority_age_s: 0.1,
            minimum_contact_confidence: 0.5,
            jacobian_epsilon: 1.0e-9,
        }
    }

    #[test]
    fn nominal_joint_authority_preserves_axis_aligned_contact_limits() {
        let assessment = assess_humanoid_contact_wrench_controllability(
            &dynamics(),
            &authority(),
            1.05,
            config(),
        )
        .unwrap();
        let site = assessment.site("right_foot").unwrap();
        assert_eq!(site.axes.len(), 6);
        assert!(site.axes.iter().all(|axis| axis.retained_fraction == 1.0));
        assert_eq!(site.minimum_retained_fraction, 1.0);
    }

    #[test]
    fn one_degraded_joint_only_reduces_axes_that_depend_on_it() {
        let mut authority = authority();
        authority.joints[0].retained_torque_fraction = 0.4;
        let assessment = assess_humanoid_contact_wrench_controllability(
            &dynamics(),
            &authority,
            1.05,
            config(),
        )
        .unwrap();
        let site = assessment.site("right_foot").unwrap();
        let mx = site
            .axes
            .iter()
            .find(|axis| axis.axis == ContactWrenchAxis::MomentX)
            .unwrap();
        let my = site
            .axes
            .iter()
            .find(|axis| axis.axis == ContactWrenchAxis::MomentY)
            .unwrap();
        assert!((mx.retained_fraction - 0.4).abs() < 1.0e-12);
        assert_eq!(mx.limiting_joint, Some(0));
        assert_eq!(my.retained_fraction, 1.0);
    }

    #[test]
    fn invalid_joint_evidence_fails_closed_for_dependent_axis() {
        let mut authority = authority();
        authority.joints[3].evidence_valid = false;
        let assessment = assess_humanoid_contact_wrench_controllability(
            &dynamics(),
            &authority,
            1.05,
            config(),
        )
        .unwrap();
        let fx = assessment
            .site("right_foot")
            .unwrap()
            .axes
            .iter()
            .find(|axis| axis.axis == ContactWrenchAxis::ForceX)
            .unwrap();
        assert_eq!(fx.retained_fraction, 0.0);
        assert_eq!(fx.retained_limit, Some(0.0));
    }

    #[test]
    fn stale_joint_authority_is_rejected() {
        assert_eq!(
            assess_humanoid_contact_wrench_controllability(
                &dynamics(),
                &authority(),
                1.2,
                config(),
            ),
            Err(ActuationControllabilityError::StaleAuthorityFrame)
        );
    }

    #[test]
    fn morphology_mismatch_is_rejected() {
        let mut authority = authority();
        authority.morphology = HumanoidMorphology::Dexterous53;
        authority.joints = vec![
            JointTorqueAuthoritySample {
                evidence_valid: true,
                retained_torque_fraction: 1.0,
            };
            HumanoidMorphology::Dexterous53.num_actuators()
        ];
        assert_eq!(
            assess_humanoid_contact_wrench_controllability(
                &dynamics(),
                &authority,
                1.05,
                config(),
            ),
            Err(ActuationControllabilityError::MorphologyMismatch)
        );
    }

    #[test]
    fn unsupported_axis_is_not_promoted_to_full_authority() {
        let mut dynamics = dynamics();
        dynamics.contacts[0].rows[5].fill(0.0);
        let assessment = assess_humanoid_contact_wrench_controllability(
            &dynamics,
            &authority(),
            1.05,
            config(),
        )
        .unwrap();
        let fz = assessment
            .site("right_foot")
            .unwrap()
            .axes
            .iter()
            .find(|axis| axis.axis == ContactWrenchAxis::ForceZ)
            .unwrap();
        assert!(!fz.actuated_support_present);
        assert_eq!(fz.retained_fraction, 0.0);
        assert_eq!(fz.nominal_limit, None);
    }
}
