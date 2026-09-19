// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MuJoCo extraction for velocity-dependent contact acceleration bias.
//!
//! This module is deliberately separate from the authoritative humanoid
//! controller. It turns one already-validated floating-base MuJoCo snapshot
//! into the stronger bias-qualified subject required by
//! `J(q) qdd + Jdot(q, qdot) qdot = a*`.
//!
//! The extractor uses MuJoCo's `mj_jacDot` at the exact current `qpos/qvel` and
//! multiplies each derivative Jacobian by the same `qvel`. Simulator-derived
//! values remain explicitly simulator-solver evidence; they are not hardware
//! measurement or an independent numerical oracle.

use std::sync::Arc;

use mujoco_rs::prelude::{MjData, MjModel, MjtObj};
use serde::Serialize;

use crate::contact_kinematics::{
    BiasQualifiedFloatingBaseDynamicsV1, ContactAccelerationBiasV1,
    ContactBiasAccelerationSource,
};
use crate::floating_base::FloatingBaseDynamicsSnapshot;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MujocoContactBiasExtractionError {
    InvalidDynamicsSnapshot,
    ModelMismatch,
    GeneralizedVelocityCountMismatch,
    SampleTimeMismatch,
    MissingSite(String),
    InvalidSiteId(String),
    InvalidBodyId(String),
    NonFiniteGeneralizedVelocity,
    NonFiniteJacobianDerivative(String),
    QualifiedBindingRejected,
}

/// Sealed MuJoCo-specific contact-bias evidence.
///
/// `BiasQualifiedFloatingBaseDynamicsV1` binds model identity, time, contact
/// sites and complete `Jdot*qdot` records, but it is backend-neutral and does
/// not carry the MuJoCo compilation signature. This wrapper preserves that
/// exact simulator-model identity for later cross-lineage evidence binding.
/// It is serialize-only and has private fields so a deserialized payload cannot
/// manufacture a verified MuJoCo signature binding.
#[derive(Debug, Clone, Serialize)]
pub struct VerifiedMujocoContactBiasEvidenceV1 {
    model_signature: u64,
    qualified: BiasQualifiedFloatingBaseDynamicsV1,
}

impl VerifiedMujocoContactBiasEvidenceV1 {
    pub fn model_signature(&self) -> u64 {
        self.model_signature
    }

    pub fn qualified(&self) -> &BiasQualifiedFloatingBaseDynamicsV1 {
        &self.qualified
    }

    pub fn model_id(&self) -> &str {
        &self.qualified.dynamics.model_id
    }

    pub fn sampled_at_s(&self) -> f64 {
        self.qualified.dynamics.sampled_at_s
    }

    pub fn bias_for_site(&self, site_id: &str) -> Option<&ContactAccelerationBiasV1> {
        self.qualified.bias_for_site(site_id)
    }

    pub fn validate_binding(&self) -> bool {
        self.qualified.validate_binding().is_ok()
            && self
                .qualified
                .contact_biases
                .iter()
                .all(|bias| bias.source == ContactBiasAccelerationSource::SimulatorSolver)
    }
}

/// Extract a complete `Jdot qdot` evidence set for every contact declared by an
/// existing MuJoCo floating-base dynamics snapshot.
///
/// The returned wrapper re-runs the subject-binding checks from
/// [`BiasQualifiedFloatingBaseDynamicsV1`]. This function does not mutate
/// `MjData`, does not advance simulation time, and does not infer missing sites.
pub fn extract_mujoco_contact_bias_qualified_dynamics(
    model: &MjModel,
    data: &MjData<Arc<MjModel>>,
    dynamics: FloatingBaseDynamicsSnapshot,
) -> Result<BiasQualifiedFloatingBaseDynamicsV1, MujocoContactBiasExtractionError> {
    if !dynamics.validate() {
        return Err(MujocoContactBiasExtractionError::InvalidDynamicsSnapshot);
    }
    if model.signature() != data.model().signature() {
        return Err(MujocoContactBiasExtractionError::ModelMismatch);
    }

    let nv = model.ffi().nv as usize;
    if nv != dynamics.generalized_velocity_count || data.qvel().len() != nv {
        return Err(MujocoContactBiasExtractionError::GeneralizedVelocityCountMismatch);
    }
    if data.qvel().iter().any(|value| !value.is_finite()) {
        return Err(MujocoContactBiasExtractionError::NonFiniteGeneralizedVelocity);
    }
    if !same_sample_time(data.time(), dynamics.sampled_at_s) {
        return Err(MujocoContactBiasExtractionError::SampleTimeMismatch);
    }

    let nsite = model.ffi().nsite as usize;
    let site_body_ids = unsafe { std::slice::from_raw_parts(model.ffi().site_bodyid, nsite) };
    let site_positions = unsafe { std::slice::from_raw_parts(data.ffi().site_xpos, 3 * nsite) };
    let qvel = data.qvel();

    let mut contact_biases = Vec::with_capacity(dynamics.contacts.len());
    for contact in &dynamics.contacts {
        let site_id = model
            .name_to_id(MjtObj::mjOBJ_SITE, &contact.site_id)
            .ok_or_else(|| MujocoContactBiasExtractionError::MissingSite(contact.site_id.clone()))?;
        if site_id >= nsite {
            return Err(MujocoContactBiasExtractionError::InvalidSiteId(
                contact.site_id.clone(),
            ));
        }
        let body_id = site_body_ids[site_id];
        if body_id < 0 || body_id >= model.ffi().nbody {
            return Err(MujocoContactBiasExtractionError::InvalidBodyId(
                contact.site_id.clone(),
            ));
        }

        let point = &site_positions[3 * site_id..3 * site_id + 3];
        let mut linear_jacobian_dot = vec![0.0f64; 3 * nv];
        let mut angular_jacobian_dot = vec![0.0f64; 3 * nv];

        // SAFETY: all pointers originate from live MuJoCo wrappers; output
        // buffers are exactly 3*nv; `point` is exactly three world-frame
        // coordinates for the validated site; body_id is bounds-checked above.
        unsafe {
            mujoco_rs::mujoco_c::mj_jacDot(
                model.ffi(),
                data.ffi(),
                linear_jacobian_dot.as_mut_ptr(),
                angular_jacobian_dot.as_mut_ptr(),
                point.as_ptr(),
                body_id,
            );
        }

        if linear_jacobian_dot
            .iter()
            .chain(angular_jacobian_dot.iter())
            .any(|value| !value.is_finite())
        {
            return Err(MujocoContactBiasExtractionError::NonFiniteJacobianDerivative(
                contact.site_id.clone(),
            ));
        }

        let angular_bias = std::array::from_fn(|axis| {
            dot(&angular_jacobian_dot[axis * nv..(axis + 1) * nv], qvel)
        });
        let linear_bias = std::array::from_fn(|axis| {
            dot(&linear_jacobian_dot[axis * nv..(axis + 1) * nv], qvel)
        });

        contact_biases.push(ContactAccelerationBiasV1 {
            site_id: contact.site_id.clone(),
            spatial_bias_acceleration: [
                angular_bias[0],
                angular_bias[1],
                angular_bias[2],
                linear_bias[0],
                linear_bias[1],
                linear_bias[2],
            ],
            source: ContactBiasAccelerationSource::SimulatorSolver,
            sampled_at_s: dynamics.sampled_at_s,
            model_id: dynamics.model_id.clone(),
        });
    }

    let qualified = BiasQualifiedFloatingBaseDynamicsV1 {
        dynamics,
        contact_biases,
    };
    qualified
        .validate_binding()
        .map_err(|_| MujocoContactBiasExtractionError::QualifiedBindingRejected)?;
    Ok(qualified)
}

/// Stronger MuJoCo evidence token that retains the exact compiled-model
/// signature in addition to the backend-neutral qualified dynamics subject.
pub fn extract_verified_mujoco_contact_bias_evidence_v1(
    model: &MjModel,
    data: &MjData<Arc<MjModel>>,
    dynamics: FloatingBaseDynamicsSnapshot,
) -> Result<VerifiedMujocoContactBiasEvidenceV1, MujocoContactBiasExtractionError> {
    let model_signature = model.signature();
    let qualified = extract_mujoco_contact_bias_qualified_dynamics(model, data, dynamics)?;
    let result = VerifiedMujocoContactBiasEvidenceV1 {
        model_signature,
        qualified,
    };
    if result.validate_binding() {
        Ok(result)
    } else {
        Err(MujocoContactBiasExtractionError::QualifiedBindingRejected)
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum()
}

fn same_sample_time(left: f64, right: f64) -> bool {
    let tolerance = 1.0e-9 * (1.0 + left.abs().max(right.abs()));
    (left - right).abs() <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::simulator::{HumanoidPhysicsSimulator, MuJoCoHumanoidSimulator};

    #[test]
    fn zero_velocity_yields_explicit_near_zero_bias_evidence() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut sim = MuJoCoHumanoidSimulator::for_morphology(morphology).unwrap();
        let dynamics = sim.floating_base_dynamics_snapshot().unwrap();
        assert!(sim.data_mut().qvel().iter().all(|value| value.abs() < 1.0e-12));
        let model = Arc::clone(sim.model_arc());

        let qualified = extract_mujoco_contact_bias_qualified_dynamics(
            model.as_ref(),
            sim.data_mut(),
            dynamics,
        )
        .unwrap();

        assert!(qualified.validate_binding().is_ok());
        assert!(qualified.contact_biases.iter().all(|bias| {
            bias.source == ContactBiasAccelerationSource::SimulatorSolver
                && bias
                    .spatial_bias_acceleration
                    .iter()
                    .all(|value| value.abs() < 1.0e-10)
        }));
    }

    #[test]
    fn articulated_velocity_produces_same_subject_nonzero_bias() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut sim = MuJoHumanoidSimulator::for_morphology(morphology).unwrap();
        let qpos = sim.data_mut().qpos().to_vec();
        let nv = sim.model_arc().ffi().nv as usize;
        let mut qvel = vec![0.0; nv];
        qvel[crate::floating_base::FLOATING_BASE_DOF] = 0.7;
        qvel[crate::floating_base::FLOATING_BASE_DOF + 1] = -0.4;
        sim.set_generalized_state(&qpos, &qvel).unwrap();

        let dynamics = sim.floating_base_dynamics_snapshot().unwrap();
        let expected_model_id = dynamics.model_id.clone();
        let expected_time = dynamics.sampled_at_s;
        let model = Arc::clone(sim.model_arc());
        let qualified = extract_mujoco_contact_bias_qualified_dynamics(
            model.as_ref(),
            sim.data_mut(),
            dynamics,
        )
        .unwrap();

        assert!(qualified.validate_binding().is_ok());
        assert!(qualified.contact_biases.iter().all(|bias| {
            bias.model_id == expected_model_id
                && (bias.sampled_at_s - expected_time).abs() < 1.0e-12
        }));
        assert!(qualified.contact_biases.iter().any(|bias| {
            bias.spatial_bias_acceleration
                .iter()
                .any(|value| value.abs() > 1.0e-10)
        }));
    }

    #[test]
    fn verified_wrapper_retains_exact_mujoco_model_signature() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut sim = MuJoCoHumanoidSimulator::for_morphology(morphology).unwrap();
        let dynamics = sim.floating_base_dynamics_snapshot().unwrap();
        let expected_model_id = dynamics.model_id.clone();
        let expected_time = dynamics.sampled_at_s;
        let model = Arc::clone(sim.model_arc());
        let expected_signature = model.signature();

        let evidence = extract_verified_mujoco_contact_bias_evidence_v1(
            model.as_ref(),
            sim.data_mut(),
            dynamics,
        )
        .unwrap();

        assert!(evidence.validate_binding());
        assert_eq!(evidence.model_signature(), expected_signature);
        assert_eq!(evidence.model_id(), expected_model_id);
        assert!((evidence.sampled_at_s() - expected_time).abs() < 1.0e-12);
        assert!(evidence
            .qualified()
            .contact_biases
            .iter()
            .all(|bias| bias.source == ContactBiasAccelerationSource::SimulatorSolver));
    }

    #[test]
    fn snapshot_from_different_sample_time_is_rejected() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut sim = MuJoCoHumanoidSimulator::for_morphology(morphology).unwrap();
        let mut dynamics = sim.floating_base_dynamics_snapshot().unwrap();
        dynamics.sampled_at_s += 1.0;
        let model = Arc::clone(sim.model_arc());

        let error = extract_mujoco_contact_bias_qualified_dynamics(
            model.as_ref(),
            sim.data_mut(),
            dynamics,
        )
        .unwrap_err();
        assert_eq!(error, MujocoContactBiasExtractionError::SampleTimeMismatch);
    }

    #[test]
    fn missing_contact_site_fails_closed_under_pinned_lookup_api() {
        let morphology = HumanoidMorphology::Dmc21;
        let mut sim = MuJoCoHumanoidSimulator::for_morphology(morphology).unwrap();
        let mut dynamics = sim.floating_base_dynamics_snapshot().unwrap();
        dynamics.contacts[0].site_id = "__missing_contact_site__".to_string();
        assert!(dynamics.validate());
        let model = Arc::clone(sim.model_arc());

        let error = extract_mujoco_contact_bias_qualified_dynamics(
            model.as_ref(),
            sim.data_mut(),
            dynamics,
        )
        .unwrap_err();
        assert_eq!(
            error,
            MujocoContactBiasExtractionError::MissingSite("__missing_contact_site__".to_string())
        );
    }
}
