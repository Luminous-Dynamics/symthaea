// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent integrity verification for sealed complete contact evidence.
//!
//! HUM-WRENCH-001D5B2 deliberately does not call the D5B producer binder or its
//! private validator. It re-derives the authority-relevant contact-kinematics
//! and complete lineage identities from the sealed token's public typed surface.
//! Successful verification is evidence integrity only; it is not D4 contact
//! authority and it does not invoke a controller or QP.

use serde::{Deserialize, Serialize};

use crate::contact_authority::ContactEvidenceClassV1;
use crate::contact_kinematics::{ContactAccelerationBiasV1, ContactBiasAccelerationSource};
use crate::full_dynamics::SpatialContactJacobian;
use crate::mujoco_complete_contact_evidence::CompleteContactEstablishmentEvidenceV1;

const TIME_TOLERANCE_SCALE: f64 = 1.0e-9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompleteContactEvidenceVerificationError {
    InvalidSubjectIdentity,
    InvalidSupportEvidence,
    SourceClassMismatch,
    InvalidGeneralizedVelocityCount,
    InvalidContactJacobian,
    JacobianSiteMismatch,
    InvalidContactBias,
    BiasSourceMismatch,
    BiasSiteMismatch,
    BiasModelMismatch,
    BiasSampleTimeMismatch,
    KinematicsLineageMismatch,
    CompleteLineageMismatch,
}

/// Independently re-derived integrity receipt for one D5B token.
///
/// This receipt is serializable/deserializable because it is diagnostic evidence,
/// not a capability token. Replaying this receipt cannot establish contact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompleteContactEvidenceVerificationV1 {
    pub model_signature: u64,
    pub model_id: String,
    pub sampled_at_s: f64,
    pub support_site_name: String,
    pub generalized_velocity_count: usize,
    pub jacobian_scalar_count: usize,
    pub bias_component_count: usize,
    pub source_class: ContactEvidenceClassV1,
    pub recomputed_contact_kinematics_lineage_id: String,
    pub recomputed_complete_lineage_id: String,
    pub producer_kinematics_lineage_matches: bool,
    pub producer_complete_lineage_matches: bool,
}

impl CompleteContactEvidenceVerificationV1 {
    pub const fn verified(&self) -> bool {
        self.producer_kinematics_lineage_matches && self.producer_complete_lineage_matches
    }
}

/// Independently verify one sealed D5B complete-contact evidence token.
pub fn verify_complete_contact_evidence_v1(
    evidence: &CompleteContactEstablishmentEvidenceV1,
) -> Result<CompleteContactEvidenceVerificationV1, CompleteContactEvidenceVerificationError> {
    verify_components(
        evidence.model_signature(),
        evidence.model_id(),
        evidence.sampled_at_s(),
        evidence.support_site_name(),
        evidence.support().support_lineage_id(),
        evidence.support().surface_support_eligible(),
        evidence.source_class(),
        evidence.generalized_velocity_count(),
        evidence.contact_jacobian(),
        evidence.contact_bias(),
        evidence.contact_kinematics_lineage_id(),
        evidence.complete_lineage_id(),
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_components(
    model_signature: u64,
    model_id: &str,
    sampled_at_s: f64,
    support_site_name: &str,
    support_lineage_id: &str,
    support_eligible: bool,
    source_class: ContactEvidenceClassV1,
    generalized_velocity_count: usize,
    jacobian: &SpatialContactJacobian,
    bias: &ContactAccelerationBiasV1,
    producer_kinematics_lineage_id: &str,
    producer_complete_lineage_id: &str,
) -> Result<CompleteContactEvidenceVerificationV1, CompleteContactEvidenceVerificationError> {
    if model_id.trim().is_empty()
        || support_site_name.trim().is_empty()
        || support_lineage_id.trim().is_empty()
        || !sampled_at_s.is_finite()
        || sampled_at_s < 0.0
    {
        return Err(CompleteContactEvidenceVerificationError::InvalidSubjectIdentity);
    }
    if !support_eligible {
        return Err(CompleteContactEvidenceVerificationError::InvalidSupportEvidence);
    }
    if source_class != ContactEvidenceClassV1::SimulatorDerived {
        return Err(CompleteContactEvidenceVerificationError::SourceClassMismatch);
    }
    if generalized_velocity_count == 0 {
        return Err(CompleteContactEvidenceVerificationError::InvalidGeneralizedVelocityCount);
    }
    if !jacobian.validate(generalized_velocity_count) {
        return Err(CompleteContactEvidenceVerificationError::InvalidContactJacobian);
    }
    if jacobian.site_id.as_str() != support_site_name {
        return Err(CompleteContactEvidenceVerificationError::JacobianSiteMismatch);
    }
    if !bias.validate() {
        return Err(CompleteContactEvidenceVerificationError::InvalidContactBias);
    }
    if bias.source != ContactBiasAccelerationSource::SimulatorSolver {
        return Err(CompleteContactEvidenceVerificationError::BiasSourceMismatch);
    }
    if bias.site_id.as_str() != support_site_name {
        return Err(CompleteContactEvidenceVerificationError::BiasSiteMismatch);
    }
    if bias.model_id.as_str() != model_id {
        return Err(CompleteContactEvidenceVerificationError::BiasModelMismatch);
    }
    if !same_sample_time(bias.sampled_at_s, sampled_at_s) {
        return Err(CompleteContactEvidenceVerificationError::BiasSampleTimeMismatch);
    }

    let recomputed_kinematics = independent_contact_kinematics_lineage_id(
        model_signature,
        model_id,
        sampled_at_s,
        generalized_velocity_count,
        jacobian,
        bias,
    );
    if recomputed_kinematics != producer_kinematics_lineage_id {
        return Err(CompleteContactEvidenceVerificationError::KinematicsLineageMismatch);
    }

    let recomputed_complete = independent_complete_lineage_id(
        support_lineage_id,
        &recomputed_kinematics,
    );
    if recomputed_complete != producer_complete_lineage_id {
        return Err(CompleteContactEvidenceVerificationError::CompleteLineageMismatch);
    }

    Ok(CompleteContactEvidenceVerificationV1 {
        model_signature,
        model_id: model_id.to_string(),
        sampled_at_s,
        support_site_name: support_site_name.to_string(),
        generalized_velocity_count,
        jacobian_scalar_count: 6 * generalized_velocity_count,
        bias_component_count: bias.spatial_bias_acceleration.len(),
        source_class,
        recomputed_contact_kinematics_lineage_id: recomputed_kinematics,
        recomputed_complete_lineage_id: recomputed_complete,
        producer_kinematics_lineage_matches: true,
        producer_complete_lineage_matches: true,
    })
}

/// Independent copy of the D5B canonicalization rule. Keep this implementation
/// local so verifier success does not reduce to calling producer-private helpers.
fn independent_contact_kinematics_lineage_id(
    model_signature: u64,
    model_id: &str,
    sampled_at_s: f64,
    generalized_velocity_count: usize,
    jacobian: &SpatialContactJacobian,
    bias: &ContactAccelerationBiasV1,
) -> String {
    format!(
        "mujoco-contact-kinematics-v1:sig:{model_signature:016x}:model:{}:site:{}:time-bits:{:016x}:nv:{}:jacobian-confidence-bits:{:016x}:jacobian-bits:{}:bias-source:{:?}:bias-bits:{}",
        independent_component(model_id),
        independent_component(&jacobian.site_id),
        sampled_at_s.to_bits(),
        generalized_velocity_count,
        jacobian.confidence.to_bits(),
        independent_jacobian_bits(&jacobian.rows),
        bias.source,
        independent_f64_bits_list(&bias.spatial_bias_acceleration),
    )
}

fn independent_complete_lineage_id(
    support_lineage_id: &str,
    contact_kinematics_lineage_id: &str,
) -> String {
    format!(
        "complete-contact-establishment-evidence-v1:support:{}:contact-kinematics:{}",
        independent_component(support_lineage_id),
        independent_component(contact_kinematics_lineage_id),
    )
}

fn independent_component(value: &str) -> String {
    format!("{}:{}", value.len(), value)
}

fn independent_f64_bits_list(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{:016x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn independent_jacobian_bits(rows: &[Vec<f64>; 6]) -> String {
    rows.iter()
        .map(|row| independent_f64_bits_list(row))
        .collect::<Vec<_>>()
        .join(";")
}

fn same_sample_time(left: f64, right: f64) -> bool {
    let tolerance = TIME_TOLERANCE_SCALE * (1.0 + left.abs().max(right.abs()));
    (left - right).abs() <= tolerance
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::mujoco_complete_contact_evidence::bind_complete_mujoco_contact_evidence_v1;
    use crate::mujoco_contact_interaction::{
        MuJoCoContactInteractionError, extract_mujoco_contact_interaction,
    };
    use crate::mujoco_contact_kinematics::extract_verified_mujoco_contact_bias_evidence_v1;
    use crate::mujoco_contact_patch::extract_mujoco_foot_patch_set;
    use crate::mujoco_support_normal::{SupportNormalPolicyV1, qualify_support_normals};
    use crate::mujoco_verified_support::bind_verified_mujoco_surface_support_v1;
    use crate::multi_contact::ContactSite;
    use crate::simulator::{HumanoidPhysicsSimulator, MuJoCoHumanoidSimulator};
    use crate::types::HumanoidCommand;

    fn live_complete_evidence() -> CompleteContactEstablishmentEvidenceV1 {
        let mut sim = MuJoCoHumanoidSimulator::for_morphology(HumanoidMorphology::Dmc21).unwrap();
        sim.step(&HumanoidCommand::zero(), 0.0);
        let dynamics = sim.floating_base_dynamics_snapshot().unwrap();
        let dynamics_model_id = dynamics.model_id.clone();
        let model = Arc::clone(sim.model_arc());
        let patches = extract_mujoco_foot_patch_set(
            model.as_ref(),
            sim.data_mut(),
            &dynamics_model_id,
        )
        .unwrap();
        let policy = SupportNormalPolicyV1 {
            minimum_abs_alignment: 0.0,
        };

        let mut support = None;
        for site in [ContactSite::RightFoot, ContactSite::LeftFoot] {
            let interaction = match extract_mujoco_contact_interaction(
                model.as_ref(),
                sim.data_mut(),
                &patches,
                site,
            ) {
                Ok(record) => record,
                Err(MuJoCoContactInteractionError::NoActiveContact)
                | Err(MuJoCoContactInteractionError::NoActiveSupportContact) => continue,
                Err(error) => panic!("unexpected interaction extraction failure: {error:?}"),
            };
            let region = qualify_support_normals(&interaction, &patches, policy).unwrap();
            if region.surface_support_eligible() {
                support = Some(
                    bind_verified_mujoco_surface_support_v1(&patches, &interaction, &region)
                        .unwrap(),
                );
                break;
            }
        }
        let support = support.expect("expected generated DMC21 areal support");
        let bias = extract_verified_mujoco_contact_bias_evidence_v1(
            model.as_ref(),
            sim.data_mut(),
            dynamics,
        )
        .unwrap();
        bind_complete_mujoco_contact_evidence_v1(&support, &bias).unwrap()
    }

    #[test]
    fn independently_rederives_valid_complete_contact_token() {
        let evidence = live_complete_evidence();
        let verification = verify_complete_contact_evidence_v1(&evidence).unwrap();
        assert!(verification.verified());
        assert_eq!(
            verification.recomputed_contact_kinematics_lineage_id,
            evidence.contact_kinematics_lineage_id()
        );
        assert_eq!(
            verification.recomputed_complete_lineage_id,
            evidence.complete_lineage_id()
        );
        assert_eq!(
            verification.jacobian_scalar_count,
            6 * evidence.generalized_velocity_count()
        );
        assert_eq!(verification.bias_component_count, 6);
    }

    #[test]
    fn changed_jacobian_coefficient_or_confidence_changes_independent_identity() {
        let evidence = live_complete_evidence();
        let base = independent_contact_kinematics_lineage_id(
            evidence.model_signature(),
            evidence.model_id(),
            evidence.sampled_at_s(),
            evidence.generalized_velocity_count(),
            evidence.contact_jacobian(),
            evidence.contact_bias(),
        );
        let mut changed = evidence.contact_jacobian().clone();
        changed.rows[0][0] += 1.0e-9;
        let changed_coefficient = independent_contact_kinematics_lineage_id(
            evidence.model_signature(),
            evidence.model_id(),
            evidence.sampled_at_s(),
            evidence.generalized_velocity_count(),
            &changed,
            evidence.contact_bias(),
        );
        changed = evidence.contact_jacobian().clone();
        changed.confidence = (changed.confidence - 0.01).max(0.0);
        let changed_confidence = independent_contact_kinematics_lineage_id(
            evidence.model_signature(),
            evidence.model_id(),
            evidence.sampled_at_s(),
            evidence.generalized_velocity_count(),
            &changed,
            evidence.contact_bias(),
        );
        assert_ne!(base, changed_coefficient);
        assert_ne!(base, changed_confidence);
    }

    #[test]
    fn changed_bias_component_or_source_changes_independent_identity() {
        let evidence = live_complete_evidence();
        let base = independent_contact_kinematics_lineage_id(
            evidence.model_signature(),
            evidence.model_id(),
            evidence.sampled_at_s(),
            evidence.generalized_velocity_count(),
            evidence.contact_jacobian(),
            evidence.contact_bias(),
        );
        let mut changed = evidence.contact_bias().clone();
        changed.spatial_bias_acceleration[0] += 1.0e-9;
        let changed_component = independent_contact_kinematics_lineage_id(
            evidence.model_signature(),
            evidence.model_id(),
            evidence.sampled_at_s(),
            evidence.generalized_velocity_count(),
            evidence.contact_jacobian(),
            &changed,
        );
        changed = evidence.contact_bias().clone();
        changed.source = ContactBiasAccelerationSource::FiniteDifferenceOracle;
        let changed_source = independent_contact_kinematics_lineage_id(
            evidence.model_signature(),
            evidence.model_id(),
            evidence.sampled_at_s(),
            evidence.generalized_velocity_count(),
            evidence.contact_jacobian(),
            &changed,
        );
        assert_ne!(base, changed_component);
        assert_ne!(base, changed_source);
    }

    #[test]
    fn malformed_jacobian_and_non_solver_bias_fail_closed() {
        let evidence = live_complete_evidence();
        let mut malformed = evidence.contact_jacobian().clone();
        malformed.rows[0].pop();
        assert!(matches!(
            verify_components(
                evidence.model_signature(),
                evidence.model_id(),
                evidence.sampled_at_s(),
                evidence.support_site_name(),
                evidence.support().support_lineage_id(),
                true,
                evidence.source_class(),
                evidence.generalized_velocity_count(),
                &malformed,
                evidence.contact_bias(),
                evidence.contact_kinematics_lineage_id(),
                evidence.complete_lineage_id(),
            ),
            Err(CompleteContactEvidenceVerificationError::InvalidContactJacobian)
        ));

        let mut non_solver = evidence.contact_bias().clone();
        non_solver.source = ContactBiasAccelerationSource::FiniteDifferenceOracle;
        assert!(matches!(
            verify_components(
                evidence.model_signature(),
                evidence.model_id(),
                evidence.sampled_at_s(),
                evidence.support_site_name(),
                evidence.support().support_lineage_id(),
                true,
                evidence.source_class(),
                evidence.generalized_velocity_count(),
                evidence.contact_jacobian(),
                &non_solver,
                evidence.contact_kinematics_lineage_id(),
                evidence.complete_lineage_id(),
            ),
            Err(CompleteContactEvidenceVerificationError::BiasSourceMismatch)
        ));
    }

    #[test]
    fn signature_time_and_site_change_independent_identity() {
        let evidence = live_complete_evidence();
        let base = independent_contact_kinematics_lineage_id(
            evidence.model_signature(),
            evidence.model_id(),
            evidence.sampled_at_s(),
            evidence.generalized_velocity_count(),
            evidence.contact_jacobian(),
            evidence.contact_bias(),
        );
        let different_signature = independent_contact_kinematics_lineage_id(
            evidence.model_signature().wrapping_add(1),
            evidence.model_id(),
            evidence.sampled_at_s(),
            evidence.generalized_velocity_count(),
            evidence.contact_jacobian(),
            evidence.contact_bias(),
        );
        let different_time = independent_contact_kinematics_lineage_id(
            evidence.model_signature(),
            evidence.model_id(),
            evidence.sampled_at_s() + 1.0e-6,
            evidence.generalized_velocity_count(),
            evidence.contact_jacobian(),
            evidence.contact_bias(),
        );
        let mut different_site = evidence.contact_jacobian().clone();
        different_site.site_id.push_str("-other");
        let different_site = independent_contact_kinematics_lineage_id(
            evidence.model_signature(),
            evidence.model_id(),
            evidence.sampled_at_s(),
            evidence.generalized_velocity_count(),
            &different_site,
            evidence.contact_bias(),
        );
        assert_ne!(base, different_signature);
        assert_ne!(base, different_time);
        assert_ne!(base, different_site);
    }
}
