// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complete simulator contact-establishment evidence before runtime authority.
//!
//! HUM-WRENCH-001D5B combines the sealed D5A surface-support token with the
//! sealed, signature-retaining HUM-DYN-002C MuJoCo contact-bias token. The
//! result proves that one exact simulator subject has both current support
//! evidence and explicit `Jdot(q, qdot) qdot` evidence for the same support
//! site. It deliberately does not mint D4 `Established` authority or touch a QP.

use serde::Serialize;

use crate::contact_authority::ContactEvidenceClassV1;
use crate::contact_kinematics::{ContactAccelerationBiasV1, ContactBiasAccelerationSource};
use crate::mujoco_contact_kinematics::VerifiedMujocoContactBiasEvidenceV1;
use crate::mujoco_verified_support::VerifiedSurfaceSupportEvidenceV1;
use crate::multi_contact::ContactSite;

const TIME_TOLERANCE_SCALE: f64 = 1.0e-9;

/// Sealed complete evidence for one simulator support contact.
///
/// This is still evidence, not runtime contact authority. Fields are private and
/// the token is serialize-only so external code cannot deserialize or fill a
/// supposedly complete evidence object.
#[derive(Debug, Clone, Serialize)]
pub struct CompleteContactEstablishmentEvidenceV1 {
    support: VerifiedSurfaceSupportEvidenceV1,
    contact_bias: ContactAccelerationBiasV1,
    contact_bias_lineage_id: String,
    complete_lineage_id: String,
    source_class: ContactEvidenceClassV1,
}

impl CompleteContactEstablishmentEvidenceV1 {
    pub fn site(&self) -> ContactSite {
        self.support.site()
    }

    pub fn model_id(&self) -> &str {
        self.support.model_id()
    }

    pub fn model_signature(&self) -> u64 {
        self.support.model_signature()
    }

    pub fn sampled_at_s(&self) -> f64 {
        self.support.sampled_at_s()
    }

    pub fn support_site_name(&self) -> &str {
        self.support.support_site_name()
    }

    pub fn support_site_id(&self) -> usize {
        self.support.support_site_id()
    }

    pub fn physical_geom_name(&self) -> &str {
        self.support.physical_geom_name()
    }

    pub fn physical_geom_id(&self) -> usize {
        self.support.physical_geom_id()
    }

    pub fn support(&self) -> &VerifiedSurfaceSupportEvidenceV1 {
        &self.support
    }

    pub fn contact_bias(&self) -> &ContactAccelerationBiasV1 {
        &self.contact_bias
    }

    pub fn contact_bias_lineage_id(&self) -> &str {
        &self.contact_bias_lineage_id
    }

    pub fn complete_lineage_id(&self) -> &str {
        &self.complete_lineage_id
    }

    pub fn source_class(&self) -> ContactEvidenceClassV1 {
        self.source_class
    }

    /// The token carries the explicit velocity-dependent spatial contact-bias
    /// term needed for `J qdd + Jdot qdot = a*`.
    pub const fn has_complete_contact_bias_evidence(&self) -> bool {
        true
    }

    fn validate(&self) -> bool {
        self.support.surface_support_eligible()
            && self.source_class == ContactEvidenceClassV1::SimulatorDerived
            && self.support.source_class() == ContactEvidenceClassV1::SimulatorDerived
            && self.contact_bias.source == ContactBiasAccelerationSource::SimulatorSolver
            && self.contact_bias.site_id == self.support.support_site_name()
            && self.contact_bias.model_id == self.support.model_id()
            && same_sample_time(self.contact_bias.sampled_at_s, self.support.sampled_at_s())
            && self.contact_bias.validate()
            && !self.contact_bias_lineage_id.trim().is_empty()
            && !self.complete_lineage_id.trim().is_empty()
            && self.contact_bias_lineage_id
                == contact_bias_lineage_id(self.support.model_signature(), &self.contact_bias)
            && self.complete_lineage_id
                == complete_lineage_id(
                    self.support.support_lineage_id(),
                    &self.contact_bias_lineage_id,
                )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompleteContactEvidenceError {
    InvalidSupportEvidence,
    InvalidBiasEvidence,
    SourceClassMismatch,
    ModelSignatureMismatch,
    ModelIdentityMismatch,
    SampleTimeMismatch,
    MissingBiasSite,
    BiasSourceMismatch,
    BiasModelMismatch,
    BiasSampleTimeMismatch,
    InvalidResult,
}

/// Bind exact D5A support evidence and exact signature-retaining DYN evidence.
///
/// Crate-internal by design: this token is an intermediate capability for a
/// later D4 minting adapter, not a public constructor for contact authority.
#[allow(dead_code, reason = "A later D4 adapter will consume the complete evidence token")]
pub(crate) fn bind_complete_mujoco_contact_evidence_v1(
    support: &VerifiedSurfaceSupportEvidenceV1,
    bias_evidence: &VerifiedMujocoContactBiasEvidenceV1,
) -> Result<CompleteContactEstablishmentEvidenceV1, CompleteContactEvidenceError> {
    if !support.surface_support_eligible() {
        return Err(CompleteContactEvidenceError::InvalidSupportEvidence);
    }
    if support.source_class() != ContactEvidenceClassV1::SimulatorDerived {
        return Err(CompleteContactEvidenceError::SourceClassMismatch);
    }
    if !bias_evidence.validate_binding() {
        return Err(CompleteContactEvidenceError::InvalidBiasEvidence);
    }
    validate_subject_binding(
        support.model_signature(),
        bias_evidence.model_signature(),
        support.model_id(),
        bias_evidence.model_id(),
        support.sampled_at_s(),
        bias_evidence.sampled_at_s(),
    )?;

    let contact_bias = bias_evidence
        .bias_for_site(support.support_site_name())
        .ok_or(CompleteContactEvidenceError::MissingBiasSite)?;
    if contact_bias.source != ContactBiasAccelerationSource::SimulatorSolver {
        return Err(CompleteContactEvidenceError::BiasSourceMismatch);
    }
    if contact_bias.model_id != support.model_id() {
        return Err(CompleteContactEvidenceError::BiasModelMismatch);
    }
    if !same_sample_time(contact_bias.sampled_at_s, support.sampled_at_s()) {
        return Err(CompleteContactEvidenceError::BiasSampleTimeMismatch);
    }

    let bias_lineage = contact_bias_lineage_id(support.model_signature(), contact_bias);
    let complete_lineage = complete_lineage_id(support.support_lineage_id(), &bias_lineage);
    let result = CompleteContactEstablishmentEvidenceV1 {
        support: support.clone(),
        contact_bias: contact_bias.clone(),
        contact_bias_lineage_id: bias_lineage,
        complete_lineage_id: complete_lineage,
        source_class: ContactEvidenceClassV1::SimulatorDerived,
    };
    if result.validate() {
        Ok(result)
    } else {
        Err(CompleteContactEvidenceError::InvalidResult)
    }
}

fn validate_subject_binding(
    support_signature: u64,
    bias_signature: u64,
    support_model_id: &str,
    bias_model_id: &str,
    support_time_s: f64,
    bias_time_s: f64,
) -> Result<(), CompleteContactEvidenceError> {
    if support_signature != bias_signature {
        return Err(CompleteContactEvidenceError::ModelSignatureMismatch);
    }
    if support_model_id != bias_model_id {
        return Err(CompleteContactEvidenceError::ModelIdentityMismatch);
    }
    if !same_sample_time(support_time_s, bias_time_s) {
        return Err(CompleteContactEvidenceError::SampleTimeMismatch);
    }
    Ok(())
}

fn contact_bias_lineage_id(model_signature: u64, bias: &ContactAccelerationBiasV1) -> String {
    format!(
        "mujoco-contact-bias-v1:sig:{model_signature:016x}:model:{}:site:{}:time-bits:{:016x}:source:{:?}:bias-bits:{}",
        component(&bias.model_id),
        component(&bias.site_id),
        bias.sampled_at_s.to_bits(),
        bias.source,
        f64_bits_list(&bias.spatial_bias_acceleration),
    )
}

fn complete_lineage_id(support_lineage_id: &str, contact_bias_lineage_id: &str) -> String {
    format!(
        "complete-contact-establishment-evidence-v1:support:{}:contact-bias:{}",
        component(support_lineage_id),
        component(contact_bias_lineage_id),
    )
}

fn component(value: &str) -> String {
    format!("{}:{}", value.len(), value)
}

fn f64_bits_list(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{:016x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
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
    use crate::mujoco_contact_interaction::{
        MuJoCoContactInteractionError, extract_mujoco_contact_interaction,
    };
    use crate::mujoco_contact_kinematics::extract_verified_mujoco_contact_bias_evidence_v1;
    use crate::mujoco_contact_patch::extract_mujoco_foot_patch_set;
    use crate::mujoco_support_normal::{SupportNormalPolicyV1, qualify_support_normals};
    use crate::mujoco_verified_support::bind_verified_mujoco_surface_support_v1;
    use crate::simulator::{HumanoidPhysicsSimulator, MuJoCoHumanoidSimulator};
    use crate::types::HumanoidCommand;

    fn live_inputs_with_support_model_id(
        support_model_id_override: Option<&str>,
    ) -> (
        VerifiedSurfaceSupportEvidenceV1,
        VerifiedMujocoContactBiasEvidenceV1,
    ) {
        let mut sim = MuJoCoHumanoidSimulator::for_morphology(HumanoidMorphology::Dmc21).unwrap();
        sim.step(&HumanoidCommand::zero(), 0.0);
        let dynamics = sim.floating_base_dynamics_snapshot().unwrap();
        let dynamics_model_id = dynamics.model_id.clone();
        let support_model_id = support_model_id_override.unwrap_or(&dynamics_model_id);
        let model = Arc::clone(sim.model_arc());
        let patches = extract_mujoco_foot_patch_set(
            model.as_ref(),
            sim.data_mut(),
            support_model_id,
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
        let support = support.expect("expected a non-degenerate generated DMC21 support region");
        let bias = extract_verified_mujoco_contact_bias_evidence_v1(
            model.as_ref(),
            sim.data_mut(),
            dynamics,
        )
        .unwrap();
        (support, bias)
    }

    #[test]
    fn same_subject_support_and_contact_bias_bind() {
        let (support, bias) = live_inputs_with_support_model_id(None);
        let complete = bind_complete_mujoco_contact_evidence_v1(&support, &bias).unwrap();
        assert!(complete.has_complete_contact_bias_evidence());
        assert_eq!(complete.model_signature(), bias.model_signature());
        assert_eq!(complete.model_id(), bias.model_id());
        assert_eq!(complete.support_site_name(), complete.contact_bias().site_id);
        assert_eq!(complete.source_class(), ContactEvidenceClassV1::SimulatorDerived);
        assert!(!complete.complete_lineage_id().is_empty());
    }

    #[test]
    fn model_identity_mismatch_fails_closed() {
        let (support, bias) = live_inputs_with_support_model_id(Some("different-model-id"));
        assert_eq!(
            bind_complete_mujoco_contact_evidence_v1(&support, &bias),
            Err(CompleteContactEvidenceError::ModelIdentityMismatch)
        );
    }

    #[test]
    fn subject_guard_rejects_signature_and_time_mismatch() {
        assert_eq!(
            validate_subject_binding(1, 2, "model", "model", 3.0, 3.0),
            Err(CompleteContactEvidenceError::ModelSignatureMismatch)
        );
        assert_eq!(
            validate_subject_binding(1, 1, "model", "model", 3.0, 3.1),
            Err(CompleteContactEvidenceError::SampleTimeMismatch)
        );
    }

    #[test]
    fn missing_support_site_bias_fails_closed() {
        let (support, _) = live_inputs_with_support_model_id(None);
        let mut sim = MuJoCoHumanoidSimulator::for_morphology(HumanoidMorphology::Dmc21).unwrap();
        sim.step(&HumanoidCommand::zero(), 0.0);
        let mut dynamics = sim.floating_base_dynamics_snapshot().unwrap();
        dynamics.contacts.retain(|contact| contact.site_id != support.support_site_name());
        assert!(dynamics.validate());
        let model = Arc::clone(sim.model_arc());
        let bias = extract_verified_mujoco_contact_bias_evidence_v1(
            model.as_ref(),
            sim.data_mut(),
            dynamics,
        )
        .unwrap();
        assert_eq!(
            bind_complete_mujoco_contact_evidence_v1(&support, &bias),
            Err(CompleteContactEvidenceError::MissingBiasSite)
        );
    }

    #[test]
    fn bias_lineage_binds_exact_vector_bits_and_source() {
        let (support, bias_evidence) = live_inputs_with_support_model_id(None);
        let bias = bias_evidence.bias_for_site(support.support_site_name()).unwrap();
        let base = contact_bias_lineage_id(support.model_signature(), bias);
        let mut changed = bias.clone();
        changed.spatial_bias_acceleration[0] += 1.0e-6;
        let changed_vector = contact_bias_lineage_id(support.model_signature(), &changed);
        changed = bias.clone();
        changed.source = ContactBiasAccelerationSource::FiniteDifferenceOracle;
        let changed_source = contact_bias_lineage_id(support.model_signature(), &changed);
        assert_ne!(base, changed_vector);
        assert_ne!(base, changed_source);
    }
}
